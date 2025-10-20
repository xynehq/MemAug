#!/usr/bin/env python3
"""
Training Script for MemAug with AST-based External Memory.
Uses the generated JSONL datasets and AST data as memory context.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel,
    HybridTransformerConfig
)


@dataclass
class ASTTrainingConfig:
    """Training configuration for AST-based memory."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # External Memory
    use_external_memory: bool = True
    external_memory_size: int = 2048
    retrieval_k: int = 4
    chunk_size: int = 4
    
    # Training
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 5e-5
    external_memory_lr: float = 1e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8
    
    # Data
    max_seq_length: int = 2048
    dataset_dir: str = "data/ast_dataset/bat"
    
    # Output
    output_dir: str = "outputs/ast_memory_training"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    
    # Device
    device: str = "auto"
    seed: int = 42


class ASTMemoryDataset(Dataset):
    """Dataset that loads JSONL sequences with AST memory context."""
    
    def __init__(self, dataset_dir: str, tokenizer, max_length: int = 2048, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = self._load_data(dataset_dir)
    
    def _load_data(self, dataset_dir: str) -> List[Dict]:
        """Load all dataset.jsonl files from commit directories."""
        data = []
        dataset_path = Path(dataset_dir)
        
        # Find all dataset.jsonl files
        for commit_dir in sorted(dataset_path.glob("commit_*")):
            dataset_file = commit_dir / "dataset.jsonl"
            ast_file = commit_dir / "ast.jsonl"
            
            if not dataset_file.exists():
                continue
            
            # Load AST data for this commit
            ast_data = {}
            if ast_file.exists():
                with open(ast_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            node = json.loads(line)
                            ast_data[node['id']] = node
            
            # Load sequences
            with open(dataset_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sequence = json.loads(line)
                        sequence['commit_dir'] = str(commit_dir)
                        sequence['ast_data'] = ast_data
                        data.append(sequence)
        
        # Split data (80/20 train/val)
        split_idx = int(len(data) * 0.8)
        if self.split == "train":
            data = data[:split_idx]
        else:
            data = data[split_idx:]
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        # Build input/output from FIM samples
        functions = sequence['functions']
        retrieval_labels = sequence.get('retrieval_labels', {})
        ast_data = sequence.get('ast_data', {})
        
        # Concatenate all FIM samples
        full_input = ""
        full_output = ""
        
        for func in functions:
            # FIM format: prefix + suffix -> middle
            full_input += f"<|fim_prefix|>{func['fim_prefix']}<|fim_suffix|>{func['fim_suffix']}<|fim_middle|>"
            full_output += func['fim_middle']
        
        # Tokenize
        input_text = full_input
        output_text = full_output
        full_text = input_text + output_text
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        
        # Create labels (mask input portion)
        labels = input_ids.clone()
        input_tokens = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
        input_length = min(len(input_tokens), len(labels))
        labels[:input_length] = -100
        
        # Prepare AST memory context (as string)
        ast_memory_str = self._prepare_ast_memory(functions, retrieval_labels, ast_data)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ast_memory": ast_memory_str,
            "retrieval_labels": retrieval_labels
        }
    
    def _prepare_ast_memory(self, functions: List[Dict], retrieval_labels: Dict, ast_data: Dict) -> str:
        """Prepare AST data as memory context string."""
        memory_parts = []
        
        for func in functions:
            func_id = func['function_id']
            
            # Add function's AST node
            if func_id in ast_data:
                node = ast_data[func_id]
                memory_parts.append(json.dumps({
                    'id': node['id'],
                    'kind': node['kind'],
                    'code': node.get('code', '')[:200]  # Truncate long code
                }))
            
            # Add retrieval labels (positive samples)
            if func_id in retrieval_labels:
                labels = retrieval_labels[func_id]
                for pos_id in labels.get('positive', []):
                    if pos_id in ast_data and pos_id != func_id:
                        node = ast_data[pos_id]
                        memory_parts.append(json.dumps({
                            'id': node['id'],
                            'kind': node['kind'],
                            'code': node.get('code', '')[:200]
                        }))
        
        return "\n".join(memory_parts[:20])  # Limit to 20 entries


class ASTMemoryTrainer:
    """Trainer with AST-based external memory."""
    
    def __init__(self, config: ASTTrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def setup_logging(self):
        """Setup logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device and seed."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        set_seed(self.config.seed)
        self.logger.info(f"Using device: {self.device}")
    
    def setup_model(self):
        """Setup model."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA
        if self.config.use_lora and PEFT_AVAILABLE:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            base_model = get_peft_model(base_model, lora_config)
            base_model.print_trainable_parameters()
        
        # Create hybrid config
        base_config_dict = base_model.config.to_dict()
        hybrid_config = HybridTransformerConfig(
            **base_config_dict,
            use_internal_memory=False,
            use_external_memory=self.config.use_external_memory,
            external_memory_size=self.config.external_memory_size,
            retrieval_k=self.config.retrieval_k,
            chunk_size=self.config.chunk_size,
            use_gpu_to_search=False,
            use_gated_retrieval=True
        )
        
        # Create hybrid model
        self.model = HybridTransformerModel(
            config=hybrid_config,
            tokenizer=self.tokenizer,
            base_model=base_model
        )
        
        self.model = self.model.to(self.device)
        
        # Enable external memory gradients
        if self.model.external_memory is not None:
            for i in range(len(self.model.external_memory.keys)):
                self.model.external_memory.keys[i].requires_grad = True
                self.model.external_memory.vals[i].requires_grad = True
        
        self.logger.info(f"Model initialized: {self.model.get_model_info()}")
    
    def setup_data(self):
        """Setup data loaders."""
        train_dataset = ASTMemoryDataset(
            self.config.dataset_dir,
            self.tokenizer,
            self.config.max_seq_length,
            split="train"
        )
        
        val_dataset = ASTMemoryDataset(
            self.config.dataset_dir,
            self.tokenizer,
            self.config.max_seq_length,
            split="val"
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
    
    def setup_optimizer(self):
        """Setup optimizer."""
        external_memory_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'external_memory' in name.lower() or 'keys' in name or 'vals' in name:
                    external_memory_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = []
        if external_memory_params:
            param_groups.append({
                'params': external_memory_params,
                'lr': self.config.external_memory_lr
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate
            })
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step."""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Use external memory on accumulation boundaries
        should_use_memory = (self.global_step + 1) % self.config.gradient_accumulation_steps == 0
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            targets=labels,
            attention_mask=attention_mask,
            use_external_memory=self.config.use_external_memory and should_use_memory
        )
        
        # Extract loss
        if isinstance(outputs, tuple):
            logits, loss, memory_info = outputs
        else:
            loss = outputs.loss
            logits = outputs.logits
        
        if loss is None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            valid_mask = (shift_labels != -100)
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
            else:
                return {"train_loss": 0.0, "skipped": True}
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient step
        metrics = {"train_loss": loss.item() * self.config.gradient_accumulation_steps}
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm
            )
            metrics["grad_norm"] = grad_norm.item()
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    targets=labels,
                    attention_mask=attention_mask,
                    use_external_memory=self.config.use_external_memory
                )
                
                if isinstance(outputs, tuple):
                    logits, loss, memory_info = outputs
                else:
                    loss = outputs.loss
                    logits = outputs.logits
                
                if loss is None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        perplexity = np.exp(avg_loss)
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity
        }
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pt"))
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        if is_best:
            best_dir = os.path.join(self.config.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(best_dir, "checkpoint.pt"))
            self.tokenizer.save_pretrained(best_dir)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(self.train_dataloader):
                # Training step
                metrics = self.train_step(batch)
                
                # Skip if batch was invalid
                if metrics.get("skipped", False):
                    continue
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    log_str = f"Step {self.global_step}: loss={metrics['train_loss']:.4f}"
                    if "grad_norm" in metrics:
                        log_str += f", grad_norm={metrics['grad_norm']:.4f}"
                    if "learning_rate" in metrics:
                        log_str += f", lr={metrics['learning_rate']:.2e}"
                    self.logger.info(log_str)
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.global_step > 0:
                    eval_metrics = self.evaluate()
                    self.logger.info(f"Evaluation: {eval_metrics}")
                    
                    # Save best model
                    is_best = eval_metrics["eval_loss"] < self.best_loss
                    if is_best:
                        self.best_loss = eval_metrics["eval_loss"]
                        self.logger.info(f"New best loss: {self.best_loss:.4f}")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step, is_best)
                
                self.global_step += 1
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            self.logger.info(f"End of epoch {epoch + 1}: {eval_metrics}")
        
        # Final save
        self.save_checkpoint(self.global_step, is_best=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AST Memory Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="data/ast_dataset/bat")
    parser.add_argument("--output_dir", type=str, default="outputs/ast_memory_training")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--config_file", type=str, help="Path to config JSON")
    
    args = parser.parse_args()
    
    # Load config
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = ASTTrainingConfig(**config_dict)
    else:
        config = ASTTrainingConfig()
    
    # Override with CLI args
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Create trainer and train
    trainer = ASTMemoryTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
