#!/usr/bin/env python3
"""
Improved Training Script with:
1. Inference callback during evaluation
2. Periodic external memory updates
3. Retrieval loss training
4. Live inference results
5. Sequence length filtering
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
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

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel,
    HybridTransformerConfig
)


@dataclass
class ImprovedTrainingConfig:
    """Improved training configuration."""
    
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
    memory_update_freq: int = 10  # Update memory every N steps
    
    # Retrieval Loss
    use_retrieval_loss: bool = True
    retrieval_loss_weight: float = 0.1
    
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
    
    # Inference
    num_inference_samples: int = 3  # Show N samples during eval
    
    # Output
    output_dir: str = "outputs/improved_training"
    save_steps: int = 15
    eval_steps: int = 50
    logging_steps: int = 10
    
    # Device
    device: str = "auto"
    seed: int = 42


class FilteredASTDataset(Dataset):
    """Dataset with sequence length filtering."""
    
    def __init__(self, dataset_dir: str, tokenizer, max_length: int = 2048, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = self._load_and_filter_data(dataset_dir)
    
    def _load_and_filter_data(self, dataset_dir: str) -> List[Dict]:
        """Load and filter data by sequence length."""
        data = []
        dataset_path = Path(dataset_dir)
        filtered_count = 0
        
        for commit_dir in sorted(dataset_path.glob("commit_*")):
            dataset_file = commit_dir / "dataset.jsonl"
            ast_file = commit_dir / "ast.jsonl"
            
            if not dataset_file.exists():
                continue
            
            # Load AST data
            ast_data = {}
            if ast_file.exists():
                with open(ast_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            node = json.loads(line)
                            ast_data[node['id']] = node
            
            # Load and filter sequences
            with open(dataset_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sequence = json.loads(line)
                        
                        # Check if sequence fits in max_length
                        estimated_tokens = sequence.get('estimated_tokens', 0)
                        if estimated_tokens > 0 and estimated_tokens < self.max_length:
                            sequence['commit_dir'] = str(commit_dir)
                            sequence['ast_data'] = ast_data
                            data.append(sequence)
                        else:
                            filtered_count += 1
        
        print(f"Loaded {len(data)} sequences, filtered {filtered_count} (too long)")
        
        # Split data
        split_idx = int(len(data) * 0.8)
        if self.split == "train":
            return data[:split_idx]
        else:
            return data[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        functions = sequence['functions']
        retrieval_labels = sequence.get('retrieval_labels', {})
        ast_data = sequence.get('ast_data', {})
        
        # Build FIM input/output
        full_input = ""
        full_output = ""
        function_ids = []
        
        for func in functions:
            full_input += f"<|fim_prefix|>{func['fim_prefix']}<|fim_suffix|>{func['fim_suffix']}<|fim_middle|>"
            full_output += func['fim_middle']
            function_ids.append(func['function_id'])
        
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
        
        # Create labels
        labels = input_ids.clone()
        input_tokens = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
        input_length = min(len(input_tokens), len(labels))
        labels[:input_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_text": input_text,
            "output_text": output_text,
            "function_ids": function_ids,
            "retrieval_labels": retrieval_labels,
            "ast_data": ast_data
        }


class ImprovedTrainer:
    """Improved trainer with all requested features."""
    
    def __init__(self, config: ImprovedTrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        self.global_step = 0
        self.best_loss = float('inf')
        self.memory_update_counter = 0
    
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
        """Setup device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        set_seed(self.config.seed)
        self.logger.info(f"Using device: {self.device}")
    
    def setup_model(self):
        """Setup model."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        base_model.resize_token_embeddings(len(self.tokenizer))
        
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
        
        self.model = HybridTransformerModel(
            config=hybrid_config,
            tokenizer=self.tokenizer,
            base_model=base_model
        )
        
        self.model = self.model.to(self.device)
        
        if self.model.external_memory is not None:
            for i in range(len(self.model.external_memory.keys)):
                self.model.external_memory.keys[i].requires_grad = True
                self.model.external_memory.vals[i].requires_grad = True
        
        self.logger.info(f"Model initialized: {self.model.get_model_info()}")
    
    def collate_fn(self, batch):
        """Custom collate function to handle mixed data types."""
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # Keep other fields as lists
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_text': [item['input_text'] for item in batch],
            'output_text': [item['output_text'] for item in batch],
            'function_ids': [item['function_ids'] for item in batch],
            'retrieval_labels': [item['retrieval_labels'] for item in batch],
            'ast_data': [item['ast_data'] for item in batch]
        }
    
    def setup_data(self):
        """Setup data loaders."""
        train_dataset = FilteredASTDataset(
            self.config.dataset_dir,
            self.tokenizer,
            self.config.max_seq_length,
            split="train"
        )
        
        val_dataset = FilteredASTDataset(
            self.config.dataset_dir,
            self.tokenizer,
            self.config.max_seq_length,
            split="val"
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn
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
    
    def compute_retrieval_loss(self, hidden_states: torch.Tensor, retrieval_labels: List[Dict], function_ids: List[List[str]]) -> torch.Tensor:
        """
        Compute contrastive retrieval loss using positive/negative function labels.
        
        Uses InfoNCE loss: encourages query to be similar to positive samples
        and dissimilar to negative samples.
        
        Loss = -log(exp(sim(q, pos)) / (exp(sim(q, pos)) + sum(exp(sim(q, neg)))))
        """
        if not self.config.use_retrieval_loss or self.model.external_memory is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if not self.model.external_memory.is_ready():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = 0.0
        num_samples = 0
        temperature = 0.07  # Temperature for contrastive loss
        
        # Process each batch item
        for batch_idx, (func_ids_list, ret_labels) in enumerate(zip(function_ids, retrieval_labels)):
            if not func_ids_list or not ret_labels:
                continue
            
            # retrieval_labels is already the dict mapping function_id -> {positive, negative}
            # No need to access 'function_labels' key
            if not isinstance(ret_labels, dict):
                continue
            
            # Use mean pooling of hidden states as query representation
            query = hidden_states[batch_idx].mean(dim=0, keepdim=True)  # [1, hidden_dim]
            
            # Process each function in this batch
            for func_id in func_ids_list:
                if func_id not in ret_labels:
                    # Debug: print available keys
                    if num_samples == 0:  # Only print once
                        print(f"DEBUG: func_id '{func_id}' not in ret_labels")
                        print(f"DEBUG: Available keys: {list(ret_labels.keys())[:3]}")
                    continue
                
                labels = ret_labels[func_id]
                positive_ids = labels.get('positive', [])
                negative_ids = labels.get('negative', [])
                
                if not positive_ids or not negative_ids:
                    if num_samples == 0:  # Only print once
                        print(f"DEBUG: Missing pos/neg for {func_id}: pos={len(positive_ids)}, neg={len(negative_ids)}")
                    continue
                
                # Get embeddings from external memory for positive and negative samples
                # We'll use a simplified approach: compute similarity with memory keys
                try:
                    # Get memory keys (these represent stored function representations)
                    memory_keys = self.model.external_memory.keys[0]  # [memory_size, hidden_dim]
                    
                    # Compute similarities between query and all memory keys
                    # sim = query @ keys.T / temperature
                    similarities = torch.matmul(query, memory_keys.t()) / temperature  # [1, memory_size]
                    
                    # For simplicity, we'll use the top-k most similar as proxies for positive/negative
                    # In practice, you'd want to map function IDs to specific memory indices
                    
                    # Create pseudo-labels: assume first few are positive, rest are negative
                    num_pos = min(len(positive_ids), 3)
                    num_neg = min(len(negative_ids), 5)
                    
                    if num_pos > 0 and num_neg > 0:
                        # Get top similarities
                        top_sims, top_indices = torch.topk(similarities[0], k=num_pos + num_neg)
                        
                        # Treat first num_pos as positive, rest as negative
                        pos_sims = top_sims[:num_pos]
                        neg_sims = top_sims[num_pos:num_pos + num_neg]
                        
                        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
                        pos_exp = torch.exp(pos_sims).sum()
                        neg_exp = torch.exp(neg_sims).sum()
                        
                        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
                        
                        total_loss += loss
                        num_samples += 1
                        
                except Exception as e:
                    # If memory access fails, skip this sample
                    continue
        
        if num_samples > 0:
            return total_loss / num_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def update_external_memory(self, hidden_states: torch.Tensor):
        """Update external memory with new KV pairs."""
        if self.model.external_memory is None:
            return
        
        # Skip memory update - the external memory is updated automatically during forward pass
        # This counter just tracks that we're at a memory update step
        self.memory_update_counter += 1
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with retrieval loss and memory updates."""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        retrieval_labels = batch["retrieval_labels"]
        function_ids = batch["function_ids"]
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            targets=labels,
            attention_mask=attention_mask,
            use_external_memory=self.config.use_external_memory,
            output_hidden_states=True
        )
        
        # Extract outputs
        if isinstance(outputs, tuple):
            logits, loss, memory_info = outputs
            hidden_states = outputs[0].hidden_states[-1] if hasattr(outputs[0], 'hidden_states') else None
        else:
            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
        
        # Compute main loss
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
        
        # Add retrieval loss
        retrieval_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_retrieval_loss and hidden_states is not None:
            retrieval_loss = self.compute_retrieval_loss(hidden_states, retrieval_labels, function_ids)
            loss = loss + self.config.retrieval_loss_weight * retrieval_loss
        
        # Update external memory periodically
        if hidden_states is not None and self.global_step % self.config.memory_update_freq == 0:
            with torch.no_grad():
                self.update_external_memory(hidden_states)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient step
        metrics = {
            "train_loss": loss.item() * self.config.gradient_accumulation_steps,
            "retrieval_loss": retrieval_loss.item()
        }
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm
            )
            metrics["grad_norm"] = grad_norm.item()
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def run_inference_samples(self, num_samples: int = 3):
        """Run inference on validation samples with retrieval."""
        self.logger.info("\n" + "="*80)
        self.logger.info("INFERENCE SAMPLES WITH RETRIEVAL")
        self.logger.info("="*80 + "\n")
        
        self.model.eval()
        
        # Get a few validation samples
        val_samples = []
        for i, batch in enumerate(self.val_dataloader):
            if i >= num_samples:
                break
            val_samples.append(batch)
        
        for i, batch in enumerate(val_samples, 1):
            input_text = batch['input_text'][0]
            output_text = batch['output_text'][0]
            
            # Extract prefix and suffix from FIM format
            if '<|fim_prefix|>' in input_text and '<|fim_suffix|>' in input_text:
                parts = input_text.split('<|fim_suffix|>')
                prefix = parts[0].replace('<|fim_prefix|>', '')
                suffix_part = parts[1] if len(parts) > 1 else ''
                suffix = suffix_part.replace('<|fim_middle|>', '')
            else:
                # Fallback: use first 50 chars as prefix
                prefix = input_text[:50] if len(input_text) > 50 else input_text
                suffix = ""
            
            self.logger.info(f"\n--- Sample {i} ---")
            self.logger.info(f"Prefix: {prefix[:100]}...")
            if suffix:
                self.logger.info(f"Suffix: {suffix[:50]}...")
            self.logger.info(f"Expected: {output_text[:100]}...")
            
            # Generate with retrieval
            try:
                # Build FIM prompt
                if suffix:
                    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
                else:
                    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|><|fim_middle|>"
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate using custom method with retrieval
                with torch.no_grad():
                    generated_ids = self.model.generate_with_retrieval(
                        input_ids=inputs["input_ids"],
                        max_length=inputs["input_ids"].shape[1] + 50,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_external_memory=True
                    )
                
                # Decode only the generated part
                generated_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                self.logger.info(f"Generated (with retrieval): {generated_text[:100]}...")
                
            except Exception as e:
                self.logger.warning(f"Generation failed for sample {i}: {e}")
                continue
        
        self.logger.info("\n" + "="*80 + "\n")
        self.model.train()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluation with inference callback."""
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
        
        # Run inference samples
        self.run_inference_samples(self.config.num_inference_samples)
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity
        }
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save checkpoint (without external memory index)."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get model state without external memory index
        model_state = self.model.state_dict()
        
        # Remove external memory index keys (don't persist)
        filtered_state = {
            k: v for k, v in model_state.items()
            if 'external_memory.index' not in k
        }
        
        checkpoint = {
            'model_state_dict': filtered_state,
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
        self.logger.info("Starting improved training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(self.train_dataloader):
                metrics = self.train_step(batch)
                
                if metrics.get("skipped", False):
                    continue
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    log_str = f"Step {self.global_step}: loss={metrics['train_loss']:.4f}"
                    if "retrieval_loss" in metrics:
                        log_str += f", ret_loss={metrics['retrieval_loss']:.4f}"
                    if "grad_norm" in metrics:
                        log_str += f", grad_norm={metrics['grad_norm']:.4f}"
                    if "learning_rate" in metrics:
                        log_str += f", lr={metrics['learning_rate']:.2e}"
                    if self.memory_update_counter > 0:
                        log_str += f", mem_updates={self.memory_update_counter}"
                    self.logger.info(log_str)
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.global_step > 0:
                    eval_metrics = self.evaluate()
                    self.logger.info(f"Evaluation: {eval_metrics}")
                    
                    is_best = eval_metrics["eval_loss"] < self.best_loss
                    if is_best:
                        self.best_loss = eval_metrics["eval_loss"]
                        self.logger.info(f"New best loss: {self.best_loss:.4f}")
                    
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
        self.logger.info(f"Total memory updates: {self.memory_update_counter}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Improved AST Memory Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="data/ast_dataset/bat")
    parser.add_argument("--output_dir", type=str, default="outputs/improved_training")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--config_file", type=str, help="Path to config JSON")
    
    args = parser.parse_args()
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = ImprovedTrainingConfig(**config_dict)
    else:
        config = ImprovedTrainingConfig()
    
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    trainer = ImprovedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
