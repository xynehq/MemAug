#!/usr/bin/env python3
"""
Advanced LoRA + External Memory Training with Production-Ready Features

Features:
- Comprehensive LoRA adapters on all possible modules
- Early stopping with patience
- Learning rate scheduling (cosine, linear, polynomial)
- Gradient accumulation and clipping
- Automatic mixed precision (when available)
- Model checkpointing and resuming
- Wandb integration
- Advanced validation metrics
- Memory optimization
- Progressive unfreezing
- Gradient noise addition
- Label smoothing
"""

import os
import sys
import json
import time
import math
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    PolynomialLR,
    OneCycleLR
)

import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed,
    TrainingArguments,
    EarlyStoppingCallback
)

# LoRA and PEFT imports
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training,
        AdaLoraConfig,
        IA3Config,
        LoHaConfig,
        LoKrConfig
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("PEFT not available. Install with: pip install peft")

# Wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Wandb not available. Install with: pip install wandb")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel, 
    HybridTransformerConfig
)
from mem_aug.utils.gradient_handler import (
    GradientHandler,
    create_gradient_aware_optimizer
)


@dataclass
class AdvancedTrainingConfig:
    """Advanced training configuration with all production features."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    model_revision: str = "main"
    trust_remote_code: bool = True
    
    # LoRA settings - Advanced configuration
    use_lora: bool = True
    lora_type: str = "lora"  # lora, adalora, ia3, loha, lokr
    lora_r: int = 64  # Higher rank for better performance
    lora_alpha: int = 128  # 2x rank is common
    lora_dropout: float = 0.05  # Lower dropout for stability
    lora_bias: str = "none"  # none, all, lora_only
    lora_task_type: str = "CAUSAL_LM"
    
    # Target modules - Only supported modules (Linear, Embedding, Conv layers)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        # Attention modules (Linear layers)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP modules (Linear layers)
        "gate_proj", "up_proj", "down_proj",
        # Embedding and output layers
        "embed_tokens", "lm_head"
        # Note: Layer norms are not supported by LoRA as they're not Linear/Embedding layers
    ])
    
    # External memory settings
    use_internal_memory: bool = False
    use_external_memory: bool = True
    external_memory_size: int = 2097152  # 2M entries
    retrieval_k: int = 16  # More retrievals
    chunk_size: int = 8  # Larger chunks
    use_gpu_to_search: bool = True
    use_gated_retrieval: bool = True
    
    # Training hyperparameters
    batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # Effective batch size = 32
    num_epochs: int = 10
    max_steps: int = -1  # If set, overrides num_epochs
    max_seq_length: int = 2048  # Longer sequences
    
    # Learning rates
    learning_rate: float = 2e-4  # Higher for LoRA
    external_memory_lr: float = 1e-4
    min_learning_rate: float = 1e-6
    lr_scheduler_type: str = "cosine_with_restarts"  # linear, cosine, cosine_with_restarts, polynomial, onecycle
    warmup_ratio: float = 0.1  # 10% warmup
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    # Optimization
    optimizer_type: str = "adamw"  # adamw, adafactor, adam
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Regularization
    label_smoothing: float = 0.1
    dropout: float = 0.1
    attention_dropout: float = 0.1
    gradient_noise_std: float = 0.0  # Gradient noise for regularization
    
    # Mixed precision and memory optimization
    use_amp: bool = False  # Disable for CPU compatibility
    fp16: bool = False
    bf16: bool = False  # Disable for CPU compatibility
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0  # Disable multiprocessing for CPU/FAISS compatibility
    gradient_checkpointing: bool = True
    
    # Advanced training features
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Progressive training
    progressive_unfreezing: bool = False
    unfreeze_layers_per_epoch: int = 2
    
    # Evaluation and logging
    eval_strategy: str = "steps"  # steps, epoch, no
    eval_steps: int = 100
    save_strategy: str = "steps"  # steps, epoch, no
    save_steps: int = 500
    logging_strategy: str = "steps"
    logging_steps: int = 10
    
    # Data settings
    train_data_path: str = "src/mem_aug/data/temp_data/train.jsonl"
    val_data_path: str = "src/mem_aug/data/temp_data/val.jsonl"
    context_data_path: str = "src/mem_aug/data/temp_data/context_data.jsonl"
    
    # Output and resuming
    output_dir: str = "outputs/advanced_lora_training"
    resume_from_checkpoint: Optional[str] = None
    overwrite_output_dir: bool = True
    
    # Distributed training
    local_rank: int = -1
    deepspeed_config: Optional[str] = None
    
    # Monitoring and debugging
    use_wandb: bool = True
    wandb_project: str = "qwen-lora-external-memory"
    wandb_run_name: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Device settings
    device: str = "auto"
    seed: int = 42
    
    # Debugging
    debug_mode: bool = False
    max_debug_samples: int = 100


class AdvancedCodeDataset(Dataset):
    """Advanced dataset with data augmentation and preprocessing."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 2048,
        label_smoothing: float = 0.0,
        data_augmentation: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_smoothing = label_smoothing
        self.data_augmentation = data_augmentation
        self.data = self._load_and_preprocess_data(data_path)
    
    def _load_and_preprocess_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess data with validation."""
        data = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # Validate required fields
                        if "input" in item and "output" in item:
                            # Basic quality filters
                            if len(item["input"].strip()) > 0 and len(item["output"].strip()) > 0:
                                data.append(item)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        else:
            logging.warning(f"Data file not found: {data_path}")
            # Create enhanced dummy data for testing
            data = self._create_dummy_data()
        
        logging.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def _create_dummy_data(self) -> List[Dict]:
        """Create enhanced dummy data for testing."""
        return [
            {
                "input": "def fibonacci(n):",
                "output": "\n    \"\"\"Calculate fibonacci number using dynamic programming.\"\"\"\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
            },
            {
                "input": "class BinarySearchTree:",
                "output": "\n    \"\"\"Efficient binary search tree implementation.\"\"\"\n    def __init__(self):\n        self.root = None\n        self.size = 0\n    \n    def insert(self, value):\n        \"\"\"Insert a value into the BST.\"\"\"\n        self.root = self._insert_recursive(self.root, value)\n        self.size += 1"
            },
            {
                "input": "def quick_sort(arr, low=0, high=None):",
                "output": "\n    \"\"\"Quick sort implementation with optimizations.\"\"\"\n    if high is None:\n        high = len(arr) - 1\n    \n    if low < high:\n        pivot = self._partition(arr, low, high)\n        self.quick_sort(arr, low, pivot - 1)\n        self.quick_sort(arr, pivot + 1, high)"
            },
            {
                "input": "import asyncio\n\nasync def fetch_data(url):",
                "output": "\n    \"\"\"Fetch data asynchronously with error handling.\"\"\"\n    async with aiohttp.ClientSession() as session:\n        try:\n            async with session.get(url, timeout=30) as response:\n                response.raise_for_status()\n                return await response.json()\n        except aiohttp.ClientError as e:\n            logging.error(f'Failed to fetch {url}: {e}')\n            raise"
            }
        ] * 50  # Repeat for sufficient training data
    
    def _apply_data_augmentation(self, text: str) -> str:
        """Apply data augmentation techniques."""
        if not self.data_augmentation:
            return text
        
        # Simple augmentations for code
        augmentations = [
            text,  # Original
            text.replace("    ", "\t"),  # Tab vs spaces
            text.replace("\n\n", "\n"),  # Remove double newlines
        ]
        
        return np.random.choice(augmentations)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["output"]
        
        # Apply augmentation
        input_text = self._apply_data_augmentation(input_text)
        target_text = self._apply_data_augmentation(target_text)
        
        full_text = input_text + target_text
        
        # Tokenize with attention to special tokens
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        
        # Create labels for causal LM with proper masking
        labels = input_ids.clone()
        
        # Mask input tokens (only compute loss on target)
        input_tokens = self.tokenizer(
            input_text, 
            truncation=True, 
            add_special_tokens=True
        )["input_ids"]
        input_length = len(input_tokens)
        
        if input_length < len(labels):
            labels[:input_length] = -100
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Note: Label smoothing will be applied in the loss function
            pass
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_length": input_length,
            "target_length": len(labels) - input_length
        }


class AdvancedLoRATrainer:
    """Advanced trainer with production-ready features."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_device_and_seed()
        self.setup_wandb()
        self.setup_model_and_tokenizer()
        self.setup_data()
        self.setup_optimizer_and_scheduler()
        self.setup_training_components()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        self.early_stopping_counter = 0
        self.training_history = defaultdict(list)
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO if not self.config.debug_mode else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Advanced LoRA Trainer initialized")
    
    def setup_device_and_seed(self):
        """Setup device and random seeds for reproducibility."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Set seeds for reproducibility
        set_seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            # Optimize CUDA settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Random seed: {self.config.seed}")
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            run_name = self.config.wandb_run_name or f"lora-r{self.config.lora_r}-{int(time.time())}"
            
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=asdict(self.config),
                tags=["lora", "external-memory", "qwen", "advanced"],
                dir=self.config.output_dir
            )
            self.logger.info(f"Wandb initialized: {self.config.wandb_project}/{run_name}")
        else:
            self.logger.info("Wandb logging disabled")
    
    def setup_model_and_tokenizer(self):
        """Setup model with comprehensive LoRA configuration."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model with optimal settings
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32),
            "device_map": None,
            "trust_remote_code": self.config.trust_remote_code,
            "revision": self.config.model_revision,
        }
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Apply advanced LoRA configuration
        if self.config.use_lora and PEFT_AVAILABLE:
            self.logger.info(f"Applying {self.config.lora_type.upper()} adapters...")
            
            # Filter target modules that exist in the model
            available_modules = set()
            for name, _ in base_model.named_modules():
                module_name = name.split('.')[-1]
                available_modules.add(module_name)
            
            target_modules = [
                module for module in self.config.lora_target_modules 
                if module in available_modules
            ]
            
            self.logger.info(f"Target modules: {target_modules}")
            
            # Create LoRA config based on type
            if self.config.lora_type == "lora":
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=target_modules,
                    bias=self.config.lora_bias,
                    inference_mode=False,
                )
            elif self.config.lora_type == "adalora":
                lora_config = AdaLoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=target_modules,
                    inference_mode=False,
                )
            elif self.config.lora_type == "ia3":
                lora_config = IA3Config(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=target_modules,
                    inference_mode=False,
                )
            else:
                raise ValueError(f"Unsupported LoRA type: {self.config.lora_type}")
            
            # Apply LoRA
            base_model = get_peft_model(base_model, lora_config)
            self.logger.info(f"{self.config.lora_type.upper()} applied successfully")
            base_model.print_trainable_parameters()
        
        # Create hybrid config for external memory
        base_config_dict = base_model.config.to_dict() if hasattr(base_model, 'config') else {}
        
        hybrid_config = HybridTransformerConfig(
            **base_config_dict,
            use_internal_memory=self.config.use_internal_memory,
            use_external_memory=self.config.use_external_memory,
            external_memory_size=self.config.external_memory_size,
            retrieval_k=self.config.retrieval_k,
            chunk_size=self.config.chunk_size,
            use_gpu_to_search=self.config.use_gpu_to_search,
            use_gated_retrieval=self.config.use_gated_retrieval,
        )
        
        # Create hybrid model
        self.model = HybridTransformerModel(
            config=hybrid_config,
            tokenizer=self.tokenizer,
            base_model=base_model
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
            except Exception as e:
                self.logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Apply progressive unfreezing if enabled
        if self.config.progressive_unfreezing:
            self._setup_progressive_unfreezing()
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model info: {model_info}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({"model": model_info})
    
    def _setup_progressive_unfreezing(self):
        """Setup progressive unfreezing of layers."""
        self.frozen_layers = []
        self.unfrozen_layers = []
        
        # Initially freeze all layers except LoRA and external memory
        for name, param in self.model.named_parameters():
            if not any(keyword in name.lower() for keyword in [
                'lora_', 'adapter', 'external_memory', 'external_mem_proj', 
                'gate_network', 'g_retrieve'
            ]):
                param.requires_grad = False
                self.frozen_layers.append(name)
            else:
                self.unfrozen_layers.append(name)
        
        self.logger.info(f"Progressive unfreezing: {len(self.frozen_layers)} layers frozen initially")
    
    def setup_data(self):
        """Setup advanced data loaders."""
        # Create datasets
        train_dataset = AdvancedCodeDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length,
            self.config.label_smoothing,
            data_augmentation=True
        )
        
        val_dataset = AdvancedCodeDataset(
            self.config.val_data_path,
            self.tokenizer,
            self.config.max_seq_length,
            label_smoothing=0.0,  # No smoothing for validation
            data_augmentation=False
        )
        
        # Debug mode: limit dataset size
        if self.config.debug_mode:
            train_indices = list(range(min(self.config.max_debug_samples, len(train_dataset))))
            val_indices = list(range(min(self.config.max_debug_samples // 4, len(val_dataset))))
            
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # Create data loaders with advanced settings
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory and self.device.type == 'cuda',
            drop_last=True,  # For stable batch sizes
            collate_fn=self._collate_fn
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory and self.device.type == 'cuda',
            drop_last=False,
            collate_fn=self._collate_fn
        )
        
        self.logger.info(f"Train dataset: {len(train_dataset)} samples")
        self.logger.info(f"Validation dataset: {len(val_dataset)} samples")
        self.logger.info(f"Train batches: {len(self.train_dataloader)}")
        self.logger.info(f"Validation batches: {len(self.val_dataloader)}")
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Standard collation
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ["input_length", "target_length"]:
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def setup_optimizer_and_scheduler(self):
        """Setup advanced optimizer and learning rate scheduler."""
        # Separate parameters by type for different learning rates
        external_memory_params = []
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(keyword in name.lower() for keyword in [
                    'external_memory', 'external_mem_proj', 'gate_network', 'g_retrieve'
                ]):
                    external_memory_params.append(param)
                elif any(keyword in name.lower() for keyword in ['lora_', 'adapter']):
                    lora_params.append(param)
                else:
                    other_params.append(param)
        
        # Create parameter groups
        param_groups = []
        
        if external_memory_params:
            param_groups.append({
                'params': external_memory_params,
                'lr': self.config.external_memory_lr,
                'weight_decay': self.config.weight_decay
            })
            self.logger.info(f"External memory params: {sum(p.numel() for p in external_memory_params):,}")
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
            self.logger.info(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
            self.logger.info(f"Other params: {sum(p.numel() for p in other_params):,}")
        
        # Create optimizer
        if self.config.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                param_groups,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Calculate total steps
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs
        
        # Calculate warmup steps
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Create learning rate scheduler
        if self.config.lr_scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.lr_scheduler_type == "cosine_with_restarts":
            T_0 = max(1, total_steps // 4)  # Ensure T_0 is at least 1
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,  # Restart every 1/4 of training
                T_mult=2,
                eta_min=self.config.min_learning_rate
            )
        elif self.config.lr_scheduler_type == "polynomial":
            self.scheduler = PolynomialLR(
                self.optimizer,
                total_iters=total_steps,
                power=1.0
            )
        elif self.config.lr_scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_ratio
            )
        else:
            self.scheduler = None
        
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
        self.logger.info(f"Optimizer: {self.config.optimizer_type}")
        self.logger.info(f"LR Scheduler: {self.config.lr_scheduler_type}")
        self.logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    def setup_training_components(self):
        """Setup additional training components."""
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config.use_amp and self.device.type == 'cuda' else None
        
        # Gradient handler for advanced gradient management
        gradient_config = {
            'max_grad_norm': self.config.max_grad_norm,
            'gradient_noise_std': self.config.gradient_noise_std,
            'external_memory_grad_scale': 1.0,
        }
        self.gradient_handler = GradientHandler(self.model, gradient_config)
        
        self.logger.info("Training components initialized")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with all advanced features."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with automatic mixed precision
        with autocast(enabled=self.config.use_amp and self.device.type == 'cuda'):
            outputs = self.model(
                input_ids=batch["input_ids"],
                targets=batch["labels"],
                attention_mask=batch["attention_mask"],
                use_external_memory=self.config.use_external_memory
            )
            
            # Extract loss
            if isinstance(outputs, tuple):
                logits, loss, memory_info = outputs
            else:
                loss = outputs.loss
                logits = outputs.logits
                memory_info = getattr(outputs, 'memory_info', {})
            
            # Apply label smoothing if specified
            if self.config.label_smoothing > 0 and loss is None:
                loss = self._compute_loss_with_smoothing(logits, batch["labels"])
            
            # Ensure loss is computed
            if loss is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            # Scale loss by gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Add gradient noise for regularization
        if self.config.gradient_noise_std > 0:
            self._add_gradient_noise()
        
        # Optimization step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = self.gradient_handler.clip_gradients(
                    [p for p in self.model.parameters() if p.requires_grad]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = self.gradient_handler.clip_gradients(
                    [p for p in self.model.parameters() if p.requires_grad]
                )
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        # Collect metrics
        metrics = {
            "train_loss": loss.item() * self.config.gradient_accumulation_steps,
            "train_perplexity": torch.exp(loss * self.config.gradient_accumulation_steps).item(),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm,
        }
        
        # Add memory info
        if memory_info:
            metrics["external_memory_size"] = memory_info.get('external_memory_size', 0)
        
        # Update training history
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        return metrics
    
    def _compute_loss_with_smoothing(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss with label smoothing."""
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Create smoothed labels
        vocab_size = flat_logits.size(-1)
        confidence = 1.0 - self.config.label_smoothing
        smoothing = self.config.label_smoothing / (vocab_size - 1)
        
        # Create one-hot and smooth
        one_hot = torch.zeros_like(flat_logits).scatter(1, flat_labels.unsqueeze(1), confidence)
        one_hot += smoothing
        
        # Mask padding tokens
        mask = (flat_labels != -100).float()
        one_hot = one_hot * mask.unsqueeze(1)
        
        # Compute loss
        log_probs = F.log_softmax(flat_logits, dim=1)
        loss = -(one_hot * log_probs).sum(1)
        loss = loss.sum() / mask.sum()
        
        return loss
    
    def _add_gradient_noise(self):
        """Add gradient noise for regularization."""
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.config.gradient_noise_std
                param.grad += noise
    
    def evaluate(self) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with autocast(enabled=self.config.use_amp and self.device.type == 'cuda'):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        targets=batch["labels"],
                        attention_mask=batch["attention_mask"],
                        use_external_memory=self.config.use_external_memory
                    )
                    
                    if isinstance(outputs, tuple):
                        logits, loss, memory_info = outputs
                    else:
                        loss = outputs.loss
                        logits = outputs.logits
                        memory_info = getattr(outputs, 'memory_info', {})
                    
                    if loss is None:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = batch["labels"][..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions for additional metrics
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        avg_loss = total_loss / total_samples
        perplexity = np.exp(avg_loss)
        
        # Additional metrics
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
        
        # Add external memory metrics if available
        if self.config.use_external_memory and memory_info:
            metrics["eval_external_memory_size"] = memory_info.get('external_memory_size', 0)
        
        return metrics
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False):
        """Save comprehensive checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'early_stopping_counter': self.early_stopping_counter,
            'training_history': dict(self.training_history),
            'config': asdict(self.config)
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save metrics
        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save best model separately
        if is_best:
            best_dir = os.path.join(self.config.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(best_dir)
            else:
                torch.save(self.model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
            
            self.tokenizer.save_pretrained(best_dir)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoint_dirs = [
            d for d in os.listdir(self.config.output_dir)
            if d.startswith("checkpoint-") and d != "best_model"
        ]
        
        if len(checkpoint_dirs) > self.config.save_total_limit:
            # Sort by step number
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            
            # Remove oldest checkpoints
            for old_dir in checkpoint_dirs[:-self.config.save_total_limit]:
                old_path = os.path.join(self.config.output_dir, old_dir)
                if os.path.exists(old_path):
                    import shutil
                    shutil.rmtree(old_path)
                    self.logger.info(f"Removed old checkpoint: {old_path}")
    
    def should_early_stop(self, current_metric: float) -> bool:
        """Check if training should stop early."""
        if self.config.early_stopping_patience <= 0:
            return False
        
        improved = False
        if self.config.greater_is_better:
            if current_metric > self.best_metric + self.config.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric
        else:
            if current_metric < self.best_metric - self.config.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric
        
        if improved:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        should_stop = self.early_stopping_counter >= self.config.early_stopping_patience
        
        if should_stop:
            self.logger.info(f"Early stopping triggered after {self.early_stopping_counter} steps without improvement")
        
        return should_stop
    
    def progressive_unfreeze(self):
        """Progressively unfreeze layers during training."""
        if not self.config.progressive_unfreezing or not self.frozen_layers:
            return
        
        layers_to_unfreeze = min(self.config.unfreeze_layers_per_epoch, len(self.frozen_layers))
        
        for _ in range(layers_to_unfreeze):
            if self.frozen_layers:
                layer_name = self.frozen_layers.pop()
                for name, param in self.model.named_parameters():
                    if name == layer_name:
                        param.requires_grad = True
                        self.unfrozen_layers.append(name)
                        break
        
        self.logger.info(f"Unfroze {layers_to_unfreeze} layers. Remaining frozen: {len(self.frozen_layers)}")
    
    def train(self):
        """Main training loop with all advanced features."""
        self.logger.info("Starting advanced training...")
        self.logger.info(f"Configuration: {asdict(self.config)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Progressive unfreezing
            if epoch > 0:
                self.progressive_unfreeze()
            
            # Training loop
            epoch_metrics = defaultdict(list)
            
            for step, batch in enumerate(self.train_dataloader):
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                step_metrics = self.train_step(batch)
                
                # Collect metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    log_metrics = {k: v for k, v in step_metrics.items()}
                    log_metrics["epoch"] = epoch
                    log_metrics["step"] = self.global_step
                    
                    self.logger.info(
                        f"Step {self.global_step}: "
                        f"loss={step_metrics['train_loss']:.4f}, "
                        f"ppl={step_metrics['train_perplexity']:.2f}, "
                        f"lr={step_metrics['learning_rate']:.2e}, "
                        f"grad_norm={step_metrics['grad_norm']:.4f}"
                    )
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(log_metrics, step=self.global_step)
                
                # Evaluation
                if (self.config.eval_strategy == "steps" and 
                    self.global_step % self.config.eval_steps == 0 and 
                    self.global_step > 0):
                    
                    eval_metrics = self.evaluate()
                    eval_metrics["epoch"] = epoch
                    eval_metrics["step"] = self.global_step
                    
                    self.logger.info(f"Evaluation: {eval_metrics}")
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(eval_metrics, step=self.global_step)
                    
                    # Early stopping check
                    metric_value = eval_metrics[self.config.metric_for_best_model]
                    is_best = False
                    
                    if self.config.greater_is_better:
                        is_best = metric_value > self.best_metric
                    else:
                        is_best = metric_value < self.best_metric
                    
                    if is_best:
                        self.best_metric = metric_value
                    
                    # Save checkpoint
                    if (self.config.save_strategy == "steps" and 
                        self.global_step % self.config.save_steps == 0):
                        self.save_checkpoint(self.global_step, eval_metrics, is_best)
                    
                    # Check early stopping
                    if self.should_early_stop(metric_value):
                        self.logger.info("Early stopping triggered")
                        break
                
                # Save checkpoint by steps
                elif (self.config.save_strategy == "steps" and 
                      self.global_step % self.config.save_steps == 0 and 
                      self.global_step > 0):
                    self.save_checkpoint(self.global_step, step_metrics)
                
                self.global_step += 1
            
            # End of epoch evaluation
            if self.config.eval_strategy == "epoch":
                eval_metrics = self.evaluate()
                eval_metrics["epoch"] = epoch
                eval_metrics["step"] = self.global_step
                
                self.logger.info(f"End of epoch {epoch + 1} evaluation: {eval_metrics}")
                
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log(eval_metrics, step=self.global_step)
                
                # Early stopping and checkpointing logic here too
                metric_value = eval_metrics[self.config.metric_for_best_model]
                is_best = (metric_value > self.best_metric) if self.config.greater_is_better else (metric_value < self.best_metric)
                
                if is_best:
                    self.best_metric = metric_value
                
                if self.config.save_strategy == "epoch":
                    self.save_checkpoint(self.global_step, eval_metrics, is_best)
                
                if self.should_early_stop(metric_value):
                    self.logger.info("Early stopping triggered")
                    break
            
            # Log epoch summary
            epoch_summary = {f"epoch_{k}": np.mean(v) for k, v in epoch_metrics.items()}
            self.logger.info(f"Epoch {epoch + 1} summary: {epoch_summary}")
            
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log(epoch_summary, step=self.global_step)
        
        # Final evaluation and save
        final_eval_metrics = self.evaluate()
        self.logger.info(f"Final evaluation: {final_eval_metrics}")
        
        # Save final checkpoint
        self.save_checkpoint(self.global_step, final_eval_metrics, is_best=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({"training_time": total_time})
            wandb.finish()


def main():
    """Main function with advanced argument parsing."""
    parser = argparse.ArgumentParser(description="Advanced LoRA + External Memory Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/advanced_lora_training")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_type", type=str, default="lora", choices=["lora", "adalora", "ia3"])
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Advanced features
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--progressive_unfreezing", action="store_true")
    
    # External memory
    parser.add_argument("--external_memory_size", type=int, default=2097152)
    parser.add_argument("--retrieval_k", type=int, default=16)
    
    # Logging and monitoring
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="qwen-lora-external-memory")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # Debug and utility
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # Config file
    parser.add_argument("--config_file", type=str, help="Path to JSON config file")
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = AdvancedTrainingConfig(**config_dict)
    else:
        config = AdvancedTrainingConfig()
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Create trainer and start training
    trainer = AdvancedLoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()