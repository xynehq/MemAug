#!/usr/bin/env python3
"""
Quick training test to verify everything works with external memory.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mem_aug.utils.advanced_lora_trainer import AdvancedLoRATrainer, AdvancedTrainingConfig

def quick_test():
    """Quick test with minimal steps."""
    
    # Create minimal config
    config = AdvancedTrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        use_internal_memory=False,
        use_external_memory=True,
        batch_size=1,
        num_epochs=1,
        max_seq_length=256,  # Shorter sequences
        logging_steps=1,  # Log every step
        eval_steps=10,    # Don't evaluate
        save_steps=10,    # Don't save
        gradient_checkpointing=False,  # Disable for speed
        train_data_path="src/mem_aug/data/temp_data/train.jsonl",
        val_data_path="src/mem_aug/data/temp_data/val.jsonl",
        output_dir="outputs/quick_test",
        use_wandb=False,  # Disable wandb
        debug_mode=True   # Enable debug mode
    )
    
    print("Starting quick training test...")
    trainer = AdvancedLoRATrainer(config)
    
    # Run quick test
    print("Running quick training test...")
    
    # Test model info
    model_info = trainer.model.get_model_info()
    print(f"Model has {model_info['trainable_params']:,} trainable parameters")
    print(f"Trainable percentage: {model_info['trainable_percentage']:.2f}%")
    
    # Test a single inference
    test_input = "def test():"
    inputs = trainer.tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
    
    import torch
    with torch.no_grad():
        outputs = trainer.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_external_memory=True
        )
        if isinstance(outputs, tuple):
            logits, _, memory_info = outputs
        else:
            logits = outputs.logits
            memory_info = getattr(outputs, 'memory_info', {})
        
        print(f"Inference successful! Output shape: {logits.shape}")
        print(f"External memory size: {memory_info.get('external_memory_size', 'N/A')}")
    
    print("✅ Quick test completed successfully!")

if __name__ == "__main__":
    quick_test()