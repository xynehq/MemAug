#!/usr/bin/env python3
"""
Test script to save and load the advanced LoRA + External Memory model
and validate that the configuration matches between training and loading.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mem_aug.utils.advanced_lora_trainer import AdvancedTrainingConfig, AdvancedLoRATrainer

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_model_with_config(trainer, save_dir):
    """Save the model with its configuration."""
    logger = logging.getLogger(__name__)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, "model_state_dict.pt")
    torch.save(trainer.model.state_dict(), model_path)
    logger.info(f"Model state dict saved to: {model_path}")
    
    # Save PEFT adapter
    adapter_path = os.path.join(save_dir, "peft_adapter")
    if hasattr(trainer.model, 'full_model') and hasattr(trainer.model.full_model, 'save_pretrained'):
        trainer.model.full_model.save_pretrained(adapter_path)
        logger.info(f"PEFT adapter saved to: {adapter_path}")
    
    # Save complete configuration
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(trainer.config), f, indent=2)
    logger.info(f"Training config saved to: {config_path}")
    
    # Save model info for validation
    model_info = trainer.model.get_model_info()
    info_path = os.path.join(save_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model info saved to: {info_path}")
    
    # Save optimizer state
    optimizer_path = os.path.join(save_dir, "optimizer_state_dict.pt")
    torch.save(trainer.optimizer.state_dict(), optimizer_path)
    logger.info(f"Optimizer state saved to: {optimizer_path}")
    
    return {
        'model_path': model_path,
        'adapter_path': adapter_path,
        'config_path': config_path,
        'info_path': info_path,
        'optimizer_path': optimizer_path
    }

def load_model_and_validate(save_dir):
    """Load the model and validate configuration match."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    original_config = AdvancedTrainingConfig(**config_dict)
    logger.info("Original configuration loaded successfully")
    
    # Create a new trainer with the same config
    new_trainer = AdvancedLoRATrainer(original_config)
    logger.info("New trainer created with original configuration")
    
    # Load model state
    model_path = os.path.join(save_dir, "model_state_dict.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=new_trainer.device)
        new_trainer.model.load_state_dict(state_dict, strict=False)
        logger.info("Model state dict loaded successfully")
    
    # Load original model info for comparison
    info_path = os.path.join(save_dir, "model_info.json")
    with open(info_path, 'r') as f:
        original_info = json.load(f)
    
    # Get new model info
    new_info = new_trainer.model.get_model_info()
    
    # Validate configuration and model match
    validation_results = {
        'config_match': True,
        'model_info_match': True,
        'differences': []
    }
    
    # Compare configurations
    original_config_dict = asdict(original_config)
    new_config_dict = asdict(new_trainer.config)
    
    for key in original_config_dict:
        if key in new_config_dict:
            if original_config_dict[key] != new_config_dict[key]:
                validation_results['config_match'] = False
                validation_results['differences'].append({
                    'type': 'config',
                    'key': key,
                    'original': original_config_dict[key],
                    'loaded': new_config_dict[key]
                })
    
    # Compare model info (excluding device-specific info)
    comparable_keys = ['trainable_params', 'total_params', 'trainable_percentage', 'has_peft', 'peft_type']
    for key in comparable_keys:
        if key in original_info and key in new_info:
            if original_info[key] != new_info[key]:
                validation_results['model_info_match'] = False
                validation_results['differences'].append({
                    'type': 'model_info',
                    'key': key,
                    'original': original_info[key],
                    'loaded': new_info[key]
                })
    
    return new_trainer, validation_results

def test_model_inference(trainer, test_input="def fibonacci(n):"):
    """Test model inference to ensure it's working."""
    logger = logging.getLogger(__name__)
    
    # Tokenize input
    inputs = trainer.tokenizer(
        test_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move to device
    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
    
    # Generate
    trainer.model.eval()
    with torch.no_grad():
        # Test forward pass
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
        
        # Get predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        
        logger.info(f"Input: {test_input}")
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Memory info: {memory_info}")
        logger.info("Model inference test successful!")
        
        return True

def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("Starting save/load model validation test")
    
    # Create a minimal config for testing
    config = AdvancedTrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        batch_size=1,
        num_epochs=1,
        external_memory_size=32,
        retrieval_k=4,
        lora_r=16,  # Smaller for faster testing
        lora_alpha=32,
        debug_mode=True,
        max_seq_length=512,  # Shorter for testing
        output_dir="outputs/save_load_test",
        use_wandb=False  # Disable wandb for testing
    )
    
    logger.info("Step 1: Creating and initializing trainer...")
    # Create trainer
    trainer = AdvancedLoRATrainer(config)
    logger.info("Trainer created successfully")
    
    # Test initial inference
    logger.info("Step 2: Testing initial model inference...")
    test_model_inference(trainer, "def hello_world():")
    
    # Save model
    logger.info("Step 3: Saving model and configuration...")
    save_dir = "outputs/saved_model_test"
    saved_paths = save_model_with_config(trainer, save_dir)
    logger.info(f"Model saved to: {save_dir}")
    
    # Load model and validate
    logger.info("Step 4: Loading model and validating configuration...")
    loaded_trainer, validation_results = load_model_and_validate(save_dir)
    
    # Test loaded model inference
    logger.info("Step 5: Testing loaded model inference...")
    test_model_inference(loaded_trainer, "def hello_world():")
    
    # Report validation results
    logger.info("Step 6: Validation Results:")
    logger.info(f"Configuration Match: {validation_results['config_match']}")
    logger.info(f"Model Info Match: {validation_results['model_info_match']}")
    
    if validation_results['differences']:
        logger.warning("Found differences:")
        for diff in validation_results['differences']:
            logger.warning(f"  {diff['type']}.{diff['key']}: {diff['original']} -> {diff['loaded']}")
    else:
        logger.info("✅ All validations passed! Model save/load is working correctly.")
    
    # Compare model parameters
    logger.info("Step 7: Comparing model parameters...")
    original_params = {name: param.data.clone() for name, param in trainer.model.named_parameters()}
    loaded_params = {name: param.data for name, param in loaded_trainer.model.named_parameters()}
    
    param_differences = 0
    for name in original_params:
        if name in loaded_params:
            if not torch.allclose(original_params[name], loaded_params[name], atol=1e-6):
                param_differences += 1
                logger.warning(f"Parameter difference in: {name}")
    
    if param_differences == 0:
        logger.info("✅ All model parameters match perfectly!")
    else:
        logger.warning(f"⚠️  Found {param_differences} parameter differences")
    
    logger.info("Save/Load validation test completed!")
    
    return validation_results, param_differences == 0

if __name__ == "__main__":
    results, params_match = main()
    
    if results['config_match'] and results['model_info_match'] and params_match:
        print("\n🎉 SUCCESS: Model save/load validation passed!")
        exit(0)
    else:
        print("\n❌ FAILURE: Model save/load validation failed!")
        exit(1)