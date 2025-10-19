#!/usr/bin/env python3
"""
Comprehensive test suite for the Advanced LoRA + External Memory Trainer.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mem_aug.utils.advanced_lora_trainer import AdvancedTrainingConfig, AdvancedLoRATrainer

def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_trainer_initialization():
    """Test that the trainer initializes correctly."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing trainer initialization...")
    
    config = AdvancedTrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        use_wandb=False,
        debug_mode=True,
        batch_size=1,
        max_seq_length=128,
        lora_r=8,  # Small for testing
        lora_alpha=16,
        external_memory_size=32,
        retrieval_k=4
    )
    
    try:
        trainer = AdvancedLoRATrainer(config)
        model_info = trainer.model.get_model_info()
        
        assert model_info['has_peft'], "Model should have PEFT"
        assert model_info['trainable_params'] > 0, "Should have trainable parameters"
        assert model_info['trainable_percentage'] < 50, "Should be parameter efficient"
        
        logger.info(f"✅ Trainer initialized successfully")
        logger.info(f"   Trainable params: {model_info['trainable_params']:,}")
        logger.info(f"   Trainable %: {model_info['trainable_percentage']:.2f}%")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trainer initialization failed: {e}")
        return False

def test_model_inference():
    """Test that model inference works."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing model inference...")
    
    config = AdvancedTrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        use_wandb=False,
        debug_mode=True,
        batch_size=1,
        max_seq_length=128,
        lora_r=8,
        lora_alpha=16,
        external_memory_size=32,
        retrieval_k=4
    )
    
    try:
        trainer = AdvancedLoRATrainer(config)
        
        # Test inference
        test_input = "def hello():"
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
            
            assert logits is not None, "Should produce logits"
            assert len(logits.shape) == 3, "Logits should be 3D"
            assert 'external_memory_size' in memory_info, "Should have memory info"
            
            logger.info(f"✅ Model inference successful")
            logger.info(f"   Output shape: {logits.shape}")
            logger.info(f"   External memory size: {memory_info.get('external_memory_size', 'N/A')}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Model inference failed: {e}")
        return False

def test_save_load_functionality():
    """Test save and load functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing save/load functionality...")
    
    config = AdvancedTrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        use_wandb=False,
        debug_mode=True,
        batch_size=1,
        max_seq_length=128,
        lora_r=8,
        lora_alpha=16,
        external_memory_size=32,
        retrieval_k=4
    )
    
    try:
        # Create and save model
        trainer = AdvancedLoRATrainer(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            import torch
            import json
            from dataclasses import asdict
            
            model_path = os.path.join(temp_dir, "model.pt")
            config_path = os.path.join(temp_dir, "config.json")
            
            torch.save(trainer.model.state_dict(), model_path)
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f)
            
            # Load model
            with open(config_path, 'r') as f:
                loaded_config_dict = json.load(f)
            loaded_config = AdvancedTrainingConfig(**loaded_config_dict)
            
            new_trainer = AdvancedLoRATrainer(loaded_config)
            state_dict = torch.load(model_path, map_location=new_trainer.device)
            new_trainer.model.load_state_dict(state_dict, strict=False)
            
            # Compare model info
            original_info = trainer.model.get_model_info()
            loaded_info = new_trainer.model.get_model_info()
            
            assert original_info['trainable_params'] == loaded_info['trainable_params'], "Trainable params should match"
            assert original_info['total_params'] == loaded_info['total_params'], "Total params should match"
            
            logger.info("✅ Save/load functionality working")
            return True
            
    except Exception as e:
        logger.error(f"❌ Save/load functionality failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing data loading...")
    
    config = AdvancedTrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        use_wandb=False,
        debug_mode=True,
        batch_size=1,
        max_seq_length=128,
        lora_r=8,
        lora_alpha=16,
        external_memory_size=32,
        retrieval_k=4,
        train_data_path="src/mem_aug/data/temp_data/train.jsonl",
        val_data_path="src/mem_aug/data/temp_data/val.jsonl"
    )
    
    try:
        trainer = AdvancedLoRATrainer(config)
        
        # Check data loaders
        assert len(trainer.train_dataloader) > 0, "Should have training data"
        assert len(trainer.val_dataloader) > 0, "Should have validation data"
        
        # Test getting a batch
        batch = next(iter(trainer.train_dataloader))
        assert 'input_ids' in batch, "Batch should have input_ids"
        assert 'labels' in batch, "Batch should have labels"
        assert 'attention_mask' in batch, "Batch should have attention_mask"
        
        logger.info("✅ Data loading working")
        logger.info(f"   Train batches: {len(trainer.train_dataloader)}")
        logger.info(f"   Val batches: {len(trainer.val_dataloader)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    logger = setup_logging()
    logger.info("🚀 Starting Advanced LoRA Trainer Test Suite")
    
    tests = [
        ("Trainer Initialization", test_trainer_initialization),
        ("Model Inference", test_model_inference),
        ("Save/Load Functionality", test_save_load_functionality),
        ("Data Loading", test_data_loading)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\\n--- Running {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\\n🎯 Test Results Summary:")
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\\n📊 Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Advanced LoRA Trainer is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)