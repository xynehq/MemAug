# Advanced LoRA + External Memory Test Suite

This directory contains comprehensive tests for the Advanced LoRA + External Memory training system.

## Test Files

### Core Tests
- **`test_advanced_lora_trainer.py`** - Comprehensive test suite covering all trainer functionality
- **`test_save_load_model.py`** - Validates model save/load functionality and configuration matching
- **`quick_train_test.py`** - Quick validation test for basic functionality

### Demonstration Scripts
- **`run_complete_training_demo.py`** - Complete training pipeline demonstration

### Legacy Tests (from original architecture)
- **`test_hybrid_architecture.py`** - Tests for hybrid memory architecture
- **`test_gated_retrieval.py`** - Tests for gated retrieval mechanism
- **`test_peft_integration.py`** - Tests for PEFT integration
- **`test_optimized_gating.py`** - Tests for optimized gating
- **`test_inference_demo.py`** - Inference demonstration

## Running Tests

### Individual Tests
```bash
# Run comprehensive test suite
python tests/test_advanced_lora_trainer.py

# Run save/load validation
python tests/test_save_load_model.py

# Run quick functionality test
python tests/quick_train_test.py

# Run training demonstration
python tests/run_complete_training_demo.py
```

### All Tests
```bash
python tests/run_all_tests.py
```

## Test Coverage

The tests cover:

### ✅ Core Functionality
- Trainer initialization with LoRA adapters
- External memory integration
- Model inference with memory retrieval
- Configuration management
- Data loading and preprocessing

### ✅ Save/Load Validation
- Model state preservation
- Configuration matching
- Parameter consistency
- PEFT adapter compatibility

### ✅ Training Pipeline
- Complete training workflow
- Advanced features (early stopping, LR scheduling)
- Monitoring and logging
- Error handling

### ✅ Production Readiness
- CPU compatibility
- Memory optimization
- Gradient handling
- Checkpointing

## Expected Output

All tests should pass with output similar to:
```
🎯 Test Results Summary:
   Trainer Initialization: ✅ PASS
   Model Inference: ✅ PASS
   Save/Load Functionality: ✅ PASS
   Data Loading: ✅ PASS

📊 Tests passed: 4/4
🎉 All tests passed! Advanced LoRA Trainer is working correctly.
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- FAISS-CPU
- NumPy

## Troubleshooting

### Common Issues
1. **FAISS Import Error**: Install with `pip install faiss-cpu`
2. **PEFT Import Error**: Install with `pip install peft`
3. **Memory Issues**: Reduce batch size or sequence length in test configs
4. **Wandb Errors**: Tests disable wandb by default, but ensure `use_wandb=False` in configs

### CPU Compatibility
All tests are optimized for CPU execution with:
- `use_amp=False`
- `fp16=False` 
- `bf16=False`
- `dataloader_num_workers=0`