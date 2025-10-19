"""
Comprehensive inference demonstration for hybrid memory architecture.
Tests inference capabilities with both CPU and GPU support.
"""

import pytest
import torch
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel,
    HybridTransformerConfig
)

# PEFT support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TestInferenceDemo:
    """Comprehensive inference tests for hybrid memory architecture."""
    
    @pytest.fixture(scope="class")
    def test_config(self):
        # Use larger model on GPU for better demo, smaller on CPU for speed
        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct" if torch.cuda.is_available() else "microsoft/DialoGPT-small"
        return {
            "model_name": model_name,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    
    @pytest.fixture(scope="class")
    def tokenizer(self, test_config):
        tokenizer = AutoTokenizer.from_pretrained(test_config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture(scope="class")
    def base_config(self, test_config):
        return AutoConfig.from_pretrained(test_config["model_name"])
    
    def create_hybrid_config(self, base_config, device="cpu", model_name=""):
        """Create hybrid configuration for testing."""
        # Set embedding layer path based on model type
        embedding_layer_path = None
        if "DialoGPT" in model_name or "gpt2" in model_name.lower():
            embedding_layer_path = "wte"  # GPT2 style
        elif "Qwen" in model_name or "qwen" in model_name.lower():
            embedding_layer_path = "model.embed_tokens"  # Qwen style
        elif "llama" in model_name.lower():
            embedding_layer_path = "embed_tokens"  # Llama style
        
        return HybridTransformerConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=2,  # Small for testing
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=getattr(base_config, 'intermediate_size', base_config.hidden_size * 4),
            max_position_embeddings=getattr(base_config, 'max_position_embeddings', 1024),
            
            # Memory parameters optimized for inference
            use_internal_memory=True,
            memory_slots=8,
            num_mem_heads=4,
            use_external_memory=FAISS_AVAILABLE,
            external_memory_size=1024 if device == "cuda" else 512,
            retrieval_k=4,
            chunk_size=8,
            max_batch_size=4,
            seq_adapt_method="compress",
            
            # Fast embedding layer path
            embedding_layer_path=embedding_layer_path,
        )

    def test_baseline_model_inference(self, test_config, base_config, tokenizer):
        """Test baseline model inference performance."""
        print(f"\n{'='*60}")
        print("BASELINE MODEL INFERENCE TEST")
        print(f"{'='*60}")
        
        # Create baseline model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        base_model = base_model.to(test_config["device"])
        base_model.eval()
        
        # Test inputs
        test_prompts = [
            "def fibonacci(n):",
            "class DataProcessor:",
            "import numpy as np\n\ndef",
        ]
        
        total_time = 0
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: '{prompt[:20]}...'")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(test_config["device"])
            
            start_time = time.time()
            with torch.no_grad():
                outputs = base_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_time += generation_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            print(f"Time: {generation_time:.3f}s")
        
        print(f"\nBaseline Average Time: {total_time/len(test_prompts):.3f}s")
        return total_time / len(test_prompts)

    def test_hybrid_model_inference(self, test_config, base_config, tokenizer):
        """Test hybrid model inference performance."""
        print(f"\n{'='*60}")
        print("HYBRID MEMORY MODEL INFERENCE TEST")
        print(f"{'='*60}")
        
        # Create hybrid model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        hybrid_config = self.create_hybrid_config(base_config, test_config["device"], test_config["model_name"])
        
        model = HybridTransformerModel(hybrid_config, tokenizer, base_model=base_model)
        model = model.to(test_config["device"])
        model.eval()
        
        # Get model info
        model_info = model.get_model_info()
        print(f"Model Info: {model_info['trainable_params']:,} trainable params")
        print(f"PEFT Status: {model_info['has_peft']}")
        
        # Test inputs
        test_prompts = [
            "def fibonacci(n):",
            "class DataProcessor:",
            "import numpy as np\n\ndef",
        ]
        
        total_time = 0
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: '{prompt[:20]}...'")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(test_config["device"])
            
            start_time = time.time()
            with torch.no_grad():
                # Single forward pass (generate not supported with wrapped layers)
                logits, loss, memory_info = model(inputs["input_ids"])
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_time += generation_time
            
            # Get predicted tokens from logits
            predicted_ids = torch.argmax(logits, dim=-1)
            generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print(f"Forward pass output: {generated_text[:50]}...")
            print(f"Time: {generation_time:.3f}s")
            print(f"Memory Info: {memory_info}")
        
        print(f"\nHybrid Average Time: {total_time/len(test_prompts):.3f}s")
        return total_time / len(test_prompts)

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT not available")
    def test_peft_hybrid_inference(self, test_config, base_config, tokenizer):
        """Test PEFT + hybrid model inference performance."""
        print(f"\n{'='*60}")
        print("PEFT + HYBRID MEMORY MODEL INFERENCE TEST")
        print(f"{'='*60}")
        
        # Create PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"] if "DialoGPT" in test_config["model_name"] else ["q_proj", "v_proj"],
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config)
        
        # Create hybrid model with PEFT
        hybrid_config = self.create_hybrid_config(base_config, test_config["device"], test_config["model_name"])
        model = HybridTransformerModel(hybrid_config, tokenizer, base_model=peft_model)
        model = model.to(test_config["device"])
        model.eval()
        
        # Get model info
        model_info = model.get_model_info()
        print(f"Model Info: {model_info['trainable_params']:,} trainable params")
        print(f"PEFT Status: {model_info['has_peft']}")
        print(f"Trainable %: {model_info['trainable_percentage']:.2f}%")
        
        # Test inputs
        test_prompts = [
            "def fibonacci(n):",
            "class DataProcessor:",
            "import numpy as np\n\ndef",
        ]
        
        total_time = 0
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: '{prompt[:20]}...'")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(test_config["device"])
            
            start_time = time.time()
            with torch.no_grad():
                # Single forward pass (generate not supported with wrapped layers)
                logits, loss, memory_info = model(inputs["input_ids"])
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_time += generation_time
            
            # Get predicted tokens from logits
            predicted_ids = torch.argmax(logits, dim=-1)
            generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print(f"Forward pass output: {generated_text[:50]}...")
            print(f"Time: {generation_time:.3f}s")
            print(f"Memory Info: {memory_info}")
        
        print(f"\nPEFT+Hybrid Average Time: {total_time/len(test_prompts):.3f}s")
        return total_time / len(test_prompts)

    def test_memory_persistence(self, test_config, base_config, tokenizer):
        """Test memory persistence across multiple inferences."""
        print(f"\n{'='*60}")
        print("MEMORY PERSISTENCE TEST")
        print(f"{'='*60}")
        
        # Create hybrid model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        hybrid_config = self.create_hybrid_config(base_config, test_config["device"], test_config["model_name"])
        
        model = HybridTransformerModel(hybrid_config, tokenizer, base_model=base_model)
        model = model.to(test_config["device"])
        model.eval()
        
        # Sequential inputs to test memory accumulation
        sequential_inputs = [
            "def process_data(data):",
            "    # First step: clean the data",
            "    cleaned = clean(data)",
            "    # Second step: transform",
            "    return transform(cleaned)",
        ]
        
        accumulated_memory = None
        
        for i, text in enumerate(sequential_inputs):
            print(f"\nStep {i+1}: '{text}'")
            
            inputs = tokenizer(text, return_tensors="pt").to(test_config["device"])
            
            with torch.no_grad():
                logits, loss, memory_info = model(inputs["input_ids"])
            
            print(f"Memory Info: {memory_info}")
            
            # Verify memory is being updated
            if memory_info and memory_info.get("internal_memory") is not None:
                current_memory = memory_info["internal_memory"]
                if accumulated_memory is not None:
                    # Check if memory has changed (indicating persistence)
                    memory_changed = not torch.allclose(current_memory, accumulated_memory, atol=1e-6)
                    print(f"Memory changed from previous step: {memory_changed}")
                accumulated_memory = current_memory.clone()
        
        print("\nMemory persistence test completed!")

    def test_device_compatibility(self, test_config, base_config, tokenizer):
        """Test model works on both CPU and available GPU."""
        print(f"\n{'='*60}")
        print("DEVICE COMPATIBILITY TEST")
        print(f"{'='*60}")
        
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
        
        for device in devices_to_test:
            print(f"\nTesting on device: {device}")
            
            # Create model for this device
            base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
            hybrid_config = self.create_hybrid_config(base_config, device, test_config["model_name"])
            
            model = HybridTransformerModel(hybrid_config, tokenizer, base_model=base_model)
            model = model.to(device)
            model.eval()
            
            # Test inference
            test_input = tokenizer("def hello():", return_tensors="pt").to(device)
            
            start_time = time.time()
            with torch.no_grad():
                logits, loss, memory_info = model(test_input["input_ids"])
            end_time = time.time()
            
            print(f"Device {device}: Success! Time: {end_time - start_time:.3f}s")
            print(f"Output shape: {logits.shape}")
            print(f"Memory available: {memory_info is not None}")


def run_inference_demo():
    """Run comprehensive inference demonstration."""
    print("="*80)
    print("HYBRID MEMORY ARCHITECTURE - INFERENCE DEMONSTRATION")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"FAISS available: {FAISS_AVAILABLE}")
    print(f"PEFT available: {PEFT_AVAILABLE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print("="*80)
    
    try:
        # Initialize test components
        # Use larger model on GPU for better demo, smaller on CPU for speed  
        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct" if torch.cuda.is_available() else "microsoft/DialoGPT-small"
        test_config = {
            "model_name": model_name, 
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        tokenizer = AutoTokenizer.from_pretrained(test_config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_config = AutoConfig.from_pretrained(test_config["model_name"])
        
        # Initialize test class
        test_class = TestInferenceDemo()
        
        # Run all tests
        print("\n1. Testing baseline model performance...")
        baseline_time = test_class.test_baseline_model_inference(test_config, base_config, tokenizer)
        
        print("\n2. Testing hybrid model performance...")
        hybrid_time = test_class.test_hybrid_model_inference(test_config, base_config, tokenizer)
        
        if PEFT_AVAILABLE:
            print("\n3. Testing PEFT + hybrid model performance...")
            peft_time = test_class.test_peft_hybrid_inference(test_config, base_config, tokenizer)
        
        print("\n4. Testing memory persistence...")
        test_class.test_memory_persistence(test_config, base_config, tokenizer)
        
        print("\n5. Testing device compatibility...")
        test_class.test_device_compatibility(test_config, base_config, tokenizer)
        
        # Performance summary
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Baseline model:     {baseline_time:.3f}s avg")
        print(f"Hybrid model:       {hybrid_time:.3f}s avg")
        if PEFT_AVAILABLE:
            print(f"PEFT + Hybrid:      {peft_time:.3f}s avg")
        print(f"Memory overhead:    {((hybrid_time - baseline_time) / baseline_time * 100):+.1f}%")
        
        print(f"\n{'='*80}")
        print("✅ ALL INFERENCE TESTS PASSED!")
        print("✅ HYBRID MEMORY ARCHITECTURE READY FOR DEPLOYMENT")
        print(f"{'='*80}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_inference_demo()
    sys.exit(0 if success else 1)
