"""
Simple test for PEFT (LoRA) integration with hybrid memory architecture.
Tests that externally created PEFT models work with the memory system.
"""

import pytest
import torch
import sys
from pathlib import Path

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


class TestPEFTIntegration:
    """Test PEFT integration with hybrid memory architecture."""
    
    @pytest.fixture(scope="class")
    def test_config(self):
        return {
            "model_name": "microsoft/DialoGPT-small",
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
    
    def create_hybrid_config(self, base_config):
        """Create hybrid configuration for testing."""
        return HybridTransformerConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=2,  # Small for testing
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=getattr(base_config, 'intermediate_size', base_config.hidden_size * 4),
            max_position_embeddings=getattr(base_config, 'max_position_embeddings', 1024),
            
            # Memory parameters
            use_internal_memory=True,
            memory_slots=4,
            num_mem_heads=2,
            use_external_memory=FAISS_AVAILABLE,
            external_memory_size=512,
            retrieval_k=2,
            chunk_size=4,
            max_batch_size=2,
            seq_adapt_method="compress",
            
            # Fast embedding layer path for DialoGPT
            embedding_layer_path="wte",  # DialoGPT uses 'wte'
        )

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT not available")
    def test_peft_model_creation(self, test_config, base_config, tokenizer):
        """Test creating PEFT model externally."""
        # Create base model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"],
            bias="none",
        )
        
        peft_model = get_peft_model(base_model, lora_config)
        
        # Verify PEFT model properties
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        assert trainable_params > 0
        assert trainable_params < total_params
        assert hasattr(peft_model, 'base_model')
        
        print(f"PEFT Model: {trainable_params:,}/{total_params:,} trainable params")

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT not available")
    def test_hybrid_model_with_peft(self, test_config, base_config, tokenizer):
        """Test hybrid model with external PEFT model."""
        # Create PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config)
        
        # Create hybrid model with PEFT model
        hybrid_config = self.create_hybrid_config(base_config)
        model = HybridTransformerModel(hybrid_config, tokenizer, base_model=peft_model)
        model = model.to(test_config["device"])
        model.eval()
        
        # Verify model info
        model_info = model.get_model_info()
        assert model_info["has_peft"] == True
        assert model_info["trainable_params"] > 0
        
        print(f"Hybrid + PEFT: {model_info['trainable_percentage']:.2f}% trainable")

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT not available")
    def test_inference_with_peft(self, test_config, base_config, tokenizer):
        """Test inference with PEFT-enhanced hybrid model."""
        # Create PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,  # Smaller for faster test
            lora_alpha=8,
            target_modules=["c_attn"],
        )
        peft_model = get_peft_model(base_model, lora_config)
        
        # Create hybrid model
        hybrid_config = self.create_hybrid_config(base_config)
        model = HybridTransformerModel(hybrid_config, tokenizer, base_model=peft_model)
        model = model.to(test_config["device"])
        model.eval()
        
        # Test inference
        test_input = tokenizer("def hello():", return_tensors="pt").to(test_config["device"])
        
        with torch.no_grad():
            logits, loss, memory_info = model(test_input["input_ids"])
        
        # Verify outputs
        assert logits.shape[0] == 1  # batch size
        assert logits.shape[1] == test_input["input_ids"].shape[1]  # sequence length
        assert memory_info is not None
        assert memory_info["internal_memory"] is not None
        
        print(f"Inference successful: output shape {logits.shape}")

    def test_hybrid_model_without_peft(self, test_config, base_config, tokenizer):
        """Test hybrid model works without PEFT too."""
        # Create regular model
        base_model = AutoModelForCausalLM.from_pretrained(test_config["model_name"])
        
        # Create hybrid model
        hybrid_config = self.create_hybrid_config(base_config)
        model = HybridTransformerModel(hybrid_config, tokenizer, base_model=base_model)
        model = model.to(test_config["device"])
        model.eval()
        
        # Verify model info
        model_info = model.get_model_info()
        assert model_info["has_peft"] == False
        
        # Test inference
        test_input = tokenizer("x = 1", return_tensors="pt").to(test_config["device"])
        
        with torch.no_grad():
            logits, loss, memory_info = model(test_input["input_ids"])
        
        assert logits.shape[0] == 1
        assert memory_info is not None
        
        print("Non-PEFT model working correctly")


def run_peft_tests():
    """Run PEFT integration tests."""
    print("="*60)
    print("PEFT INTEGRATION TESTS")
    print("="*60)
    
    if not PEFT_AVAILABLE:
        print("❌ PEFT not available. Skipping tests.")
        return False
    
    try:
        # Initialize test components
        test_config = {"model_name": "microsoft/DialoGPT-small", "device": "cpu"}
        
        tokenizer = AutoTokenizer.from_pretrained(test_config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_config = AutoConfig.from_pretrained(test_config["model_name"])
        
        # Initialize test class
        test_class = TestPEFTIntegration()
        
        print("\n1. Testing PEFT model creation...")
        test_class.test_peft_model_creation(test_config, base_config, tokenizer)
        
        print("\n2. Testing hybrid model with PEFT...")
        test_class.test_hybrid_model_with_peft(test_config, base_config, tokenizer)
        
        print("\n3. Testing inference with PEFT...")
        test_class.test_inference_with_peft(test_config, base_config, tokenizer)
        
        print("\n4. Testing hybrid model without PEFT...")
        test_class.test_hybrid_model_without_peft(test_config, base_config, tokenizer)
        
        print("\n" + "="*60)
        print("✅ ALL PEFT TESTS PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ PEFT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_peft_tests()
    sys.exit(0 if success else 1)