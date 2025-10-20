#!/usr/bin/env python3
"""
Inference script using the full hybrid model WITH external memory retrieval.
Tests the fix for generation compatibility.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel,
    HybridTransformerConfig
)


def load_hybrid_model(checkpoint_dir: str, device: str = "auto"):
    """Load the full hybrid model with external memory."""
    print(f"Loading hybrid model from: {checkpoint_dir}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_name = config_dict.get("model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    else:
        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load full state (including LoRA)
        model_state = checkpoint.get('model_state_dict', {})
        try:
            base_model.load_state_dict(model_state, strict=False)
            print(f"Loaded checkpoint with {len(model_state)} parameters")
        except Exception as e:
            print(f"Warning: Could not load some parameters: {e}")
    
    # Create hybrid config
    base_config_dict = base_model.config.to_dict()
    hybrid_config = HybridTransformerConfig(
        **base_config_dict,
        use_internal_memory=False,
        use_external_memory=True,
        external_memory_size=2048,
        retrieval_k=4,
        chunk_size=4,
        use_gpu_to_search=False,
        use_gated_retrieval=True
    )
    
    # Create hybrid model
    print("Creating hybrid model with external memory...")
    hybrid_model = HybridTransformerModel(
        config=hybrid_config,
        tokenizer=tokenizer,
        base_model=base_model
    )
    
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()
    
    print(f"Model info: {hybrid_model.get_model_info()}")
    
    return hybrid_model, tokenizer, device


def generate_with_retrieval(
    model,
    tokenizer,
    device,
    prefix: str,
    suffix: str = "",
    max_length: int = 100,
    temperature: float = 0.7,
):
    """Generate code completion using FIM format with retrieval."""
    
    # Build FIM prompt
    if suffix:
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    else:
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|><|fim_middle|>"
    
    print(f"\n{'='*80}")
    print("INPUT:")
    print(f"{'='*80}")
    print(f"Prefix: {prefix}")
    if suffix:
        print(f"Suffix: {suffix}")
    print(f"{'='*80}\n")
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    print("Generating with hybrid model (retrieval enabled)...")
    
    # Generate using the hybrid model
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"{'='*80}")
        print("COMPLETION (with retrieval):")
        print(f"{'='*80}")
        print(generated_text)
        print(f"{'='*80}\n")
        
        return generated_text
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run inference with hybrid model + retrieval")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/improved_training/best_model",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="fn main() {",
        help="Code prefix for completion"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="}",
        help="Code suffix for FIM"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TESTING HYBRID MODEL GENERATION WITH RETRIEVAL")
    print("="*80)
    print("\nThis tests if the __getattr__ fix resolves the generation issue.\n")
    
    # Load hybrid model
    model, tokenizer, device = load_hybrid_model(args.checkpoint, args.device)
    
    # Test generation
    result = generate_with_retrieval(
        model, tokenizer, device,
        prefix=args.prefix,
        suffix=args.suffix,
        max_length=args.max_length
    )
    
    if result is not None:
        print("✅ SUCCESS! Generation with hybrid model works!")
        print("The __getattr__ fix resolved the compatibility issue.")
    else:
        print("❌ FAILED! Generation still has issues.")
        print("Additional debugging needed.")


if __name__ == "__main__":
    main()
