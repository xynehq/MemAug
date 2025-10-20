#!/usr/bin/env python3
"""
Inference script using generate_with_retrieval method.
This uses the custom generation loop with active external memory retrieval.
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


def load_model_with_retrieval(checkpoint_dir: str, device: str = "auto"):
    """Load hybrid model with external memory for generation."""
    print(f"Loading model from: {checkpoint_dir}")
    
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
        
        # Load state dict
        model_state = checkpoint.get('model_state_dict', {})
        try:
            base_model.load_state_dict(model_state, strict=False)
            print(f"Loaded checkpoint with {len(model_state)} parameters")
        except Exception as e:
            print(f"Warning: Could not load some parameters: {e}")
    
    # Create hybrid config with external memory enabled
    base_config_dict = base_model.config.to_dict()
    hybrid_config = HybridTransformerConfig(
        **base_config_dict,
        use_internal_memory=False,  # Disable for simplicity
        use_external_memory=True,   # Enable retrieval
        external_memory_size=2048,
        retrieval_k=4,
        chunk_size=4,
        use_gpu_to_search=False,
        use_gated_retrieval=False  # Disable gating for now
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


def generate_completion(
    model,
    tokenizer,
    device,
    prefix: str,
    suffix: str = "",
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    use_retrieval: bool = True,
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
    
    input_ids = inputs["input_ids"]
    
    print(f"Generating with retrieval={'ENABLED' if use_retrieval else 'DISABLED'}...")
    
    # Generate using custom generation loop with retrieval
    try:
        with torch.no_grad():
            generated_ids = model.generate_with_retrieval(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_external_memory=use_retrieval
            )
        
        # Decode only the generated part
        generated_tokens = generated_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"{'='*80}")
        print(f"COMPLETION (retrieval={'ON' if use_retrieval else 'OFF'}):")
        print(f"{'='*80}")
        print(generated_text)
        print(f"{'='*80}\n")
        
        return generated_text
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode(model, tokenizer, device):
    """Interactive generation mode."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("="*80 + "\n")
    
    while True:
        try:
            prefix = input("\nEnter prefix (or 'quit'): ").strip()
            if prefix.lower() == 'quit':
                break
            
            suffix = input("Enter suffix (optional): ").strip()
            
            # Generate with retrieval
            print("\n--- WITH RETRIEVAL ---")
            generate_completion(
                model, tokenizer, device,
                prefix=prefix,
                suffix=suffix,
                max_length=50,
                use_retrieval=True
            )
            
            # Generate without retrieval for comparison
            print("\n--- WITHOUT RETRIEVAL (for comparison) ---")
            generate_completion(
                model, tokenizer, device,
                prefix=prefix,
                suffix=suffix,
                max_length=50,
                use_retrieval=False
            )
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Inference with retrieval")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/improved_training/best_model",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Code prefix for completion"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Code suffix for FIM"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable retrieval (for comparison)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("INFERENCE WITH RETRIEVAL")
    print("="*80)
    print("\nThis uses the custom generate_with_retrieval method")
    print("that supports active external memory retrieval.\n")
    
    # Load model
    model, tokenizer, device = load_model_with_retrieval(args.checkpoint, args.device)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation
        if args.prefix is None:
            # Default examples
            examples = [
                {
                    "prefix": "fn main() {",
                    "suffix": "}",
                    "description": "Simple main function"
                },
                {
                    "prefix": "pub struct User {\n    name: String,",
                    "suffix": "}",
                    "description": "Struct definition"
                },
                {
                    "prefix": "fn add(a: i32, b: i32) -> i32 {",
                    "suffix": "}",
                    "description": "Function implementation"
                }
            ]
            
            for i, example in enumerate(examples, 1):
                print(f"\n{'='*80}")
                print(f"EXAMPLE {i}: {example['description']}")
                print(f"{'='*80}")
                
                # With retrieval
                print("\n--- WITH RETRIEVAL ---")
                generate_completion(
                    model, tokenizer, device,
                    prefix=example["prefix"],
                    suffix=example["suffix"],
                    max_length=args.max_length,
                    temperature=args.temperature,
                    use_retrieval=True
                )
                
                # Without retrieval for comparison
                print("\n--- WITHOUT RETRIEVAL (for comparison) ---")
                generate_completion(
                    model, tokenizer, device,
                    prefix=example["prefix"],
                    suffix=example["suffix"],
                    max_length=args.max_length,
                    temperature=args.temperature,
                    use_retrieval=False
                )
        else:
            # User-provided example
            result = generate_completion(
                model, tokenizer, device,
                prefix=args.prefix,
                suffix=args.suffix,
                max_length=args.max_length,
                temperature=args.temperature,
                use_retrieval=not args.no_retrieval
            )
            
            if result is not None:
                print("✅ Generation successful!")
            else:
                print("❌ Generation failed!")


if __name__ == "__main__":
    main()
