#!/usr/bin/env python3
"""
Quick inference script to test a trained model.
Load LoRA weights and generate responses.
"""
import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.model_loader import load_4bit_engine
from src.core.lora import inject_lora_layers
from src.utils.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--lora-weights",
        type=str,
        required=True,
        help="Path to LoRA weights file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRPO Inference")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading base model...")
    model, tokenizer = load_4bit_engine()
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Inject LoRA
    print("\n[2/3] Injecting LoRA layers...")
    inject_lora_layers(
        model,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        rank=16,
        verbose=False
    )
    
    # Load trained weights
    print(f"\n[3/3] Loading LoRA weights from {args.lora_weights}...")
    checkpoint_manager = CheckpointManager(checkpoint_dir=".")
    
    try:
        metadata = checkpoint_manager.load_lora_weights(args.lora_weights, model)
        print(f"   Loaded weights from step {metadata.get('step', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return
    
    # Format prompt
    prompt = f"<|User|>{args.prompt}<|endoftext|><|Assistant|><think>\n"
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Generate
    print("Generating...\n")
    model.eval()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the response (after the prompt)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    response = generated_text[len(prompt_text):]
    
    print("Response:")
    print(response)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
