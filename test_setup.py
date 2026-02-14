#!/usr/bin/env python3
"""
Test script to verify system setup and model loading.
Run this first to ensure everything is configured correctly.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.core.model_loader import load_4bit_engine
from src.core.lora import inject_lora_layers, get_lora_parameters
from src.core.memory_manager import MemoryManager, print_model_memory_usage


def test_model_loading():
    """Test loading model in 4-bit."""
    print("\n" + "=" * 60)
    print("Test 1: Loading Model in 4-bit")
    print("=" * 60)
    
    try:
        model, tokenizer = load_4bit_engine()
        
        if model is None:
            print("❌ FAILED: Model loading returned None")
            return False
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {model.device}")
        
        # Check memory usage
        mem_footprint = model.get_memory_footprint() / (1024**3)
        print(f"   Memory footprint: {mem_footprint:.2f} GB")
        
        if mem_footprint > 2.0:
            print(f"⚠️  WARNING: Model using more memory than expected for 4-bit")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_injection():
    """Test injecting LoRA layers."""
    print("\n" + "=" * 60)
    print("Test 2: Injecting LoRA Layers")
    print("=" * 60)
    
    try:
        model, tokenizer = load_4bit_engine()
        
        print("Injecting LoRA (rank=16, alpha=32)...")
        num_injected = inject_lora_layers(
            model,
            target_modules=["q_proj", "v_proj"],
            rank=16,
            alpha=32,
            verbose=True
        )
        
        if num_injected == 0:
            print("❌ FAILED: No LoRA layers injected")
            return False
        
        print(f"✅ Injected {num_injected} LoRA layers")
        
        # Check trainable parameters
        trainable_params = get_lora_parameters(model)
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"✅ Trainable parameters: {total_trainable:,}")
        
        # Memory usage
        print_model_memory_usage(model, "Model with LoRA")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    """Test text generation."""
    print("\n" + "=" * 60)
    print("Test 3: Text Generation")
    print("=" * 60)
    
    try:
        model, tokenizer = load_4bit_engine()
        inject_lora_layers(model, target_modules=["q_proj", "v_proj"], rank=16, verbose=False)
        
        # Test prompt
        prompt = "What is 2 + 2?"
        print(f"\nPrompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        print("Generating...")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        print("✅ Generation successful")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_management():
    """Test memory management."""
    print("\n" + "=" * 60)
    print("Test 4: Memory Management")
    print("=" * 60)
    
    try:
        memory_manager = MemoryManager(device="cuda")
        
        print("Initial memory state:")
        memory_manager.print_memory_stats("  ")
        
        # Test cache clearing
        memory_manager.clear_cache(aggressive=True)
        print("✅ Cache cleared successfully")
        
        print("\nMemory state after clearing:")
        memory_manager.print_memory_stats("  ")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("GRPO Training System Test Suite")
    print("RTX 3060 Ti 8GB VRAM Setup")
    print("=" * 60)
    
    # Check CUDA
    print("\n[System] Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("❌ CRITICAL: CUDA not available!")
        print("   Make sure you have installed PyTorch with CUDA support.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA available: {gpu_name}")
    
    # Run tests
    tests = [
        ("Model Loading", test_model_loading),
        ("LoRA Injection", test_lora_injection),
        ("Text Generation", test_generation),
        ("Memory Management", test_memory_management),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready for training.")
        print("\nNext steps:")
        print("  1. Run: python train.py --dry-run")
        print("  2. Start training: python train.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - CUDA version mismatch: Check PyTorch CUDA version")
        print("  - Out of memory: Close other GPU applications")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
