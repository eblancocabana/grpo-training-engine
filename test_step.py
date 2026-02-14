#!/usr/bin/env python3
"""
Diagnostic script to test a single GRPO step and verify logic.
"""
import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from src.grpo.trainer import GRPOTrainerLoop
from src.utils.config import get_8gb_vram_config
from src.data.gsm8k_loader import create_grpo_dataloader

def run_diagnostic():
    print("🚀 Starting GRPO Diagnostic Step...")
    
    config = get_8gb_vram_config()
    # Reduce lengths for quick test
    config.training.max_prompt_length = 128
    config.training.max_response_length = 128
    config.grpo.group_size = 4
    config.training.enable_gradient_checkpointing = False  # Disable for debug
    
    trainer = GRPOTrainerLoop(config)
    trainer.setup()
    
    # Load a single batch
    print("\n[Test] Loading data...")
    dataloader = create_grpo_dataloader(
        tokenizer=trainer.tokenizer,
        split="train",
        batch_size=1,
        max_prompt_length=config.training.max_prompt_length
    )
    batch = next(iter(dataloader))
    
    print(f"\n[Test] Question: {batch['questions'][0]}")
    print(f"[Test] Ground Truth: {batch['answers'][0]}")
    
    # Execute training step with logs
    print("\n[Test] Executing training step (FULL forward + backward)...")
    
    try:
        metrics = trainer.training_step(batch)
        print("\n✅ Training step completed successfully!")
        print("Metrics:", metrics)
    except Exception as e:
        print(f"\n❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Diagnostic complete! The engine is working correctly.")

if __name__ == "__main__":
    run_diagnostic()
