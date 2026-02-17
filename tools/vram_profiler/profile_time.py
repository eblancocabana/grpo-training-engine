"""
Latency/Performance Profiler for PyTorch (Corrected for 4-bit loading & Dataloader).
Generates a 'trace.json' compatible with chrome://tracing or brave://tracing.
"""
import sys
import os
import argparse
import traceback
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 1. Setup Path EXACTLY like vram_profiler.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Performance Profiler")
    parser.add_argument("--steps", type=int, default=5, help="Total steps")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup steps")
    parser.add_argument("--active", type=int, default=1, help="Active steps")
    parser.add_argument("--output", type=str, default="trace_final.json", help="Output trace file")
    args = parser.parse_args()

    print(f"[Profiler] Initializing (Compatible Mode)...")
    
    # Force Unbuffered Output
    sys.stdout.reconfigure(line_buffering=True)

    # Force CUDA initialization
    if torch.cuda.is_available():
        torch.cuda.init()

    try:
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config
        from tqdm import tqdm
        import src.grpo.trainer as trainer_module
        
        # MONKEY PATCH: Disable num_workers to prevent hangs
        original_create_dataloader = trainer_module.create_grpo_dataloader
        def patched_create_dataloader(*args, **kwargs):
            print("[Profiler] Creating Dataloader (Forced num_workers=0)...")
            loader = original_create_dataloader(*args, **kwargs)
            loader.num_workers = 0 
            loader.prefetch_factor = None # prefetch_factor requires num_workers > 0
            return loader
        
        trainer_module.create_grpo_dataloader = patched_create_dataloader

        # Load Config
        config = get_8gb_vram_config()
        config.wandb.enabled = False 
        config.training.output_dir = os.path.join(PROJECT_ROOT, "profiler_output_time")
        # Ensure we only run 1 epoch so trainer.train() finishes
        config.training.num_epochs = 1 
        
        # REDUCE MEMORY FOR PROFILER OVERHEAD
        print("[Profiler] Tuning config for VRAM limits + Profiler Overhead...")
        config.training.max_prompt_length = 512   # Reduced from 1024 to save VRAM
        config.training.max_response_length = 128 # Reduced from 256
        config.grpo.group_size = 1                # Reduced from 2 (Critical for OOM)
        # We need to ensure we don't break the model loading, but these are runtime configs for data/generation.
        
        # Initialize Trainer
        print("[Profiler] Creating Trainer...")
        trainer = GRPOTrainerLoop(config)
        
        # Define the patched training loop
        # This will receive the dataloader correctly created by trainer.train()
        def profiled_train_epoch(dataloader, epoch):
            print(f"[Profiler] Starting Capture (Wait=1, Warmup={args.warmup}, Active={args.active})...")
            
            trainer.model.train()
            it = iter(dataloader)
            
            # Profiler Context
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=args.warmup, active=args.active, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=False # Disabled to reduce file size (was 2GB+)
            ) as prof:
                
                # Run exactly enough steps for the schedule
                total_steps = 1 + args.warmup + args.active
                pbar = tqdm(range(total_steps), desc="Profiling")
                
                for step in pbar:
                    try:
                        batch = next(it)
                        
                        # Wrap the training step
                        with record_function("model_training_step"):
                            trainer.training_step(batch)
                            
                        prof.step()  # Signal step completion to profiler
                        
                    except StopIteration:
                        break
            
            print(f"[Profiler] Exporting chrome trace to {args.output}...")
            prof.export_chrome_trace(args.output)
            
            # Gzip the file
            if os.path.exists(args.output):
                print(f"[Profiler] Compressing trace to {args.output}.gz...")
                import gzip
                import shutil
                with open(args.output, 'rb') as f_in:
                    with gzip.open(f"{args.output}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(args.output)
                print(f"[Profiler] Done! Open brave://tracing and load {os.path.abspath(args.output)}.gz")
            
            return

        # Monkey-patch
        trainer.train_epoch = profiled_train_epoch

        # Setup Model (triggers 4-bit loading)
        print("[Profiler] Loading Model (4-bit)...")
        trainer.setup()
        
        # Disable side effects
        trainer.save_checkpoint = lambda *a, **k: None
        trainer.save_lora_weights = lambda *a, **k: None
        trainer._finish_wandb = lambda *a, **k: None
        if trainer.benchmark:
            trainer.benchmark.run = lambda *a, **k: {}

        # Run Trainer (this will call create_dataloader and then our patched train_epoch)
        print("[Profiler] Running active training session...")
        try:
            if hasattr(trainer, 'train'):
                trainer.train()
            else:
                print("[Error] Trainer has no 'train' method.")

        except Exception as e:
            if "Profiling complete" not in str(e): raise e

    except Exception as e:
        print(f"\n[Error] Profiling failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
