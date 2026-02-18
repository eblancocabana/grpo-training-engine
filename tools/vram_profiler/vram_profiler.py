
"""
VRAM Profiler for PyTorch.
Captures low-level memory events using torch.cuda.memory._record_memory_history.
"""
import sys
import os
import argparse
import traceback
import json
import torch

# Module-level constants
AUTO_COMPRESS_THRESHOLD_MB = 150  # Threshold in MB for auto-compressing snapshot files

# 1. Early Startup Hook
def start_profiling(context="all", max_entries=2000000):
    if not torch.cuda.is_available():
        print("[Error] CUDA not available.")
        sys.exit(1)

    if not hasattr(torch.cuda.memory, "_record_memory_history"):
        print("[Error] PyTorch version does not support _record_memory_history.")
        sys.exit(1)

    print(f"[Profiler] Starting recording (context={context})...")
    try:
        torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=max_entries,
            trace_alloc_record_context=True if context != "none" else False
        )
    except Exception as e:
        print(f"[Warning] Failed to start with context={context}: {e}")
        torch.cuda.memory._record_memory_history(True, trace_alloc_record_context=False)

def stop_and_save(output_file, metadata=None):
    print(f"[Profiler] Saving snapshot to {output_file}...")
    try:
        if hasattr(torch.cuda.memory, "_snapshot"):
            s = torch.cuda.memory._snapshot()
            
            # Inject Metadata if applicable
            if metadata and isinstance(s, dict):
                s['user_metadata'] = metadata
            
            with open(output_file, 'w') as f:
                json.dump(s, f)
            print("[Profiler] Saved JSON snapshot.")
        else:
            torch.cuda.memory._dump_snapshot(output_file)
            print("[Profiler] Saved Pickle snapshot.")
    except Exception as e:
        print(f"[Error] Failed to dump snapshot: {e}")
    finally:
        torch.cuda.memory._record_memory_history(False)

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="VRAM Profiler")
    parser.add_argument("--steps", type=int, default=3, help="Training steps to run")
    parser.add_argument("--output", type=str, default="vram_snapshot.json", help="Output JSON file")
    parser.add_argument("--context", choices=["all", "state", "none"], default="all", help="Stack trace level")
    parser.add_argument("--max-entries", type=int, default=2000000, help="Max history entries")
    args = parser.parse_args()

    start_profiling(args.context, args.max_entries)

    try:
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config
        import src.grpo.trainer as trainer_module
        
        try:
            import compress_snapshot
        except ImportError:
            # Try to import from local dir if running from root
            sys.path.append(os.path.dirname(__file__))
            try:
                import compress_snapshot
            except ImportError:
                print("[Error] Could not import 'compress_snapshot'. "
                      "Ensure compress_snapshot.py is in the same directory as vram_profiler.py "
                      "or that the module is available on PYTHONPATH.")
                sys.exit(1)

        print("[Profiler] Initializing Trainer...")
        config = get_8gb_vram_config() # Function name kept for compatibility
        config.wandb.enabled = False
        config.training.num_epochs = 1
        config.training.save_interval = 999999
        config.training.output_dir = os.path.join(PROJECT_ROOT, "profiler_output")
        
        trainer = GRPOTrainerLoop(config)

        # Custom mini-loop by patching train_epoch
        from tqdm import tqdm

        # Custom mini-loop by patching train_epoch
        original_train_epoch = trainer.train_epoch
        def limited_train_epoch(dataloader, epoch):
            print(f"[Profiler] Training for {args.steps} steps...")
            
            mock_self = trainer
            mock_self.model.train()
            it = iter(dataloader)
            
            # Use tqdm for progress feedback
            pbar = tqdm(range(args.steps), desc="Profiling Steps", unit="step")
            
            # Reset peak stats before loop
            torch.cuda.reset_peak_memory_stats()
            
            for i in pbar:
                try:
                    batch = next(it)
                    mock_self.training_step(batch)
                    
                    # Update description with Peak VRAM (more useful for OOMs)
                    if torch.cuda.is_available():
                        peak = torch.cuda.max_memory_allocated() / 1024**3
                        curr = torch.cuda.memory_allocated() / 1024**3
                        pbar.set_postfix(vram=f"{curr:.2f}GB", peak=f"{peak:.2f}GB")
                         
                except StopIteration:
                    break
            print("[Profiler] Limited epoch complete.")
            
        trainer.train_epoch = limited_train_epoch

        print("[Profiler] Setting up model...")
        trainer.setup()
        
        # Disable side effects after setup (since setup() initializes these components)
        trainer.save_checkpoint = lambda *a, **k: None
        trainer.save_lora_weights = lambda *a, **k: None
        trainer._finish_wandb = lambda *a, **k: None
        if trainer.benchmark:
            trainer.benchmark.run = lambda *a, **k: {}
        
        print("[Profiler] Running active training session...")
        trainer.train()
        print("[Profiler] Done.")

    except Exception as e:
        print(f"\n[Profiler] Crashed: {e}")
        traceback.print_exc()
    finally:
        # Collect Metadata
        meta = {
            "steps": args.steps,
            "context": args.context,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "timestamp": "auto-generated" 
        }
        
        stop_and_save(args.output, metadata=meta)
        
        if os.path.exists(args.output):
            file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"[Profiler] Snapshot size: {file_size_mb:.2f} MB")
            
            if file_size_mb > AUTO_COMPRESS_THRESHOLD_MB:
                print("[Profiler] File is large. Auto-compressing...")
                compressed_path = args.output + ".tmp"
                try:
                    compress_snapshot.compress_snapshot(args.output, compressed_path)
                    
                    # Replace original with compressed
                    os.remove(args.output)
                    os.rename(compressed_path, args.output)
                    
                    new_size = os.path.getsize(args.output) / (1024 * 1024)
                    print(f"[Profiler] Compression complete: {new_size:.2f} MB. Original deleted.")
                except Exception as e:
                    print(f"[Profiler] Compression failed: {e}")
            else:
                print("[Profiler] File size is within limits. No compression needed.")

        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            print(f"Final VRAM: {stats.get('allocated_bytes.all.current', 0)/1024**3:.2f}GB allocated")

if __name__ == "__main__":
    main()
