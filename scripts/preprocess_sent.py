"""
SENT Preprocessing Script.

Computes semantic entropy for GSM8K dataset and generates sorted cache.
Can resume from checkpoint if interrupted.
"""
import os
import sys
import argparse
import logging
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from datasets import load_dataset

from src.core.model_loader import load_4bit_engine
from src.data.sent_calculator import SemanticEntropyCalculator
from src.grpo.verifier import RuleBasedVerifier
from src.utils.config import get_8gb_vram_config
from src.utils.logging_utils import setup_logging, get_logger

logger = get_logger("preprocess_sent")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K with SENT")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Process only N queries")
    parser.add_argument("--M", type=int, default=4, help="Number of samples per query")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--cache-path", type=str, default="data/cache/gsm8k_sent_sorted.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./logs")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (number of queries)")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.cache_path) or "data/cache", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File logging
    log_file = os.path.join(args.output_dir, "preprocess_sent.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", log_file)
    
    logger.info("=" * 60)
    logger.info("SENT Preprocessing")
    logger.info("=" * 60)
    logger.info("Cache path: %s", args.cache_path)
    logger.info("Resume: %s", args.resume)
    logger.info("Max steps: %s", args.max_steps)
    logger.info("Samples (M): %s", args.M)
    logger.info("Temperature: %s", args.temperature)
    logger.info("Batch size: %s", args.batch_size)
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("GPU: %s (%.1f GB)", gpu_name, vram_gb)
    
    config = get_8gb_vram_config()
    model_id = args.model_id or config.model.model_id
    
    logger.info("Loading model: %s...", model_id)
    start_time = time.time()
    model, tokenizer = load_4bit_engine(model_id)
    
    if model is None:
        logger.error("Failed to load model")
        return 1
    
    load_time = time.time() - start_time
    logger.info("Model loaded in %.1fs", load_time)
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info("Peak VRAM after model load: %.2f GB", peak_mem)
    
    verifier = RuleBasedVerifier()
    
    config.sent.num_samples = args.M
    config.sent.temperature = args.temperature
    config.sent.checkpoint_interval = args.checkpoint_interval
    config.sent.seed = args.seed
    
    calculator = SemanticEntropyCalculator(model, tokenizer, verifier, config)
    
    logger.info("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main", split="train")
    dataset = [{"id": i, "question": item["question"]} for i, item in enumerate(gsm8k)]
    logger.info("Loaded %d examples", len(dataset))
    
    if args.max_steps:
        dataset = dataset[:args.max_steps]
        logger.info("Processing subset: %d examples", len(dataset))
    
    logger.info("Starting entropy calculation...")
    process_start = time.time()
    
    pbar = tqdm(total=len(dataset), desc="Computing SENT", unit="query")
    
    def progress_callback(current, total):
        pbar.update(current - pbar.n)
        if current % 10 == 0:
            elapsed = time.time() - process_start
            qps = current / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"q/s": f"{qps:.2f}", "VRAM": f"{torch.cuda.max_memory_allocated()/(1024**3):.1f}GB"})
    
    try:
        calculator.process_dataset(dataset, args.cache_path, resume=args.resume, progress_callback=progress_callback, batch_size=args.batch_size)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Partial progress is saved in periodic checkpoints.")
        pbar.close()
        return 130
    except Exception as e:
        logger.error("Error during processing: %s", e)
        pbar.close()
        raise
    
    pbar.close()
    
    process_time = time.time() - process_start
    logger.info("Processing complete in %.1fs", process_time)
    
    queries_per_sec = len(dataset) / process_time if process_time > 0 else 0
    logger.info("Speed: %.2f queries/second", queries_per_sec)
    
    if os.path.exists(args.cache_path):
        if args.cache_path.endswith('.json'):
            import json
            with open(args.cache_path, 'r') as f:
                cache_data = json.load(f)
        else:
            cache_data = torch.load(args.cache_path, weights_only=False)
        logger.info("Cache saved: %d entries", len(cache_data.get("entropies", [])))
        logger.info("Cache status: %s", cache_data.get("metadata", {}).get("status"))
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info("Peak VRAM during processing: %.2f GB", peak_mem)
    
    total_time = time.time() - start_time
    logger.info("Total time: %.1fs", total_time)
    logger.info("=" * 60)
    logger.info("Preprocessing complete. Cache saved to: %s", args.cache_path)
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
