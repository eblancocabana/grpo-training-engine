"""
SENT Preprocessing Script — vLLM backend.

Drop-in replacement for preprocess_sent.py using vLLM for ~3-5× faster
inference via continuous batching, PagedAttention, and optimized CUDA kernels.

Produces the *exact same* cache format (.pt or .json) so the rest of the
training pipeline (SENTGSM8KDataset, create_grpo_dataloader, etc.) works
without changes.

Usage:
    python scripts/preprocess_sent_vllm.py
    python scripts/preprocess_sent_vllm.py --M 8 --max-steps 100
    python scripts/preprocess_sent_vllm.py --resume  # resume from partial cache
"""
import os
import sys
import argparse
import logging
import time
import math
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from datasets import load_dataset

from src.grpo.verifier import RuleBasedVerifier
from src.utils.config import get_8gb_vram_config
from src.utils.logging_utils import setup_logging, get_logger
from src.data.sent_calculator import (
    cluster_by_answer,
    compute_semantic_entropy,
    save_sent_cache,
    load_sent_cache,
    make_sent_metadata,
)

logger = get_logger("preprocess_sent_vllm")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K with SENT (vLLM)")
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
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory for vLLM (default: 0.90)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.cache_path) or "data/cache", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # File logging
    log_file = os.path.join(args.output_dir, "preprocess_sent_vllm.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", log_file)

    logger.info("=" * 60)
    logger.info("SENT Preprocessing (vLLM backend)")
    logger.info("=" * 60)
    logger.info("Cache path: %s", args.cache_path)
    logger.info("Resume: %s", args.resume)
    logger.info("Max steps: %s", args.max_steps)
    logger.info("Samples (M): %s", args.M)
    logger.info("Temperature: %s", args.temperature)
    logger.info("Batch size: %s", args.batch_size)

    config = get_8gb_vram_config()
    model_id = args.model_id or config.model.model_id

    logger.info("Loading vLLM engine: %s ...", model_id)
    start_time = time.time()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vllm is not installed. Install with: pip install vllm")
        return 1

    llm = LLM(
        model=model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed or 42,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=args.M,
        temperature=args.temperature,
        max_tokens=config.training.max_response_length,
        top_p=1.0,
    )

    load_time = time.time() - start_time
    logger.info("vLLM engine loaded in %.1fs", load_time)

    verifier = RuleBasedVerifier()

    logger.info("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main", split="train")
    dataset = [{"id": i, "question": item["question"]} for i, item in enumerate(gsm8k)]
    logger.info("Loaded %d examples", len(dataset))

    if args.max_steps:
        dataset = dataset[:args.max_steps]
        logger.info("Processing subset: %d examples", len(dataset))

    start_idx = 0
    entropies: List[float] = []
    clusters_list: List[Any] = []
    indices: List[Any] = []

    if args.resume and os.path.exists(args.cache_path):
        try:
            data = load_sent_cache(args.cache_path)
            entropies = data.get("entropies", [])
            clusters_list = data.get("clusters", [])
            indices = data.get("indices", [])
            start_idx = len(entropies)
            logger.info("Resumed from checkpoint: %d entries already processed", start_idx)
        except Exception as e:
            logger.warning("Could not resume from checkpoint: %s", e)
            start_idx = 0

    if start_idx >= len(dataset):
        logger.info("Dataset already fully processed. Exiting.")
        return 0

    remaining = dataset[start_idx:]
    total = len(dataset)

    logger.info("Starting entropy calculation (%d remaining queries)...", len(remaining))
    process_start = time.time()

    pbar = tqdm(
        total=len(remaining),
        initial=0,
        desc="Computing SENT (vLLM)",
        unit="query",
    )

    batch_size = args.batch_size

    try:
        for batch_start in range(0, len(remaining), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining))
            batch = remaining[batch_start:batch_end]

            # Build prompts for this batch
            prompts = [ex["question"] for ex in batch]

            # vLLM generates n=M samples per prompt
            outputs = llm.generate(prompts, sampling_params)

            # Process each query's outputs
            for query_idx, request_output in enumerate(outputs):
                global_idx = start_idx + batch_start + query_idx
                ex = batch[query_idx]

                responses: List[str] = []
                for completion in request_output.outputs:
                    responses.append(completion.text)

                # Cluster & compute entropy (shared functions from sent_calculator)
                clusters = cluster_by_answer(verifier, responses)
                num_samples = len(responses)
                h = compute_semantic_entropy(clusters, num_samples)

                entropies.append(h)
                clusters_list.append([
                    {
                        "answer": c["answer"],
                        "count": len(c["indices"]),
                    }
                    for c in clusters
                ])
                indices.append(ex.get("id", global_idx))

            pbar.update(len(batch))

            # Update progress bar postfix
            elapsed = time.time() - process_start
            processed = batch_start + len(batch)
            qps = processed / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"q/s": f"{qps:.1f}", "done": f"{start_idx + processed}/{total}"})

            # Periodic checkpoint
            if (start_idx + batch_start + len(batch)) % args.checkpoint_interval < batch_size:
                checkpoint = {
                    "metadata": make_sent_metadata(config, status="in_progress", backend="vllm"),
                    "indices": indices,
                    "entropies": entropies,
                    "clusters": clusters_list,
                }
                save_sent_cache(args.cache_path, checkpoint)
                logger.info(
                    "Checkpoint saved: %d / %d queries",
                    len(entropies), total,
                )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving partial progress...")
        partial = {
            "metadata": make_sent_metadata(config, status="in_progress", backend="vllm"),
            "indices": indices,
            "entropies": entropies,
            "clusters": clusters_list,
        }
        save_sent_cache(args.cache_path, partial)
        pbar.close()
        logger.info("Partial progress saved (%d / %d queries)", len(entropies), total)
        return 130

    pbar.close()

    paired = list(zip(indices, entropies, clusters_list))
    paired_sorted = sorted(paired, key=lambda x: (math.isnan(x[1]), x[1]))

    final_cache = {
        "metadata": make_sent_metadata(config, status="complete", backend="vllm"),
        "indices": [p[0] for p in paired_sorted],
        "entropies": [p[1] for p in paired_sorted],
        "clusters": [p[2] for p in paired_sorted],
    }

    save_sent_cache(args.cache_path, final_cache)

    process_time = time.time() - process_start
    queries_per_sec = len(remaining) / process_time if process_time > 0 else 0

    logger.info("=" * 60)
    logger.info("Processing complete in %.1fs", process_time)
    logger.info("Speed: %.2f queries/second", queries_per_sec)
    logger.info("Total queries processed: %d", len(entropies))
    logger.info("Cache saved to: %s", args.cache_path)
    logger.info("=" * 60)

    total_time = time.time() - start_time
    logger.info("Total time (incl. model load): %.1fs", total_time)

    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
