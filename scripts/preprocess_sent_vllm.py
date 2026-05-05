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
from src.data.gsm8k_loader import format_grpo_prompt, resolve_sent_cache_path
from src.data.math_dataset import load_math_split_rows, supported_dataset_names

logger = get_logger("preprocess_sent_vllm")


def _format_prompt_with_cap(tokenizer: Any, question: str, max_prompt_length: int) -> str:
    prompt = format_grpo_prompt(tokenizer, question)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
        padding=False,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"]
    return tokenizer.decode(input_ids, skip_special_tokens=False)


def _assert_resume_cache_compatible(
    cache_path: str,
    expected_key: str,
    model_id: str,
) -> Dict[str, Any]:
    data = load_sent_cache(cache_path)
    metadata = data.get("metadata", {})
    cached_key = metadata.get("sent_cache_key")
    if not cached_key:
        raise ValueError(
            "Resume cache is missing a SENT compatibility key; regenerate it before resuming."
        )
    if cached_key != expected_key:
        raise ValueError(
            "Resume cache is incompatible with the current model/SENT settings."
        )
    cached_model_id = metadata.get("model_id")
    if cached_model_id and cached_model_id != model_id:
        raise ValueError(
            f"Resume cache was generated for model '{cached_model_id}', not '{model_id}'."
        )
    return data


def main():
    parser = argparse.ArgumentParser(description="Preprocess math dataset train split with SENT (vLLM)")
    parser.add_argument("--dataset-name", choices=supported_dataset_names(), default="gsm8k")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Process only N queries")
    parser.add_argument("--M", type=int, default=4, help="Number of samples per query")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--cache-path", type=str, default="data/cache/gsm8k_sent_sorted.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./logs")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (number of queries)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model context length")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Alias for the prompt length used in SENT cache metadata.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=None,
        help="Maximum sampled response length for SENT generations.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70,
                        help="Fraction of GPU memory for vLLM (default: 0.70)")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=48,
        help="Maximum active vLLM sequences. Lower this on 8GB GPUs to avoid sampler warmup OOM.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per vLLM scheduler batch.",
    )

    args = parser.parse_args()

    args.cache_path = resolve_sent_cache_path(args.dataset_name, args.cache_path)

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
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Cache path: %s", args.cache_path)
    logger.info("Resume: %s", args.resume)
    logger.info("Max steps: %s", args.max_steps)
    logger.info("Samples (M): %s", args.M)
    logger.info("Temperature: %s", args.temperature)
    logger.info("Batch size: %s", args.batch_size)
    logger.info("Max num seqs: %s", args.max_num_seqs)
    logger.info("Max num batched tokens: %s", args.max_num_batched_tokens)

    config = get_8gb_vram_config()
    model_id = args.model_id or config.model.model_id
    config.model.model_id = model_id
    config.training.dataset_name = args.dataset_name
    config.sent.num_samples = args.M
    config.sent.temperature = args.temperature
    config.sent.cache_path = args.cache_path
    config.sent.checkpoint_interval = args.checkpoint_interval
    config.sent.seed = args.seed
    if args.max_response_length is not None:
        config.training.max_response_length = args.max_response_length
    if args.max_prompt_length is not None:
        config.training.max_prompt_length = args.max_prompt_length
    if args.max_model_len is None:
        args.max_model_len = (
            config.training.max_prompt_length + config.training.max_response_length
        )
    else:
        config.training.max_prompt_length = max(
            1,
            args.max_model_len - config.training.max_response_length,
        )

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
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    expected_cache_key = make_sent_metadata(
        config,
        status="in_progress",
        backend="vllm",
        tokenizer=tokenizer,
        max_prompt_length=config.training.max_prompt_length,
    )["sent_cache_key"]

    logger.info("Loading %s train split...", args.dataset_name)
    rows, split_metadata = load_math_split_rows(
        args.dataset_name,
        "train",
        split_seed=config.training.split_seed,
        split_ratios=config.training.split_ratios,
        strict_filter_invalid=config.training.drop_invalid_dataset_rows,
    )
    dataset = [
        {"id": row.get("original_id", i), "question": row["question"]}
        for i, row in enumerate(rows)
    ]
    logger.info("Loaded %d examples", len(dataset))
    logger.info("Split sizes: %s", split_metadata.get("split_sizes"))

    if args.max_steps:
        dataset = dataset[:args.max_steps]
        logger.info("Processing subset: %d examples", len(dataset))

    start_idx = 0
    entropies: List[float] = []
    clusters_list: List[Any] = []
    indices: List[int] = []
    example_ids: List[Any] = []

    if args.resume and os.path.exists(args.cache_path):
        try:
            data = _assert_resume_cache_compatible(
                args.cache_path,
                expected_key=expected_cache_key,
                model_id=model_id,
            )
            entropies = data.get("entropies", [])
            clusters_list = data.get("clusters", [])
            indices = data.get("indices", [])
            example_ids = data.get("example_ids", [])
            if not example_ids and len(indices) == len(entropies):
                example_ids = list(indices)
                indices = list(range(len(entropies)))
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
            prompts = [
                _format_prompt_with_cap(
                    tokenizer,
                    ex["question"],
                    config.training.max_prompt_length,
                )
                for ex in batch
            ]

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
                indices.append(global_idx)
                example_ids.append(ex.get("id", global_idx))

            pbar.update(len(batch))

            # Update progress bar postfix
            elapsed = time.time() - process_start
            processed = batch_start + len(batch)
            qps = processed / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"q/s": f"{qps:.1f}", "done": f"{start_idx + processed}/{total}"})

            # Periodic checkpoint
            if (start_idx + batch_start + len(batch)) % args.checkpoint_interval < batch_size:
                checkpoint = {
                    "metadata": make_sent_metadata(
                        config,
                        status="in_progress",
                        backend="vllm",
                        tokenizer=tokenizer,
                        max_prompt_length=config.training.max_prompt_length,
                    ),
                    "indices": indices,
                    "example_ids": example_ids,
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
            "metadata": make_sent_metadata(
                config,
                status="in_progress",
                backend="vllm",
                tokenizer=tokenizer,
                max_prompt_length=config.training.max_prompt_length,
            ),
            "indices": indices,
            "example_ids": example_ids,
            "entropies": entropies,
            "clusters": clusters_list,
        }
        save_sent_cache(args.cache_path, partial)
        pbar.close()
        logger.info("Partial progress saved (%d / %d queries)", len(entropies), total)
        return 130

    pbar.close()

    paired = list(zip(indices, example_ids, entropies, clusters_list))
    paired_sorted = sorted(paired, key=lambda x: (math.isnan(x[2]), x[2]))

    final_cache = {
        "metadata": make_sent_metadata(
            config,
            status="complete",
            backend="vllm",
            tokenizer=tokenizer,
            max_prompt_length=config.training.max_prompt_length,
        ),
        "indices": [p[0] for p in paired_sorted],
        "example_ids": [p[1] for p in paired_sorted],
        "entropies": [p[2] for p in paired_sorted],
        "clusters": [p[3] for p in paired_sorted],
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
