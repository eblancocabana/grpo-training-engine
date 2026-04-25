"""
Build local length-filtered math datasets with vLLM generation.

This is an offline data filtering utility, not GRPO. It samples completions
from the base model and keeps prompts whose sampled responses usually finish
before the configured response length cap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from src.data.gsm8k_loader import format_grpo_prompt
from src.data.math_dataset import SUPPORTED_SPLITS, load_math_split_rows
from src.data.sent_calculator import (
    cluster_by_answer,
    compute_semantic_entropy,
    make_sent_metadata,
    save_sent_cache,
)
from src.grpo.verifier import RuleBasedVerifier
from src.utils.config import get_8gb_vram_config
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("filter_dataset_by_response_length_vllm")

SUPPORTED_SOURCE_DATASETS = ("dapo-math-17k", "open-rs", "open-deepscaler")
PRESERVED_FIELDS = (
    "question",
    "answer",
    "solution",
    "original_id",
    "raw_index",
    "split",
    "dataset_name",
    "source",
)
DEFAULT_SOURCE_DATASETS = ("dapo-math-17k", "open-rs")
DEFAULT_OUTPUT_DATASET_NAME = "dapo-open-rs-lenfilter-640"


def completion_is_finished(
    finish_reason: Any,
    output_token_count: int,
    max_response_length: int,
) -> bool:
    """Return whether a sampled completion finished before the token cap."""
    return (
        normalize_finish_reason(finish_reason) != "length"
        and int(output_token_count) < int(max_response_length)
    )


def normalize_finish_reason(finish_reason: Any) -> str | None:
    """Convert vLLM finish reasons to stable JSON-safe values."""
    if finish_reason is None:
        return None
    value = getattr(finish_reason, "value", finish_reason)
    return str(value)


def should_keep_prompt(
    response_lengths: Sequence[int],
    finish_reasons: Sequence[Any],
    *,
    max_response_length: int,
    keep_min_finished: int,
) -> tuple[bool, int]:
    """Apply the length-filter decision rule for one prompt."""
    finished_count = sum(
        1
        for length, reason in zip(response_lengths, finish_reasons)
        if completion_is_finished(reason, int(length), max_response_length)
    )
    return finished_count >= int(keep_min_finished), finished_count


def output_dataset_dir(output_dir: str, output_dataset_name: str) -> str:
    """Return the output directory for one filtered dataset."""
    directory_name = output_dataset_name.replace("-lenfilter-", "_lenfilter_")
    return os.path.join(output_dir, directory_name)


def _progress_path(split_dir: str, split: str) -> str:
    return os.path.join(split_dir, f"{split}.progress.jsonl")


def _row_key(row: Mapping[str, Any]) -> str:
    source = row.get("source") or row.get("dataset_name") or "unknown"
    return f"{source}:{row.get('original_id', row.get('raw_index'))}"


def _load_processed_keys(path: str) -> set[str]:
    keys: set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(_row_key(row))
    return keys


def _load_kept_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl_row(fh, row: Mapping[str, Any]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def summarize_progress(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from processed progress rows."""
    total_seen = len(rows)
    total_kept = sum(1 for row in rows if row.get("length_filter_kept") is True)
    finished_counts = [int(row.get("length_filter_finished_count", 0)) for row in rows]
    response_lengths = [
        int(length)
        for row in rows
        for length in row.get("length_filter_response_lengths", [])
    ]
    finish_reasons = [
        reason
        for row in rows
        for reason in row.get("length_filter_finish_reasons", [])
    ]
    length_reasons = sum(1 for reason in finish_reasons if reason == "length")
    return {
        "total_seen": total_seen,
        "total_kept": total_kept,
        "keep_rate": (total_kept / total_seen) if total_seen else 0.0,
        "mean_finished_count": statistics.fmean(finished_counts) if finished_counts else 0.0,
        "mean_response_length": statistics.fmean(response_lengths) if response_lengths else 0.0,
        "p50_response_length": _percentile(response_lengths, 0.50),
        "p90_response_length": _percentile(response_lengths, 0.90),
        "length_finish_reason_rate": (
            length_reasons / len(finish_reasons) if finish_reasons else 0.0
        ),
    }


def format_eta(processed: int, total: int, elapsed_seconds: float) -> tuple[str, str]:
    """Return remaining duration and wall-clock finish estimate."""
    if processed <= 0 or elapsed_seconds <= 0:
        return "unknown", "unknown"
    remaining = max(0, total - processed)
    seconds_remaining = remaining * (elapsed_seconds / processed)
    finish_time = datetime.now().astimezone().timestamp() + seconds_remaining
    finish_dt = datetime.fromtimestamp(finish_time).astimezone()
    return (
        time.strftime("%H:%M:%S", time.gmtime(seconds_remaining)),
        finish_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )


def _make_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "source_datasets": list(args.dataset_name),
        "output_dataset_name": args.output_dataset_name,
        "max_response_length": args.max_response_length,
        "max_prompt_length": args.max_prompt_length,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }


def output_dataset_name(dataset_name: Sequence[str] | str, max_response_length: int) -> str:
    """Return the registered name for a filtered local dataset."""
    if isinstance(dataset_name, str):
        return f"{dataset_name}-lenfilter-{int(max_response_length)}"
    joined = "-".join(name.replace("-math-17k", "").replace("-rs", "rs") for name in dataset_name)
    return f"{joined}-lenfilter-{int(max_response_length)}"


def resolve_output_dataset_name(args: argparse.Namespace) -> str:
    """Return the explicit output dataset name for this filtering run."""
    if args.output_dataset_name:
        return args.output_dataset_name
    if tuple(args.dataset_name) == DEFAULT_SOURCE_DATASETS and args.max_response_length == 640:
        return DEFAULT_OUTPUT_DATASET_NAME
    if len(args.dataset_name) == 1:
        return output_dataset_name(args.dataset_name[0], args.max_response_length)
    joined = "-".join(args.dataset_name)
    return f"{joined}-lenfilter-{int(args.max_response_length)}"


def default_sent_cache_path(output_name: str) -> str:
    """Return the default SENT cache path for the filtered dataset identity."""
    stem = output_name.replace("/", "_").replace("-", "_")
    return os.path.join("data", "cache", f"{stem}_sent_sorted.pt")


def compute_sent_from_completions(
    verifier: RuleBasedVerifier,
    responses: Sequence[str],
) -> tuple[float, list[dict[str, Any]]]:
    """Compute SENT entropy metadata from already-generated completions."""
    clusters = cluster_by_answer(verifier, list(responses))
    entropy = compute_semantic_entropy(clusters, len(responses))
    return entropy, [
        {
            "answer": cluster["answer"],
            "count": len(cluster["indices"]),
        }
        for cluster in clusters
    ]


def build_filtered_row(
    row: Mapping[str, Any],
    *,
    response_lengths: Sequence[int],
    finish_reasons: Sequence[Any],
    finished_count: int,
    generation_config: Mapping[str, Any],
    keep_min_finished: int,
    completions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create the persisted filtered row from a normalized source row."""
    output = {field: row.get(field) for field in PRESERVED_FIELDS}
    output["original_dataset_name"] = row.get("dataset_name")
    output["dataset_name"] = generation_config["output_dataset_name"]
    output.update(
        {
            "length_filter_num_samples": len(response_lengths),
            "length_filter_keep_min_finished": int(keep_min_finished),
            "length_filter_response_lengths": [int(v) for v in response_lengths],
            "length_filter_finish_reasons": list(finish_reasons),
            "length_filter_finished_count": int(finished_count),
            "length_filter_kept": True,
            "length_filter_generation_config": dict(generation_config),
        }
    )
    if completions is not None:
        output["length_filter_completions"] = list(completions)
    return output


def _progress_row(
    row: Mapping[str, Any],
    *,
    kept: bool,
    response_lengths: Sequence[int],
    finish_reasons: Sequence[Any],
    finished_count: int,
    generation_config: Mapping[str, Any],
    sent_entropy: float | None = None,
    sent_clusters: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output = {field: row.get(field) for field in PRESERVED_FIELDS}
    output["original_dataset_name"] = row.get("dataset_name")
    output["dataset_name"] = generation_config["output_dataset_name"]
    output.update(
        {
            "length_filter_response_lengths": [int(v) for v in response_lengths],
            "length_filter_finish_reasons": list(finish_reasons),
            "length_filter_finished_count": int(finished_count),
            "length_filter_kept": bool(kept),
            "length_filter_generation_config": dict(generation_config),
        }
    )
    if sent_entropy is not None:
        output["length_filter_sent_entropy"] = float(sent_entropy)
        output["length_filter_sent_clusters"] = [dict(cluster) for cluster in sent_clusters or []]
    return output


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


def _iter_batches(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def make_sampling_params(sampling_params_cls: Any, args: argparse.Namespace) -> Any:
    """Create vLLM SamplingParams while tolerating versions without per-request seed."""
    kwargs = {
        "n": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_response_length,
        "seed": args.seed,
    }
    try:
        return sampling_params_cls(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return sampling_params_cls(**kwargs)


def write_sent_cache_from_progress(
    *,
    progress_rows: Sequence[Mapping[str, Any]],
    cache_path: str,
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any]:
    """Write a SENT cache over kept train rows using filtered-dataset positions."""
    output_name = args.output_dataset_name
    paired: list[tuple[int, Any, float, Any]] = []
    kept_position = 0
    for row in progress_rows:
        if row.get("length_filter_kept") is not True:
            continue
        entropy = row.get("length_filter_sent_entropy")
        clusters = row.get("length_filter_sent_clusters")
        if entropy is None or clusters is None:
            kept_position += 1
            continue
        paired.append((kept_position, row.get("original_id", kept_position), float(entropy), clusters))
        kept_position += 1

    paired_sorted = sorted(paired, key=lambda item: (math.isnan(item[2]), item[2]))

    config = get_8gb_vram_config()
    config.model.model_id = args.model_id
    config.training.dataset_name = output_name
    config.training.split_seed = args.split_seed
    config.training.max_prompt_length = args.max_prompt_length
    config.training.max_response_length = args.max_response_length
    config.sent.num_samples = args.num_samples
    config.sent.temperature = args.temperature
    config.sent.cache_path = cache_path
    config.sent.seed = args.seed

    cache = {
        "metadata": make_sent_metadata(
            config,
            status="complete",
            backend="vllm_length_filter",
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            dataset_name=output_name,
            split="train",
        ),
        "indices": [item[0] for item in paired_sorted],
        "example_ids": [item[1] for item in paired_sorted],
        "entropies": [item[2] for item in paired_sorted],
        "clusters": [item[3] for item in paired_sorted],
    }
    os.makedirs(os.path.dirname(cache_path) or "data/cache", exist_ok=True)
    save_sent_cache(cache_path, cache)
    return {
        "sent_cache_path": cache_path,
        "sent_cache_entries": len(paired_sorted),
    }


def _process_split(
    *,
    args: argparse.Namespace,
    split: str,
    out_dir: str,
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    verifier: RuleBasedVerifier | None,
) -> dict[str, Any]:
    kept_path = os.path.join(out_dir, f"{split}.jsonl")
    progress_path = _progress_path(out_dir, split)
    if not args.resume:
        for path in (kept_path, progress_path):
            if os.path.exists(path):
                os.remove(path)

    processed_keys = _load_processed_keys(progress_path) if args.resume else set()
    kept_keys = _load_processed_keys(kept_path) if args.resume else set()
    source_metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for source_dataset in args.dataset_name:
        source_rows, metadata = load_math_split_rows(
            source_dataset,
            split,
            split_seed=args.split_seed,
            strict_filter_invalid=True,
        )
        if args.limit is not None:
            source_rows = source_rows[: args.limit]
        rows.extend(source_rows)
        source_metadata[source_dataset] = metadata
    remaining = [row for row in rows if _row_key(row) not in processed_keys]
    generation_config = _make_generation_config(args)

    logger.info(
        "Processing %s split for %s: %d rows (%d remaining)",
        split,
        ", ".join(args.dataset_name),
        len(rows),
        len(remaining),
    )

    split_start = time.time()
    last_eta_log = 0.0
    with open(kept_path, "a", encoding="utf-8") as kept_fh, open(
        progress_path, "a", encoding="utf-8"
    ) as progress_fh:
        pbar = tqdm(total=len(remaining), desc=f"Filtering {split}", unit="prompt")
        for batch in _iter_batches(remaining, args.batch_size):
            prompts = [
                _format_prompt_with_cap(tokenizer, row["question"], args.max_prompt_length)
                for row in batch
            ]
            request_outputs = llm.generate(prompts, sampling_params)

            for row, request_output in zip(batch, request_outputs):
                response_lengths: list[int] = []
                finish_reasons: list[Any] = []
                completions: list[str] = []
                for completion in request_output.outputs:
                    token_ids = getattr(completion, "token_ids", None) or []
                    response_lengths.append(len(token_ids))
                    finish_reasons.append(
                        normalize_finish_reason(getattr(completion, "finish_reason", None))
                    )
                    completions.append(getattr(completion, "text", ""))

                kept, finished_count = should_keep_prompt(
                    response_lengths,
                    finish_reasons,
                    max_response_length=args.max_response_length,
                    keep_min_finished=args.keep_min_finished,
                )
                sent_entropy = None
                sent_clusters = None
                if kept and split == "train" and args.write_sent_cache:
                    if verifier is None:
                        raise ValueError("SENT cache writing requires a verifier")
                    sent_entropy, sent_clusters = compute_sent_from_completions(
                        verifier,
                        completions,
                    )
                progress = _progress_row(
                    row,
                    kept=kept,
                    response_lengths=response_lengths,
                    finish_reasons=finish_reasons,
                    finished_count=finished_count,
                    generation_config=generation_config,
                    sent_entropy=sent_entropy,
                    sent_clusters=sent_clusters,
                )
                _write_jsonl_row(progress_fh, progress)
                if kept and _row_key(row) not in kept_keys:
                    kept_row = build_filtered_row(
                        row,
                        response_lengths=response_lengths,
                        finish_reasons=finish_reasons,
                        finished_count=finished_count,
                        generation_config=generation_config,
                        keep_min_finished=args.keep_min_finished,
                        completions=completions if args.save_completions else None,
                    )
                    _write_jsonl_row(kept_fh, kept_row)
                    kept_keys.add(_row_key(row))
            pbar.update(len(batch))
            processed = min(pbar.n, len(remaining))
            elapsed = time.time() - split_start
            eta_remaining, eta_finish = format_eta(processed, len(remaining), elapsed)
            pbar.set_postfix({"eta": eta_remaining, "finish": eta_finish})
            now = time.time()
            if now - last_eta_log >= 60 or processed == len(remaining):
                logger.info(
                    "%s ETA: %s remaining, estimated finish %s (%d/%d prompts)",
                    split,
                    eta_remaining,
                    eta_finish,
                    processed,
                    len(remaining),
                )
                last_eta_log = now
        pbar.close()

    progress_rows = _load_kept_rows(progress_path)
    kept_rows = _load_kept_rows(kept_path)
    with open(os.path.join(out_dir, f"kept_indices_{split}.json"), "w", encoding="utf-8") as fh:
        json.dump([row.get("raw_index") for row in kept_rows], fh, indent=2)
    summary = summarize_progress(progress_rows)
    summary["source_datasets"] = list(args.dataset_name)
    summary["source_metadata"] = source_metadata
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter math datasets by vLLM response length")
    parser.add_argument(
        "--dataset-name",
        nargs="+",
        choices=SUPPORTED_SOURCE_DATASETS,
        default=list(DEFAULT_SOURCE_DATASETS),
        help="One or more source datasets to filter into one output dataset.",
    )
    parser.add_argument(
        "--output-dataset-name",
        default=None,
        help="Name of the mixed filtered local dataset.",
    )
    parser.add_argument("--splits", nargs="+", choices=SUPPORTED_SPLITS, default=list(SUPPORTED_SPLITS))
    parser.add_argument("--output-dir", default="data/filtered")
    parser.add_argument("--max-response-length", type=int, default=640)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--keep-min-finished", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-completions", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--write-sent-cache",
        action="store_true",
        help="Reuse kept train completions to write a SENT cache for the filtered dataset.",
    )
    parser.add_argument(
        "--sent-cache-path",
        type=str,
        default=None,
        help="Override SENT cache path. Defaults to data/cache/{filtered_dataset}_sent_sorted.pt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_8gb_vram_config()
    args.model_id = config.model.model_id
    args.output_dataset_name = resolve_output_dataset_name(args)
    if args.keep_min_finished > args.num_samples:
        raise ValueError("--keep-min-finished cannot exceed --num-samples")
    if args.batch_size is None:
        args.batch_size = 32

    out_dir = output_dataset_dir(args.output_dir, args.output_dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        logger.error("Missing dependency for vLLM filtering: %s", exc)
        return 1

    logger.info("Loading tokenizer: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    logger.info("Loading vLLM engine: %s", args.model_id)
    llm_kwargs = {
        "model": args.model_id,
        "max_model_len": args.max_prompt_length + args.max_response_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "trust_remote_code": True,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)

    sampling_params = make_sampling_params(SamplingParams, args)
    verifier = RuleBasedVerifier() if args.write_sent_cache else None

    split_summaries: dict[str, Any] = {}
    started = time.time()
    for split in args.splits:
        split_summaries[split] = _process_split(
            args=args,
            split=split,
            out_dir=out_dir,
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            verifier=verifier,
        )

    sent_summary: dict[str, Any] = {}
    if args.write_sent_cache and "train" in args.splits:
        sent_cache_path = args.sent_cache_path or default_sent_cache_path(args.output_dataset_name)
        progress_rows = _load_kept_rows(_progress_path(out_dir, "train"))
        sent_summary = write_sent_cache_from_progress(
            progress_rows=progress_rows,
            cache_path=sent_cache_path,
            args=args,
            tokenizer=tokenizer,
        )
        logger.info(
            "Wrote SENT cache with %d entries to %s",
            sent_summary["sent_cache_entries"],
            sent_summary["sent_cache_path"],
        )

    summary = {
        "dataset_name": list(args.dataset_name),
        "source_datasets": list(args.dataset_name),
        "output_dataset_name": args.output_dataset_name,
        "split_seed": args.split_seed,
        "max_response_length": args.max_response_length,
        "max_prompt_length": args.max_prompt_length,
        "num_samples": args.num_samples,
        "keep_min_finished": args.keep_min_finished,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "total_seen": {split: stats["total_seen"] for split, stats in split_summaries.items()},
        "total_kept": {split: stats["total_kept"] for split, stats in split_summaries.items()},
        "keep_rate": {split: stats["keep_rate"] for split, stats in split_summaries.items()},
        "mean_finished_count": {
            split: stats["mean_finished_count"] for split, stats in split_summaries.items()
        },
        "mean_response_length": {
            split: stats["mean_response_length"] for split, stats in split_summaries.items()
        },
        "p50_response_length": {
            split: stats["p50_response_length"] for split, stats in split_summaries.items()
        },
        "p90_response_length": {
            split: stats["p90_response_length"] for split, stats in split_summaries.items()
        },
        "length_finish_reason_rate": {
            split: stats["length_finish_reason_rate"] for split, stats in split_summaries.items()
        },
        "sent_cache": sent_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote filtered dataset to %s", out_dir)
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
