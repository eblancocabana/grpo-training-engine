"""
Temporary vLLM probe for math dataset difficulty and response-length feasibility.

This utility samples normalized math prompts from several Hugging Face datasets,
generates multiple base-model completions with vLLM, and reports aggregate
length, truncation, extraction, and verifier metrics. It does not run GRPO and
does not modify training code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.gsm8k_loader import format_grpo_prompt
from src.data.math_dataset import MathDatasetError, extract_answer, extract_question
from src.grpo.verifier import RuleBasedVerifier


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATASETS = "gsm8k,gsm-plus,gsm-hard,svamp,asdiv,multiarith"
SUPPORTED_SPLIT_PRIORITY = ("test", "validation", "train")
MATH_EASY_CONFIGS = ("prealgebra",)
NUMBER_PATTERN = re.compile(r"-?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?")


@dataclass(frozen=True)
class DatasetCandidate:
    """One Hugging Face loading option for a logical benchmark dataset."""

    path: str
    config: str | None = None


@dataclass(frozen=True)
class NormalizedExample:
    """Dataset row normalized for benchmarking."""

    dataset: str
    split: str
    question: str
    answer: str
    source_id: str


DATASET_ADAPTERS: dict[str, tuple[DatasetCandidate, ...]] = {
    "gsm8k": (DatasetCandidate("gsm8k", "main"),),
    "gsm-plus": (
        DatasetCandidate("qintongli/GSM-Plus"),
        DatasetCandidate("reasoning-machines/gsm-plus"),
    ),
    "gsm-hard": (
        DatasetCandidate("reasoning-machines/gsm-hard"),
        DatasetCandidate("lighteval/GSM-Hard"),
    ),
    "svamp": (
        DatasetCandidate("ChilleD/SVAMP"),
        DatasetCandidate("svamp"),
    ),
    "asdiv": (
        DatasetCandidate("EleutherAI/asdiv"),
        DatasetCandidate("asdiv"),
    ),
    "multiarith": (
        DatasetCandidate("ChilleD/MultiArith"),
        DatasetCandidate("multi_arith"),
    ),
    "math-easy": tuple(DatasetCandidate("hendrycks/competition_math", config) for config in MATH_EASY_CONFIGS),
    "math-prealgebra": (DatasetCandidate("hendrycks/competition_math", "prealgebra"),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--limit-per-dataset", "--samples-per-dataset", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-response-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", "--output-path", default="outputs/dataset_difficulty_probe_t1_len768")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits", default="test,validation,train")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--print-samples",
        type=int,
        default=0,
        help="Print this many per-completion inspection rows with question/gold/extracted/correct.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_first_available_dataset(name: str, trust_remote_code: bool) -> tuple[Any, DatasetCandidate]:
    errors: list[str] = []
    for candidate in DATASET_ADAPTERS[name]:
        try:
            kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
            if candidate.config is None:
                return load_dataset(candidate.path, **kwargs), candidate
            return load_dataset(candidate.path, candidate.config, **kwargs), candidate
        except Exception as exc:
            errors.append(f"{candidate.path}/{candidate.config}: {exc}")
    raise RuntimeError(f"Could not load dataset '{name}'. Tried: {' | '.join(errors)}")


def iter_split_names(raw: Any, requested_splits: Sequence[str]) -> list[str]:
    if isinstance(raw, DatasetDict) or isinstance(raw, Mapping):
        available = [split for split in requested_splits if split in raw]
        if available:
            return available
        return list(raw.keys())
    return ["train"]


def row_get(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def normalize_gold_answer_text(answer: str) -> str:
    """Keep dataset gold answers compatible with the numeric verifier."""
    answer = str(answer).strip()
    if not answer:
        return answer
    if "####" in answer:
        answer = answer.rsplit("####", 1)[1].strip()
    cleaned = answer.replace("$", "").replace("%", "").strip()
    try:
        float(cleaned.replace(",", ""))
        return cleaned
    except ValueError:
        pass
    numbers = NUMBER_PATTERN.findall(cleaned)
    if numbers:
        return numbers[-1].strip()
    return answer


def normalize_row(row: Mapping[str, Any], dataset: str, split: str, raw_index: int) -> NormalizedExample | None:
    question = ""
    concat_question = row_get(row, ("question_concat", "Question_Concat"))
    if concat_question is not None:
        question = str(concat_question).strip()
    else:
        body = row_get(row, ("Body", "body"))
        direct_question = row_get(row, ("Question", "question"))
        if body is not None and direct_question is not None:
            question = f"{str(body).strip()} {str(direct_question).strip()}".strip()

    if not question:
        try:
            question = extract_question(row)
        except Exception:
            question_value = row_get(row, ("Question", "Body", "problem", "input", "query"))
            question = str(question_value).strip() if question_value is not None else ""

    try:
        answer = extract_answer(row)
    except Exception:
        answer_value = row_get(
            row,
            (
                "Answer",
                "final_ans",
                "final_answer",
                "target",
                "label",
                "gold",
                "gold_answer",
                "correct",
            ),
        )
        answer = str(answer_value).strip() if answer_value is not None else ""
    answer = normalize_gold_answer_text(answer)

    if not question or not answer:
        return None

    source_id = row_get(row, ("id", "idx", "uid", "qid", "source_id"))
    if source_id is None:
        source_id = str(raw_index)
    source_id = f"{split}:{source_id}"

    return NormalizedExample(
        dataset=dataset,
        split=split,
        question=question,
        answer=answer,
        source_id=str(source_id),
    )


def load_normalized_examples(
    dataset_name: str,
    *,
    limit: int,
    seed: int,
    requested_splits: Sequence[str],
    trust_remote_code: bool,
) -> tuple[list[NormalizedExample], dict[str, Any]]:
    raw, candidate = load_first_available_dataset(dataset_name, trust_remote_code)
    split_names = iter_split_names(raw, requested_splits)
    normalized: list[NormalizedExample] = []
    invalid = 0

    for split in split_names:
        source = raw[split] if isinstance(raw, (DatasetDict, Mapping)) else raw
        for raw_index, row in enumerate(source):
            if not isinstance(row, Mapping):
                invalid += 1
                continue
            example = normalize_row(row, dataset_name, split, raw_index)
            if example is None:
                invalid += 1
                continue
            normalized.append(example)

    rng = random.Random(seed)
    rng.shuffle(normalized)
    sampled = normalized[: max(0, int(limit))]
    metadata = {
        "dataset": dataset_name,
        "hf_path": candidate.path,
        "hf_config": candidate.config,
        "splits_seen": split_names,
        "valid_rows_seen": len(normalized),
        "invalid_rows_skipped": invalid,
        "sampled_rows": len(sampled),
    }
    return sampled, metadata


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower))


def finished_before_cap(finish_reason: Any, token_count: int, max_tokens: int) -> bool:
    reason = getattr(finish_reason, "value", finish_reason)
    return str(reason) != "length" and int(token_count) < int(max_tokens)


def format_prompt_with_cap(tokenizer: Any, question: str, max_prompt_length: int) -> str:
    """Format prompts like training/eval, then cap them like the vLLM filter utility."""
    prompt = format_grpo_prompt(tokenizer, question)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
        padding=False,
        return_tensors=None,
    )
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)


def summarize_dataset(
    dataset: str,
    examples_attempted: int,
    completion_rows: Sequence[Mapping[str, Any]],
    n: int,
    split: str | None = None,
) -> dict[str, Any]:
    lengths = [int(row["response_token_length"]) for row in completion_rows]
    truncated = [bool(row["truncated"]) for row in completion_rows]
    extracted = [row.get("extracted_answer") for row in completion_rows]
    correct_values = [row.get("correct") for row in completion_rows if row.get("correct") is not None]
    rewards = [float(row["reward"]) for row in completion_rows if row.get("reward") is not None]

    by_prompt: dict[str, list[Mapping[str, Any]]] = {}
    for row in completion_rows:
        by_prompt.setdefault(str(row["source_id"]), []).append(row)

    all_hit_cap = 0
    at_least_three_finish = 0
    all_finish = 0
    nonzero_reward_std = 0
    for rows in by_prompt.values():
        prompt_truncated = [bool(row["truncated"]) for row in rows]
        finished_count = len(prompt_truncated) - sum(prompt_truncated)
        prompt_rewards = [float(row["reward"]) for row in rows if row.get("reward") is not None]
        if len(rows) == n and all(prompt_truncated):
            all_hit_cap += 1
        if finished_count >= min(3, n):
            at_least_three_finish += 1
        if len(rows) == n and finished_count == n:
            all_finish += 1
        if len(prompt_rewards) > 1 and statistics.pstdev(prompt_rewards) > 0.0:
            nonzero_reward_std += 1

    prompt_count = len(by_prompt)
    successful_generations = len(completion_rows)
    return {
        "dataset": dataset,
        "split": split,
        "examples_attempted": examples_attempted,
        "successful_generations": successful_generations,
        "average_response_token_length": statistics.fmean(lengths) if lengths else 0.0,
        "median_response_token_length": statistics.median(lengths) if lengths else 0.0,
        "p75_response_token_length": percentile(lengths, 0.75),
        "p90_response_token_length": percentile(lengths, 0.90),
        "max_response_token_length": max(lengths) if lengths else 0,
        "truncation_ratio": sum(truncated) / len(truncated) if truncated else 0.0,
        "all_completions_hit_cap_ratio": all_hit_cap / prompt_count if prompt_count else 0.0,
        "at_least_3_of_4_finish_before_cap_ratio": at_least_three_finish / prompt_count if prompt_count else 0.0,
        "all_4_finish_before_cap_ratio": all_finish / prompt_count if prompt_count else 0.0,
        "answer_extraction_success_rate": (
            sum(value is not None for value in extracted) / len(extracted) if extracted else 0.0
        ),
        "accuracy": (
            sum(1 for value in correct_values if value is True) / len(correct_values)
            if correct_values
            else None
        ),
        "average_reward": statistics.fmean(rewards) if rewards else None,
        "reward_variance": statistics.pvariance(rewards) if len(rewards) > 1 else 0.0 if rewards else None,
        "nonzero_reward_std_ratio": nonzero_reward_std / prompt_count if prompt_count else 0.0,
    }


def summarize_by_split(
    dataset: str,
    examples: Sequence[NormalizedExample],
    completion_rows: Sequence[Mapping[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """Return aggregate and per-split summaries for one logical dataset."""
    summaries = [summarize_dataset(dataset, len(examples), completion_rows, n)]
    split_counts: dict[str, int] = {}
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1
    for split in sorted(split_counts):
        split_rows = [row for row in completion_rows if row.get("split") == split]
        summaries.append(
            summarize_dataset(
                dataset,
                split_counts[split],
                split_rows,
                n,
                split=split,
            )
        )
    return summaries


def classify_dataset(row: Mapping[str, Any], gsm8k: Mapping[str, Any] | None) -> str:
    all_finish = float(row.get("all_4_finish_before_cap_ratio") or 0.0)
    three_finish = float(row.get("at_least_3_of_4_finish_before_cap_ratio") or 0.0)
    truncation = float(row.get("truncation_ratio") or 0.0)
    extraction = float(row.get("answer_extraction_success_rate") or 0.0)
    accuracy = row.get("accuracy")
    gsm_acc = gsm8k.get("accuracy") if gsm8k else None

    if truncation > 0.25 or three_finish < 0.80:
        return "infeasible: high truncation"
    if extraction < 0.60:
        return "infeasible: low extraction"
    if accuracy is None or gsm_acc is None:
        return "unknown difficulty"
    if float(accuracy) >= float(gsm_acc) - 0.02:
        return "too easy/saturated vs GSM8K"
    if all_finish >= 0.70:
        return "harder and feasible"
    return "harder but length-risky"


def print_ranked_table(summaries: Sequence[Mapping[str, Any]]) -> None:
    gsm8k = next((row for row in summaries if row.get("dataset") == "gsm8k"), None)

    def sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
        accuracy = row.get("accuracy")
        return (float(accuracy) if accuracy is not None else 1.0, -float(row.get("truncation_ratio") or 0.0))

    rows = sorted(summaries, key=sort_key)
    print("\nDataset difficulty / feasibility ranking")
    print(
        f"{'dataset':<22} {'acc':>7} {'extract':>8} {'avg_len':>8} {'p90':>7} "
        f"{'trunc':>7} {'all<cap':>8} {'3/4<cap':>8} {'rwvar':>7}  status"
    )
    for row in rows:
        accuracy = row.get("accuracy")
        acc_text = f"{accuracy:.3f}" if accuracy is not None else "n/a"
        label = row["dataset"] if row.get("split") is None else f"{row['dataset']}/{row['split']}"
        print(
            f"{label:<22} {acc_text:>7} "
            f"{row['answer_extraction_success_rate']:>8.3f} "
            f"{row['average_response_token_length']:>8.1f} "
            f"{row['p90_response_token_length']:>7.1f} "
            f"{row['truncation_ratio']:>7.3f} "
            f"{row['all_4_finish_before_cap_ratio']:>8.3f} "
            f"{row['at_least_3_of_4_finish_before_cap_ratio']:>8.3f} "
            f"{row['nonzero_reward_std_ratio']:>7.3f}  "
            f"{classify_dataset(row, gsm8k)}"
        )


def print_inspection_rows(rows: Sequence[Mapping[str, Any]], limit: int) -> None:
    if limit <= 0:
        return
    print(f"\nManual extraction inspection: first {min(limit, len(rows))} completions")
    for idx, row in enumerate(rows[:limit], start=1):
        print("-" * 80)
        print(f"{idx}. {row['dataset']}/{row['split']} {row['source_id']} completion={row['completion_index']}")
        print(f"Q: {row['question']}")
        print(f"Gold: {row['answer']}")
        print(f"Extracted: {row.get('extracted_answer')}")
        print(f"Correct: {row.get('correct')}  reward={row.get('reward')}  len={row.get('response_token_length')} truncated={row.get('truncated')}")


def main() -> int:
    args = parse_args()
    dataset_names = split_csv(args.datasets)
    requested_splits = split_csv(args.splits) or list(SUPPORTED_SPLIT_PRIORITY)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    if not args.overwrite and (samples_path.exists() or summary_path.exists()):
        raise SystemExit(
            f"Output already exists under {output_dir}. Pass --overwrite or choose a new --output-dir."
        )

    if args.overwrite:
        samples_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    print("Loading tokenizer and vLLM model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    verifier = RuleBasedVerifier()

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": (
            args.max_model_len
            if args.max_model_len is not None
            else args.max_prompt_length + args.max_response_length
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )

    dataset_metadata: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.time()

    with samples_path.open("w", encoding="utf-8") as samples_fh:
        for dataset_name in dataset_names:
            if dataset_name not in DATASET_ADAPTERS:
                raise MathDatasetError(
                    f"Unsupported dataset '{dataset_name}'. Known: {', '.join(sorted(DATASET_ADAPTERS))}"
                )
            examples, metadata = load_normalized_examples(
                dataset_name,
                limit=args.limit_per_dataset,
                seed=args.seed,
                requested_splits=requested_splits,
                trust_remote_code=args.trust_remote_code,
            )
            dataset_metadata.append(metadata)
            print(
                f"Benchmarking {dataset_name}: {len(examples)} sampled examples "
                f"from {metadata['hf_path']} ({metadata.get('hf_config')})"
            )

            completion_rows: list[dict[str, Any]] = []
            printed_rows = 0
            for start in tqdm(range(0, len(examples), args.batch_size), desc=dataset_name, unit="batch"):
                batch = examples[start : start + args.batch_size]
                prompts = [
                    format_prompt_with_cap(tokenizer, example.question, args.max_prompt_length)
                    for example in batch
                ]
                outputs = llm.generate(prompts, sampling_params)

                for example, request_output in zip(batch, outputs):
                    for completion_index, completion in enumerate(request_output.outputs):
                        token_ids = getattr(completion, "token_ids", None) or []
                        token_count = len(token_ids)
                        truncated = not finished_before_cap(
                            getattr(completion, "finish_reason", None),
                            token_count,
                            args.max_response_length,
                        )
                        reward, info = verifier.verify(completion.text, example.answer)
                        extracted_answer = info.get("extracted_answer")
                        row = {
                            "dataset": example.dataset,
                            "split": example.split,
                            "source_id": example.source_id,
                            "question": example.question,
                            "answer": example.answer,
                            "completion_index": completion_index,
                            "completion_text": completion.text,
                            "response_token_length": token_count,
                            "truncated": truncated,
                            "finish_reason": str(getattr(completion, "finish_reason", None)),
                            "extracted_answer": extracted_answer,
                            "correct": bool(info.get("match")) if extracted_answer is not None else False,
                            "reward": reward,
                        }
                        completion_rows.append(row)
                        samples_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if printed_rows < args.print_samples:
                            print_inspection_rows([row], 1)
                            printed_rows += 1
                samples_fh.flush()

            summaries.extend(summarize_by_split(dataset_name, examples, completion_rows, args.n))

    summary = {
        "config": {
            "model": args.model,
            "datasets": dataset_names,
            "limit_per_dataset": args.limit_per_dataset,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_model_len": llm_kwargs["max_model_len"],
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "splits": requested_splits,
        },
        "dataset_load_metadata": dataset_metadata,
        "summaries": summaries,
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_ranked_table(summaries)
    print(f"\nWrote samples to {samples_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
