"""Main evaluation orchestration."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
from transformers import AutoTokenizer

from src.reasoning_eval.datasets import DATASET_REGISTRY, load_examples
from src.reasoning_eval.io import append_jsonl, completed_example_keys, read_jsonl, write_csv
from src.reasoning_eval.models import ModelSpec
from src.reasoning_eval.prompts import build_prompt
from src.reasoning_eval.schema import EvalExample, Generation
from src.reasoning_eval.scoring import avg_at_k, median, pass_at_k, score_generation
from src.reasoning_eval.vllm_engine import GenerationProtocol, MockGenerator, VLLMGenerator


HARD_MATH_DATASETS = {
    "aime24",
    "aime25",
    "amc23",
    "math500",
    "minerva",
    "olympiadbench",
    "gaokao",
    "omni-math",
}


def protocol_plan(
    *,
    selected_protocol: str,
    tier: str,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    n_samples: int | None,
    seed: int,
    dataset_name: str,
) -> list[GenerationProtocol]:
    protocols: list[GenerationProtocol] = []
    if selected_protocol in {"deterministic", "both"}:
        protocols.append(
            GenerationProtocol(
                name="deterministic",
                temperature=0.0,
                top_p=1.0,
                n=1,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        )
    if selected_protocol in {"sampled", "both"}:
        if n_samples is not None:
            n = n_samples
        elif tier == "strong":
            n = 8
        elif tier == "maximal":
            n = 32 if dataset_name in {"aime24", "aime25", "amc23"} else 16
        else:
            n = 4
        protocols.append(
            GenerationProtocol(
                name="sampled",
                temperature=0.6 if temperature is None else temperature,
                top_p=0.95 if top_p is None else top_p,
                n=n,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        )
    return protocols


def _generation_row(generation: Generation, model: ModelSpec) -> dict[str, Any]:
    return {
        **asdict(generation),
        "adapter": model.adapter,
        "selection_source": model.selection_source,
    }


def _score_rows(
    *,
    model: ModelSpec,
    examples: list[EvalExample],
    generations: list[Generation],
    protocol: GenerationProtocol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_example = {example.example_id: example for example in examples}
    parsed_rows: list[dict[str, Any]] = []
    scores_by_generation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        example = by_example[generation.example_id]
        score = score_generation(example, generation)
        parsed = {
            "model_key": model.key,
            "dataset": example.dataset,
            "family": example.family,
            "protocol": protocol.name,
            "example_id": example.example_id,
            "sample_index": generation.sample_index,
            "parsed_answer": score.parsed,
            "normalized_prediction": score.normalized_prediction,
            "normalized_gold": score.normalized_gold,
            "correct": score.correct,
            "parsable": score.parsable,
            "format_ok": score.format_ok,
            "finish_reason": generation.finish_reason,
            "token_count": generation.token_count,
            "latency_s": generation.latency_s,
            "details": score.details,
        }
        parsed_rows.append(parsed)
        scores_by_generation[example.example_id].append(parsed)

    score_rows: list[dict[str, Any]] = []
    for example in examples:
        rows = sorted(scores_by_generation.get(example.example_id, []), key=lambda row: row["sample_index"])
        correct_values = [bool(row["correct"]) for row in rows]
        token_counts = [int(row["token_count"]) for row in rows]
        finish_reasons = [row["finish_reason"] for row in rows]
        score_rows.append(
            {
                "model_key": model.key,
                "dataset": example.dataset,
                "family": example.family,
                "protocol": protocol.name,
                "example_id": example.example_id,
                "task_type": example.task_type,
                "sample_count": len(rows),
                "correct": pass_at_k(correct_values, min(1, len(correct_values))),
                "pass_at_k": pass_at_k(correct_values, protocol.n),
                "avg_at_k": avg_at_k(correct_values, protocol.n),
                "parsable": any(bool(row["parsable"]) for row in rows),
                "format_ok": any(bool(row["format_ok"]) for row in rows),
                "unparsable_count": sum(1 for row in rows if not row["parsable"]),
                "truncated_count": sum(1 for reason in finish_reasons if str(reason).lower() == "length"),
                "avg_response_length": sum(token_counts) / max(1, len(token_counts)),
                "median_response_length": median(token_counts),
                "total_tokens": sum(token_counts),
                "avg_latency_s": statistics.mean([float(row["latency_s"] or 0.0) for row in rows]) if rows else 0.0,
            }
        )
    return parsed_rows, score_rows


def summarize_scores(score_rows: list[dict[str, Any]], parsed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(row["model_key"], row["dataset"], row["protocol"])].append(row)

    parsed_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        parsed_grouped[(row["model_key"], row["dataset"], row["protocol"])].append(row)

    dataset_summary: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        model_key, dataset, protocol = key
        generations = parsed_grouped.get(key, [])
        total_tokens = sum(int(row.get("token_count") or 0) for row in generations)
        total_latency = sum(float(row.get("latency_s") or 0.0) for row in generations)
        correct_examples = sum(1 for row in rows if row["pass_at_k"])
        sample_count = sum(int(row["sample_count"]) for row in rows)
        dataset_summary.append(
            {
                "model_key": model_key,
                "dataset": dataset,
                "family": rows[0]["family"],
                "protocol": protocol,
                "examples": len(rows),
                "actual_sample_count": sample_count,
                "pass@1_exact_accuracy": sum(1 for row in rows if row["correct"]) / max(1, len(rows)),
                "pass@k": correct_examples / max(1, len(rows)),
                "avg@k": statistics.mean([float(row["avg_at_k"]) for row in rows]) if rows else 0.0,
                "format_compliance": sum(1 for row in rows if row["format_ok"]) / max(1, len(rows)),
                "invalid_unparsable_rate": sum(int(row["unparsable_count"]) for row in rows) / max(1, sample_count),
                "average_response_length": statistics.mean([float(row["avg_response_length"]) for row in rows]) if rows else 0.0,
                "median_response_length": median([float(row["median_response_length"]) for row in rows]),
                "truncation_rate": sum(int(row["truncated_count"]) for row in rows) / max(1, sample_count),
                "tokens_per_correct_answer": total_tokens / max(1, correct_examples),
                "average_generation_latency": statistics.mean([float(row["avg_latency_s"]) for row in rows]) if rows else 0.0,
                "total_tokens_generated": total_tokens,
                "tokens_per_second": total_tokens / max(1e-9, total_latency),
            }
        )

    by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_summary:
        by_model[(row["model_key"], row["protocol"])].append(row)
        by_family[(row["model_key"], row["family"], row["protocol"])].append(row)

    model_summary = [
        {
            "model_key": model_key,
            "protocol": protocol,
            "datasets": len(rows),
            "overall_macro_average": statistics.mean([float(row["pass@k"]) for row in rows]),
            "overall_macro_avg_at_k": statistics.mean([float(row["avg@k"]) for row in rows]),
            "total_tokens_generated": sum(int(row["total_tokens_generated"]) for row in rows),
        }
        for (model_key, protocol), rows in sorted(by_model.items())
    ]
    family_summary = [
        {
            "model_key": model_key,
            "family": family,
            "protocol": protocol,
            "datasets": len(rows),
            "macro_pass@k": statistics.mean([float(row["pass@k"]) for row in rows]),
            "macro_avg@k": statistics.mean([float(row["avg@k"]) for row in rows]),
        }
        for (model_key, family, protocol), rows in sorted(by_family.items())
    ]
    return dataset_summary, model_summary, family_summary


def _iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


class EvaluationRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        models: list[ModelSpec],
        dataset_names: list[str],
        tier: str,
        selected_protocol: str,
        max_new_tokens: int,
        temperature: float | None,
        top_p: float | None,
        n_samples: int | None,
        limit_per_dataset: int | None,
        batch_size: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str,
        quantization: str | None,
        seed: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        use_mock_generator: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.models = models
        self.dataset_names = dataset_names
        self.tier = tier
        self.selected_protocol = selected_protocol
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_samples = n_samples
        self.limit_per_dataset = limit_per_dataset
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.quantization = quantization
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.use_mock_generator = use_mock_generator

    def run(self) -> None:
        examples_by_dataset: dict[str, list[EvalExample]] = {}
        dataset_metadata: dict[str, Any] = {}
        for dataset_name in self.dataset_names:
            examples, metadata = load_examples(dataset_name, limit=self.limit_per_dataset, seed=self.seed)
            examples_by_dataset[dataset_name] = examples
            dataset_metadata[dataset_name] = metadata

        completed = completed_example_keys(self.output_dir / "scores_by_example.jsonl")
        raw_path = self.output_dir / "raw_generations.jsonl"
        parsed_path = self.output_dir / "parsed_predictions.jsonl"
        scores_path = self.output_dir / "scores_by_example.jsonl"

        models_by_base: dict[str, list[ModelSpec]] = defaultdict(list)
        for model in self.models:
            models_by_base[model.model_id].append(model)

        adapter_ids = {model.key: idx + 1 for idx, model in enumerate(self.models)}
        total_units = 0
        for model in self.models:
            for dataset_name, examples in examples_by_dataset.items():
                for protocol in protocol_plan(
                    selected_protocol=self.selected_protocol,
                    tier=self.tier,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n_samples=self.n_samples,
                    seed=self.seed,
                    dataset_name=dataset_name,
                ):
                    total_units += sum(
                        1
                        for example in examples
                        if (model.key, dataset_name, protocol.name, example.example_id) not in completed
                    )

        pbar = tqdm(total=total_units, desc="Evaluating", unit="example")
        started = time.time()
        for base_model_id, base_models in models_by_base.items():
            tokenizer = None if self.use_mock_generator else AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            enable_lora = any(model.adapter for model in base_models)
            if self.use_mock_generator:
                generator: Any = MockGenerator(tokenizer)
            else:
                generator = VLLMGenerator(
                    model_id=base_model_id,
                    tokenizer=tokenizer,
                    max_model_len=self.max_new_tokens + 2048,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    dtype=self.dtype,
                    quantization=self.quantization,
                    seed=self.seed,
                    enable_lora=enable_lora,
                    max_num_seqs=self.max_num_seqs,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                )

            for model in base_models:
                for dataset_name, examples in examples_by_dataset.items():
                    prompts = {example.example_id: build_prompt(example, tokenizer) for example in examples}
                    for protocol in protocol_plan(
                        selected_protocol=self.selected_protocol,
                        tier=self.tier,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        n_samples=self.n_samples,
                        seed=self.seed,
                        dataset_name=dataset_name,
                    ):
                        remaining = [
                            example
                            for example in examples
                            if (model.key, dataset_name, protocol.name, example.example_id) not in completed
                        ]
                        for batch in _iter_batches(remaining, self.batch_size):
                            batch_prompts = [prompts[example.example_id] for example in batch]
                            batch_ids = [example.example_id for example in batch]
                            batch_started = time.time()
                            generations = generator.generate(
                                prompts=batch_prompts,
                                model_key=model.key,
                                dataset=dataset_name,
                                protocol=protocol,
                                example_ids=batch_ids,
                                adapter_path=model.adapter,
                                adapter_id=adapter_ids[model.key],
                            )
                            parsed_rows, score_rows = _score_rows(
                                model=model,
                                examples=batch,
                                generations=generations,
                                protocol=protocol,
                            )
                            append_jsonl(raw_path, (_generation_row(gen, model) for gen in generations))
                            append_jsonl(parsed_path, parsed_rows)
                            append_jsonl(scores_path, score_rows)
                            completed.update((model.key, dataset_name, protocol.name, example.example_id) for example in batch)
                            pbar.update(len(batch))
                            elapsed = time.time() - started
                            examples_per_s = pbar.n / max(1e-9, elapsed)
                            eta_s = (total_units - pbar.n) / examples_per_s if examples_per_s > 0 else 0.0
                            generated_tokens = sum(gen.token_count for gen in generations)
                            pbar.set_postfix(
                                {
                                    "ex/s": f"{examples_per_s:.2f}",
                                    "tok/s": f"{generated_tokens / max(1e-9, time.time() - batch_started):.1f}",
                                    "elapsed": f"{elapsed / 60:.1f}m",
                                    "eta": f"{eta_s / 60:.1f}m",
                                }
                            )
        pbar.close()

        parsed_rows = read_jsonl(parsed_path)
        score_rows = read_jsonl(scores_path)
        dataset_summary, model_summary, family_summary = summarize_scores(score_rows, parsed_rows)
        write_csv(self.output_dir / "summary_by_dataset.csv", dataset_summary)
        write_csv(self.output_dir / "summary_by_model.csv", model_summary)
        write_csv(self.output_dir / "macro_family_summary.csv", family_summary)
        (self.output_dir / "dataset_metadata.json").write_text(
            json.dumps(dataset_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

