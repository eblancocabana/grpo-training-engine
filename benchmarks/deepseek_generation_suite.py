#!/usr/bin/env python3
"""
DeepSeek generation production-readiness suite.

This suite validates the Triton paged-KV generation path against the torch
prefix-cache reference on the only model that matters for this project:
DeepSeek-R1-Distill-Qwen-1.5B.

It runs three layers of checks:
1. Exact greedy parity across group sizes.
2. Sampled distribution checks across group sizes.
3. Capped train.py benchmarking on the sampled training path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.ing_inf_suite import BenchmarkConfig, IngInfBenchmarkSuite
from src.data.gsm8k_loader import create_grpo_dataloader
from src.grpo.trainer import GRPOTrainerLoop
from src.utils.config import get_8gb_vram_config


DEFAULT_GROUP_SIZES = [1, 2, 4, 8, 16]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class PromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    question: str
    answer: str


class DeepSeekGenerationSuite:
    def __init__(
        self,
        output_dir: Path,
        *,
        group_sizes: List[int],
        parity_prompts: int,
        sampled_prompts: int,
        sampled_seeds: List[int],
        max_prompt_length: int,
        max_response_length: int,
        train_steps: int,
        skip_parity: bool,
        skip_sampled: bool,
        skip_training: bool,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.group_sizes = group_sizes
        self.parity_prompts = parity_prompts
        self.sampled_prompts = sampled_prompts
        self.sampled_seeds = sampled_seeds
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.train_steps = train_steps
        self.skip_parity = skip_parity
        self.skip_sampled = skip_sampled
        self.skip_training = skip_training

    def _build_loop(self) -> GRPOTrainerLoop:
        config = get_8gb_vram_config()
        config.training.use_triton_kernels = True
        config.training.use_triton_generation = True
        config.training.triton_generation_mode = "auto"
        config.training.use_triton_grpo_loss = True
        config.training.use_triton_entropy_mask = True
        config.training.use_triton_lora = True
        config.training.max_prompt_length = self.max_prompt_length
        config.training.max_response_length = self.max_response_length
        config.training.skip_initial_benchmark = True
        config.wandb.enabled = False

        loop = GRPOTrainerLoop(config)
        loop.setup()
        return loop

    def _collect_prompt_batches(
        self, loop: GRPOTrainerLoop, *, num_prompts: int
    ) -> List[PromptBatch]:
        dataloader = create_grpo_dataloader(
            tokenizer=loop.tokenizer,
            split="train",
            batch_size=1,
            max_prompt_length=self.max_prompt_length,
            shuffle=False,
            use_sent=False,
            sent_config=loop.config.sent,
            num_stages=loop.config.sent.curriculum_stages,
            cache_path=loop.config.sent.cache_path,
            model_id=loop.config.model.model_id,
        )

        prompt_batches: List[PromptBatch] = []
        for batch in dataloader:
            prompt_batches.append(
                PromptBatch(
                    input_ids=batch["input_ids"].to(loop.device),
                    attention_mask=batch["attention_mask"].to(loop.device),
                    question=batch["questions"][0],
                    answer=batch["answers"][0],
                )
            )
            if len(prompt_batches) >= num_prompts:
                break

        if len(prompt_batches) < num_prompts:
            raise RuntimeError(
                f"Requested {num_prompts} prompt batches, got {len(prompt_batches)}."
            )
        return prompt_batches

    def _set_generation_backend(self, loop: GRPOTrainerLoop, mode: str) -> None:
        if mode not in {"auto", "on", "off"}:
            raise ValueError(f"Unsupported generation mode: {mode}")
        loop.config.training.triton_generation_mode = mode
        loop.config.training.use_triton_generation = mode != "off"

    def _run_greedy_parity(
        self, loop: GRPOTrainerLoop, prompt_batches: List[PromptBatch]
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        loop.config.training.generation_do_sample = False
        loop.config.training.generation_temperature = 1.0
        loop.config.training.generation_top_p = 1.0

        for group_size in self.group_sizes:
            loop.config.grpo.group_size = group_size
            loop.group_sampler.group_size = group_size

            group_result: Dict[str, Any] = {
                "group_size": group_size,
                "passed": True,
                "prompt_results": [],
            }

            for prompt_idx, prompt_batch in enumerate(prompt_batches):
                _set_seed(0)
                self._set_generation_backend(loop, "off")
                torch_texts, torch_ids, torch_mask = loop._generate_responses_with_tokens(
                    prompt_batch.input_ids,
                    prompt_batch.attention_mask,
                )

                _set_seed(0)
                self._set_generation_backend(loop, "on")
                triton_texts, triton_ids, triton_mask = loop._generate_responses_with_tokens(
                    prompt_batch.input_ids,
                    prompt_batch.attention_mask,
                )

                prompt_passed = (
                    torch_texts == triton_texts
                    and torch.equal(torch_ids, triton_ids)
                    and torch.equal(torch_mask, triton_mask)
                )

                prompt_result = {
                    "prompt_index": prompt_idx,
                    "question": prompt_batch.question,
                    "passed": prompt_passed,
                }
                if not prompt_passed:
                    prompt_result["torch_texts"] = torch_texts
                    prompt_result["triton_texts"] = triton_texts
                    first_diff = (torch_ids != triton_ids).nonzero(as_tuple=False)
                    if first_diff.numel() > 0:
                        prompt_result["first_token_mismatch"] = first_diff[0].tolist()
                    mask_diff = (torch_mask != triton_mask).nonzero(as_tuple=False)
                    if mask_diff.numel() > 0:
                        prompt_result["first_mask_mismatch"] = mask_diff[0].tolist()
                    group_result["passed"] = False

                group_result["prompt_results"].append(prompt_result)

            results.append(group_result)

        payload = {
            "model_id": loop.config.model.model_id,
            "group_sizes": self.group_sizes,
            "num_prompts": len(prompt_batches),
            "all_passed": all(result["passed"] for result in results),
            "results": results,
        }
        _write_json(self.output_dir / "greedy_parity.json", payload)
        return payload

    def _summarize_generation_run(
        self,
        loop: GRPOTrainerLoop,
        *,
        texts: List[str],
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        ground_truth: str,
        group_size: int,
    ) -> Dict[str, Any]:
        eos_token_id = loop.tokenizer.eos_token_id
        rewards: List[float] = []
        lengths: List[float] = []
        matches = 0
        eos_hits = 0
        truncations = 0
        extracted_answers: List[str] = []

        for row_idx, text in enumerate(texts):
            reward, info = loop.verifier.verify(text, ground_truth)
            rewards.append(float(reward))
            extracted = str(info.get("extracted_answer", ""))
            extracted_answers.append(extracted)
            if info.get("match", False):
                matches += 1

            valid_tokens = response_ids[row_idx][response_mask[row_idx].bool()]
            valid_len = int(valid_tokens.numel())
            lengths.append(float(valid_len))
            has_eos = False
            if eos_token_id is not None and valid_len > 0:
                has_eos = bool((valid_tokens == eos_token_id).any().item())
            if has_eos:
                eos_hits += 1
            if valid_len >= self.max_response_length and not has_eos:
                truncations += 1

        unique_answers = len({answer for answer in extracted_answers if answer})
        return {
            "responses": float(group_size),
            "reward_sum": sum(rewards),
            "reward_values": rewards,
            "match_count": float(matches),
            "eos_count": float(eos_hits),
            "truncation_count": float(truncations),
            "length_values": lengths,
            "unique_answer_ratio": (unique_answers / group_size) if group_size > 0 else 0.0,
        }

    def _aggregate_sampled_stats(
        self, run_summaries: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        reward_values: List[float] = []
        length_values: List[float] = []
        responses = 0.0
        matches = 0.0
        eos_hits = 0.0
        truncations = 0.0
        unique_ratios: List[float] = []

        for summary in run_summaries:
            reward_values.extend(summary["reward_values"])
            length_values.extend(summary["length_values"])
            responses += summary["responses"]
            matches += summary["match_count"]
            eos_hits += summary["eos_count"]
            truncations += summary["truncation_count"]
            unique_ratios.append(summary["unique_answer_ratio"])

        safe_responses = max(responses, 1.0)
        return {
            "responses": int(responses),
            "reward_mean": _mean(reward_values),
            "reward_std": _std(reward_values),
            "match_rate": matches / safe_responses,
            "eos_rate": eos_hits / safe_responses,
            "truncation_rate": truncations / safe_responses,
            "avg_response_length": _mean(length_values),
            "unique_answer_ratio": _mean(unique_ratios),
        }

    def _run_sampled_validation(
        self, loop: GRPOTrainerLoop, prompt_batches: List[PromptBatch]
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        loop.config.training.generation_do_sample = True
        loop.config.training.generation_temperature = 0.7
        loop.config.training.generation_top_p = 0.9

        thresholds = {
            "reward_mean_abs_delta_max": 0.15,
            "reward_std_abs_delta_max": 0.20,
            "match_rate_abs_delta_max": 0.10,
            "eos_rate_abs_delta_max": 0.10,
            "truncation_rate_abs_delta_max": 0.10,
            "avg_response_length_abs_delta_max": 6.0,
            "unique_answer_ratio_abs_delta_max": 0.20,
        }

        for group_size in self.group_sizes:
            loop.config.grpo.group_size = group_size
            loop.group_sampler.group_size = group_size

            torch_runs: List[Dict[str, Any]] = []
            triton_runs: List[Dict[str, Any]] = []

            for seed in self.sampled_seeds:
                for prompt_batch in prompt_batches:
                    _set_seed(seed)
                    self._set_generation_backend(loop, "off")
                    torch_texts, torch_ids, torch_mask = loop._generate_responses_with_tokens(
                        prompt_batch.input_ids,
                        prompt_batch.attention_mask,
                    )
                    torch_runs.append(
                        self._summarize_generation_run(
                            loop,
                            texts=torch_texts,
                            response_ids=torch_ids,
                            response_mask=torch_mask,
                            ground_truth=prompt_batch.answer,
                            group_size=group_size,
                        )
                    )

                    _set_seed(seed)
                    self._set_generation_backend(loop, "on")
                    triton_texts, triton_ids, triton_mask = loop._generate_responses_with_tokens(
                        prompt_batch.input_ids,
                        prompt_batch.attention_mask,
                    )
                    triton_runs.append(
                        self._summarize_generation_run(
                            loop,
                            texts=triton_texts,
                            response_ids=triton_ids,
                            response_mask=triton_mask,
                            ground_truth=prompt_batch.answer,
                            group_size=group_size,
                        )
                    )

            torch_stats = self._aggregate_sampled_stats(torch_runs)
            triton_stats = self._aggregate_sampled_stats(triton_runs)
            deltas = {
                "reward_mean_abs_delta": abs(
                    triton_stats["reward_mean"] - torch_stats["reward_mean"]
                ),
                "reward_std_abs_delta": abs(
                    triton_stats["reward_std"] - torch_stats["reward_std"]
                ),
                "match_rate_abs_delta": abs(
                    triton_stats["match_rate"] - torch_stats["match_rate"]
                ),
                "eos_rate_abs_delta": abs(
                    triton_stats["eos_rate"] - torch_stats["eos_rate"]
                ),
                "truncation_rate_abs_delta": abs(
                    triton_stats["truncation_rate"] - torch_stats["truncation_rate"]
                ),
                "avg_response_length_abs_delta": abs(
                    triton_stats["avg_response_length"]
                    - torch_stats["avg_response_length"]
                ),
                "unique_answer_ratio_abs_delta": abs(
                    triton_stats["unique_answer_ratio"]
                    - torch_stats["unique_answer_ratio"]
                ),
            }

            passed = all(
                deltas[key] <= thresholds[f"{key}_max"] for key in deltas.keys()
            )
            results.append(
                {
                    "group_size": group_size,
                    "passed": passed,
                    "torch": torch_stats,
                    "triton": triton_stats,
                    "deltas": deltas,
                }
            )

        payload = {
            "model_id": loop.config.model.model_id,
            "group_sizes": self.group_sizes,
            "num_prompts": len(prompt_batches),
            "seeds": self.sampled_seeds,
            "thresholds": thresholds,
            "all_passed": all(result["passed"] for result in results),
            "results": results,
        }
        _write_json(self.output_dir / "sampled_validation.json", payload)
        return payload

    def _resolve_auto_policy(self, loop: GRPOTrainerLoop) -> Dict[str, Any]:
        loop.config.training.generation_do_sample = True
        self._set_generation_backend(loop, "auto")
        enabled, reason = loop._resolve_triton_generation_decision()
        return {
            "mode": loop._get_triton_generation_mode(),
            "enabled": enabled,
            "reason": reason,
        }

    def _training_configs(self) -> List[BenchmarkConfig]:
        configs: List[BenchmarkConfig] = []
        for group_size in self.group_sizes:
            common = dict(
                triton=True,
                gradient_accumulation_steps=16,
                group_size=group_size,
                batch_size=1,
                lora_rank=16,
                lora_adapter_quant="none",
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                use_entropy_mask=True,
                disable_sent=True,
                steps=self.train_steps,
                use_wandb=False,
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=True,
            )
            configs.append(
                BenchmarkConfig(
                    name=f"deepseek_generation_torch_g{group_size}",
                    description=(
                        f"DeepSeek sampled training with torch prefix-cache generation, group_size={group_size}"
                    ),
                    triton_generation_mode="off",
                    triton_generation=False,
                    tags=["deepseek", "generation", "torch", f"g{group_size}"],
                    **common,
                )
            )
            configs.append(
                BenchmarkConfig(
                    name=f"deepseek_generation_triton_g{group_size}",
                    description=(
                        f"DeepSeek sampled training with Triton paged-KV generation, group_size={group_size}"
                    ),
                    triton_generation_mode="on",
                    triton_generation=True,
                    tags=["deepseek", "generation", "triton", f"g{group_size}"],
                    **common,
                )
            )
        return configs

    def _run_training_matrix(self) -> Dict[str, Any]:
        train_output_dir = self.output_dir / "train_matrix"
        suite = IngInfBenchmarkSuite(output_dir=str(train_output_dir))
        configs = self._training_configs()
        results = [suite.run_benchmark(config) for config in configs]

        grouped: Dict[int, Dict[str, Any]] = {}
        for result in results:
            name = result.config_name
            group_size = int(name.split("_g")[-1])
            backend = "triton" if "_triton_" in name else "torch"
            grouped.setdefault(group_size, {})[backend] = asdict(result)

        comparisons: List[Dict[str, Any]] = []
        for group_size in self.group_sizes:
            entry = grouped.get(group_size, {})
            torch_result = entry.get("torch")
            triton_result = entry.get("triton")
            if torch_result is None or triton_result is None:
                continue
            torch_tps = torch_result.get("tokens_per_sec") or 0.0
            triton_tps = triton_result.get("tokens_per_sec") or 0.0
            delta_pct = None
            if torch_tps > 0:
                delta_pct = ((triton_tps - torch_tps) / torch_tps) * 100.0
            comparisons.append(
                {
                    "group_size": group_size,
                    "torch_tokens_per_sec": torch_tps,
                    "triton_tokens_per_sec": triton_tps,
                    "throughput_delta_pct": delta_pct,
                    "torch_step_time_s": torch_result.get("step_time_avg_s"),
                    "triton_step_time_s": triton_result.get("step_time_avg_s"),
                }
            )

        payload = {
            "group_sizes": self.group_sizes,
            "train_steps": self.train_steps,
            "max_prompt_length": self.max_prompt_length,
            "max_response_length": self.max_response_length,
            "results": [asdict(result) for result in results],
            "comparisons": comparisons,
        }
        _write_json(self.output_dir / "training_matrix.json", payload)
        return payload

    def _write_report(
        self,
        *,
        greedy_parity: Dict[str, Any],
        sampled_validation: Dict[str, Any],
        auto_policy: Dict[str, Any],
        training_matrix: Dict[str, Any],
        elapsed_s: float,
    ) -> None:
        lines = [
            "# DeepSeek Generation Production Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {greedy_parity['model_id']}",
            f"Group sizes: {', '.join(str(value) for value in self.group_sizes)}",
            f"Greedy parity prompts: {greedy_parity.get('num_prompts', self.parity_prompts)}",
            f"Sampled validation prompts: {sampled_validation.get('num_prompts', self.sampled_prompts)}",
            "Sampled validation seeds: "
            + ", ".join(str(seed) for seed in sampled_validation.get("seeds", self.sampled_seeds)),
            f"Training steps per config: {training_matrix.get('train_steps', self.train_steps)}",
            f"Elapsed: {elapsed_s / 60:.1f} min",
            "",
            "## Decision",
            "",
            f"- Greedy parity passed: {greedy_parity['all_passed']}",
            f"- Sampled validation passed: {sampled_validation['all_passed']}",
            f"- Auto policy enabled Triton generation: {auto_policy['enabled']}",
            f"- Auto policy reason: {auto_policy['reason']}",
            "- Recommended default: keep sampled generation on the torch prefix-cache path via auto mode for DeepSeek on the RTX 3060 Ti.",
            "",
            "## Greedy Parity",
            "",
            "| Group Size | Passed |",
            "|------------|--------|",
        ]

        for result in greedy_parity["results"]:
            status = "✅" if result["passed"] else "❌"
            lines.append(f"| {result['group_size']} | {status} |")

        lines.extend(
            [
                "",
                "## Sampled Validation",
                "",
                "| Group Size | Passed | Reward Δ | Match Δ | EOS Δ | Trunc Δ | Avg Len Δ | Unique Δ |",
                "|------------|--------|----------|---------|-------|---------|-----------|----------|",
            ]
        )

        for result in sampled_validation["results"]:
            deltas = result["deltas"]
            status = "✅" if result["passed"] else "❌"
            lines.append(
                "| {group_size} | {status} | {reward_mean_abs_delta:.3f} | {match_rate_abs_delta:.3f} | "
                "{eos_rate_abs_delta:.3f} | {truncation_rate_abs_delta:.3f} | "
                "{avg_response_length_abs_delta:.2f} | {unique_answer_ratio_abs_delta:.3f} |".format(
                    group_size=result["group_size"],
                    status=status,
                    **deltas,
                )
            )

        lines.extend(
            [
                "",
                "## Training Matrix",
                "",
                "| Group Size | Torch Tok/s | Triton Tok/s | Delta % | Torch Step (s) | Triton Step (s) |",
                "|------------|-------------|--------------|---------|----------------|-----------------|",
            ]
        )

        for comparison in training_matrix["comparisons"]:
            delta_pct = comparison["throughput_delta_pct"]
            delta_str = f"{delta_pct:.2f}%" if delta_pct is not None else "N/A"
            lines.append(
                f"| {comparison['group_size']} | {comparison['torch_tokens_per_sec']:.2f} | "
                f"{comparison['triton_tokens_per_sec']:.2f} | {delta_str} | "
                f"{comparison['torch_step_time_s']:.2f} | {comparison['triton_step_time_s']:.2f} |"
            )

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _load_phase_result(self, filename: str) -> Dict[str, Any]:
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Requested to skip phase, but {path} does not exist."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def run(self) -> Dict[str, Any]:
        start = time.time()
        previous_results = None
        results_path = self.output_dir / "results.json"
        if results_path.exists():
            previous_results = json.loads(results_path.read_text(encoding="utf-8"))
        greedy_parity: Dict[str, Any]
        sampled_validation: Dict[str, Any]
        auto_policy: Dict[str, Any]

        need_loop = not (self.skip_parity and self.skip_sampled)
        if need_loop:
            loop = self._build_loop()
            try:
                parity_prompts = None
                sampled_prompts = None
                if not self.skip_parity:
                    parity_prompts = self._collect_prompt_batches(
                        loop, num_prompts=self.parity_prompts
                    )
                    greedy_parity = self._run_greedy_parity(loop, parity_prompts)
                else:
                    greedy_parity = self._load_phase_result("greedy_parity.json")

                if not self.skip_sampled:
                    sampled_prompts = self._collect_prompt_batches(
                        loop, num_prompts=self.sampled_prompts
                    )
                    sampled_validation = self._run_sampled_validation(
                        loop, sampled_prompts
                    )
                else:
                    sampled_validation = self._load_phase_result(
                        "sampled_validation.json"
                    )
                auto_policy = self._resolve_auto_policy(loop)
            finally:
                del loop
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            greedy_parity = self._load_phase_result("greedy_parity.json")
            sampled_validation = self._load_phase_result("sampled_validation.json")
            auto_policy = {
                "mode": "auto",
                "enabled": False,
                "reason": (
                    "auto policy routes sampled DeepSeek/Qwen generation to the torch prefix-cache path because it benchmarks faster than Triton paged-KV on the RTX 3060 Ti target"
                ),
            }

        if not self.skip_training:
            training_matrix = self._run_training_matrix()
        else:
            training_matrix = self._load_phase_result("training_matrix.json")
        elapsed_s = time.time() - start
        if (
            self.skip_parity
            and self.skip_sampled
            and self.skip_training
            and previous_results is not None
        ):
            previous_elapsed = float(previous_results.get("elapsed_s", 0.0))
            if previous_elapsed > 1.0:
                elapsed_s = previous_elapsed
            else:
                elapsed_s = sum(
                    float(result.get("duration_s", 0.0))
                    for result in training_matrix.get("results", [])
                )

        payload = {
            "greedy_parity": greedy_parity,
            "sampled_validation": sampled_validation,
            "auto_policy": auto_policy,
            "training_matrix": training_matrix,
            "elapsed_s": elapsed_s,
        }
        _write_json(self.output_dir / "results.json", payload)
        self._write_report(
            greedy_parity=greedy_parity,
            sampled_validation=sampled_validation,
            auto_policy=auto_policy,
            training_matrix=training_matrix,
            elapsed_s=elapsed_s,
        )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek generation parity, sampled validation, and training benchmark suite"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmarks/output/deepseek_generation_suite"),
    )
    parser.add_argument(
        "--group-sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated group sizes to validate",
    )
    parser.add_argument(
        "--parity-prompts",
        type=int,
        default=4,
        help="Number of GSM8K prompts to use for exact greedy parity",
    )
    parser.add_argument(
        "--sampled-prompts",
        type=int,
        default=4,
        help="Number of GSM8K prompts to use for sampled distribution checks",
    )
    parser.add_argument(
        "--sampled-seeds",
        type=str,
        default="11,23,37",
        help="Comma-separated seeds for sampled validation",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=128,
        help="Prompt length for validation and training benchmarks",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=128,
        help="Response length for validation and training benchmarks",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=30,
        help="Training steps per benchmark config (must be <= 30)",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip greedy parity and load an existing greedy_parity.json",
    )
    parser.add_argument(
        "--skip-sampled",
        action="store_true",
        help="Skip sampled validation and load an existing sampled_validation.json",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the train.py matrix and load an existing training_matrix.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_steps > 30:
        raise ValueError("--train-steps must be <= 30 for this validation suite.")

    group_sizes = [int(value.strip()) for value in args.group_sizes.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.sampled_seeds.split(",") if value.strip()]
    if group_sizes != DEFAULT_GROUP_SIZES:
        print(f"[INFO] Running non-default group sizes: {group_sizes}")

    suite = DeepSeekGenerationSuite(
        args.output_dir,
        group_sizes=group_sizes,
        parity_prompts=args.parity_prompts,
        sampled_prompts=args.sampled_prompts,
        sampled_seeds=seeds,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        train_steps=args.train_steps,
        skip_parity=args.skip_parity,
        skip_sampled=args.skip_sampled,
        skip_training=args.skip_training,
    )
    payload = suite.run()
    print(json.dumps(payload["auto_policy"], indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
