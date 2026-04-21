#!/usr/bin/env python3
"""
CDIA endurance benchmark suite for the PFG.

This suite is the strict final winner selector before the later heavy/long
training run. It is intentionally narrow:

- only the two remaining real finalist families are included
- every run uses the same trainer-step budget and evaluation cadence
- SENT stays disabled for all runs
- the held-out benchmark path stays on the trainer's existing GSM8K test
  benchmark flow

The suite defaults to 240 trainer steps because its purpose is to surface late
instability, not to run a broad exploratory sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarks" / "output" / "cdia_endurance"
DEFAULT_TRAINER_STEPS = 240
BENCHMARK_CADENCE_STEPS = 100
TOKEN_CAP_FAILURE_FRACTION = 0.95
COMPARISON_GROUPS = (
    "length_penalty_0",
    "length_penalty_0005_mask_truncated_off",
)


def note_cleanup_moved() -> None:
    """Match the runtime cleanup note used by the other benchmark suites."""
    print("[CLEANUP] train.py handles stale-process cleanup and GPU cache.")


def get_conda_python() -> str:
    """Resolve the Python executable for the configured conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "grpo-3060ti")
    possible_paths = [
        f"{os.environ.get('HOME', '/home/ndk')}/.conda/envs/{conda_env}/bin/python",
        f"/opt/anaconda/envs/{conda_env}/bin/python",
        f"{os.environ.get('CONDA_PREFIX', '')}/bin/python",
        sys.executable,
    ]
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    return sys.executable


def run_with_conda(cmd: list[str], **kwargs) -> subprocess.Popen[str]:
    """Run a command using the conda Python and a clean environment."""
    python_exe = get_conda_python()
    if cmd and cmd[0] in ["python", "python3", sys.executable]:
        cmd[0] = python_exe

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    return subprocess.Popen(cmd, env=env, **kwargs)


@dataclass(frozen=True)
class BaselineConfig:
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    group_size: int = 4
    batch_size: int = 1
    max_prompt_length: int = 4096
    max_response_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    entropy_mask: bool = True
    entropy_percentile: float = 0.5
    entropy_min_tokens: int = 10
    clip_epsilon: float = 0.2
    epsilon_high: float = 0.3
    delta: float = 1.5
    length_penalty_coef: float = 0.001
    mask_truncated_completions: bool = True
    trainer_steps: int = DEFAULT_TRAINER_STEPS
    seed: int = 1
    sent_enabled: bool = False
    difficulty_weighting_mode: str = "off"
    difficulty_weighting_min_weight: float = 1.0
    difficulty_weighting_max_weight: float = 1.2

    @property
    def effective_samples_per_update(self) -> int:
        return self.batch_size * self.group_size * self.gradient_accumulation_steps


BASELINE = BaselineConfig()


@dataclass
class BenchmarkResult:
    """Runtime summary for a single CDIA endurance run."""

    config_name: str
    comparison_group: str
    seed: int
    success: bool
    duration_s: float
    steps_completed: int
    step_time_avg_s: float | None = None
    step_time_min_s: float | None = None
    step_time_max_s: float | None = None
    tokens_per_sec: float | None = None
    tokens_per_sec_min: float | None = None
    tokens_per_sec_max: float | None = None
    vram_peak_gb: float | None = None
    vram_avg_gb: float | None = None
    loss_avg: float | None = None
    loss_final: float | None = None
    reward_avg: float | None = None
    reward_final: float | None = None
    reward_peak: float | None = None
    reward_peak_step: int | None = None
    reward_drop_from_peak_to_final: float | None = None
    reward_std_avg: float | None = None
    reward_std_final: float | None = None
    reward_min_avg: float | None = None
    reward_max_avg: float | None = None
    avg_response_length_avg: float | None = None
    avg_response_length_final: float | None = None
    response_length_peak: float | None = None
    response_length_peak_step: int | None = None
    response_length_at_reward_peak: float | None = None
    response_length_drop_from_peak_to_final: float | None = None
    response_budget_usage_avg: float | None = None
    response_budget_usage_final: float | None = None
    entropy_masked_ratio_avg: float | None = None
    truncated_completions_ratio_avg: float | None = None
    actual_truncated_completions_ratio_avg: float | None = None
    actual_truncated_completions_ratio_final: float | None = None
    actual_truncated_completions_ratio_peak: float | None = None
    actual_truncated_completions_ratio_peak_step: int | None = None
    truncated_completions_masked_out_of_loss_ratio_avg: float | None = None
    truncated_completions_masked_out_of_loss_ratio_final: float | None = None
    truncation_masking_active_final: float | None = None
    positive_advantages_ratio_avg: float | None = None
    oom_backoff_count_final: float | None = None
    optimizer_step_final: int | None = None
    optimizer_steps_completed: int | None = None
    optimizer_step_fraction_of_trainer_steps: float | None = None
    trainer_steps_per_optimizer_step: float | None = None
    in_training_heldout_benchmark_accuracy_values: list[float] = field(
        default_factory=list
    )
    in_training_heldout_benchmark_accuracy_steps: list[int] = field(
        default_factory=list
    )
    in_training_heldout_benchmark_accuracy_final: float | None = None
    in_training_heldout_benchmark_accuracy_peak: float | None = None
    in_training_heldout_benchmark_accuracy_peak_step: int | None = None
    final_checkpoint_heldout_benchmark_accuracy: float | None = None
    final_checkpoint_heldout_benchmark_step: int | None = None
    final_checkpoint_heldout_benchmark_format_compliance: float | None = None
    final_checkpoint_heldout_benchmark_avg_len: float | None = None
    error_message: str | None = None
    oom_events: int = 0
    log_path: str | None = None
    metrics_path: str | None = None


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    comparison_group: str
    description: str
    overrides: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def resolved(self) -> dict[str, Any]:
        resolved = asdict(BASELINE)
        resolved.update(self.overrides)
        return resolved

    def train_args(
        self,
        output_root: Path,
        trainer_steps: int,
        use_wandb: bool,
        run_prefix: str = "cdia",
    ) -> list[str]:
        resolved = self.resolved()
        normalized_prefix = run_prefix.strip().strip("_-")
        run_name = f"{normalized_prefix}_{self.name}" if normalized_prefix else self.name
        today = time.strftime("%Y-%m-%d")
        wandb_run_name = f"{today}_CDIA_{self.name}"
        run_dir = output_root / run_name
        args = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--output-dir",
            str(run_dir),
            "--max-steps",
            str(trainer_steps),
            "--group-size",
            str(resolved["group_size"]),
            "--batch-size",
            str(resolved["batch_size"]),
            "--learning-rate",
            str(resolved["learning_rate"]),
            "--gradient-accumulation-steps",
            str(resolved["gradient_accumulation_steps"]),
            "--max-prompt-length",
            str(resolved["max_prompt_length"]),
            "--max-response-length",
            str(resolved["max_response_length"]),
            "--clip-epsilon",
            str(resolved["clip_epsilon"]),
            "--epsilon-high",
            str(resolved["epsilon_high"]),
            "--delta",
            str(resolved["delta"]),
            "--length-penalty-coef",
            str(resolved["length_penalty_coef"]),
            "--entropy-percentile",
            str(resolved["entropy_percentile"]),
            "--entropy-min-tokens",
            str(resolved["entropy_min_tokens"]),
            "--temperature",
            str(resolved["temperature"]),
            "--top-p",
            str(resolved["top_p"]),
            "--seed",
            str(resolved["seed"]),
            "--no-sent",
            "--no-initial-benchmark",
            "--log-metrics-jsonl",
        ]
        if resolved["entropy_mask"]:
            args.append("--use-entropy-mask")
        else:
            args.append("--no-entropy-mask")
        if resolved["mask_truncated_completions"] is False:
            args.append("--no-mask-truncated")
        if resolved["do_sample"] is False:
            args.append("--greedy")
        if use_wandb:
            args.extend(
                [
                    "--wandb",
                    "--wandb-project",
                    "grpo-training",
                    "--wandb-run-name",
                    wandb_run_name,
                    "--wandb-tags",
                    "cdia",
                    "endurance",
                    self.comparison_group,
                    f"seed-{resolved['seed']}",
                ]
            )
        else:
            args.append("--no-wandb")
        return args


def build_suite() -> list[SuiteConfig]:
    return [
        SuiteConfig(
            name="length_penalty_0_seed1",
            comparison_group="length_penalty_0",
            description="Endurance finalist comparison: no length penalty, seed 1",
            overrides={
                "length_penalty_coef": 0.0,
                "seed": 1,
                "sent_enabled": False,
            },
            notes=[
                "Strict finalist run.",
                "Raw reward is not directly comparable against penalized runs.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0_seed2",
            comparison_group="length_penalty_0",
            description="Endurance finalist comparison: no length penalty, seed 2",
            overrides={
                "length_penalty_coef": 0.0,
                "seed": 2,
                "sent_enabled": False,
            },
            notes=[
                "Strict finalist run.",
                "Raw reward is not directly comparable against penalized runs.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0_seed3",
            comparison_group="length_penalty_0",
            description="Endurance finalist comparison: no length penalty, seed 3",
            overrides={
                "length_penalty_coef": 0.0,
                "seed": 3,
                "sent_enabled": False,
            },
            notes=[
                "Strict finalist run.",
                "Raw reward is not directly comparable against penalized runs.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0005_mask_truncated_off_seed1",
            comparison_group="length_penalty_0005_mask_truncated_off",
            description=(
                "Endurance finalist comparison: len_penalty=0.0005, "
                "mask_truncated=False, seed 1"
            ),
            overrides={
                "length_penalty_coef": 0.0005,
                "mask_truncated_completions": False,
                "seed": 1,
                "sent_enabled": False,
            },
            notes=[
                "Strict finalist run.",
                "Penalized raw reward includes a length-cost subtraction.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0005_mask_truncated_off_seed2",
            comparison_group="length_penalty_0005_mask_truncated_off",
            description=(
                "Endurance finalist comparison: len_penalty=0.0005, "
                "mask_truncated=False, seed 2"
            ),
            overrides={
                "length_penalty_coef": 0.0005,
                "mask_truncated_completions": False,
                "seed": 2,
                "sent_enabled": False,
            },
            notes=[
                "Strict finalist run.",
                "Penalized raw reward includes a length-cost subtraction.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0005_mask_truncated_off_seed3",
            comparison_group="length_penalty_0005_mask_truncated_off",
            description=(
                "Endurance finalist comparison: len_penalty=0.0005, "
                "mask_truncated=False, seed 3"
            ),
            overrides={
                "length_penalty_coef": 0.0005,
                "mask_truncated_completions": False,
                "seed": 3,
                "sent_enabled": False,
            },
            notes=[
                "Strict finalist run.",
                "Penalized raw reward includes a length-cost subtraction.",
            ],
        ),
    ]


def _parse_selected_runs(run_args: list[str] | None) -> list[str]:
    if not run_args:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    for raw in run_args:
        for item in raw.split(","):
            name = item.strip()
            if not name or name in seen:
                continue
            selected.append(name)
            seen.add(name)
    return selected


def filter_suite(
    configs: list[SuiteConfig], selected_names: list[str] | None = None
) -> list[SuiteConfig]:
    """Filter suite configs by explicit run names."""
    if not selected_names:
        return configs

    registry = {cfg.name: cfg for cfg in configs}
    filtered: list[SuiteConfig] = []
    unknown: list[str] = []
    for name in selected_names:
        cfg = registry.get(name)
        if cfg is None:
            unknown.append(name)
            continue
        filtered.append(cfg)

    if unknown:
        available = ", ".join(cfg.name for cfg in configs)
        missing = ", ".join(unknown)
        raise ValueError(f"Unknown run name(s): {missing}. Available: {available}")

    return filtered


def _print_config_section(title: str, configs: list[SuiteConfig]) -> None:
    print(title)
    print("-" * len(title))
    for cfg in configs:
        resolved = cfg.resolved()
        print(f"[RUNNABLE] {cfg.name} ({cfg.comparison_group})")
        print(f"  {cfg.description}")
        print(
            "  "
            f"seed={resolved['seed']} sent={resolved['sent_enabled']} "
            f"len_penalty={resolved['length_penalty_coef']} "
            f"mask_truncated={resolved['mask_truncated_completions']} "
            f"grad_accum={resolved['gradient_accumulation_steps']} "
            f"group={resolved['group_size']} "
            f"entropy_mask={resolved['entropy_mask']}"
        )
        for note in cfg.notes:
            print(f"  note={note}")
        print()


def print_summary(configs: list[SuiteConfig], trainer_steps: int) -> None:
    by_group = {
        group: [cfg for cfg in configs if cfg.comparison_group == group]
        for group in COMPARISON_GROUPS
    }
    print("CDIA endurance suite")
    print(f"Trainer steps per config: {trainer_steps}")
    print(f"Configs in suite: {len(configs)}")
    print(f"Default run set size: 6")
    print(f"Baseline effective samples/update: {BASELINE.effective_samples_per_update}")
    print(
        "Scope: strict finalist selection only. No difficulty weighting, no eps_high "
        "variants, no batch-size variants, no exploratory additions."
    )
    print(
        "Held-out benchmark cadence: same for every run, periodic every "
        f"{BENCHMARK_CADENCE_STEPS} trainer steps plus one final post-checkpoint benchmark."
    )
    print(
        "Interpretation rule: raw reward is not directly comparable between the "
        "penalized and unpenalized finalist families."
    )
    print()
    for group in COMPARISON_GROUPS:
        _print_config_section(group, by_group[group])


def write_plan(configs: list[SuiteConfig], output_path: Path, trainer_steps: int) -> None:
    payload = {
        "trainer_steps": trainer_steps,
        "benchmark_cadence_trainer_steps": BENCHMARK_CADENCE_STEPS,
        "final_checkpoint_benchmark": True,
        "baseline": asdict(BASELINE),
        "comparison_groups": list(COMPARISON_GROUPS),
        "default_run_names": [cfg.name for cfg in build_suite()],
        "heldout_benchmark_source": (
            "trainer GSM8KBenchmark test split from src/grpo/benchmark.py"
        ),
        "configs": [
            {
                "name": cfg.name,
                "comparison_group": cfg.comparison_group,
                "description": cfg.description,
                "overrides": cfg.overrides,
                "resolved": cfg.resolved(),
                "notes": cfg.notes,
            }
            for cfg in configs
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CdiaEnduranceSuite:
    """Strict CDIA finalist endurance suite."""

    def __init__(
        self,
        output_dir: Path,
        trainer_steps: int,
        use_wandb: bool,
        run_prefix: str,
        plan_path: Path,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trainer_steps = trainer_steps
        self.use_wandb = use_wandb
        self.run_prefix = self._normalize_prefix(run_prefix)
        self.plan_path = plan_path
        self.results: list[BenchmarkResult] = []
        self.python_exe = get_conda_python()
        self.suite_name = "CDIA Endurance Suite"
        self.report_title = "CDIA Endurance Report"
        self._config_registry: dict[str, SuiteConfig] = {}

    def _normalize_prefix(self, prefix: Optional[str]) -> str:
        if prefix is None:
            return ""
        return prefix.strip().strip("_-")

    def _prefixed_name(self, name: str) -> str:
        if not self.run_prefix:
            return name
        return f"{self.run_prefix}_{name}"

    def _config_for_result(self, result: BenchmarkResult) -> SuiteConfig | None:
        return self._config_registry.get(result.config_name)

    def _parse_log(self, log_path: Path) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "steps_completed": 0,
            "step_times": [],
            "tokens_per_sec": [],
            "losses": [],
            "rewards": [],
            "vram_samples": [],
            "oom_events": 0,
            "error": None,
        }
        if not log_path.exists():
            return metrics

        content = log_path.read_text(encoding="utf-8", errors="ignore")
        step_records: dict[int, dict[str, float]] = {}
        for segment in re.split(r"[\r\n]+", content):
            step_match = re.search(r"step=(\d+)", segment)
            if not step_match:
                continue

            step = int(step_match.group(1))
            record = step_records.setdefault(step, {})

            step_time_match = re.search(r"(\d+\.?\d*)s/it", segment)
            if step_time_match:
                record["step_time"] = float(step_time_match.group(1))

            tokens_match = re.search(r"tokens_per_sec[=:]\s*([\d.]+)", segment)
            if tokens_match:
                record["tokens_per_sec"] = float(tokens_match.group(1))

            loss_match = re.search(r"loss[=:]\s*(-?[\d.]+)", segment)
            if loss_match:
                record["loss"] = float(loss_match.group(1))

            reward_match = re.search(r"reward[=:]\s*(-?[\d.]+)", segment)
            if reward_match:
                record["reward"] = float(reward_match.group(1))

        metrics["steps_completed"] = len(step_records)
        metrics["step_times"] = [
            record["step_time"]
            for _, record in sorted(step_records.items())
            if "step_time" in record
        ]
        metrics["tokens_per_sec"] = [
            record["tokens_per_sec"]
            for _, record in sorted(step_records.items())
            if "tokens_per_sec" in record
        ]
        metrics["losses"] = [
            record["loss"] for _, record in sorted(step_records.items()) if "loss" in record
        ]
        metrics["rewards"] = [
            record["reward"]
            for _, record in sorted(step_records.items())
            if "reward" in record
        ]

        for match in re.finditer(r"VRAM:\s*([\d.]+)\s*GB", content):
            try:
                metrics["vram_samples"].append(float(match.group(1)))
            except ValueError:
                pass

        if "CUDA out of memory" in content or "OOM" in content:
            metrics["oom_events"] += 1

        if "Traceback" in content or "Error" in content or "Exception" in content:
            for line in reversed(content.splitlines()):
                if "Error" in line or "Exception" in line:
                    metrics["error"] = line.strip()
                    break

        return metrics

    @staticmethod
    def _series_mean(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    @staticmethod
    def _series_std(values: list[float]) -> float | None:
        if not values:
            return None
        if len(values) == 1:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _peak_with_step(
        values: list[float], steps: list[int]
    ) -> tuple[float | None, int | None, int | None]:
        if not values or not steps or len(values) != len(steps):
            return None, None, None
        peak_idx = max(range(len(values)), key=values.__getitem__)
        return values[peak_idx], steps[peak_idx], peak_idx

    @staticmethod
    def _format_peak(value: float | None, step: int | None, precision: int = 4) -> str:
        if value is None or step is None:
            return "N/A"
        return f"{value:.{precision}f}@{step}"

    @staticmethod
    def _format_optional(value: float | None, precision: int = 4) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{precision}f}"

    def _format_mean_spread(self, values: list[float], precision: int = 4) -> str:
        if not values:
            return "N/A"
        mean = self._series_mean(values)
        spread = self._series_std(values)
        if mean is None or spread is None:
            return "N/A"
        return f"{mean:.{precision}f} +- {spread:.{precision}f}"

    @staticmethod
    def _format_series_with_steps(values: list[float], steps: list[int], precision: int = 4) -> str:
        if not values or not steps or len(values) != len(steps):
            return "N/A"
        return ", ".join(
            f"{value:.{precision}f}@{step}" for value, step in zip(values, steps, strict=False)
        )

    def _parse_metrics_jsonl(self, metrics_path: Path) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "steps": [],
            "reward_avg_values": [],
            "reward_avg_steps": [],
            "reward_std_values": [],
            "reward_min_values": [],
            "reward_max_values": [],
            "avg_response_length_values": [],
            "avg_response_length_steps": [],
            "entropy_masked_ratio_values": [],
            "truncated_completions_ratio_values": [],
            "actual_truncated_completions_ratio_values": [],
            "actual_truncated_completions_ratio_steps": [],
            "truncated_completions_masked_out_of_loss_ratio_values": [],
            "truncation_masking_active_values": [],
            "positive_advantages_ratio_values": [],
            "oom_backoff_count_values": [],
            "optimizer_step_values": [],
            "optimizer_step_steps": [],
            "periodic_benchmark_accuracy_values": [],
            "periodic_benchmark_accuracy_steps": [],
            "final_checkpoint_benchmark_accuracy": None,
            "final_checkpoint_benchmark_step": None,
            "final_checkpoint_benchmark_format_compliance": None,
            "final_checkpoint_benchmark_avg_len": None,
        }
        if not metrics_path.exists():
            return stats

        for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = entry.get("event")
            step = int(entry.get("step", 0))

            if event == "train_metrics":
                stats["steps"].append(step)

                if "train/avg_reward" in entry:
                    stats["reward_avg_values"].append(float(entry["train/avg_reward"]))
                    stats["reward_avg_steps"].append(step)
                if "train/reward_std" in entry:
                    stats["reward_std_values"].append(float(entry["train/reward_std"]))
                if "train/reward_min" in entry:
                    stats["reward_min_values"].append(float(entry["train/reward_min"]))
                if "train/reward_max" in entry:
                    stats["reward_max_values"].append(float(entry["train/reward_max"]))
                if "train/avg_response_length" in entry:
                    stats["avg_response_length_values"].append(
                        float(entry["train/avg_response_length"])
                    )
                    stats["avg_response_length_steps"].append(step)
                if "train/entropy_masked_ratio" in entry:
                    stats["entropy_masked_ratio_values"].append(
                        float(entry["train/entropy_masked_ratio"])
                    )
                if "train/truncated_completions_ratio" in entry:
                    stats["truncated_completions_ratio_values"].append(
                        float(entry["train/truncated_completions_ratio"])
                    )
                if "train/actual_truncated_completions_ratio" in entry:
                    stats["actual_truncated_completions_ratio_values"].append(
                        float(entry["train/actual_truncated_completions_ratio"])
                    )
                    stats["actual_truncated_completions_ratio_steps"].append(step)
                if "train/truncated_completions_masked_out_of_loss_ratio" in entry:
                    stats["truncated_completions_masked_out_of_loss_ratio_values"].append(
                        float(entry["train/truncated_completions_masked_out_of_loss_ratio"])
                    )
                if "train/truncation_masking_active" in entry:
                    stats["truncation_masking_active_values"].append(
                        float(entry["train/truncation_masking_active"])
                    )
                if "train/positive_advantages_ratio" in entry:
                    stats["positive_advantages_ratio_values"].append(
                        float(entry["train/positive_advantages_ratio"])
                    )
                if "train/oom_backoff_count" in entry:
                    stats["oom_backoff_count_values"].append(
                        float(entry["train/oom_backoff_count"])
                    )
                if "train/optimizer_step" in entry:
                    stats["optimizer_step_values"].append(float(entry["train/optimizer_step"]))
                    stats["optimizer_step_steps"].append(step)
                continue

            if event != "benchmark_metrics":
                continue

            phase = str(entry.get("benchmark_phase", ""))
            if phase == "periodic" and "val/acc" in entry:
                stats["periodic_benchmark_accuracy_values"].append(float(entry["val/acc"]))
                stats["periodic_benchmark_accuracy_steps"].append(step)
            elif phase == "final_checkpoint":
                if "val/acc" in entry:
                    stats["final_checkpoint_benchmark_accuracy"] = float(entry["val/acc"])
                    stats["final_checkpoint_benchmark_step"] = step
                if "val/format_compliance" in entry:
                    stats["final_checkpoint_benchmark_format_compliance"] = float(
                        entry["val/format_compliance"]
                    )
                if "val/avg_len" in entry:
                    stats["final_checkpoint_benchmark_avg_len"] = float(
                        entry["val/avg_len"]
                    )

        return stats

    def _estimate_total_minutes(self, configs: list[SuiteConfig]) -> float:
        est_time_min = 0.0
        for cfg in configs:
            resolved = cfg.resolved()
            base_time = 25.0
            base_time *= self.trainer_steps / 30.0
            base_time *= resolved["max_response_length"] / 1024.0
            base_time *= resolved["group_size"] / 4.0
            base_time *= resolved["gradient_accumulation_steps"] / 4.0
            est_time_min += base_time
        return est_time_min

    def _save_results(self) -> None:
        results_path = self.output_dir / "results.json"
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "suite_name": self.suite_name,
            "trainer_steps": self.trainer_steps,
            "benchmark_cadence_trainer_steps": BENCHMARK_CADENCE_STEPS,
            "total_runs": len(self.results),
            "successful_runs": sum(1 for result in self.results if result.success),
            "results": [asdict(result) for result in self.results],
        }
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _results_by_group(self) -> dict[str, list[BenchmarkResult]]:
        return {
            group: [result for result in self.results if result.comparison_group == group]
            for group in COMPARISON_GROUPS
        }

    def _is_cap_drift_failure(self, result: BenchmarkResult) -> bool:
        if (
            result.response_budget_usage_final is not None
            and result.response_budget_usage_final >= TOKEN_CAP_FAILURE_FRACTION
        ):
            return True
        if (
            result.response_length_peak is not None
            and result.response_length_peak
            >= TOKEN_CAP_FAILURE_FRACTION * BASELINE.max_response_length
        ):
            return True
        return False

    def _is_late_collapse(self, result: BenchmarkResult) -> bool:
        if (
            result.in_training_heldout_benchmark_accuracy_peak is not None
            and result.final_checkpoint_heldout_benchmark_accuracy is not None
            and (
                result.in_training_heldout_benchmark_accuracy_peak
                - result.final_checkpoint_heldout_benchmark_accuracy
            )
            >= 0.05
        ):
            return True
        if (
            result.reward_drop_from_peak_to_final is not None
            and result.reward_drop_from_peak_to_final >= 0.25
            and self._is_cap_drift_failure(result)
        ):
            return True
        return False

    def _summarize_group(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        success_results = [result for result in results if result.success]
        summary = {
            "runs": len(results),
            "successes": len(success_results),
            "final_checkpoint_accuracy_values": [
                result.final_checkpoint_heldout_benchmark_accuracy
                for result in success_results
                if result.final_checkpoint_heldout_benchmark_accuracy is not None
            ],
            "in_training_final_accuracy_values": [
                result.in_training_heldout_benchmark_accuracy_final
                for result in success_results
                if result.in_training_heldout_benchmark_accuracy_final is not None
            ],
            "in_training_peak_accuracy_values": [
                result.in_training_heldout_benchmark_accuracy_peak
                for result in success_results
                if result.in_training_heldout_benchmark_accuracy_peak is not None
            ],
            "tokens_per_sec_values": [
                result.tokens_per_sec for result in success_results if result.tokens_per_sec is not None
            ],
            "duration_s_values": [
                result.duration_s for result in success_results if result.duration_s is not None
            ],
            "reward_avg_values": [
                result.reward_avg for result in success_results if result.reward_avg is not None
            ],
            "reward_final_values": [
                result.reward_final for result in success_results if result.reward_final is not None
            ],
            "reward_drop_values": [
                result.reward_drop_from_peak_to_final
                for result in success_results
                if result.reward_drop_from_peak_to_final is not None
            ],
            "response_length_final_values": [
                result.avg_response_length_final
                for result in success_results
                if result.avg_response_length_final is not None
            ],
            "response_budget_final_values": [
                result.response_budget_usage_final
                for result in success_results
                if result.response_budget_usage_final is not None
            ],
            "actual_truncation_final_values": [
                result.actual_truncated_completions_ratio_final
                for result in success_results
                if result.actual_truncated_completions_ratio_final is not None
            ],
            "cap_drift_failures": sum(
                1 for result in success_results if self._is_cap_drift_failure(result)
            ),
            "late_collapse_flags": sum(
                1 for result in success_results if self._is_late_collapse(result)
            ),
        }
        return summary

    def _winner_key(self, summary: dict[str, Any]) -> tuple[float, float, int, int, float, float]:
        final_ckpt_mean = self._series_mean(summary["final_checkpoint_accuracy_values"])
        in_train_final_mean = self._series_mean(summary["in_training_final_accuracy_values"])
        final_ckpt_spread = self._series_std(summary["final_checkpoint_accuracy_values"])
        tok_s_mean = self._series_mean(summary["tokens_per_sec_values"])
        return (
            final_ckpt_mean if final_ckpt_mean is not None else -1.0,
            in_train_final_mean if in_train_final_mean is not None else -1.0,
            -summary["late_collapse_flags"],
            -summary["cap_drift_failures"],
            -(final_ckpt_spread if final_ckpt_spread is not None else 1e9),
            tok_s_mean if tok_s_mean is not None else -1.0,
        )

    def _append_summary_table(self, lines: list[str], results: list[BenchmarkResult]) -> None:
        lines.extend(
            [
                "## Summary",
                "",
                "| Config | Group | Seed | Status | Steps | Opt Steps | Tok/s | In-Train Bench | Final Ckpt Bench | Reward Final | Reward Peak@Step | Drop | Resp Final | Resp Peak@Step | Trunc Avg | Masked-Out Final |",
                "|--------|-------|------|--------|-------|-----------|-------|----------------|------------------|--------------|------------------|------|------------|----------------|-----------|------------------|",
            ]
        )
        if not results:
            lines.extend(
                ["| _none_ | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |", ""]
            )
            return

        for result in results:
            status = "OK" if result.success else "FAIL"
            opt_steps = (
                str(result.optimizer_step_final)
                if result.optimizer_step_final is not None
                else "N/A"
            )
            tok_s = f"{result.tokens_per_sec:.2f}" if result.tokens_per_sec else "N/A"
            in_train_bench = (
                f"{result.in_training_heldout_benchmark_accuracy_final:.4f}"
                if result.in_training_heldout_benchmark_accuracy_final is not None
                else "N/A"
            )
            final_ckpt_bench = (
                f"{result.final_checkpoint_heldout_benchmark_accuracy:.4f}"
                if result.final_checkpoint_heldout_benchmark_accuracy is not None
                else "N/A"
            )
            reward_final = (
                f"{result.reward_final:.4f}" if result.reward_final is not None else "N/A"
            )
            reward_peak = self._format_peak(
                result.reward_peak, result.reward_peak_step, precision=4
            )
            reward_drop = (
                f"{result.reward_drop_from_peak_to_final:.4f}"
                if result.reward_drop_from_peak_to_final is not None
                else "N/A"
            )
            final_resp_len = (
                f"{result.avg_response_length_final:.1f}"
                if result.avg_response_length_final is not None
                else "N/A"
            )
            peak_resp_len = self._format_peak(
                result.response_length_peak, result.response_length_peak_step, precision=1
            )
            actual_trunc_avg = (
                f"{result.actual_truncated_completions_ratio_avg:.4f}"
                if result.actual_truncated_completions_ratio_avg is not None
                else "N/A"
            )
            masked_out_final = (
                f"{result.truncated_completions_masked_out_of_loss_ratio_final:.4f}"
                if result.truncated_completions_masked_out_of_loss_ratio_final is not None
                else "N/A"
            )
            lines.append(
                f"| {result.config_name} | {result.comparison_group} | {result.seed} | "
                f"{status} | {result.steps_completed} | {opt_steps} | {tok_s} | "
                f"{in_train_bench} | {final_ckpt_bench} | {reward_final} | "
                f"{reward_peak} | {reward_drop} | {final_resp_len} | {peak_resp_len} | "
                f"{actual_trunc_avg} | {masked_out_final} |"
            )
        lines.append("")

    def _append_selection_guidance(self, lines: list[str]) -> None:
        grouped = self._results_by_group()
        summaries = {group: self._summarize_group(results) for group, results in grouped.items()}

        lines.extend(
            [
                "## Selection Guidance",
                "",
                "Compare only the two finalist families below.",
                "",
                "- Raw reward is not directly comparable between the penalized and unpenalized families because the penalized objective subtracts length cost.",
                "- Final winner selection should weigh held-out benchmark accuracy, repeatability across seeds, late collapse or survival, response-length and truncation behavior, throughput/runtime cost, and reward only in context.",
                f"- This report explicitly treats drift toward the {BASELINE.max_response_length} token cap as a failure signal. A run is flagged when final or peak average response length reaches at least {TOKEN_CAP_FAILURE_FRACTION:.0%} of the cap.",
                "",
                "| Finalist Family | Success | Final Ckpt Acc Mean+-Spread | In-Train Final Acc Mean+-Spread | In-Train Peak Acc Mean+-Spread | Late Collapse Flags | Cap-Drift Flags | Final Resp Len Mean+-Spread | Final Trunc Mean+-Spread | Tok/s Mean+-Spread | Reward Mean+-Spread (Context Only) |",
                "|-----------------|---------|-----------------------------|----------------------------------|---------------------------------|--------------------|-----------------|-----------------------------|--------------------------|--------------------|------------------------------------|",
            ]
        )

        for group in COMPARISON_GROUPS:
            summary = summaries[group]
            lines.append(
                f"| {group} | {summary['successes']}/{summary['runs']} | "
                f"{self._format_mean_spread(summary['final_checkpoint_accuracy_values'])} | "
                f"{self._format_mean_spread(summary['in_training_final_accuracy_values'])} | "
                f"{self._format_mean_spread(summary['in_training_peak_accuracy_values'])} | "
                f"{summary['late_collapse_flags']} | "
                f"{summary['cap_drift_failures']} | "
                f"{self._format_mean_spread(summary['response_length_final_values'], precision=1)} | "
                f"{self._format_mean_spread(summary['actual_truncation_final_values'])} | "
                f"{self._format_mean_spread(summary['tokens_per_sec_values'], precision=2)} | "
                f"{self._format_mean_spread(summary['reward_final_values'])} |"
            )

        winner = max(COMPARISON_GROUPS, key=lambda group: self._winner_key(summaries[group]))
        runner_up = next(group for group in COMPARISON_GROUPS if group != winner)
        winner_summary = summaries[winner]
        runner_up_summary = summaries[runner_up]

        winner_final_ckpt = self._series_mean(winner_summary["final_checkpoint_accuracy_values"])
        runner_up_final_ckpt = self._series_mean(
            runner_up_summary["final_checkpoint_accuracy_values"]
        )
        winner_tok_s = self._series_mean(winner_summary["tokens_per_sec_values"])
        runner_up_tok_s = self._series_mean(runner_up_summary["tokens_per_sec_values"])

        lines.extend(
            [
                "",
                f"Recommended single winner for the later heavy run: `{winner}`.",
                "",
                f"- Benchmark accuracy: `{winner}` leads on the primary selector signal when compared against `{runner_up}` using final-checkpoint held-out accuracy ({self._format_optional(winner_final_ckpt)} vs {self._format_optional(runner_up_final_ckpt)}).",
                f"- Repeatability and survival: `{winner}` has {winner_summary['late_collapse_flags']} late-collapse flag(s) across {winner_summary['runs']} seed(s) versus {runner_up_summary['late_collapse_flags']} for `{runner_up}`.",
                f"- Response-length drift: `{winner}` has {winner_summary['cap_drift_failures']} cap-drift failure signal(s) versus {runner_up_summary['cap_drift_failures']} for `{runner_up}`. Any drift toward the {BASELINE.max_response_length}-token cap is treated here as negative evidence, not as progress.",
                f"- Throughput/runtime context: `{winner}` averages {winner_tok_s:.2f} tok/s versus {runner_up_tok_s:.2f} tok/s for `{runner_up}`." if winner_tok_s is not None and runner_up_tok_s is not None else "- Throughput/runtime context: insufficient tok/s data to separate the finalists on runtime cost alone.",
                "- Reward context only: keep reward in the report for within-family drift and collapse reading, but do not use raw reward alone to pick between the penalized and unpenalized finalists.",
                "",
            ]
        )

    def _generate_report(self) -> None:
        report_path = self.output_dir / "report.md"
        lines = [
            f"# {self.report_title}",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Plan: {self.plan_path}",
            f"Trainer steps: {self.trainer_steps}",
            f"Total runs: {len(self.results)}",
            f"Successful: {sum(1 for result in self.results if result.success)}",
            "Derived from benchmarks/cdia_final_validation_suite.py with the same output flow and per-run layout, but narrowed to a strict 6-run endurance selector across the two real finalist families.",
            "",
            "## Methodology",
            "",
            "- Strict finalist-only scope: `length_penalty_0` versus `length_penalty_0005_mask_truncated_off`, three seeds each, no other variants.",
            f"- Apples-to-apples budget: every run gets {self.trainer_steps} trainer steps, the same held-out benchmark path, the same benchmark cadence, and SENT disabled.",
            f"- Held-out benchmark cadence: periodic every {BENCHMARK_CADENCE_STEPS} trainer steps plus one final benchmark immediately after the final checkpoint is saved.",
            "- Hardware/output assumptions match the current CDIA suites: same train.py path, same per-run folder structure, same checkpoint markers, same metrics JSONL logging.",
            "- Critical interpretation rule: raw reward is not directly comparable between the penalized and unpenalized finalist families. A penalized run can show lower raw reward simply because the objective subtracts length cost.",
            "",
        ]

        self._append_summary_table(lines, self.results)

        lines.extend(["## Detailed Results", ""])
        for result in self.results:
            lines.extend(
                [
                    f"### {result.config_name}",
                    "",
                    f"- Comparison group: {result.comparison_group}",
                    f"- Seed: {result.seed}",
                    f"- Success: {result.success}",
                    f"- Duration: {result.duration_s:.1f}s",
                    f"- Steps completed: {result.steps_completed}",
                    f"- Log: {result.log_path or 'N/A'}",
                    f"- Metrics: {result.metrics_path or 'N/A'}",
                ]
            )
            if result.step_time_avg_s is not None:
                lines.append(
                    f"- Step time: {result.step_time_avg_s:.2f}s (min: {result.step_time_min_s:.2f}, max: {result.step_time_max_s:.2f})"
                )
            if result.tokens_per_sec is not None:
                lines.append(f"- Throughput: {result.tokens_per_sec:.2f} tok/s")
            if result.vram_peak_gb is not None:
                lines.append(f"- VRAM peak: {result.vram_peak_gb:.2f} GB")
            if result.optimizer_step_final is not None:
                lines.append(f"- Optimizer steps completed: {result.optimizer_step_final}")
            if result.trainer_steps_per_optimizer_step is not None:
                lines.append(
                    "- Trainer steps per optimizer step: "
                    f"{result.trainer_steps_per_optimizer_step:.2f}"
                )
            if result.loss_final is not None:
                lines.append(f"- Final loss: {result.loss_final:.4f}")
            if result.reward_avg is not None:
                lines.append(f"- Whole-run reward avg: {result.reward_avg:.4f}")
            if result.reward_final is not None:
                lines.append(f"- Final reward: {result.reward_final:.4f}")
            if result.reward_peak is not None and result.reward_peak_step is not None:
                lines.append(
                    f"- Peak reward: {result.reward_peak:.4f} at step {result.reward_peak_step}"
                )
            if result.reward_drop_from_peak_to_final is not None:
                lines.append(
                    "- Reward drop from peak to final: "
                    f"{result.reward_drop_from_peak_to_final:.4f}"
                )
            if result.reward_std_final is not None:
                lines.append(f"- Final reward std: {result.reward_std_final:.4f}")
            if result.reward_std_avg is not None:
                lines.append(f"- Whole-run reward std avg: {result.reward_std_avg:.4f}")
            if result.reward_min_avg is not None and result.reward_max_avg is not None:
                lines.append(
                    f"- Reward range avg: [{result.reward_min_avg:.4f}, {result.reward_max_avg:.4f}]"
                )
            if result.avg_response_length_avg is not None:
                lines.append(
                    f"- Whole-run avg response length: {result.avg_response_length_avg:.1f}"
                )
            if result.avg_response_length_final is not None:
                lines.append(
                    f"- Final avg response length: {result.avg_response_length_final:.1f}"
                )
            if (
                result.response_length_peak is not None
                and result.response_length_peak_step is not None
            ):
                lines.append(
                    "- Peak avg response length: "
                    f"{result.response_length_peak:.1f} at step {result.response_length_peak_step}"
                )
            if (
                result.response_length_at_reward_peak is not None
                and result.reward_peak_step is not None
            ):
                lines.append(
                    "- Avg response length at reward peak step "
                    f"{result.reward_peak_step}: {result.response_length_at_reward_peak:.1f}"
                )
            if result.response_length_drop_from_peak_to_final is not None:
                lines.append(
                    "- Response length drop from peak to final: "
                    f"{result.response_length_drop_from_peak_to_final:.1f}"
                )
            if result.response_budget_usage_avg is not None:
                lines.append(
                    f"- Response budget usage avg: {result.response_budget_usage_avg:.4f}"
                )
            if result.response_budget_usage_final is not None:
                lines.append(
                    f"- Final response budget usage: {result.response_budget_usage_final:.4f}"
                )
            if self._is_cap_drift_failure(result):
                lines.append(
                    f"- Cap-drift failure signal: True (drift toward the {BASELINE.max_response_length}-token cap)"
                )
            if result.entropy_masked_ratio_avg is not None:
                lines.append(
                    f"- Entropy-masked ratio avg: {result.entropy_masked_ratio_avg:.4f}"
                )
            if result.truncated_completions_ratio_avg is not None:
                lines.append(
                    "- Legacy truncated-completions ratio avg "
                    f"(only when masking active): {result.truncated_completions_ratio_avg:.4f}"
                )
            if result.actual_truncated_completions_ratio_avg is not None:
                lines.append(
                    "- Actual truncated completions ratio avg: "
                    f"{result.actual_truncated_completions_ratio_avg:.4f}"
                )
            if (
                result.actual_truncated_completions_ratio_peak is not None
                and result.actual_truncated_completions_ratio_peak_step is not None
            ):
                lines.append(
                    "- Peak actual truncated completions ratio: "
                    f"{result.actual_truncated_completions_ratio_peak:.4f} at step "
                    f"{result.actual_truncated_completions_ratio_peak_step}"
                )
            if result.actual_truncated_completions_ratio_final is not None:
                lines.append(
                    "- Final actual truncated completions ratio: "
                    f"{result.actual_truncated_completions_ratio_final:.4f}"
                )
            if result.truncated_completions_masked_out_of_loss_ratio_avg is not None:
                lines.append(
                    "- Truncated completions masked out of loss ratio avg: "
                    f"{result.truncated_completions_masked_out_of_loss_ratio_avg:.4f}"
                )
            if result.truncated_completions_masked_out_of_loss_ratio_final is not None:
                lines.append(
                    "- Final truncated completions masked out of loss ratio: "
                    f"{result.truncated_completions_masked_out_of_loss_ratio_final:.4f}"
                )
            if result.truncation_masking_active_final is not None:
                lines.append(
                    "- Truncation masking active in loss: "
                    f"{bool(int(result.truncation_masking_active_final))}"
                )
            if result.positive_advantages_ratio_avg is not None:
                lines.append(
                    f"- Positive advantages ratio avg: {result.positive_advantages_ratio_avg:.4f}"
                )
            if result.oom_backoff_count_final is not None:
                lines.append(f"- OOM backoff count final: {result.oom_backoff_count_final:.0f}")
            lines.append(
                "- In-training held-out benchmark accuracy values: "
                + self._format_series_with_steps(
                    result.in_training_heldout_benchmark_accuracy_values,
                    result.in_training_heldout_benchmark_accuracy_steps,
                )
            )
            if result.in_training_heldout_benchmark_accuracy_final is not None:
                lines.append(
                    "- Final in-training held-out benchmark accuracy: "
                    f"{result.in_training_heldout_benchmark_accuracy_final:.4f}"
                )
            if (
                result.in_training_heldout_benchmark_accuracy_peak is not None
                and result.in_training_heldout_benchmark_accuracy_peak_step is not None
            ):
                lines.append(
                    "- Peak in-training held-out benchmark accuracy: "
                    f"{result.in_training_heldout_benchmark_accuracy_peak:.4f} at step "
                    f"{result.in_training_heldout_benchmark_accuracy_peak_step}"
                )
            if result.final_checkpoint_heldout_benchmark_accuracy is not None:
                lines.append(
                    "- Final checkpoint held-out benchmark accuracy: "
                    f"{result.final_checkpoint_heldout_benchmark_accuracy:.4f}"
                )
            if result.final_checkpoint_heldout_benchmark_step is not None:
                lines.append(
                    "- Final checkpoint held-out benchmark step: "
                    f"{result.final_checkpoint_heldout_benchmark_step}"
                )
            if result.final_checkpoint_heldout_benchmark_format_compliance is not None:
                lines.append(
                    "- Final checkpoint benchmark format compliance: "
                    f"{result.final_checkpoint_heldout_benchmark_format_compliance:.4f}"
                )
            if result.final_checkpoint_heldout_benchmark_avg_len is not None:
                lines.append(
                    "- Final checkpoint benchmark avg length: "
                    f"{result.final_checkpoint_heldout_benchmark_avg_len:.1f}"
                )
            if self._is_late_collapse(result):
                lines.append("- Late-collapse flag: True")
            if result.error_message:
                lines.append(f"- Error: {result.error_message}")
            lines.append("")

        self._append_selection_guidance(lines)

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

    def run_benchmark(self, config: SuiteConfig) -> BenchmarkResult:
        self._config_registry[config.name] = config
        resolved = config.resolved()
        run_name = self._prefixed_name(config.name)
        run_dir = self.output_dir / run_name.replace(" ", "_").replace("/", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        metrics_path = run_dir / "metrics.jsonl"

        cmd = config.train_args(
            output_root=self.output_dir,
            trainer_steps=self.trainer_steps,
            use_wandb=self.use_wandb,
            run_prefix=self.run_prefix,
        )

        print(f"\n{'=' * 60}")
        print(f"Running: {config.name}")
        print(f"Description: {config.description}")
        print(
            "Resolved: "
            f"group={config.comparison_group} seed={resolved['seed']} "
            f"sent={resolved['sent_enabled']} "
            f"lr={resolved['learning_rate']} grad_accum={resolved['gradient_accumulation_steps']} "
            f"group_size={resolved['group_size']} entropy_mask={resolved['entropy_mask']} "
            f"len_penalty={resolved['length_penalty_coef']} "
            f"mask_truncated={resolved['mask_truncated_completions']}"
        )
        print(
            "Effective samples/update: "
            f"{resolved['batch_size'] * resolved['group_size'] * resolved['gradient_accumulation_steps']}"
        )
        print(f"{'=' * 60}")
        note_cleanup_moved()
        print(f"Python: {self.python_exe}")
        print(f"Command: {' '.join(cmd[:6])} ...")
        print(f"Log: {log_path}")

        start_time = time.time()
        success = False
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = run_with_conda(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    text=True,
                )
                process.wait()
                success = process.returncode == 0
        except Exception as exc:
            print(f"[ERROR] {exc}")

        duration = time.time() - start_time
        metrics = self._parse_log(log_path)
        metrics_jsonl = self._parse_metrics_jsonl(metrics_path)
        step_times = metrics.get("step_times", [])
        tokens_per_sec = metrics.get("tokens_per_sec", [])
        losses = metrics.get("losses", [])
        rewards = metrics.get("rewards", [])
        vram_samples = metrics.get("vram_samples", [])
        step_values = metrics_jsonl.get("steps", [])
        reward_avg_values = metrics_jsonl.get("reward_avg_values", [])
        reward_avg_steps = metrics_jsonl.get("reward_avg_steps", [])
        reward_std_values = metrics_jsonl.get("reward_std_values", [])
        reward_min_values = metrics_jsonl.get("reward_min_values", [])
        reward_max_values = metrics_jsonl.get("reward_max_values", [])
        avg_response_length_values = metrics_jsonl.get("avg_response_length_values", [])
        avg_response_length_steps = metrics_jsonl.get("avg_response_length_steps", [])
        entropy_masked_ratio_values = metrics_jsonl.get("entropy_masked_ratio_values", [])
        truncated_ratio_values = metrics_jsonl.get("truncated_completions_ratio_values", [])
        actual_truncated_ratio_values = metrics_jsonl.get(
            "actual_truncated_completions_ratio_values", []
        )
        actual_truncated_ratio_steps = metrics_jsonl.get(
            "actual_truncated_completions_ratio_steps", []
        )
        truncated_masked_out_ratio_values = metrics_jsonl.get(
            "truncated_completions_masked_out_of_loss_ratio_values", []
        )
        truncation_masking_active_values = metrics_jsonl.get(
            "truncation_masking_active_values", []
        )
        positive_advantages_ratio_values = metrics_jsonl.get(
            "positive_advantages_ratio_values", []
        )
        oom_backoff_count_values = metrics_jsonl.get("oom_backoff_count_values", [])
        optimizer_step_values = metrics_jsonl.get("optimizer_step_values", [])
        optimizer_step_steps = metrics_jsonl.get("optimizer_step_steps", [])
        periodic_benchmark_accuracy_values = metrics_jsonl.get(
            "periodic_benchmark_accuracy_values", []
        )
        periodic_benchmark_accuracy_steps = metrics_jsonl.get(
            "periodic_benchmark_accuracy_steps", []
        )

        reward_steps = reward_avg_steps or list(range(1, len(reward_avg_values) + 1))
        response_length_steps = avg_response_length_steps or list(
            range(1, len(avg_response_length_values) + 1)
        )
        truncation_steps = actual_truncated_ratio_steps or list(
            range(1, len(actual_truncated_ratio_values) + 1)
        )

        reward_peak, reward_peak_step, _ = self._peak_with_step(
            reward_avg_values, reward_steps
        )
        response_length_peak, response_length_peak_step, _ = self._peak_with_step(
            avg_response_length_values, response_length_steps
        )
        actual_truncation_peak, actual_truncation_peak_step, _ = self._peak_with_step(
            actual_truncated_ratio_values, truncation_steps
        )
        benchmark_peak, benchmark_peak_step, _ = self._peak_with_step(
            periodic_benchmark_accuracy_values,
            periodic_benchmark_accuracy_steps,
        )
        response_length_by_step = dict(
            zip(response_length_steps, avg_response_length_values, strict=False)
        )
        response_length_at_reward_peak = (
            response_length_by_step.get(reward_peak_step)
            if reward_peak_step is not None
            else None
        )
        reward_final = (
            reward_avg_values[-1] if reward_avg_values else (rewards[-1] if rewards else None)
        )
        avg_response_length_final = (
            avg_response_length_values[-1] if avg_response_length_values else None
        )
        optimizer_step_final = (
            int(optimizer_step_values[-1]) if optimizer_step_values else None
        )
        steps_completed = metrics.get("steps_completed", 0) or max(
            step_values
            or reward_avg_steps
            or avg_response_length_steps
            or optimizer_step_steps
            or periodic_benchmark_accuracy_steps
            or [0]
        )

        result = BenchmarkResult(
            config_name=config.name,
            comparison_group=config.comparison_group,
            seed=int(resolved["seed"]),
            success=success,
            duration_s=duration,
            steps_completed=steps_completed,
            step_time_avg_s=self._series_mean(step_times),
            step_time_min_s=min(step_times) if step_times else None,
            step_time_max_s=max(step_times) if step_times else None,
            tokens_per_sec=self._series_mean(tokens_per_sec),
            tokens_per_sec_min=min(tokens_per_sec) if tokens_per_sec else None,
            tokens_per_sec_max=max(tokens_per_sec) if tokens_per_sec else None,
            vram_peak_gb=max(vram_samples) if vram_samples else None,
            vram_avg_gb=self._series_mean(vram_samples),
            loss_avg=self._series_mean(losses),
            loss_final=losses[-1] if losses else None,
            reward_avg=self._series_mean(reward_avg_values)
            if reward_avg_values
            else self._series_mean(rewards),
            reward_final=reward_final,
            reward_peak=reward_peak,
            reward_peak_step=reward_peak_step,
            reward_drop_from_peak_to_final=(
                reward_peak - reward_final
                if reward_peak is not None and reward_final is not None
                else None
            ),
            reward_std_avg=self._series_mean(reward_std_values),
            reward_std_final=reward_std_values[-1] if reward_std_values else None,
            reward_min_avg=self._series_mean(reward_min_values),
            reward_max_avg=self._series_mean(reward_max_values),
            avg_response_length_avg=self._series_mean(avg_response_length_values),
            avg_response_length_final=avg_response_length_final,
            response_length_peak=response_length_peak,
            response_length_peak_step=response_length_peak_step,
            response_length_at_reward_peak=response_length_at_reward_peak,
            response_length_drop_from_peak_to_final=(
                response_length_peak - avg_response_length_final
                if response_length_peak is not None
                and avg_response_length_final is not None
                else None
            ),
            response_budget_usage_avg=(
                self._series_mean(avg_response_length_values) / resolved["max_response_length"]
            )
            if avg_response_length_values and resolved["max_response_length"] > 0
            else None,
            response_budget_usage_final=(
                avg_response_length_final / resolved["max_response_length"]
            )
            if avg_response_length_final is not None and resolved["max_response_length"] > 0
            else None,
            entropy_masked_ratio_avg=self._series_mean(entropy_masked_ratio_values),
            truncated_completions_ratio_avg=self._series_mean(truncated_ratio_values),
            actual_truncated_completions_ratio_avg=self._series_mean(
                actual_truncated_ratio_values
            ),
            actual_truncated_completions_ratio_final=(
                actual_truncated_ratio_values[-1] if actual_truncated_ratio_values else None
            ),
            actual_truncated_completions_ratio_peak=actual_truncation_peak,
            actual_truncated_completions_ratio_peak_step=actual_truncation_peak_step,
            truncated_completions_masked_out_of_loss_ratio_avg=self._series_mean(
                truncated_masked_out_ratio_values
            ),
            truncated_completions_masked_out_of_loss_ratio_final=(
                truncated_masked_out_ratio_values[-1]
                if truncated_masked_out_ratio_values
                else None
            ),
            truncation_masking_active_final=(
                truncation_masking_active_values[-1]
                if truncation_masking_active_values
                else None
            ),
            positive_advantages_ratio_avg=self._series_mean(
                positive_advantages_ratio_values
            ),
            oom_backoff_count_final=(
                oom_backoff_count_values[-1] if oom_backoff_count_values else None
            ),
            optimizer_step_final=optimizer_step_final,
            optimizer_steps_completed=optimizer_step_final,
            optimizer_step_fraction_of_trainer_steps=(
                optimizer_step_final / steps_completed
                if optimizer_step_final is not None and steps_completed > 0
                else None
            ),
            trainer_steps_per_optimizer_step=(
                steps_completed / optimizer_step_final
                if optimizer_step_final is not None and optimizer_step_final > 0
                else None
            ),
            in_training_heldout_benchmark_accuracy_values=periodic_benchmark_accuracy_values,
            in_training_heldout_benchmark_accuracy_steps=periodic_benchmark_accuracy_steps,
            in_training_heldout_benchmark_accuracy_final=(
                periodic_benchmark_accuracy_values[-1]
                if periodic_benchmark_accuracy_values
                else None
            ),
            in_training_heldout_benchmark_accuracy_peak=benchmark_peak,
            in_training_heldout_benchmark_accuracy_peak_step=benchmark_peak_step,
            final_checkpoint_heldout_benchmark_accuracy=metrics_jsonl.get(
                "final_checkpoint_benchmark_accuracy"
            ),
            final_checkpoint_heldout_benchmark_step=metrics_jsonl.get(
                "final_checkpoint_benchmark_step"
            ),
            final_checkpoint_heldout_benchmark_format_compliance=metrics_jsonl.get(
                "final_checkpoint_benchmark_format_compliance"
            ),
            final_checkpoint_heldout_benchmark_avg_len=metrics_jsonl.get(
                "final_checkpoint_benchmark_avg_len"
            ),
            error_message=metrics.get("error"),
            oom_events=metrics.get("oom_events", 0),
            log_path=str(log_path),
            metrics_path=str(metrics_path),
        )

        print(f"\nResults for {config.name}:")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_s:.1f}s")
        print(f"  Steps: {result.steps_completed}")
        if result.step_time_avg_s is not None:
            print(f"  Step time: {result.step_time_avg_s:.2f}s")
        if result.tokens_per_sec is not None:
            print(f"  Throughput: {result.tokens_per_sec:.2f} tok/s")
        if result.vram_peak_gb is not None:
            print(f"  VRAM peak: {result.vram_peak_gb:.2f} GB")
        if result.in_training_heldout_benchmark_accuracy_values:
            print(
                "  In-training benchmark acc: "
                + self._format_series_with_steps(
                    result.in_training_heldout_benchmark_accuracy_values,
                    result.in_training_heldout_benchmark_accuracy_steps,
                )
            )
        if result.final_checkpoint_heldout_benchmark_accuracy is not None:
            print(
                "  Final checkpoint benchmark acc: "
                f"{result.final_checkpoint_heldout_benchmark_accuracy:.4f}"
            )
        if result.reward_final is not None:
            if result.reward_std_final is not None:
                print(
                    f"  Reward: {result.reward_final:.4f} (std: {result.reward_std_final:.4f})"
                )
            else:
                print(f"  Reward: {result.reward_final:.4f}")
        if result.reward_peak is not None and result.reward_peak_step is not None:
            print(
                f"  Peak reward: {result.reward_peak:.4f} at step {result.reward_peak_step}"
            )
        if result.avg_response_length_final is not None:
            usage = (
                f", budget usage: {result.response_budget_usage_final:.3f}"
                if result.response_budget_usage_final is not None
                else ""
            )
            print(f"  Avg response length: {result.avg_response_length_final:.1f}{usage}")
        if result.actual_truncated_completions_ratio_avg is not None:
            print(
                "  Actual truncated completions ratio avg: "
                f"{result.actual_truncated_completions_ratio_avg:.4f}"
            )
        if result.optimizer_step_final is not None:
            print(f"  Optimizer steps completed: {result.optimizer_step_final}")
        if result.positive_advantages_ratio_avg is not None:
            print(
                f"  Positive advantages ratio avg: {result.positive_advantages_ratio_avg:.4f}"
            )
        if result.oom_backoff_count_final is not None:
            print(f"  OOM backoff count final: {result.oom_backoff_count_final:.0f}")
        if self._is_cap_drift_failure(result):
            print(
                f"  Cap-drift failure signal: avg response length drifted toward {BASELINE.max_response_length}"
            )
        if result.error_message:
            print(f"  Error: {result.error_message}")

        return result

    def run_all(self, configs: list[SuiteConfig]) -> int:
        if not configs:
            print("[INFO] No runs selected.")
            return 0

        est_time_min = self._estimate_total_minutes(configs)
        print(f"\n{'=' * 60}")
        print(self.suite_name)
        print(f"{'=' * 60}")
        print(f"Total configs: {len(configs)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Plan path: {self.plan_path}")
        print(f"Estimated time: ~{(est_time_min / 60):.1f} hours ({est_time_min:.0f} minutes)")
        print(f"Python executable: {self.python_exe}")
        print(f"{'=' * 60}")
        print("[INFO] Auto-starting in 3 seconds...")
        time.sleep(3)

        start_time = time.time()
        failures = 0
        for index, config in enumerate(configs, 1):
            print(f"\n[{index}/{len(configs)}] ", end="")
            result = self.run_benchmark(config)
            self.results.append(result)
            if not result.success:
                failures += 1

            self._save_results()

            elapsed = (time.time() - start_time) / 60.0
            remaining = max(est_time_min - elapsed, 0.0) if index < len(configs) else 0.0
            print(f"\n[PROGRESS] {index}/{len(configs)} complete")
            print(f"[PROGRESS] Elapsed: {elapsed:.1f} min, Est. remaining: {remaining:.1f} min")

        total_elapsed = (time.time() - start_time) / 60.0
        print(f"\n{'=' * 60}")
        print("All CDIA endurance runs complete!")
        print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed / 60:.1f} hours)")
        print(f"{'=' * 60}")

        self._generate_report()
        return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDIA endurance planner/runner")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAINER_STEPS,
        help="Trainer steps per config for the endurance suite. Fixed default: 240.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for plan files and optional runs.",
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=None,
        help="Optional path for the generated JSON plan.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run only the named config(s). Repeat the flag or pass a comma-separated list.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB while running the suite.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the generated plan as JSON and exit without running.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="cdia",
        help="Prefix for per-run output directories and WandB run names.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train.py not found at {TRAIN_SCRIPT}")

    configs = build_suite()
    selected_runs = _parse_selected_runs(args.run)
    configs = filter_suite(configs, selected_runs)
    normalized_prefix = args.run_prefix.strip().strip("_-")
    default_plan_name = (
        f"{normalized_prefix}_endurance_suite_plan.json"
        if normalized_prefix
        else "cdia_endurance_suite_plan.json"
    )
    plan_path = args.plan_path or (args.output_root / default_plan_name)
    write_plan(configs, plan_path, trainer_steps=args.steps)

    if args.json:
        print(plan_path.read_text(encoding="utf-8"))
        return 0

    print_summary(configs, trainer_steps=args.steps)
    print(f"Plan written to: {plan_path}")

    suite = CdiaEnduranceSuite(
        output_dir=args.output_root,
        trainer_steps=args.steps,
        use_wandb=args.wandb,
        run_prefix=args.run_prefix,
        plan_path=plan_path,
    )
    failures = suite.run_all(configs)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
