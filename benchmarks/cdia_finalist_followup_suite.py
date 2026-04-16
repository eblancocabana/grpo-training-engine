#!/usr/bin/env python3
"""
CDIA finalist follow-up benchmark suite for the PFG.

This suite is a finalist-selection follow-up derived from the reduced
`cdia_suite.py`. It preserves the same runner style and output layout, but it
replaces the broad shortlist screen with a tighter set of boundary and
interaction checks.

This is still not a final-claims benchmark. Its role is to answer:
- whether `length_penalty_0005` is genuinely better than no penalty
- whether a lower boundary like `0.00025` improves on `0.0005`
- whether `batch_eff_32` and `eps_high_02` remain useful under the current best
  length-penalty regime
- whether those same interactions help when length penalty is removed
- whether `mask_truncated_completions=False` still helps under the current best
  length-penalty regime

Interpretation improvements over the original `cdia_suite`:
- actual truncation incidence is tracked even when truncation masking is off
- the report surfaces whether truncations were masked out of loss or not
- optimizer-step counts are shown explicitly so fixed-trainer-step runs remain
  comparable when grad accumulation changes
- peak-vs-final reward and response-length behavior is reported directly
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarks" / "output" / "cdia_finalist_followup"
DEFAULT_TRAINER_STEPS = 120


def note_cleanup_moved() -> None:
    """Match ING INF messaging about runtime cleanup ownership."""
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
    max_prompt_length: int = 128
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

    @property
    def effective_samples_per_update(self) -> int:
        return self.batch_size * self.group_size * self.gradient_accumulation_steps


BASELINE = BaselineConfig()


@dataclass
class BenchmarkResult:
    """Runtime summary for a single finalist follow-up CDIA config."""

    config_name: str
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
    error_message: str | None = None
    oom_events: int = 0
    log_path: str | None = None
    metrics_path: str | None = None


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    family: str
    description: str
    overrides: dict[str, Any]
    runnable: bool = True
    blocked_reason: str | None = None
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
        if not self.runnable:
            raise ValueError(f"{self.name} is blocked: {self.blocked_reason}")

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
                    "finalist-followup",
                    self.family,
                ]
            )
        else:
            args.append("--no-wandb")
        return args


def build_suite() -> list[SuiteConfig]:
    return [
        SuiteConfig(
            name="baseline",
            family="baseline",
            description="Reference config inherited unchanged from cdia_suite",
            overrides={},
            notes=[
                "Default finalist screen still uses trainer steps, not matched optimizer updates.",
                "Compare optimizer_step_final before making stability claims.",
                "The suite disables SENT with --no-sent to keep finalist checks comparable and fast.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0",
            family="length_penalty_boundary",
            description="Remove the length penalty entirely",
            overrides={"length_penalty_coef": 0.0},
        ),
        SuiteConfig(
            name="length_penalty_00025",
            family="length_penalty_boundary",
            description="Lower boundary check below the current best penalty",
            overrides={"length_penalty_coef": 0.00025},
            notes=[
                "Tests whether 0.0005 is too strong and a milder penalty improves stability.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0005",
            family="length_penalty_boundary",
            description="Current best mild length-penalty regime from the reduced CDIA screen",
            overrides={"length_penalty_coef": 0.0005},
        ),
        SuiteConfig(
            name="length_penalty_0005_batch_eff_32",
            family="interaction_batch_size",
            description="Check whether batch_eff_32 stays helpful under the current best penalty",
            overrides={
                "length_penalty_coef": 0.0005,
                "gradient_accumulation_steps": 8,
            },
            notes=[
                "Uses grad_accum=8, so trainer steps and optimizer updates are not the same as baseline.",
                "Interpret with optimizer_step_final, not trainer steps alone.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0005_eps_high_02",
            family="interaction_epsilon_high",
            description="Check whether tighter upper clipping stays helpful under the current best penalty",
            overrides={
                "length_penalty_coef": 0.0005,
                "epsilon_high": 0.2,
            },
        ),
        SuiteConfig(
            name="length_penalty_0_batch_eff_32",
            family="interaction_batch_size",
            description="Check whether batch_eff_32 is still helpful when length penalty is removed",
            overrides={
                "length_penalty_coef": 0.0,
                "gradient_accumulation_steps": 8,
            },
            notes=[
                "Tests whether the apparent batch-size win depended on the length-penalty regime.",
                "Interpret with optimizer_step_final, not trainer steps alone.",
            ],
        ),
        SuiteConfig(
            name="length_penalty_0_eps_high_02",
            family="interaction_epsilon_high",
            description="Check whether tighter upper clipping is still helpful with no length penalty",
            overrides={
                "length_penalty_coef": 0.0,
                "epsilon_high": 0.2,
            },
        ),
        SuiteConfig(
            name="length_penalty_0005_mask_truncated_off",
            family="interaction_truncation_mask",
            description="Check whether disabling truncation masking still helps under the current best penalty",
            overrides={
                "length_penalty_coef": 0.0005,
                "mask_truncated_completions": False,
            },
            notes=[
                "Actual truncation incidence is still logged even though truncated samples remain in the loss.",
                "Compare actual_truncated_completions_ratio against truncated_completions_masked_out_of_loss_ratio.",
            ],
        ),
    ]


def filter_suite(configs: list[SuiteConfig], selected_names: list[str] | None = None) -> list[SuiteConfig]:
    """Filter suite configs by explicit config names."""
    if not selected_names:
        return configs
    selected = {name.strip() for name in selected_names if name.strip()}
    if not selected:
        return configs
    return [cfg for cfg in configs if cfg.name in selected]


def print_summary(configs: list[SuiteConfig], trainer_steps: int) -> None:
    print("CDIA finalist follow-up shortlist suite")
    print(f"Trainer steps per config: {trainer_steps}")
    print(f"Configs in finalist suite: {len(configs)}")
    print(f"Baseline effective samples/update: {BASELINE.effective_samples_per_update}")
    print(
        "Focus: lower length-penalty boundary checks, interaction checks with "
        "effective batch and upper clipping, and truncation-mask interaction "
        "under the current best length-penalty regime"
    )
    print(
        "Interpretation guardrails: compare optimizer_step_final, peak-vs-final "
        "reward, response length, and actual truncation incidence before drawing conclusions"
    )
    print()
    for cfg in configs:
        status = "RUNNABLE" if cfg.runnable else "BLOCKED"
        print(f"[{status}] {cfg.name} ({cfg.family})")
        print(f"  {cfg.description}")
        if cfg.runnable:
            resolved = cfg.resolved()
            print(
                "  "
                f"lr={resolved['learning_rate']} grad_accum={resolved['gradient_accumulation_steps']} "
                f"group={resolved['group_size']} max_prompt={resolved['max_prompt_length']} "
                f"max_response={resolved['max_response_length']} entropy_mask={resolved['entropy_mask']} "
                f"entropy_pct={resolved['entropy_percentile']} eps_high={resolved['epsilon_high']} "
                f"len_penalty={resolved['length_penalty_coef']} mask_truncated={resolved['mask_truncated_completions']}"
            )
        if cfg.blocked_reason:
            print(f"  blocked_reason={cfg.blocked_reason}")
        for note in cfg.notes:
            print(f"  note={note}")
        print()


def write_plan(configs: list[SuiteConfig], output_path: Path, trainer_steps: int) -> None:
    payload = {
        "trainer_steps": trainer_steps,
        "baseline": asdict(BASELINE),
        "configs": [
            {
                "name": cfg.name,
                "family": cfg.family,
                "description": cfg.description,
                "overrides": cfg.overrides,
                "resolved": cfg.resolved() if cfg.runnable else None,
                "runnable": cfg.runnable,
                "blocked_reason": cfg.blocked_reason,
                "notes": cfg.notes,
            }
            for cfg in configs
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CdiaBenchmarkSuite:
    """Finalist follow-up CDIA suite with the original runner/reporting flow."""

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
        self.suite_name = "CDIA Finalist Follow-Up Suite"
        self.report_title = "CDIA Finalist Follow-Up Report"

    def _normalize_prefix(self, prefix: Optional[str]) -> str:
        if prefix is None:
            return ""
        return prefix.strip().strip("_-")

    def _prefixed_name(self, name: str) -> str:
        if not self.run_prefix:
            return name
        return f"{self.run_prefix}_{name}"

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
            record["loss"]
            for _, record in sorted(step_records.items())
            if "loss" in record
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

            if entry.get("event") != "train_metrics":
                continue

            step = int(entry.get("step", 0))
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
            "trainer_steps": self.trainer_steps,
            "total_runs": len(self.results),
            "successful_runs": sum(1 for result in self.results if result.success),
            "results": [asdict(result) for result in self.results],
        }
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
            "Derived from benchmarks/cdia_suite.py with the same output flow but finalist-specific config coverage and stronger observability.",
            "",
            "## Summary",
            "",
            "| Config | Status | Steps | Opt Steps | Tok/s | Reward Final | Reward Peak@Step | Drop | Resp Final | Resp Peak@Step | Trunc Avg | Masked-Out Avg |",
            "|--------|--------|-------|-----------|-------|--------------|------------------|------|------------|----------------|-----------|----------------|",
        ]
        for result in self.results:
            status = "OK" if result.success else "FAIL"
            opt_steps = (
                str(result.optimizer_step_final)
                if result.optimizer_step_final is not None
                else "N/A"
            )
            tok_s = f"{result.tokens_per_sec:.2f}" if result.tokens_per_sec else "N/A"
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
            masked_out_avg = (
                f"{result.truncated_completions_masked_out_of_loss_ratio_avg:.4f}"
                if result.truncated_completions_masked_out_of_loss_ratio_avg is not None
                else "N/A"
            )
            lines.append(
                f"| {result.config_name} | {status} | {result.steps_completed} | {opt_steps} | {tok_s} | {reward_final} | {reward_peak} | {reward_drop} | {final_resp_len} | {peak_resp_len} | {actual_trunc_avg} | {masked_out_avg} |"
            )

        lines.extend(["", "## Detailed Results", ""])
        for result in self.results:
            lines.extend(
                [
                    f"### {result.config_name}",
                    "",
                    f"- Success: {result.success}",
                    f"- Duration: {result.duration_s:.1f}s",
                    f"- Steps completed: {result.steps_completed}",
                    f"- Log: {result.log_path or 'N/A'}",
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
            if result.avg_response_length_final is not None:
                lines.append(f"- Final avg response length: {result.avg_response_length_final:.1f}")
            if (
                result.response_length_peak is not None
                and result.response_length_peak_step is not None
            ):
                lines.append(
                    "- Peak avg response length: "
                    f"{result.response_length_peak:.1f} at step {result.response_length_peak_step}"
                )
            if result.response_length_at_reward_peak is not None and result.reward_peak_step is not None:
                lines.append(
                    "- Avg response length at reward peak step "
                    f"{result.reward_peak_step}: {result.response_length_at_reward_peak:.1f}"
                )
            if result.response_length_drop_from_peak_to_final is not None:
                lines.append(
                    "- Response length drop from peak to final: "
                    f"{result.response_length_drop_from_peak_to_final:.1f}"
                )
            if result.response_budget_usage_final is not None:
                lines.append(f"- Final response budget usage: {result.response_budget_usage_final:.4f}")
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
            if result.error_message:
                lines.append(f"- Error: {result.error_message}")
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

    def run_benchmark(self, config: SuiteConfig) -> BenchmarkResult:
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
            f"lr={resolved['learning_rate']} grad_accum={resolved['gradient_accumulation_steps']} "
            f"group={resolved['group_size']} entropy_mask={resolved['entropy_mask']} "
            f"entropy_pct={resolved['entropy_percentile']} eps_high={resolved['epsilon_high']} "
            f"len_penalty={resolved['length_penalty_coef']} mask_truncated={resolved['mask_truncated_completions']}"
        )
        print(f"Effective samples/update: {resolved['batch_size'] * resolved['group_size'] * resolved['gradient_accumulation_steps']}")
        print(f"{'=' * 60}")
        note_cleanup_moved()
        print(f"Python: {self.python_exe}")
        print(f"Command: {' '.join(cmd[:6])} ...")
        print(f"Log: {log_path}")

        start_time = time.time()
        process = None
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
        positive_advantages_ratio_values = metrics_jsonl.get("positive_advantages_ratio_values", [])
        oom_backoff_count_values = metrics_jsonl.get("oom_backoff_count_values", [])
        optimizer_step_values = metrics_jsonl.get("optimizer_step_values", [])
        optimizer_step_steps = metrics_jsonl.get("optimizer_step_steps", [])

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
            step_values or reward_avg_steps or avg_response_length_steps or optimizer_step_steps or [0]
        )

        result = BenchmarkResult(
            config_name=config.name,
            success=success,
            duration_s=duration,
            steps_completed=steps_completed,
            step_time_avg_s=(sum(step_times) / len(step_times)) if step_times else None,
            step_time_min_s=min(step_times) if step_times else None,
            step_time_max_s=max(step_times) if step_times else None,
            tokens_per_sec=(sum(tokens_per_sec) / len(tokens_per_sec)) if tokens_per_sec else None,
            tokens_per_sec_min=min(tokens_per_sec) if tokens_per_sec else None,
            tokens_per_sec_max=max(tokens_per_sec) if tokens_per_sec else None,
            vram_peak_gb=max(vram_samples) if vram_samples else None,
            vram_avg_gb=(sum(vram_samples) / len(vram_samples)) if vram_samples else None,
            loss_avg=(sum(losses) / len(losses)) if losses else None,
            loss_final=losses[-1] if losses else None,
            reward_avg=(sum(reward_avg_values) / len(reward_avg_values))
            if reward_avg_values
            else ((sum(rewards) / len(rewards)) if rewards else None),
            reward_final=reward_final,
            reward_peak=reward_peak,
            reward_peak_step=reward_peak_step,
            reward_drop_from_peak_to_final=(
                reward_peak - reward_final
                if reward_peak is not None and reward_final is not None
                else None
            ),
            reward_std_avg=(sum(reward_std_values) / len(reward_std_values))
            if reward_std_values
            else None,
            reward_std_final=reward_std_values[-1] if reward_std_values else None,
            reward_min_avg=(sum(reward_min_values) / len(reward_min_values))
            if reward_min_values
            else None,
            reward_max_avg=(sum(reward_max_values) / len(reward_max_values))
            if reward_max_values
            else None,
            avg_response_length_avg=(
                sum(avg_response_length_values) / len(avg_response_length_values)
            )
            if avg_response_length_values
            else None,
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
                (sum(avg_response_length_values) / len(avg_response_length_values))
                / resolved["max_response_length"]
            )
            if avg_response_length_values and resolved["max_response_length"] > 0
            else None,
            response_budget_usage_final=(
                avg_response_length_values[-1] / resolved["max_response_length"]
            )
            if avg_response_length_values and resolved["max_response_length"] > 0
            else None,
            entropy_masked_ratio_avg=(
                sum(entropy_masked_ratio_values) / len(entropy_masked_ratio_values)
            )
            if entropy_masked_ratio_values
            else None,
            truncated_completions_ratio_avg=(
                sum(truncated_ratio_values) / len(truncated_ratio_values)
            )
            if truncated_ratio_values
            else None,
            actual_truncated_completions_ratio_avg=(
                sum(actual_truncated_ratio_values) / len(actual_truncated_ratio_values)
            )
            if actual_truncated_ratio_values
            else None,
            actual_truncated_completions_ratio_final=(
                actual_truncated_ratio_values[-1] if actual_truncated_ratio_values else None
            ),
            actual_truncated_completions_ratio_peak=actual_truncation_peak,
            actual_truncated_completions_ratio_peak_step=actual_truncation_peak_step,
            truncated_completions_masked_out_of_loss_ratio_avg=(
                sum(truncated_masked_out_ratio_values)
                / len(truncated_masked_out_ratio_values)
            )
            if truncated_masked_out_ratio_values
            else None,
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
            positive_advantages_ratio_avg=(
                sum(positive_advantages_ratio_values) / len(positive_advantages_ratio_values)
            )
            if positive_advantages_ratio_values
            else None,
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
            print(
                f"  Avg response length: {result.avg_response_length_final:.1f}{usage}"
            )
        if result.entropy_masked_ratio_avg is not None:
            print(f"  Entropy-masked ratio avg: {result.entropy_masked_ratio_avg:.4f}")
        if result.truncated_completions_ratio_avg is not None:
            print(
                f"  Truncated completions ratio avg: {result.truncated_completions_ratio_avg:.4f}"
            )
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
        if result.error_message:
            print(f"  Error: {result.error_message}")

        return result

    def run_all(self, configs: list[SuiteConfig]) -> int:
        runnable = [cfg for cfg in configs if cfg.runnable]
        if not runnable:
            print("[INFO] No runnable configs selected.")
            return 0

        est_time_min = self._estimate_total_minutes(runnable)
        print(f"\n{'=' * 60}")
        print(self.suite_name)
        print(f"{'=' * 60}")
        print(f"Total configs: {len(runnable)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Plan path: {self.plan_path}")
        print(f"Estimated time: ~{(est_time_min / 60):.1f} hours ({est_time_min:.0f} minutes)")
        print(f"Python executable: {self.python_exe}")
        print(f"{'=' * 60}")
        print("[INFO] Auto-starting in 3 seconds...")
        time.sleep(3)

        start_time = time.time()
        failures = 0
        for index, config in enumerate(runnable, 1):
            print(f"\n[{index}/{len(runnable)}] ", end="")
            result = self.run_benchmark(config)
            self.results.append(result)
            if not result.success:
                failures += 1

            self._save_results()

            elapsed = (time.time() - start_time) / 60.0
            remaining = max(est_time_min - elapsed, 0.0) if index < len(runnable) else 0.0
            print(f"\n[PROGRESS] {index}/{len(runnable)} complete")
            print(f"[PROGRESS] Elapsed: {elapsed:.1f} min, Est. remaining: {remaining:.1f} min")

        total_elapsed = (time.time() - start_time) / 60.0
        print(f"\n{'=' * 60}")
        print("All CDIA runs complete!")
        print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed / 60:.1f} hours)")
        print(f"{'=' * 60}")

        self._generate_report()
        return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CDIA finalist follow-up shortlist planner/runner"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAINER_STEPS,
        help="Trainer steps per config for the finalist follow-up suite. Recommended: 100-150.",
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
        "--no-wandb",
        action="store_true",
        help="Disable WandB while running the suite.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the generated plan as JSON and exit without running.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the finalist follow-up suite summary and exit without running.",
    )
    parser.add_argument(
        "--filter-configs",
        type=str,
        help="Comma-separated list of config names to run.",
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
    filter_configs = None
    if args.filter_configs:
        filter_configs = [name.strip() for name in args.filter_configs.split(",") if name.strip()]
    configs = filter_suite(configs, filter_configs)
    normalized_prefix = args.run_prefix.strip().strip("_-")
    default_plan_name = (
        f"{normalized_prefix}_finalist_followup_suite_plan.json"
        if normalized_prefix
        else "cdia_finalist_followup_suite_plan.json"
    )
    plan_path = args.plan_path or (args.output_root / default_plan_name)
    write_plan(configs, plan_path, trainer_steps=args.steps)

    if args.json:
        print(plan_path.read_text(encoding="utf-8"))
        return 0

    if args.list:
        print_summary(configs, trainer_steps=args.steps)
        print(f"Plan written to: {plan_path}")
        return 0

    suite = CdiaBenchmarkSuite(
        output_dir=args.output_root,
        trainer_steps=args.steps,
        use_wandb=not args.no_wandb,
        run_prefix=args.run_prefix,
        plan_path=plan_path,
    )
    print_summary(configs, trainer_steps=args.steps)
    print(f"Plan written to: {plan_path}")
    failures = suite.run_all(configs)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
