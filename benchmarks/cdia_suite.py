#!/usr/bin/env python3
"""
CDIA quick-screen benchmark suite for the PFG.

This suite is a shortlist generator for longer-run CDIA experiments, not a
final-claims benchmark. It focuses on project-specific controls rather than an
exhaustive generic sweep.

Design constraints for this reduced suite:
- Keep the baseline fixed and screen only the most thesis-relevant knobs.
- Response length 512 is intentionally excluded because local benchmark
  analysis already found it too truncation-heavy for this quick screen.
- `group_8`, broad sampling sweeps, and broad prompt-budget sweeps are
  intentionally omitted from this reduced suite.

Long-running experiments to keep separate from this shortlist suite:
- Full SENT curriculum effect: run >=1000 trainer steps across curriculum stages.
- Convergence study: train shortlisted configs to >=1000 trainer steps.
- Multiple epochs on the same data: isolate repeated-exposure effects vs. fresh-step gains.
- Reward distribution analysis: track mean/std/min/max and failure examples over time.
- Seed sensitivity study: rerun shortlisted configs with 3 seeds before thesis conclusions.
- LR x effective-batch interaction study: do not trust one-factor results alone for finalists.
- Stability analysis: compare truncation ratio, entropy-masked ratio, OOM backoff count.
- Current repo default uses no reference-model KL. Stability mainly comes from:
  old-policy ratio clipping, asymmetric upper clip, hard delta cap, truncation masking,
  entropy masking, and length penalty.

Why this suite uses 120 trainer steps by default:
- 20 trainer steps is only 5 optimizer updates at grad_accum=4, which is too short.
- 100-150 trainer steps is enough for a quick screen, not for final thesis claims.
- Final thesis comparisons should still use longer runs and repeated seeds.
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarks" / "output" / "cdia"
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
    """Runtime summary for a single CDIA config."""

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
    reward_std_avg: float | None = None
    reward_std_final: float | None = None
    reward_min_avg: float | None = None
    reward_max_avg: float | None = None
    avg_response_length_avg: float | None = None
    avg_response_length_final: float | None = None
    response_budget_usage_avg: float | None = None
    response_budget_usage_final: float | None = None
    entropy_masked_ratio_avg: float | None = None
    truncated_completions_ratio_avg: float | None = None
    positive_advantages_ratio_avg: float | None = None
    oom_backoff_count_final: float | None = None
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
                    "quick-screen",
                    self.family,
                ]
            )
        else:
            args.append("--no-wandb")
        return args


def _batch_cfg(effective_samples: int) -> SuiteConfig:
    if effective_samples % (BASELINE.group_size * BASELINE.batch_size) != 0:
        raise ValueError(f"effective_samples={effective_samples} is not divisible by baseline samples per step")
    grad_accum = effective_samples // (BASELINE.group_size * BASELINE.batch_size)
    return SuiteConfig(
        name=f"batch_eff_{effective_samples}",
        family="batch_size",
        description=f"Effective samples/update = {effective_samples} via grad_accum={grad_accum}, batch_size=1, group_size=4",
        overrides={"gradient_accumulation_steps": grad_accum},
        notes=[
            "Uses grad_accum rather than prompt batch to avoid 3060 Ti OOM pressure.",
            "This is the recommended batch-size interpretation for this codebase.",
        ],
    )


def build_suite() -> list[SuiteConfig]:
    return [
        SuiteConfig(
            name="baseline",
            family="baseline",
            description="Reduced-suite baseline for shortlist generation",
            overrides={},
            notes=[
                "Default quick screen uses trainer steps, not optimizer updates.",
                "The suite disables SENT with --no-sent to keep shortlist runs comparable and fast.",
            ],
        ),
        SuiteConfig(
            name="lr_5e-5",
            family="learning_rate",
            description="Minimal lower-LR sanity check",
            overrides={"learning_rate": 5e-5},
        ),
        _batch_cfg(32),
        SuiteConfig(
            name="entropy_mask_off",
            family="entropy_mask",
            description="Disable entropy mask",
            overrides={"entropy_mask": False},
            notes=[
                "Implemented with train.py --no-entropy-mask.",
                "This does not disable truncation masking; mask_truncated_completions stays at the baseline value.",
            ],
        ),
        SuiteConfig(
            name="entropy_pct_03",
            family="entropy_percentile",
            description="Keep top 30% highest-entropy tokens",
            overrides={"entropy_mask": True, "entropy_percentile": 0.3},
            notes=["Entropy percentile ablations always force entropy mask on."],
        ),
        SuiteConfig(
            name="entropy_pct_07",
            family="entropy_percentile",
            description="Keep top 70% highest-entropy tokens",
            overrides={"entropy_mask": True, "entropy_percentile": 0.7},
            notes=["Entropy percentile ablations always force entropy mask on."],
        ),
        SuiteConfig(
            name="eps_high_02",
            family="epsilon_high",
            description="Tighter upper clip bound",
            overrides={"epsilon_high": 0.2},
        ),
        SuiteConfig(
            name="length_penalty_0005",
            family="length_penalty",
            description="Milder response-length penalty",
            overrides={"length_penalty_coef": 0.0005},
        ),
        SuiteConfig(
            name="length_penalty_002",
            family="length_penalty",
            description="Stronger response-length penalty",
            overrides={"length_penalty_coef": 0.002},
        ),
        SuiteConfig(
            name="mask_truncated_off",
            family="mask_truncated_completions",
            description="Let truncated completions contribute to loss",
            overrides={"mask_truncated_completions": False},
            notes=[
                "Keeps entropy masking at the baseline setting.",
                "Isolates truncation-mask behavior from entropy-mask behavior.",
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
    print("CDIA quick-screen shortlist suite")
    print(f"Trainer steps per config: {trainer_steps}")
    print(f"Configs in reduced suite: {len(configs)}")
    print(f"Baseline effective samples/update: {BASELINE.effective_samples_per_update}")
    print("Focus: entropy masking, epsilon_high, length penalty, truncation masking, minimal LR/effective-batch checks")
    print("Omitted intentionally: group_8, generic sampling sweeps, generic clip/delta sweeps, prompt-budget sweeps, response_length=512")
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
    """Reduced CDIA shortlist suite with ING INF-style runtime behavior."""

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
        self.suite_name = "CDIA Quick-Screen Suite"
        self.report_title = "CDIA Quick-Screen Report"

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

    def _parse_metrics_jsonl(self, metrics_path: Path) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "reward_avg_values": [],
            "reward_std_values": [],
            "reward_min_values": [],
            "reward_max_values": [],
            "avg_response_length_values": [],
            "entropy_masked_ratio_values": [],
            "truncated_completions_ratio_values": [],
            "positive_advantages_ratio_values": [],
            "oom_backoff_count_values": [],
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

            if "train/avg_reward" in entry:
                stats["reward_avg_values"].append(float(entry["train/avg_reward"]))
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
            if "train/entropy_masked_ratio" in entry:
                stats["entropy_masked_ratio_values"].append(
                    float(entry["train/entropy_masked_ratio"])
                )
            if "train/truncated_completions_ratio" in entry:
                stats["truncated_completions_ratio_values"].append(
                    float(entry["train/truncated_completions_ratio"])
                )
            if "train/positive_advantages_ratio" in entry:
                stats["positive_advantages_ratio_values"].append(
                    float(entry["train/positive_advantages_ratio"])
                )
            if "train/oom_backoff_count" in entry:
                stats["oom_backoff_count_values"].append(
                    float(entry["train/oom_backoff_count"])
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
            "",
            "## Summary",
            "",
            "| Config | Status | Steps | Step Time (s) | Tok/s | VRAM (GB) | Reward | Reward Std | Avg Resp Len | Trunc Ratio |",
            "|--------|--------|-------|---------------|-------|-----------|--------|------------|--------------|-------------|",
        ]
        for result in self.results:
            status = "OK" if result.success else "FAIL"
            step_time = f"{result.step_time_avg_s:.2f}" if result.step_time_avg_s else "N/A"
            tok_s = f"{result.tokens_per_sec:.2f}" if result.tokens_per_sec else "N/A"
            vram = f"{result.vram_peak_gb:.2f}" if result.vram_peak_gb else "N/A"
            reward = f"{result.reward_final:.4f}" if result.reward_final is not None else "N/A"
            reward_std = (
                f"{result.reward_std_final:.4f}" if result.reward_std_final is not None else "N/A"
            )
            avg_resp_len = (
                f"{result.avg_response_length_final:.1f}"
                if result.avg_response_length_final is not None
                else "N/A"
            )
            trunc_ratio = (
                f"{result.truncated_completions_ratio_avg:.4f}"
                if result.truncated_completions_ratio_avg is not None
                else "N/A"
            )
            lines.append(
                f"| {result.config_name} | {status} | {result.steps_completed} | {step_time} | {tok_s} | {vram} | {reward} | {reward_std} | {avg_resp_len} | {trunc_ratio} |"
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
            if result.loss_final is not None:
                lines.append(f"- Final loss: {result.loss_final:.4f}")
            if result.reward_final is not None:
                lines.append(f"- Final reward: {result.reward_final:.4f}")
            if result.reward_std_final is not None:
                lines.append(f"- Final reward std: {result.reward_std_final:.4f}")
            if result.reward_min_avg is not None and result.reward_max_avg is not None:
                lines.append(
                    f"- Reward range avg: [{result.reward_min_avg:.4f}, {result.reward_max_avg:.4f}]"
                )
            if result.avg_response_length_final is not None:
                lines.append(f"- Final avg response length: {result.avg_response_length_final:.1f}")
            if result.response_budget_usage_final is not None:
                lines.append(f"- Final response budget usage: {result.response_budget_usage_final:.4f}")
            if result.entropy_masked_ratio_avg is not None:
                lines.append(
                    f"- Entropy-masked ratio avg: {result.entropy_masked_ratio_avg:.4f}"
                )
            if result.truncated_completions_ratio_avg is not None:
                lines.append(
                    f"- Truncated completions ratio avg: {result.truncated_completions_ratio_avg:.4f}"
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
        reward_avg_values = metrics_jsonl.get("reward_avg_values", [])
        reward_std_values = metrics_jsonl.get("reward_std_values", [])
        reward_min_values = metrics_jsonl.get("reward_min_values", [])
        reward_max_values = metrics_jsonl.get("reward_max_values", [])
        avg_response_length_values = metrics_jsonl.get("avg_response_length_values", [])
        entropy_masked_ratio_values = metrics_jsonl.get("entropy_masked_ratio_values", [])
        truncated_ratio_values = metrics_jsonl.get("truncated_completions_ratio_values", [])
        positive_advantages_ratio_values = metrics_jsonl.get("positive_advantages_ratio_values", [])
        oom_backoff_count_values = metrics_jsonl.get("oom_backoff_count_values", [])

        result = BenchmarkResult(
            config_name=config.name,
            success=success,
            duration_s=duration,
            steps_completed=metrics.get("steps_completed", 0),
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
            reward_final=reward_avg_values[-1] if reward_avg_values else (rewards[-1] if rewards else None),
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
            avg_response_length_final=(
                avg_response_length_values[-1] if avg_response_length_values else None
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
            positive_advantages_ratio_avg=(
                sum(positive_advantages_ratio_values) / len(positive_advantages_ratio_values)
            )
            if positive_advantages_ratio_values
            else None,
            oom_backoff_count_final=(
                oom_backoff_count_values[-1] if oom_backoff_count_values else None
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
        description="CDIA reduced quick-screen shortlist planner/runner"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAINER_STEPS,
        help="Trainer steps per config for the reduced shortlist screen. Recommended: 100-150.",
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
        help="Print the reduced suite summary and exit without running.",
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
        f"{normalized_prefix}_suite_plan.json" if normalized_prefix else "cdia_suite_plan.json"
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
