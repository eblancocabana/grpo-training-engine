#!/usr/bin/env python3
"""
CDIA quick-screen benchmark suite for the PFG.

Long-running experiments to keep separate from this quick suite:
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
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarks" / "output" / "cdia"
DEFAULT_TRAINER_STEPS = 120


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
            "--no-checkpoints",
            "--no-initial-benchmark",
            "--log-metrics-jsonl",
        ]
        if resolved["entropy_mask"]:
            args.append("--use-entropy-mask")
        else:
            args.extend(["--use-entropy-mask", "--no-mask-truncated"])
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
                    run_name,
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
            description="User-requested quick-screen baseline",
            overrides={},
            notes=[
                "Default quick screen uses trainer steps, not optimizer updates.",
                "SENT/default trainer behavior is preserved because the suite does not override it.",
            ],
        ),
        SuiteConfig(
            name="lr_5e-5",
            family="learning_rate",
            description="Lower LR",
            overrides={"learning_rate": 5e-5},
        ),
        SuiteConfig(
            name="lr_3e-4",
            family="learning_rate",
            description="Higher LR",
            overrides={"learning_rate": 3e-4},
        ),
        _batch_cfg(16),
        _batch_cfg(32),
        _batch_cfg(64),
        SuiteConfig(
            name="group_2",
            family="group_size",
            description="Smaller GRPO group size",
            overrides={"group_size": 2},
            notes=["Changes intra-group comparison quality and memory pressure."],
        ),
        SuiteConfig(
            name="group_8",
            family="group_size",
            description="Larger GRPO group size",
            overrides={"group_size": 8},
            notes=["Changes intra-group comparison quality and memory pressure."],
        ),
        SuiteConfig(
            name="entropy_mask_off",
            family="entropy_mask",
            description="Disable entropy mask",
            overrides={"entropy_mask": False},
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
            name="clip_eps_01",
            family="clip_epsilon",
            description="Tighter lower clip bound",
            overrides={"clip_epsilon": 0.1},
        ),
        SuiteConfig(
            name="clip_eps_03",
            family="clip_epsilon",
            description="Looser lower clip bound",
            overrides={"clip_epsilon": 0.3},
        ),
        SuiteConfig(
            name="eps_high_02",
            family="epsilon_high",
            description="Tighter upper clip bound",
            overrides={"epsilon_high": 0.2},
        ),
        SuiteConfig(
            name="eps_high_05",
            family="epsilon_high",
            description="Looser upper clip bound",
            overrides={"epsilon_high": 0.5},
        ),
        SuiteConfig(
            name="delta_12",
            family="delta",
            description="Stricter hard ratio cap",
            overrides={"delta": 1.2},
        ),
        SuiteConfig(
            name="delta_20",
            family="delta",
            description="Looser hard ratio cap",
            overrides={"delta": 2.0},
        ),
        SuiteConfig(
            name="length_penalty_0",
            family="length_penalty",
            description="Disable response-length penalty",
            overrides={"length_penalty_coef": 0.0},
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
        ),
        SuiteConfig(
            name="temp_03",
            family="temperature",
            description="Lower sampling temperature",
            overrides={"temperature": 0.3},
        ),
        SuiteConfig(
            name="temp_10",
            family="temperature",
            description="Higher sampling temperature",
            overrides={"temperature": 1.0},
        ),
        SuiteConfig(
            name="top_p_07",
            family="top_p",
            description="Lower top-p nucleus threshold",
            overrides={"top_p": 0.7},
        ),
        SuiteConfig(
            name="top_p_095",
            family="top_p",
            description="Higher top-p nucleus threshold",
            overrides={"top_p": 0.95},
        ),
        SuiteConfig(
            name="resp_256",
            family="response_length",
            description="Shorter response budget",
            overrides={"max_response_length": 256},
        ),
        SuiteConfig(
            name="resp_512",
            family="response_length",
            description="Shorter response budget",
            overrides={"max_response_length": 512},
        ),
        SuiteConfig(
            name="prompt_64",
            family="max_prompt_length",
            description="Shorter prompt budget",
            overrides={"max_prompt_length": 64},
        ),
        SuiteConfig(
            name="prompt_256",
            family="max_prompt_length",
            description="Longer prompt budget",
            overrides={"max_prompt_length": 256},
        ),
    ]


def print_summary(configs: list[SuiteConfig], trainer_steps: int) -> None:
    print(f"CDIA quick screen: {trainer_steps} trainer steps per config")
    print(f"Baseline effective samples/update: {BASELINE.effective_samples_per_update}")
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
                f"clip={resolved['clip_epsilon']} eps_high={resolved['epsilon_high']} delta={resolved['delta']}"
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


def run_suite(
    configs: list[SuiteConfig],
    output_root: Path,
    trainer_steps: int,
    use_wandb: bool,
    run_prefix: str,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    runnable = [cfg for cfg in configs if cfg.runnable]
    failures = 0
    for cfg in runnable:
        cmd = cfg.train_args(
            output_root=output_root,
            trainer_steps=trainer_steps,
            use_wandb=use_wandb,
            run_prefix=run_prefix,
        )
        print(f"[RUN] {cfg.name}")
        print("      " + " ".join(cmd))
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            failures += 1
            print(f"[FAIL] {cfg.name} exit_code={completed.returncode}")
        else:
            print(f"[OK] {cfg.name}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDIA quick-screen suite planner/runner")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAINER_STEPS,
        help="Trainer steps per config for the quick screen. Recommended: 100-150.",
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
        action="store_true",
        help="Execute runnable configs sequentially. Default behavior only prints/writes the plan.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB when using --run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the generated plan as JSON instead of a readable summary.",
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
    normalized_prefix = args.run_prefix.strip().strip("_-")
    default_plan_name = (
        f"{normalized_prefix}_suite_plan.json" if normalized_prefix else "cdia_suite_plan.json"
    )
    plan_path = args.plan_path or (args.output_root / default_plan_name)
    write_plan(configs, plan_path, trainer_steps=args.steps)

    if args.json:
        print(plan_path.read_text(encoding="utf-8"))
    else:
        print_summary(configs, trainer_steps=args.steps)
        print(f"Plan written to: {plan_path}")

    if not args.run:
        return 0

    failures = run_suite(
        configs=configs,
        output_root=args.output_root,
        trainer_steps=args.steps,
        use_wandb=args.wandb,
        run_prefix=args.run_prefix,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
