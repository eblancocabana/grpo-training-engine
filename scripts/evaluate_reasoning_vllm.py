#!/usr/bin/env python
"""Evaluate reasoning checkpoints with vLLM across benchmark suites."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reasoning_eval.datasets import DATASET_REGISTRY, resolve_dataset_names
from src.reasoning_eval.io import prepare_output_dir
from src.reasoning_eval.models import load_models_config
from src.reasoning_eval.runner import EvaluationRunner


def _default_max_tokens(tier: str) -> int:
    if tier == "minimal":
        return 8192
    if tier == "strong":
        return 16384
    return 32768


def _default_protocol(tier: str) -> str:
    return "deterministic" if tier == "minimal" else "both"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SOTA-style vLLM evaluation for reasoning checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-config", required=True, help="Path to models YAML/JSON config")
    parser.add_argument("--models", default=None, help="Optional comma-separated subset of model keys")
    parser.add_argument("--datasets", required=True, help="'all' or comma-separated dataset names")
    parser.add_argument("--tier", choices=("minimal", "strong", "maximal"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("deterministic", "sampled", "both"), default=None)
    parser.add_argument("--prompt-style", choices=("reasoning", "training"), default="reasoning")
    parser.add_argument("--max-new-tokens", type=int, choices=(768, 8192, 16384, 32768), default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Number of prompts submitted to one blocking vLLM generate call. "
            "Defaults to min(max_num_seqs, 32) for frequent global progress updates."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-generator", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.n_samples is not None and args.n_samples < 1:
        parser.error("--n-samples must be >= 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.top_p is not None and not (0 < args.top_p <= 1):
        parser.error("--top-p must be in (0, 1]")
    args.protocol = args.protocol or _default_protocol(args.tier)
    args.max_new_tokens = args.max_new_tokens or _default_max_tokens(args.tier)
    if args.batch_size is None:
        args.batch_size = max(1, min(int(args.max_num_seqs), 32))
    return args


def _readme_text(args: argparse.Namespace, model_keys: list[str], dataset_names: list[str]) -> str:
    command = " ".join(sys.argv)
    local_envs = [
        (name, DATASET_REGISTRY[name].local_path_env)
        for name in dataset_names
        if DATASET_REGISTRY[name].local_path_env
    ]
    env_lines = "\n".join(f"- `{env}`: optional local JSON/JSONL/CSV override for `{name}`" for name, env in local_envs)
    return f"""# Reasoning Evaluation Run

Command:

```bash
{command}
```

Models: {", ".join(model_keys)}

Datasets: {", ".join(dataset_names)}

The best-checkpoint labels are accepted from `models.yaml` only when marked as
`selection_source: in_training_validation`; this script does not select best
checkpoints from benchmark results.

Optional local dataset overrides:

{env_lines or "- None for this run."}
"""


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    model_keys: list[str],
    dataset_names: list[str],
) -> None:
    payload: dict[str, Any] = {
        "args": vars(args),
        "model_keys": model_keys,
        "dataset_names": dataset_names,
        "repository": str(REPO_ROOT),
        "environment": {
            "python": sys.version,
            "cwd": os.getcwd(),
        },
    }
    run_config_path = output_dir / "run_config.json"
    readme_path = output_dir / "README.md"
    if args.resume and run_config_path.exists() and readme_path.exists():
        return
    run_config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    readme_path.write_text(_readme_text(args, model_keys, dataset_names), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_subset = [item.strip() for item in args.models.split(",")] if args.models else None
    models = load_models_config(args.models_config, only=model_subset)
    dataset_names = resolve_dataset_names(args.datasets, args.tier)
    output_dir = prepare_output_dir(args.output_dir, resume=args.resume, overwrite=args.overwrite)
    write_run_config(output_dir, args, [model.key for model in models], dataset_names)

    runner = EvaluationRunner(
        output_dir=output_dir,
        models=models,
        dataset_names=dataset_names,
        tier=args.tier,
        selected_protocol=args.protocol,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples=args.n_samples,
        limit_per_dataset=args.limit_per_dataset,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        quantization=args.quantization,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prompt_style=args.prompt_style,
        use_mock_generator=args.mock_generator,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
