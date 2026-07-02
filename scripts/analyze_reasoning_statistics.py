#!/usr/bin/env python
"""Paired statistical analysis for prompt-matched reasoning evaluations.

The script operates on ``scores_by_example.jsonl`` emitted by
``scripts/evaluate_reasoning_vllm.py``. It does not run model inference or
select checkpoints. Each row is treated as one prompt-matched problem summary
for one model, dataset, and protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "correct",
    "pass_at_k",
    "avg_at_k",
    "avg_response_length",
    "truncation_rate",
    "format_ok",
)


@dataclass(frozen=True)
class ExampleScore:
    model_key: str
    dataset: str
    protocol: str
    example_id: str
    correct: float
    pass_at_k: float
    avg_at_k: float
    avg_response_length: float
    truncation_rate: float
    format_ok: float
    sample_count: int


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def load_scores(path: Path, *, protocol: str) -> list[ExampleScore]:
    rows: list[ExampleScore] = []
    for row in _read_jsonl(path):
        if str(row.get("protocol")) != protocol:
            continue
        sample_count = int(row.get("sample_count") or 1)
        truncated_count = _to_float(row.get("truncated_count"))
        rows.append(
            ExampleScore(
                model_key=str(row["model_key"]),
                dataset=str(row["dataset"]),
                protocol=str(row["protocol"]),
                example_id=str(row["example_id"]),
                correct=_to_float(row.get("correct")),
                pass_at_k=_to_float(row.get("pass_at_k")),
                avg_at_k=_to_float(row.get("avg_at_k")),
                avg_response_length=_to_float(row.get("avg_response_length")),
                truncation_rate=truncated_count / max(1, sample_count),
                format_ok=1.0 if bool(row.get("format_ok")) else 0.0,
                sample_count=sample_count,
            )
        )
    if not rows:
        raise ValueError(f"No rows found for protocol={protocol!r} in {path}")
    return rows


def _metric(score: ExampleScore, name: str) -> float:
    return float(getattr(score, name))


def group_scores(rows: list[ExampleScore]) -> dict[tuple[str, str, str], ExampleScore]:
    grouped: dict[tuple[str, str, str], ExampleScore] = {}
    for row in rows:
        key = (row.model_key, row.dataset, row.example_id)
        if key in grouped:
            raise ValueError(f"Duplicate score row for {key}")
        grouped[key] = row
    return grouped


def common_datasets(
    rows: list[ExampleScore],
    *,
    baseline: str,
    models: list[str],
) -> dict[str, list[str]]:
    by_dataset_model: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.model_key in {baseline, *models}:
            by_dataset_model[(row.dataset, row.model_key)].add(row.example_id)

    datasets = sorted({row.dataset for row in rows})
    common: dict[str, list[str]] = {}
    for dataset in datasets:
        model_sets = [by_dataset_model.get((dataset, model), set()) for model in [baseline, *models]]
        if not model_sets or any(not ids for ids in model_sets):
            continue
        shared = set.intersection(*model_sets)
        if shared:
            common[dataset] = sorted(shared, key=lambda item: (len(item), item))
    if not common:
        raise ValueError("No prompt-matched examples found across baseline and requested models")
    return common


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def dataset_metric(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    model: str,
    dataset: str,
    example_ids: list[str],
    metric: str,
) -> float:
    return mean(_metric(score_map[(model, dataset, example_id)], metric) for example_id in example_ids)


def macro_metric(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    model: str,
    matched: dict[str, list[str]],
    metric: str,
) -> float:
    return mean(
        dataset_metric(score_map, model=model, dataset=dataset, example_ids=example_ids, metric=metric)
        for dataset, example_ids in matched.items()
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
    metrics: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    point_values: dict[tuple[str, str], float] = {}
    for model in [baseline, *models]:
        for metric in metrics:
            point_values[(model, metric)] = macro_metric(score_map, model=model, matched=matched, metric=metric)

    for model in models:
        for metric in metrics:
            boot_values: list[float] = []
            for _ in range(n_bootstrap):
                dataset_values: list[float] = []
                for dataset, example_ids in matched.items():
                    sampled_ids = [rng.choice(example_ids) for _ in example_ids]
                    deltas = [
                        _metric(score_map[(model, dataset, example_id)], metric)
                        - _metric(score_map[(baseline, dataset, example_id)], metric)
                        for example_id in sampled_ids
                    ]
                    dataset_values.append(mean(deltas))
                boot_values.append(mean(dataset_values))
            rows.append(
                {
                    "model_key": model,
                    "baseline": baseline,
                    "metric": metric,
                    "baseline_value": point_values[(baseline, metric)],
                    "model_value": point_values[(model, metric)],
                    "delta": point_values[(model, metric)] - point_values[(baseline, metric)],
                    "ci95_low": percentile(boot_values, 0.025),
                    "ci95_high": percentile(boot_values, 0.975),
                    "bootstrap_samples": n_bootstrap,
                }
            )
    return rows


def summary_metrics(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_values = {
        metric: macro_metric(score_map, model=baseline, matched=matched, metric=metric)
        for metric in metrics
    }
    for model in [baseline, *models]:
        for metric in metrics:
            value = macro_metric(score_map, model=model, matched=matched, metric=metric)
            rows.append(
                {
                    "model_key": model,
                    "metric": metric,
                    "value": value,
                    "delta_vs_baseline": value - baseline_values[metric],
                    "datasets": len(matched),
                    "matched_examples": sum(len(ids) for ids in matched.values()),
                }
            )
    return rows


def per_dataset_intervals(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
    metrics: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 17)
    rows: list[dict[str, Any]] = []
    for model in [baseline, *models]:
        for dataset, example_ids in matched.items():
            for metric in metrics:
                point = dataset_metric(
                    score_map,
                    model=model,
                    dataset=dataset,
                    example_ids=example_ids,
                    metric=metric,
                )
                boot_values: list[float] = []
                for _ in range(n_bootstrap):
                    sampled_ids = [rng.choice(example_ids) for _ in example_ids]
                    boot_values.append(
                        mean(_metric(score_map[(model, dataset, example_id)], metric) for example_id in sampled_ids)
                    )
                rows.append(
                    {
                        "model_key": model,
                        "dataset": dataset,
                        "metric": metric,
                        "value": point,
                        "ci95_low": percentile(boot_values, 0.025),
                        "ci95_high": percentile(boot_values, 0.975),
                        "matched_examples": len(example_ids),
                        "bootstrap_samples": n_bootstrap,
                    }
                )
    return rows


def per_dataset_paired_bootstrap(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
    metrics: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 31)
    rows: list[dict[str, Any]] = []
    for model in models:
        for dataset, example_ids in matched.items():
            for metric in metrics:
                base_value = dataset_metric(
                    score_map,
                    model=baseline,
                    dataset=dataset,
                    example_ids=example_ids,
                    metric=metric,
                )
                model_value = dataset_metric(
                    score_map,
                    model=model,
                    dataset=dataset,
                    example_ids=example_ids,
                    metric=metric,
                )
                boot_values: list[float] = []
                for _ in range(n_bootstrap):
                    sampled_ids = [rng.choice(example_ids) for _ in example_ids]
                    boot_values.append(
                        mean(
                            _metric(score_map[(model, dataset, example_id)], metric)
                            - _metric(score_map[(baseline, dataset, example_id)], metric)
                            for example_id in sampled_ids
                        )
                    )
                rows.append(
                    {
                        "model_key": model,
                        "baseline": baseline,
                        "dataset": dataset,
                        "metric": metric,
                        "baseline_value": base_value,
                        "model_value": model_value,
                        "delta": model_value - base_value,
                        "ci95_low": percentile(boot_values, 0.025),
                        "ci95_high": percentile(boot_values, 0.975),
                        "matched_examples": len(example_ids),
                        "bootstrap_samples": n_bootstrap,
                    }
                )
    return rows


def paired_problem_outcomes(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for dataset, example_ids in matched.items():
            for metric in metrics:
                improved = worsened = tied = 0
                for example_id in example_ids:
                    delta = _metric(score_map[(model, dataset, example_id)], metric) - _metric(
                        score_map[(baseline, dataset, example_id)],
                        metric,
                    )
                    if delta > 0:
                        improved += 1
                    elif delta < 0:
                        worsened += 1
                    else:
                        tied += 1
                rows.append(
                    {
                        "model_key": model,
                        "baseline": baseline,
                        "dataset": dataset,
                        "metric": metric,
                        "improved": improved,
                        "worsened": worsened,
                        "tied": tied,
                        "matched_examples": len(example_ids),
                    }
                )
    return rows


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials <= 0:
        return float("nan")
    observed = math.comb(trials, successes) * (0.5**trials)
    p_value = 0.0
    for k in range(trials + 1):
        probability = math.comb(trials, k) * (0.5**trials)
        if probability <= observed + 1e-15:
            p_value += probability
    return min(1.0, p_value)


def mcnemar_pass1(
    score_map: dict[tuple[str, str, str], ExampleScore],
    *,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_datasets = {"__macro__": [example_id for ids in matched.values() for example_id in ids]}
    for model in models:
        for dataset, example_ids in {**matched, **all_datasets}.items():
            baseline_wrong_model_right = 0
            baseline_right_model_wrong = 0
            comparable = 0
            if dataset == "__macro__":
                iterator = [(ds, eid) for ds, ids in matched.items() for eid in ids]
            else:
                iterator = [(dataset, eid) for eid in example_ids]
            for ds, example_id in iterator:
                base_correct = bool(score_map[(baseline, ds, example_id)].correct)
                model_correct = bool(score_map[(model, ds, example_id)].correct)
                comparable += 1
                if (not base_correct) and model_correct:
                    baseline_wrong_model_right += 1
                elif base_correct and (not model_correct):
                    baseline_right_model_wrong += 1
            discordant = baseline_wrong_model_right + baseline_right_model_wrong
            rows.append(
                {
                    "model_key": model,
                    "baseline": baseline,
                    "dataset": dataset,
                    "baseline_wrong_model_right": baseline_wrong_model_right,
                    "baseline_right_model_wrong": baseline_right_model_wrong,
                    "discordant_pairs": discordant,
                    "matched_examples": comparable,
                    "exact_binomial_p": exact_binomial_two_sided(baseline_wrong_model_right, discordant),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint_sanity(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    baseline: str,
    models: list[str],
) -> None:
    values: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["model_key"]), str(row["metric"])): row for row in summary_rows
    }
    labels = {
        "sent_best": "validation-selected SENT checkpoint",
        "sent_final": "SENT training endpoint",
        "no_sent_final": "non-SENT endpoint, not a perfect control",
    }
    lines = [
        "# Checkpoint Selection Sanity Check",
        "",
        "This analysis separates the selected checkpoint from endpoint comparisons. "
        "It should be used to bound claims rather than to imply a new training result.",
        "",
        "| model | role | pass@1 delta | avg@k delta | pass@k delta | length delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        role = labels.get(model, "comparison checkpoint")
        lines.append(
            "| {model} | {role} | {pass1:.6f} | {avgk:.6f} | {passk:.6f} | {length:.3f} |".format(
                model=model,
                role=role,
                pass1=float(values[(model, "correct")]["delta_vs_baseline"]),
                avgk=float(values[(model, "avg_at_k")]["delta_vs_baseline"]),
                passk=float(values[(model, "pass_at_k")]["delta_vs_baseline"]),
                length=float(values[(model, "avg_response_length")]["delta_vs_baseline"]),
            )
        )
    lines.extend(
        [
            "",
            f"Baseline: `{baseline}`.",
            "",
            "Interpretation guardrail: `sent_best` is valid as a selected checkpoint only when its selection source is "
            "external to this benchmark. `sent_final` is the endpoint comparison. `no_sent_final` helps separate "
            "entropy-curriculum effects from generic GRPO training drift, but it is not a controlled ablation unless "
            "all other training conditions are identical and independently documented.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    input_path: Path,
    protocol: str,
    baseline: str,
    models: list[str],
    matched: dict[str, list[str]],
    bootstrap_rows: list[dict[str, Any]],
    mcnemar_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Prompt-Matched Statistical Analysis",
        "",
        f"Input: `{input_path}`",
        f"Protocol: `{protocol}`",
        f"Baseline: `{baseline}`",
        f"Models: `{', '.join(models)}`",
        "",
        "## Matched Evaluation Set",
        "",
        "| dataset | matched examples |",
        "|---|---:|",
    ]
    for dataset, ids in matched.items():
        lines.append(f"| {dataset} | {len(ids)} |")
    lines.extend(["", "## Macro Bootstrap Deltas", "", "| model | metric | delta | 95% CI |", "|---|---|---:|---:|"])
    for row in bootstrap_rows:
        lines.append(
            "| {model_key} | {metric} | {delta:.6f} | [{lo:.6f}, {hi:.6f}] |".format(
                model_key=row["model_key"],
                metric=row["metric"],
                delta=float(row["delta"]),
                lo=float(row["ci95_low"]),
                hi=float(row["ci95_high"]),
            )
        )
    lines.extend(["", "## McNemar / Exact Binomial Pass@1", "", "| model | dataset | b | c | p |", "|---|---|---:|---:|---:|"])
    for row in mcnemar_rows:
        lines.append(
            "| {model_key} | {dataset} | {b} | {c} | {p:.6f} |".format(
                model_key=row["model_key"],
                dataset=row["dataset"],
                b=row["baseline_wrong_model_right"],
                c=row["baseline_right_model_wrong"],
                p=float(row["exact_binomial_p"]) if not math.isnan(float(row["exact_binomial_p"])) else float("nan"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap CIs, per-problem outcomes, McNemar tests, and checkpoint sanity tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", required=True, type=Path, help="Path to scores_by_example.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for analysis outputs")
    parser.add_argument("--protocol", default="sampled")
    parser.add_argument("--baseline", default="base")
    parser.add_argument(
        "--models",
        default="sent_best,sent_final,no_sent_final",
        help="Comma-separated checkpoint/model keys compared against the baseline",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = load_scores(args.scores, protocol=args.protocol)
    score_map = group_scores(rows)
    matched = common_datasets(rows, baseline=args.baseline, models=models)

    summary_rows = summary_metrics(
        score_map,
        baseline=args.baseline,
        models=models,
        matched=matched,
        metrics=METRICS,
    )
    bootstrap_rows = paired_bootstrap(
        score_map,
        baseline=args.baseline,
        models=models,
        matched=matched,
        metrics=METRICS,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    dataset_ci_rows = per_dataset_intervals(
        score_map,
        baseline=args.baseline,
        models=models,
        matched=matched,
        metrics=METRICS,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    dataset_delta_rows = per_dataset_paired_bootstrap(
        score_map,
        baseline=args.baseline,
        models=models,
        matched=matched,
        metrics=METRICS,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    problem_rows = paired_problem_outcomes(
        score_map,
        baseline=args.baseline,
        models=models,
        matched=matched,
        metrics=("correct", "pass_at_k", "avg_at_k", "truncation_rate", "format_ok"),
    )
    mcnemar_rows = mcnemar_pass1(score_map, baseline=args.baseline, models=models, matched=matched)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "bootstrap_deltas.csv", bootstrap_rows)
    write_csv(args.output_dir / "per_dataset_confidence_intervals.csv", dataset_ci_rows)
    write_csv(args.output_dir / "per_dataset_bootstrap_deltas.csv", dataset_delta_rows)
    write_csv(args.output_dir / "paired_problem_outcomes.csv", problem_rows)
    write_csv(args.output_dir / "mcnemar_pass1.csv", mcnemar_rows)
    write_checkpoint_sanity(
        args.output_dir / "checkpoint_selection_sanity.md",
        summary_rows=summary_rows,
        baseline=args.baseline,
        models=models,
    )
    write_report(
        args.output_dir / "analysis_report.md",
        input_path=args.scores,
        protocol=args.protocol,
        baseline=args.baseline,
        models=models,
        matched=matched,
        bootstrap_rows=bootstrap_rows,
        mcnemar_rows=mcnemar_rows,
    )
    print(f"Wrote statistical analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
