from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer.evaluation.acceptance import AcceptanceDecision, decide_acceptance
from optimizer.evaluation.benchmark_gate import classify_run
from optimizer.frontier import Frontier, FrontierEntry, FrontierTransition
from optimizer.plot_step_time import (
    load_ledger,
    write_report as write_step_time_report,
    write_svg_plot as write_step_time_svg_plot,
)
from optimizer.plot_tokens_per_sec import (
    write_report as write_tokens_per_sec_report,
    write_svg_plot as write_tokens_per_sec_svg_plot,
)
from optimizer.records import (
    BenchmarkComparisonRecord,
    BenchmarkRunRecord,
    DecisionRecord,
    ExperimentLedgerRecord,
    ExperimentSnapshotRecord,
    GenerationReviewRecord,
    JsonValue,
    TestRunRecord,
    append_jsonl_record,
    load_benchmark_report,
    load_generation_review_record,
    load_json_record,
    load_last_jsonl_record,
    load_test_run_record,
    utc_timestamp,
    write_json_record,
)


@dataclass(frozen=True)
class AttemptRecord:
    candidate_id: str
    frontier_target: str
    worktree_path: str
    selected_target: str | None
    hypothesis: str
    change_summary: str
    frontier_id: str | None = None
    attempt_markdown_path: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce benchmark output into frontier and ledger updates"
    )
    parser.add_argument("--attempt-markdown", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("optimizer/artifacts/records"),
    )
    parser.add_argument(
        "--allow-recovered-oom-promotion",
        action="store_true",
        default=False,
    )
    parser.add_argument("--test-result", type=Path, required=True)
    parser.add_argument("--generation-review", type=Path)
    args = parser.parse_args(argv)

    result = reduce_attempt(
        attempt_markdown_path=args.attempt_markdown,
        benchmark_report_path=args.benchmark_report,
        artifacts_dir=args.artifacts_dir,
        allow_recovered_oom_promotion=args.allow_recovered_oom_promotion,
        test_result_path=args.test_result,
        generation_review_path=args.generation_review,
    )
    print(f"decision_id={result['decision_id']}")
    print(f"accepted={result['accepted']}")
    print(f"reason={result['reason']}")
    print(f"frontier_state_path={result['frontier_state_path']}")
    print(f"ledger_path={result['ledger_path']}")
    print(f"plot_path={result['plot_path']}")
    return 0


def reduce_attempt(
    *,
    attempt_markdown_path: Path,
    benchmark_report_path: Path,
    artifacts_dir: Path,
    allow_recovered_oom_promotion: bool = True,
    test_result_path: Path | None = None,
    generation_review_path: Path | None = None,
) -> dict[str, str | bool]:
    artifacts_dir = artifacts_dir.resolve()
    attempt = parse_attempt_markdown(attempt_markdown_path)
    comparison = load_benchmark_report(benchmark_report_path)
    test_result = _load_required_test_run(test_result_path)
    generation_review = _load_optional_generation_review(generation_review_path)
    _enforce_sequential_report(
        comparison, attempt.frontier_target, attempt.worktree_path
    )

    frontier = Frontier()
    frontier_state_path = artifacts_dir / "frontier" / "frontier_state.json"
    current_frontier = _restore_frontier(
        frontier, frontier_state_path, allow_recovered_oom_promotion
    )

    if current_frontier is None:
        frontier_run = comparison.require_run(attempt.frontier_target)
        transition = frontier.seed(
            FrontierEntry.from_run(
                candidate_id=attempt.frontier_id or attempt.frontier_target,
                target=attempt.frontier_target,
                benchmark=frontier_run,
                allow_recovered_oom_promotion=allow_recovered_oom_promotion,
            )
        )
        current_frontier = frontier.current
        if current_frontier is None:
            raise ValueError("Failed to seed frontier from benchmark report.")
    else:
        transition = FrontierTransition(
            accepted=True,
            previous_candidate_id=None,
            current_candidate_id=current_frontier.candidate_id,
            reason="frontier_restored",
        )

    frontier_target = _resolve_frontier_target_for_comparison(
        frontier_entry=current_frontier,
        comparison=comparison,
        artifacts_dir=artifacts_dir,
    )
    frontier_run = comparison.require_run(frontier_target)
    frontier.current = FrontierEntry.from_run(
        candidate_id=current_frontier.candidate_id,
        target=frontier_target,
        benchmark=frontier_run,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    current_frontier = frontier.current
    if current_frontier is None:
        raise ValueError(
            "Failed to refresh restored frontier from current benchmark run."
        )
    candidate_run = comparison.require_run(attempt.worktree_path)
    acceptance = _decide_with_hard_gates(
        candidate_run=candidate_run,
        frontier_run=frontier_run,
        test_result=test_result,
        generation_review=generation_review,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    transition = frontier.apply_decision(
        candidate_id=attempt.candidate_id,
        target=attempt.worktree_path,
        benchmark=candidate_run,
        decision=acceptance,
    )

    decision_id = utc_timestamp()
    stored_benchmark_path = write_json_record(
        artifacts_dir / "benchmarks" / f"{decision_id}_generation_hf.json",
        comparison,
    )
    frontier_payload = _frontier_payload(frontier)
    stored_frontier_state_path = write_json_record(
        frontier_state_path, frontier_payload
    )
    decision = DecisionRecord(
        decision_id=decision_id,
        campaign="generation_hf",
        candidate_id=attempt.candidate_id,
        candidate_target=attempt.worktree_path,
        frontier_id=current_frontier.candidate_id,
        frontier_target=current_frontier.target,
        change_summary=attempt.change_summary,
        accepted=acceptance.accepted,
        reason=acceptance.reason,
        candidate_status=candidate_run.status,
        frontier_status=frontier_run.status,
        candidate_comparability=acceptance.candidate_classification.comparability,
        frontier_comparability=acceptance.frontier_classification.comparability,
        reward_delta=None,
        loss_delta=None,
        benchmark_report_path=str(stored_benchmark_path),
        frontier_state_path=str(stored_frontier_state_path),
        diagnostics={
            **acceptance.diagnostics,
            "frontier_transition_reason": transition.reason,
            "sequential_only": True,
            "test_result_path": str(test_result_path) if test_result_path else None,
            "generation_review_path": (
                str(generation_review_path) if generation_review_path else None
            ),
        },
    )
    safe_candidate_id = attempt.candidate_id.replace("/", "-")
    decision_path = write_json_record(
        artifacts_dir / "decisions" / f"{decision_id}_{safe_candidate_id}.json",
        decision,
    )
    ledger_record = _build_experiment_ledger(
        artifacts_dir=artifacts_dir,
        attempt=attempt,
        decision=decision,
        decision_path=decision_path,
        frontier_entry=current_frontier,
        frontier_run=frontier_run,
        candidate_run=candidate_run,
    )
    ledger_path = append_jsonl_record(
        artifacts_dir / "experiments" / "experiment_ledger.jsonl",
        ledger_record,
    )
    _regenerate_step_time_report(artifacts_dir)
    return {
        "decision_id": decision_id,
        "accepted": decision.accepted,
        "reason": decision.reason,
        "frontier_state_path": str(stored_frontier_state_path),
        "ledger_path": str(ledger_path),
        "plot_path": str(artifacts_dir.parent / "reports" / "step_time.svg"),
    }


def _load_required_test_run(path: Path | None) -> TestRunRecord:
    if path is None:
        raise ValueError("Reducer requires a test result artifact.")
    return load_test_run_record(path)


def _load_optional_generation_review(
    path: Path | None,
) -> GenerationReviewRecord | None:
    if path is None:
        return None
    return load_generation_review_record(path)


def _decide_with_hard_gates(
    *,
    candidate_run: BenchmarkRunRecord,
    frontier_run: BenchmarkRunRecord,
    test_result: TestRunRecord,
    generation_review: GenerationReviewRecord | None,
    allow_recovered_oom_promotion: bool,
) -> AcceptanceDecision:
    benchmark_decision = decide_acceptance(
        candidate_run,
        frontier_run,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    diagnostics = dict(benchmark_decision.diagnostics)
    diagnostics["test_command"] = test_result.command
    diagnostics["test_passed"] = test_result.passed
    diagnostics["test_exit_code"] = test_result.exit_code
    diagnostics["test_failed_count"] = test_result.failed_count
    diagnostics["test_xfailed_count"] = test_result.xfailed_count
    diagnostics["test_log_path"] = test_result.log_path

    candidate_classification = classify_run(
        candidate_run,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    frontier_classification = classify_run(
        frontier_run,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )

    if not test_result.passed:
        return _forced_rejection(
            reason="candidate_tests_failed",
            candidate_run=candidate_run,
            frontier_run=frontier_run,
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            diagnostics=diagnostics,
        )

    if (candidate_run.failed_response_count or 0) > 0:
        if generation_review is None:
            return _forced_rejection(
                reason="candidate_generation_review_missing",
                candidate_run=candidate_run,
                frontier_run=frontier_run,
                candidate_classification=candidate_classification,
                frontier_classification=frontier_classification,
                diagnostics=diagnostics,
            )
        diagnostics["generation_review_verdict"] = generation_review.verdict
        diagnostics["generation_review_reason"] = generation_review.reason
        diagnostics["generation_review_examples"] = generation_review.examples_reviewed
        diagnostics["generation_review_reviewer"] = generation_review.reviewer
        if generation_review.candidate_target != candidate_run.input:
            return _forced_rejection(
                reason="candidate_generation_review_target_mismatch",
                candidate_run=candidate_run,
                frontier_run=frontier_run,
                candidate_classification=candidate_classification,
                frontier_classification=frontier_classification,
                diagnostics=diagnostics,
            )
        if generation_review.verdict != "not_gibberish":
            return _forced_rejection(
                reason="candidate_failed_generations_gibberish",
                candidate_run=candidate_run,
                frontier_run=frontier_run,
                candidate_classification=candidate_classification,
                frontier_classification=frontier_classification,
                diagnostics=diagnostics,
            )

    return AcceptanceDecision(
        accepted=benchmark_decision.accepted,
        reason=benchmark_decision.reason,
        candidate_classification=benchmark_decision.candidate_classification,
        frontier_classification=benchmark_decision.frontier_classification,
        step_time_delta=benchmark_decision.step_time_delta,
        diagnostics=diagnostics,
    )


def _forced_rejection(
    *,
    reason: str,
    candidate_run: BenchmarkRunRecord,
    frontier_run: BenchmarkRunRecord,
    candidate_classification,
    frontier_classification,
    diagnostics: dict[str, JsonValue],
) -> AcceptanceDecision:
    return AcceptanceDecision(
        accepted=False,
        reason=reason,
        candidate_classification=candidate_classification,
        frontier_classification=frontier_classification,
        step_time_delta=_maybe_delta(candidate_run.time_avg_s, frontier_run.time_avg_s),
        diagnostics=diagnostics,
    )


def parse_attempt_markdown(path: Path) -> AttemptRecord:
    lines = path.read_text(encoding="utf-8").splitlines()
    fields: dict[str, str] = {}
    hypothesis_lines: list[str] = []
    in_hypothesis = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("Change summary:"):
            fields["change_summary"] = line.split(":", 1)[1].strip()
            in_hypothesis = False
            continue
        if line.startswith("## "):
            in_hypothesis = line == "## Hypothesis"
            continue
        if in_hypothesis:
            if line:
                hypothesis_lines.append(line)
            continue
        if line.startswith("- ") and ": `" in line and line.endswith("`"):
            label, value = line[2:].split(": `", 1)
            fields[label.strip().lower().replace(" ", "_")] = value[:-1]
    candidate_id = fields.get("candidate_id")
    frontier_target = fields.get("frontier_target")
    worktree_path = fields.get("candidate_worktree")
    if candidate_id is None or frontier_target is None or worktree_path is None:
        raise ValueError(
            "Attempt markdown must define frontier target, candidate id, and candidate worktree."
        )
    if "change_summary" in fields:
        change_summary = fields["change_summary"]
    elif hypothesis_lines:
        change_summary = hypothesis_lines[0]
    else:
        change_summary = ""
    hypothesis = " ".join(hypothesis_lines).strip()
    return AttemptRecord(
        candidate_id=candidate_id,
        frontier_target=frontier_target,
        worktree_path=worktree_path,
        selected_target=fields.get("selected_target"),
        hypothesis=hypothesis,
        change_summary=change_summary,
        attempt_markdown_path=str(path.resolve()),
    )


def _restore_frontier(
    frontier: Frontier,
    frontier_state_path: Path,
    allow_recovered_oom_promotion: bool,
) -> FrontierEntry | None:
    if not frontier_state_path.exists():
        return None
    payload = load_json_record(frontier_state_path)
    current_obj = payload.get("current")
    if not isinstance(current_obj, dict):
        return None
    target = current_obj.get("target")
    candidate_id = current_obj.get("candidate_id")
    comparability = current_obj.get("comparability")
    benchmark_obj = current_obj.get("benchmark")
    if (
        not isinstance(target, str)
        or not isinstance(candidate_id, str)
        or not isinstance(comparability, str)
    ):
        raise ValueError("Frontier state current entry is malformed.")
    effective_target = target
    if isinstance(benchmark_obj, dict) and isinstance(benchmark_obj.get("input"), str):
        effective_target = str(benchmark_obj.get("input"))
        benchmark = BenchmarkRunRecord.from_dict(
            {str(key): item for key, item in benchmark_obj.items()}
        )
    else:
        tokens_per_sec = current_obj.get("tokens_per_sec")
        vram_peak_gb = current_obj.get("vram_peak_gb")
        if tokens_per_sec is None or vram_peak_gb is None:
            raise ValueError(
                "Legacy frontier state must include tokens_per_sec and vram_peak_gb."
            )
        benchmark = BenchmarkRunRecord(
            input=target,
            label=target,
            commit="unknown",
            status="ok",
            valid=True,
            steps_requested=10,
            steps_observed=10,
            tokens_per_sec=float(tokens_per_sec),
            time_avg_s=_maybe_float(current_obj.get("time_avg_s")),
            vram_peak_gb=float(vram_peak_gb),
            loss_avg=_maybe_float(current_obj.get("loss_avg")),
            reward_avg=_maybe_float(current_obj.get("reward_avg")),
            effective_batch=_maybe_int(current_obj.get("effective_batch")) or 16,
            oom_events=_maybe_int(current_obj.get("oom_events")) or 0,
        )
    entry = FrontierEntry.from_run(
        candidate_id=candidate_id,
        target=effective_target,
        benchmark=benchmark,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    frontier.current = entry
    return entry


def _build_experiment_ledger(
    *,
    artifacts_dir: Path,
    attempt: AttemptRecord,
    decision: DecisionRecord,
    decision_path: Path,
    frontier_entry: FrontierEntry,
    frontier_run: BenchmarkRunRecord,
    candidate_run: BenchmarkRunRecord,
) -> ExperimentLedgerRecord:
    ledger_path = artifacts_dir / "experiments" / "experiment_ledger.jsonl"
    last_record = load_last_jsonl_record(ledger_path)
    previous_experiment_number = 0
    running_best_step_time = None
    running_best_experiment_number = None
    if last_record is not None:
        previous_experiment_number = (
            _maybe_int(last_record.get("experiment_number")) or 0
        )
        running_best_step_time = _maybe_float(
            last_record.get("running_best_step_time")
        )
        running_best_experiment_number = _maybe_int(
            last_record.get("running_best_experiment_number")
        )
    experiment_number = previous_experiment_number + 1
    baseline_snapshot = _build_snapshot(
        frontier_entry.target, frontier_run, decision.frontier_comparability
    )
    candidate_snapshot = _build_snapshot(
        decision.candidate_target, candidate_run, decision.candidate_comparability
    )
    incumbent_step_time_after_decision = (
        candidate_run.time_avg_s
        if decision.accepted
        else frontier_run.time_avg_s
    )
    # For step time, lower is better (unlike tokens/sec where higher is better)
    if incumbent_step_time_after_decision is not None and (
        running_best_step_time is None
        or incumbent_step_time_after_decision < running_best_step_time
    ):
        running_best_step_time = incumbent_step_time_after_decision
        running_best_experiment_number = experiment_number
    return ExperimentLedgerRecord(
        experiment_number=experiment_number,
        decision_id=decision.decision_id,
        campaign=decision.campaign,
        change_summary=decision.change_summary,
        candidate_id=decision.candidate_id,
        candidate_target=decision.candidate_target,
        frontier_id=decision.frontier_id,
        frontier_target=decision.frontier_target,
        outcome="kept" if decision.accepted else "discarded",
        reason=decision.reason,
        primary_metric="step_time",
        baseline_snapshot=baseline_snapshot,
        candidate_snapshot=candidate_snapshot,
        incumbent_step_time_after_decision=incumbent_step_time_after_decision,
        running_best_step_time=running_best_step_time,
        running_best_experiment_number=running_best_experiment_number,
        benchmark_report_path=decision.benchmark_report_path,
        frontier_state_path=decision.frontier_state_path,
        decision_path=str(decision_path),
        timestamp=decision.decision_id,
        attempt_number=experiment_number,
        selected_target=attempt.selected_target,
        hypothesis=attempt.hypothesis,
        worktree_path=attempt.worktree_path,
        attempt_markdown_path=attempt.attempt_markdown_path,
        step_time_delta=_maybe_delta(
            candidate_snapshot.time_avg_s, baseline_snapshot.time_avg_s
        ),
        step_time_pct_change=_percent_change(
            candidate_snapshot.time_avg_s, baseline_snapshot.time_avg_s
        ),
        incumbent_tokens_per_sec_after_decision=(
            candidate_run.tokens_per_sec
            if decision.accepted
            else frontier_run.tokens_per_sec
        ),
        running_best_tokens_per_sec=_running_best_tokens_per_sec(
            last_record=last_record,
            accepted=decision.accepted,
            candidate_run=candidate_run,
            frontier_run=frontier_run,
        ),
        tokens_per_sec_delta=_maybe_delta(
            candidate_snapshot.tokens_per_sec, baseline_snapshot.tokens_per_sec
        ),
        tokens_per_sec_pct_change=_percent_change(
            candidate_snapshot.tokens_per_sec, baseline_snapshot.tokens_per_sec
        ),
        sequential_only=True,
        generation_only=True,
    )


def _build_snapshot(
    target: str, benchmark: BenchmarkRunRecord, comparability: str
) -> ExperimentSnapshotRecord:
    return ExperimentSnapshotRecord(
        target=target,
        status=benchmark.status,
        comparability=comparability,
        tokens_per_sec=benchmark.tokens_per_sec,
        time_avg_s=benchmark.time_avg_s,
        reward_avg=benchmark.reward_avg,
        loss_avg=benchmark.loss_avg,
        vram_peak_gb=benchmark.vram_peak_gb,
        oom_events=benchmark.oom_events,
        effective_batch=benchmark.effective_batch,
    )


def _frontier_payload(frontier: Frontier) -> dict[str, JsonValue]:
    current = frontier.current
    return {
        "schema_version": 1,
        "updated_at": utc_timestamp(),
        "sequential_only": True,
        "campaign": "generation_hf",
        "current": None
        if current is None
        else {
            "candidate_id": current.candidate_id,
            "target": current.target,
            "comparability": current.comparability,
            "benchmark": current.benchmark.to_dict(),
        },
        "history": [
            {
                "accepted": transition.accepted,
                "previous_candidate_id": transition.previous_candidate_id,
                "current_candidate_id": transition.current_candidate_id,
                "reason": transition.reason,
            }
            for transition in frontier.history
        ],
    }


def _enforce_sequential_report(
    comparison: BenchmarkComparisonRecord,
    frontier_target: str,
    candidate_target: str,
) -> None:
    targets = {run.input for run in comparison.runs}
    expected_targets = {frontier_target, candidate_target}
    if targets != expected_targets or len(comparison.runs) != 2:
        raise ValueError(
            "Sequential reducer expects exactly one frontier and one candidate run. "
            f"Expected {expected_targets}, got {targets}."
        )


def _resolve_frontier_target_for_comparison(
    *,
    frontier_entry: FrontierEntry,
    comparison: BenchmarkComparisonRecord,
    artifacts_dir: Path,
) -> str:
    available_targets = {run.input for run in comparison.runs}
    if frontier_entry.target in available_targets:
        return frontier_entry.target
    if frontier_entry.candidate_id in available_targets:
        return frontier_entry.candidate_id
    worktree_record_path = (
        artifacts_dir / "worktrees" / f"{frontier_entry.candidate_id}.json"
    )
    if worktree_record_path.exists():
        payload = load_json_record(worktree_record_path)
        worktree_path = payload.get("worktree_path")
        if isinstance(worktree_path, str) and worktree_path in available_targets:
            return worktree_path
    benchmark_input = frontier_entry.benchmark.input
    if benchmark_input in available_targets:
        return benchmark_input
    raise KeyError(
        f"Run for frontier target '{frontier_entry.target}' not found in benchmark report."
    )


def _regenerate_step_time_report(artifacts_dir: Path) -> None:
    ledger_path = artifacts_dir / "experiments" / "experiment_ledger.jsonl"
    reports_dir = artifacts_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = load_ledger(ledger_path)
    svg_path = reports_dir / "step_time.svg"
    md_path = reports_dir / "step_time.md"
    write_step_time_svg_plot(rows, svg_path)
    write_step_time_report(rows, md_path, svg_path)
    tps_svg_path = reports_dir / "tokens_per_sec.svg"
    tps_md_path = reports_dir / "tokens_per_sec.md"
    write_tokens_per_sec_svg_plot(rows, tps_svg_path)
    write_tokens_per_sec_report(rows, tps_md_path, tps_svg_path)


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"Expected float-compatible value, got {type(value)!r}.")


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected int-compatible value, got {type(value)!r}.")


def _maybe_delta(
    candidate_value: float | None, baseline_value: float | None
) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _percent_change(
    candidate_value: float | None, baseline_value: float | None
) -> float | None:
    if candidate_value is None or baseline_value is None or baseline_value == 0:
        return None
    return ((candidate_value - baseline_value) / baseline_value) * 100.0


def _running_best_tokens_per_sec(
    *,
    last_record: dict[str, object] | None,
    accepted: bool,
    candidate_run: BenchmarkRunRecord,
    frontier_run: BenchmarkRunRecord,
) -> float | None:
    running_best = None
    if last_record is not None:
        running_best = _maybe_float(last_record.get("running_best_tokens_per_sec"))
    incumbent = candidate_run.tokens_per_sec if accepted else frontier_run.tokens_per_sec
    if incumbent is None:
        return running_best
    if running_best is None or incumbent > running_best:
        return incumbent
    return running_best


if __name__ == "__main__":
    raise SystemExit(main())
