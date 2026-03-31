from __future__ import annotations
from dataclasses import dataclass, field

from optimizer.evaluation.benchmark_gate import BenchmarkClassification, classify_run
from optimizer.records import BenchmarkRunRecord, JsonValue


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    candidate_classification: BenchmarkClassification
    frontier_classification: BenchmarkClassification
    step_time_delta: float | None
    diagnostics: dict[str, JsonValue] = field(default_factory=dict)


def decide_acceptance(
    candidate: BenchmarkRunRecord,
    frontier: BenchmarkRunRecord,
    *,
    step_time_tolerance: float = 1e-9,
    allow_recovered_oom_promotion: bool = True,
) -> AcceptanceDecision:
    candidate_classification = classify_run(
        candidate,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    frontier_classification = classify_run(
        frontier,
        allow_recovered_oom_promotion=allow_recovered_oom_promotion,
    )
    step_time_delta = _delta(candidate.time_avg_s, frontier.time_avg_s)

    diagnostics: dict[str, JsonValue] = {
        "candidate_status": candidate.status,
        "frontier_status": frontier.status,
        "candidate_oom_events": candidate.oom_events,
        "frontier_oom_events": frontier.oom_events,
        "candidate_triton_mode": candidate.triton_mode,
        "frontier_triton_mode": frontier.triton_mode,
        "candidate_triton_arg": candidate.triton_arg,
        "frontier_triton_arg": frontier.triton_arg,
        "candidate_step_time_s": candidate.time_avg_s,
        "frontier_step_time_s": frontier.time_avg_s,
        "candidate_tokens_per_sec": candidate.tokens_per_sec,
        "frontier_tokens_per_sec": frontier.tokens_per_sec,
        "candidate_reward_avg": candidate.reward_avg,
        "frontier_reward_avg": frontier.reward_avg,
        "candidate_failed_response_count": candidate.failed_response_count,
        "frontier_failed_response_count": frontier.failed_response_count,
        "candidate_failed_response_examples": candidate.failed_response_examples or [],
        "frontier_failed_response_examples": frontier.failed_response_examples or [],
        "authority": "benchmark_only",
        "allow_recovered_oom_promotion": allow_recovered_oom_promotion,
        "diagnostic_only_fields": [
            "triton_mode",
            "triton_arg",
            "oom_events",
            "vram_peak_gb",
            "time_avg_s",
            "failed_response_examples",
        ],
    }

    if not frontier_classification.promotable:
        return AcceptanceDecision(
            accepted=False,
            reason="frontier_not_comparable",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            step_time_delta=step_time_delta,
            diagnostics=diagnostics,
        )

    if not candidate_classification.promotable:
        return AcceptanceDecision(
            accepted=False,
            reason="candidate_not_comparable",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            step_time_delta=step_time_delta,
            diagnostics=diagnostics,
        )

    if candidate.effective_batch != frontier.effective_batch:
        return AcceptanceDecision(
            accepted=False,
            reason="effective_batch_mismatch",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            step_time_delta=step_time_delta,
            diagnostics=diagnostics,
        )

    if step_time_delta is None:
        return AcceptanceDecision(
            accepted=False,
            reason="step_time_missing",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            step_time_delta=step_time_delta,
            diagnostics=diagnostics,
        )

    if step_time_delta < -step_time_tolerance:
        return AcceptanceDecision(
            accepted=True,
            reason="step_time_improved",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            step_time_delta=step_time_delta,
            diagnostics=diagnostics,
        )

    return AcceptanceDecision(
        accepted=False,
        reason="step_time_not_improved",
        candidate_classification=candidate_classification,
        frontier_classification=frontier_classification,
        step_time_delta=step_time_delta,
        diagnostics=diagnostics,
    )


def _delta(candidate_value: float | None, frontier_value: float | None) -> float | None:
    if candidate_value is None or frontier_value is None:
        return None
    return candidate_value - frontier_value
