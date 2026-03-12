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
    reward_delta: float | None
    loss_delta: float | None
    diagnostics: dict[str, JsonValue] = field(default_factory=dict)


def decide_acceptance(
    candidate: BenchmarkRunRecord,
    frontier: BenchmarkRunRecord,
    *,
    reward_tolerance: float = 1e-9,
    loss_tolerance: float = 1e-9,
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
    reward_delta = _delta(candidate.reward_avg, frontier.reward_avg)
    loss_delta = _delta(candidate.loss_avg, frontier.loss_avg)

    diagnostics: dict[str, JsonValue] = {
        "candidate_status": candidate.status,
        "frontier_status": frontier.status,
        "candidate_oom_events": candidate.oom_events,
        "frontier_oom_events": frontier.oom_events,
        "candidate_triton_mode": candidate.triton_mode,
        "frontier_triton_mode": frontier.triton_mode,
        "authority": "benchmark_only",
        "allow_recovered_oom_promotion": allow_recovered_oom_promotion,
        "diagnostic_only_fields": [
            "triton_mode",
            "triton_arg",
            "oom_events",
            "vram_peak_gb",
            "time_avg_s",
        ],
    }

    if not frontier_classification.promotable:
        return AcceptanceDecision(
            accepted=False,
            reason="frontier_not_comparable",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            reward_delta=reward_delta,
            loss_delta=loss_delta,
            diagnostics=diagnostics,
        )

    if not candidate_classification.promotable:
        return AcceptanceDecision(
            accepted=False,
            reason="candidate_not_comparable",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            reward_delta=reward_delta,
            loss_delta=loss_delta,
            diagnostics=diagnostics,
        )

    if candidate.effective_batch != frontier.effective_batch:
        return AcceptanceDecision(
            accepted=False,
            reason="effective_batch_mismatch",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            reward_delta=reward_delta,
            loss_delta=loss_delta,
            diagnostics=diagnostics,
        )

    if reward_delta is None:
        return AcceptanceDecision(
            accepted=False,
            reason="reward_missing",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            reward_delta=reward_delta,
            loss_delta=loss_delta,
            diagnostics=diagnostics,
        )

    if reward_delta > reward_tolerance:
        return AcceptanceDecision(
            accepted=True,
            reason="reward_improved",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            reward_delta=reward_delta,
            loss_delta=loss_delta,
            diagnostics=diagnostics,
        )

    if (
        abs(reward_delta) <= reward_tolerance
        and loss_delta is not None
        and loss_delta < -loss_tolerance
    ):
        return AcceptanceDecision(
            accepted=True,
            reason="reward_tied_loss_improved",
            candidate_classification=candidate_classification,
            frontier_classification=frontier_classification,
            reward_delta=reward_delta,
            loss_delta=loss_delta,
            diagnostics=diagnostics,
        )

    return AcceptanceDecision(
        accepted=False,
        reason="benchmark_not_better",
        candidate_classification=candidate_classification,
        frontier_classification=frontier_classification,
        reward_delta=reward_delta,
        loss_delta=loss_delta,
        diagnostics=diagnostics,
    )


def _delta(candidate_value: float | None, frontier_value: float | None) -> float | None:
    if candidate_value is None or frontier_value is None:
        return None
    return candidate_value - frontier_value
