from __future__ import annotations

from dataclasses import dataclass

from optimizer.records import BenchmarkRunRecord


@dataclass(frozen=True)
class BenchmarkClassification:
    comparability: str
    promotable: bool
    complete: bool
    reasons: tuple[str, ...]


def classify_run(
    run: BenchmarkRunRecord,
    *,
    allow_recovered_oom_promotion: bool = True,
) -> BenchmarkClassification:
    reasons: list[str] = []
    complete = _is_complete(run)

    if not complete:
        reasons.append("benchmark_incomplete")

    if not run.valid:
        reasons.append("benchmark_invalid")

    if run.status == "ok":
        comparability = (
            "clean_comparable" if complete and run.valid else "not_comparable"
        )
    elif run.status == "oom_recovered":
        comparability = (
            "recovered_comparable"
            if complete and run.valid and allow_recovered_oom_promotion
            else "not_comparable"
        )
        reasons.append("oom_backoff_recovered")
        if not allow_recovered_oom_promotion:
            reasons.append("recovered_oom_promotion_disabled")
    else:
        comparability = "not_comparable"
        reasons.append(f"status:{run.status}")

    if run.reward_avg is None:
        reasons.append("reward_missing")
        comparability = "not_comparable"

    if run.effective_batch is None:
        reasons.append("effective_batch_missing")
        comparability = "not_comparable"

    promotable = comparability in {"clean_comparable", "recovered_comparable"}
    return BenchmarkClassification(
        comparability=comparability,
        promotable=promotable,
        complete=complete,
        reasons=tuple(reasons),
    )


def _is_complete(run: BenchmarkRunRecord) -> bool:
    if run.steps_requested is None:
        return True
    if run.steps_observed is None:
        return False
    return run.steps_observed >= run.steps_requested
