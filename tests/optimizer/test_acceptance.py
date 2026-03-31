from optimizer.evaluation.acceptance import decide_acceptance
from optimizer.records import BenchmarkRunRecord, JsonValue


def _run(name: str, **overrides: JsonValue) -> BenchmarkRunRecord:
    payload: dict[str, JsonValue] = {
        "input": name,
        "label": name,
        "commit": "abc123",
        "status": "ok",
        "valid": True,
        "steps_requested": 10,
        "steps_observed": 10,
        "tokens_per_sec": 10.0,
        "reward_avg": 0.4,
        "loss_avg": 0.9,
        "effective_batch": 16,
        "oom_events": 0,
        "triton_mode": "on",
        "triton_arg": "--use-triton",
    }
    payload.update(overrides)
    return BenchmarkRunRecord.from_dict(payload)


def test_accepts_clean_improvement() -> None:
    frontier = _run("frontier", tokens_per_sec=10.0, time_avg_s=1.2)
    candidate = _run("candidate", tokens_per_sec=12.0, time_avg_s=1.0)

    decision = decide_acceptance(candidate, frontier)

    assert decision.accepted is True
    assert decision.reason == "step_time_improved"
    assert decision.diagnostics["authority"] == "benchmark_only"


def test_accepts_recovered_comparable_when_tokens_per_sec_improves() -> None:
    frontier = _run("frontier", tokens_per_sec=10.0, time_avg_s=1.2)
    candidate = _run(
        "candidate",
        status="oom_recovered",
        valid=True,
        tokens_per_sec=10.5,
        time_avg_s=1.1,
        oom_events=1,
    )

    decision = decide_acceptance(candidate, frontier)

    assert decision.accepted is True
    assert decision.candidate_classification.comparability == "recovered_comparable"


def test_rejects_effective_batch_mismatch_and_non_improvement() -> None:
    frontier = _run("frontier", tokens_per_sec=10.0, time_avg_s=1.2)
    mismatched = _run(
        "candidate-a", tokens_per_sec=11.0, time_avg_s=1.0, effective_batch=8
    )
    weaker = _run("candidate-b", tokens_per_sec=12.0, time_avg_s=1.25)

    mismatch_decision = decide_acceptance(mismatched, frontier)
    weaker_decision = decide_acceptance(weaker, frontier)

    assert mismatch_decision.accepted is False
    assert mismatch_decision.reason == "effective_batch_mismatch"
    assert weaker_decision.accepted is False
    assert weaker_decision.reason == "step_time_not_improved"
