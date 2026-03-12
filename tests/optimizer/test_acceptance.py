from optimizer.evaluation.acceptance import decide_acceptance
from optimizer.records import BenchmarkRunRecord, JsonValue


def _run(name: str, **overrides: JsonValue) -> BenchmarkRunRecord:
    payload: dict[str, JsonValue] = {
        "input": name,
        "label": name,
        "commit": "abc123",
        "status": "ok",
        "valid": True,
        "steps_requested": 5,
        "steps_observed": 5,
        "reward_avg": 0.4,
        "loss_avg": 0.9,
        "effective_batch": 16,
        "oom_events": 0,
        "triton_mode": "auto",
    }
    payload.update(overrides)
    return BenchmarkRunRecord.from_dict(payload)


def test_accepts_clean_improvement() -> None:
    frontier = _run("frontier", reward_avg=0.4, loss_avg=0.9)
    candidate = _run("candidate", reward_avg=0.5, loss_avg=1.1)

    decision = decide_acceptance(candidate, frontier)

    assert decision.accepted is True
    assert decision.reason == "reward_improved"
    assert decision.diagnostics["authority"] == "benchmark_only"


def test_accepts_recovered_comparable_when_reward_improves() -> None:
    frontier = _run("frontier", reward_avg=0.4, loss_avg=0.9)
    candidate = _run(
        "candidate",
        status="oom_recovered",
        valid=True,
        reward_avg=0.45,
        oom_events=1,
    )

    decision = decide_acceptance(candidate, frontier)

    assert decision.accepted is True
    assert decision.candidate_classification.comparability == "recovered_comparable"


def test_rejects_effective_batch_mismatch_and_non_improvement() -> None:
    frontier = _run("frontier", reward_avg=0.4, loss_avg=0.9)
    mismatched = _run("candidate-a", reward_avg=0.5, effective_batch=8)
    weaker = _run("candidate-b", reward_avg=0.39, loss_avg=0.7)

    mismatch_decision = decide_acceptance(mismatched, frontier)
    weaker_decision = decide_acceptance(weaker, frontier)

    assert mismatch_decision.accepted is False
    assert mismatch_decision.reason == "effective_batch_mismatch"
    assert weaker_decision.accepted is False
    assert weaker_decision.reason == "benchmark_not_better"
