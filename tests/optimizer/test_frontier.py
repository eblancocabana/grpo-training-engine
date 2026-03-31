from optimizer.evaluation.acceptance import decide_acceptance
from optimizer.frontier import Frontier, FrontierEntry
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
        "time_avg_s": 1.0,
        "reward_avg": 0.4,
        "loss_avg": 0.9,
        "effective_batch": 16,
        "oom_events": 0,
        "triton_mode": "on",
        "triton_arg": "--use-triton",
    }
    payload.update(overrides)
    return BenchmarkRunRecord.from_dict(payload)


def test_frontier_promotes_on_acceptance() -> None:
    frontier = Frontier()
    _ = frontier.seed(
        FrontierEntry.from_run(
            candidate_id="baseline",
            target="main",
            benchmark=_run("main", tokens_per_sec=10.0),
        )
    )
    current = frontier.current
    assert current is not None

    candidate_run = _run("candidate", tokens_per_sec=12.0, time_avg_s=0.9)
    decision = decide_acceptance(candidate_run, current.benchmark)
    transition = frontier.apply_decision(
        candidate_id="candidate-1",
        target="feat/candidate",
        benchmark=candidate_run,
        decision=decision,
    )

    assert transition.accepted is True
    assert frontier.current is not None
    assert frontier.current.candidate_id == "candidate-1"
    assert frontier.history[-1].reason == "step_time_improved"


def test_frontier_holds_incumbent_on_rejection() -> None:
    frontier = Frontier()
    _ = frontier.seed(
        FrontierEntry.from_run(
            candidate_id="baseline",
            target="main",
            benchmark=_run("main", tokens_per_sec=10.0),
        )
    )
    current = frontier.current
    assert current is not None

    candidate_run = _run("candidate", tokens_per_sec=9.0, time_avg_s=1.1)
    decision = decide_acceptance(candidate_run, current.benchmark)
    transition = frontier.apply_decision(
        candidate_id="candidate-2",
        target="feat/weaker",
        benchmark=candidate_run,
        decision=decision,
    )

    assert transition.accepted is False
    assert frontier.current is not None
    assert frontier.current.candidate_id == "baseline"
    assert transition.current_candidate_id == "baseline"
