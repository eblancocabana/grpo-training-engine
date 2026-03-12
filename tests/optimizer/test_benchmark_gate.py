from pathlib import Path

from optimizer.evaluation.benchmark_gate import classify_run
from optimizer.records import BenchmarkRunRecord, JsonValue


def _run(**overrides: JsonValue) -> BenchmarkRunRecord:
    payload: dict[str, JsonValue] = {
        "input": "candidate",
        "label": "candidate",
        "commit": "abc123",
        "status": "ok",
        "valid": True,
        "steps_requested": 5,
        "steps_observed": 5,
        "reward_avg": 0.4,
        "loss_avg": 0.9,
        "effective_batch": 16,
        "oom_events": 0,
    }
    payload.update(overrides)
    return BenchmarkRunRecord.from_dict(payload)


def test_classifies_ok_run_as_clean_comparable() -> None:
    classification = classify_run(_run(status="ok", valid=True))

    assert classification.comparability == "clean_comparable"
    assert classification.promotable is True
    assert classification.complete is True


def test_classifies_completed_oom_recovery_as_recovered_comparable() -> None:
    classification = classify_run(
        _run(status="oom_recovered", valid=True, oom_events=2)
    )

    assert classification.comparability == "recovered_comparable"
    assert classification.promotable is True
    assert "oom_backoff_recovered" in classification.reasons


def test_rejects_incomplete_or_missing_metrics() -> None:
    incomplete = classify_run(_run(steps_observed=3))
    missing_batch = classify_run(_run(effective_batch=None))
    invalid = classify_run(_run(valid=False))
    incomplete_recovered = classify_run(
        _run(status="oom_recovered", valid=True, oom_events=1, steps_observed=3)
    )

    assert incomplete.comparability == "not_comparable"
    assert "benchmark_incomplete" in incomplete.reasons
    assert missing_batch.comparability == "not_comparable"
    assert "effective_batch_missing" in missing_batch.reasons
    assert invalid.comparability == "not_comparable"
    assert "benchmark_invalid" in invalid.reasons
    assert incomplete_recovered.comparability == "not_comparable"
    assert "oom_backoff_recovered" in incomplete_recovered.reasons
    assert "benchmark_incomplete" in incomplete_recovered.reasons


def test_compare_bench_script_marks_complete_recovered_oom_runs_valid() -> None:
    script = Path(__file__).resolve().parents[2] / "tools" / "compare_bench.sh"
    content = script.read_text(encoding="utf-8")

    assert 'return "oom_recovered", True, None, parsed["terminal_error"]' in content
