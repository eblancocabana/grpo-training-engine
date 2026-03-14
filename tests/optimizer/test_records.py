import json
from pathlib import Path

from optimizer.records import (
    BenchmarkComparisonRecord,
    BenchmarkRunRecord,
    ExperimentLedgerRecord,
    append_jsonl_record,
    load_benchmark_report,
    load_last_jsonl_record,
)


def test_benchmark_run_record_round_trips_tokens_per_sec() -> None:
    run = BenchmarkRunRecord.from_dict(
        {
            "input": "main",
            "label": "main",
            "commit": "abc123",
            "status": "ok",
            "valid": True,
            "steps_requested": 5,
            "steps_observed": 5,
            "tokens_per_sec": 123.5,
            "time_avg_s": 1.2,
            "reward_avg": 0.4,
            "loss_avg": 0.9,
            "effective_batch": 16,
            "oom_events": 0,
        }
    )

    assert run.tokens_per_sec == 123.5
    assert run.to_dict()["tokens_per_sec"] == 123.5


def test_load_benchmark_report_reads_tokens_per_sec(tmp_path: Path) -> None:
    report_path = tmp_path / "compare.json"
    _ = report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "20260312T120000Z",
                "runs": [
                    {
                        "input": "main",
                        "label": "main",
                        "commit": "abc123",
                        "status": "ok",
                        "valid": True,
                        "steps_requested": 5,
                        "steps_observed": 5,
                        "tokens_per_sec": 88.0,
                        "time_avg_s": 1.1,
                        "reward_avg": 0.4,
                        "loss_avg": 0.9,
                        "effective_batch": 16,
                        "oom_events": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = load_benchmark_report(report_path)

    assert isinstance(report, BenchmarkComparisonRecord)
    assert report.runs[0].tokens_per_sec == 88.0


def test_jsonl_helpers_preserve_latest_experiment_record(tmp_path: Path) -> None:
    ledger_path = tmp_path / "experiments" / "experiment_ledger.jsonl"
    first = ExperimentLedgerRecord.from_dict(
        {
            "experiment_number": 1,
            "decision_id": "d1",
            "campaign": "generation_hf",
            "change_summary": "first change",
            "candidate_id": "c1",
            "candidate_target": "feat/a",
            "frontier_id": "f0",
            "frontier_target": "main",
            "outcome": "kept",
            "reason": "tokens_per_sec_improved",
            "primary_metric": "tokens_per_sec",
            "baseline_snapshot": {
                "target": "main",
                "status": "ok",
                "comparability": "clean_comparable",
                "tokens_per_sec": 90.0,
                "reward_avg": 0.4,
                "loss_avg": 0.9,
                "vram_peak_gb": 6.1,
                "oom_events": 0,
                "effective_batch": 16,
            },
            "candidate_snapshot": {
                "target": "feat/a",
                "status": "ok",
                "comparability": "clean_comparable",
                "tokens_per_sec": 100.0,
                "reward_avg": 0.5,
                "loss_avg": 0.8,
                "vram_peak_gb": 6.2,
                "oom_events": 0,
                "effective_batch": 16,
            },
            "incumbent_tokens_per_sec_after_decision": 100.0,
            "running_best_tokens_per_sec": 100.0,
            "running_best_experiment_number": 1,
            "benchmark_report_path": "/tmp/bench1.json",
            "frontier_state_path": "/tmp/frontier.json",
            "decision_path": "/tmp/decision1.json",
            "sequential_only": True,
        }
    )
    second = ExperimentLedgerRecord.from_dict(
        {
            "experiment_number": 2,
            "decision_id": "d2",
            "campaign": "generation_hf",
            "change_summary": "second change",
            "candidate_id": "c2",
            "candidate_target": "feat/b",
            "frontier_id": "c1",
            "frontier_target": "feat/a",
            "outcome": "discarded",
            "reason": "tokens_per_sec_not_improved",
            "primary_metric": "tokens_per_sec",
            "baseline_snapshot": {
                "target": "feat/a",
                "status": "ok",
                "comparability": "clean_comparable",
                "tokens_per_sec": 100.0,
                "reward_avg": 0.5,
                "loss_avg": 0.8,
                "vram_peak_gb": 6.2,
                "oom_events": 0,
                "effective_batch": 16,
            },
            "candidate_snapshot": {
                "target": "feat/b",
                "status": "ok",
                "comparability": "clean_comparable",
                "tokens_per_sec": 95.0,
                "reward_avg": 0.4,
                "loss_avg": 0.85,
                "vram_peak_gb": 6.0,
                "oom_events": 0,
                "effective_batch": 16,
            },
            "incumbent_tokens_per_sec_after_decision": 100.0,
            "running_best_tokens_per_sec": 100.0,
            "running_best_experiment_number": 1,
            "benchmark_report_path": "/tmp/bench2.json",
            "frontier_state_path": "/tmp/frontier.json",
            "decision_path": "/tmp/decision2.json",
            "sequential_only": True,
        }
    )

    _ = append_jsonl_record(ledger_path, first)
    _ = append_jsonl_record(ledger_path, second)

    latest = load_last_jsonl_record(ledger_path)

    assert latest is not None
    assert latest["experiment_number"] == 2
    assert latest["change_summary"] == "second change"


def test_experiment_ledger_record_reads_legacy_row_with_flat_fields() -> None:
    record = ExperimentLedgerRecord.from_dict(
        {
            "experiment_number": 3,
            "timestamp": "20260313T124741Z",
            "campaign": "generation_general",
            "candidate_id": "feat/example-candidate-legacy",
            "change_summary": "Legacy attempt",
            "outcome": "kept",
            "reason": "tokens_per_sec_improved",
            "primary_metric": "tokens_per_sec",
            "baseline_snapshot": {
                "target": "main",
                "status": "oom_recovered",
                "tokens_per_sec": 12.15,
                "time_avg_s": 83.68,
                "loss_avg": 0.42,
                "reward_avg": 0.75,
                "oom_events": 2,
            },
            "candidate_snapshot": {
                "target": "feat/example-candidate-legacy",
                "status": "oom_recovered",
                "tokens_per_sec": 25.35,
                "time_avg_s": 47.39,
                "loss_avg": 0.17,
                "reward_avg": 0.52,
                "oom_events": 1,
            },
            "tokens_per_sec_delta": 13.2,
            "tokens_per_sec_pct_change": 108.64,
            "benchmark_report_path": "/tmp/bench.json",
        }
    )

    assert record.timestamp == "20260313T124741Z"
    assert record.decision_id == "20260313T124741Z"
    assert record.baseline_snapshot.comparability == "unknown"
    assert record.candidate_snapshot.time_avg_s == 47.39
    assert record.tokens_per_sec_delta == 13.2
