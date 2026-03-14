import json
from pathlib import Path
from typing import cast

from optimizer.records import JsonValue, load_json_record
from optimizer.reducer import parse_attempt_markdown, reduce_attempt


def _benchmark_report(frontier_target: str, candidate_target: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generated_at": "20260314T120000Z",
        "runs": [
            {
                "input": frontier_target,
                "label": frontier_target,
                "commit": "abc123",
                "status": "ok",
                "valid": True,
                "steps_requested": 10,
                "steps_observed": 10,
                "tokens_per_sec": 100.0,
                "time_avg_s": 1.2,
                "reward_avg": 0.40,
                "loss_avg": 0.90,
                "effective_batch": 16,
                "oom_events": 0,
            },
            {
                "input": candidate_target,
                "label": candidate_target,
                "commit": "def456",
                "status": "ok",
                "valid": True,
                "steps_requested": 10,
                "steps_observed": 10,
                "tokens_per_sec": 140.0,
                "time_avg_s": 1.0,
                "reward_avg": 0.50,
                "loss_avg": 0.80,
                "effective_batch": 16,
                "oom_events": 0,
            },
        ],
    }


def test_parse_attempt_markdown_reads_minimal_contract(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.md"
    attempt.write_text(
        "# Optimization Attempt\n\n"
        "## Frontier\n\n"
        "- Frontier target: `main`\n"
        "- Candidate id: `cand-1`\n"
        "- Candidate worktree: `/tmp/cand-1`\n"
        "- Selected target: `generation_general`\n\n"
        "## Hypothesis\n\n"
        "Reducing generation overhead should improve throughput.\n\n"
        "Change summary: tighten decode path\n",
        encoding="utf-8",
    )

    parsed = parse_attempt_markdown(attempt)

    assert parsed.frontier_target == "main"
    assert parsed.candidate_id == "cand-1"
    assert parsed.worktree_path == "/tmp/cand-1"
    assert parsed.selected_target == "generation_general"
    assert parsed.change_summary == "tighten decode path"


def test_reducer_updates_frontier_ledger_and_plot(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "optimizer-artifacts"
    benchmark_dir = tmp_path / "benchmarks"
    reports_dir = artifacts_dir.parent / "reports"
    benchmark_dir.mkdir(parents=True)
    candidate_worktree = tmp_path / "candidate-worktree"
    candidate_worktree.mkdir(parents=True)

    attempt = tmp_path / "attempt.md"
    attempt.write_text(
        "# Optimization Attempt\n\n"
        "## Frontier\n\n"
        "- Frontier target: `main`\n"
        "- Candidate id: `cand-1`\n"
        f"- Candidate worktree: `{candidate_worktree}`\n"
        "- Selected target: `generation_general`\n\n"
        "## Hypothesis\n\n"
        "Reducing generation overhead should improve throughput.\n\n"
        "Change summary: tighten decode path\n",
        encoding="utf-8",
    )
    report_path = benchmark_dir / "compare.json"
    report_path.write_text(
        json.dumps(_benchmark_report("main", str(candidate_worktree))),
        encoding="utf-8",
    )

    result = reduce_attempt(
        attempt_markdown_path=attempt,
        benchmark_report_path=report_path,
        artifacts_dir=artifacts_dir,
        allow_recovered_oom_promotion=True,
    )

    assert result["accepted"] is True
    frontier = cast(
        dict[str, object],
        load_json_record(artifacts_dir / "frontier" / "frontier_state.json"),
    )
    current = cast(dict[str, object], frontier["current"])
    assert current["candidate_id"] == "cand-1"
    ledger_lines = [
        line
        for line in (artifacts_dir / "experiments" / "experiment_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(ledger_lines) == 1
    ledger = cast(dict[str, JsonValue], json.loads(ledger_lines[0]))
    assert ledger["outcome"] == "kept"
    assert (
        ledger["hypothesis"]
        == "Reducing generation overhead should improve throughput."
    )
    assert ledger["change_summary"] == "tighten decode path"
    assert (reports_dir / "tokens_per_sec.svg").exists()
    assert (reports_dir / "tokens_per_sec.md").exists()


def test_reducer_rejects_non_sequential_report(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "optimizer-artifacts"
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    candidate_worktree = tmp_path / "candidate-worktree"
    candidate_worktree.mkdir(parents=True)
    attempt = tmp_path / "attempt.md"
    attempt.write_text(
        "# Optimization Attempt\n\n"
        "## Frontier\n\n"
        "- Frontier target: `main`\n"
        "- Candidate id: `cand-1`\n"
        f"- Candidate worktree: `{candidate_worktree}`\n\n"
        "## Hypothesis\n\n"
        "One hypothesis.\n\n"
        "Change summary: one change\n",
        encoding="utf-8",
    )
    report = _benchmark_report("main", str(candidate_worktree))
    cast(list[object], report["runs"]).append(
        {
            "input": "extra",
            "label": "extra",
            "commit": "ghi789",
            "status": "ok",
            "valid": True,
            "steps_requested": 10,
            "steps_observed": 10,
            "tokens_per_sec": 50.0,
            "reward_avg": 0.2,
            "loss_avg": 1.0,
            "effective_batch": 16,
            "oom_events": 0,
        }
    )
    report_path = benchmark_dir / "compare.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    try:
        _ = reduce_attempt(
            attempt_markdown_path=attempt,
            benchmark_report_path=report_path,
            artifacts_dir=artifacts_dir,
            allow_recovered_oom_promotion=True,
        )
    except ValueError as error:
        assert "exactly one frontier and one candidate run" in str(error)
    else:
        raise AssertionError("Expected reducer to reject non-sequential report.")


def test_reducer_sanitizes_decision_filename_for_slash_candidate_ids(
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / "optimizer-artifacts"
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    candidate_worktree = tmp_path / "candidate-worktree"
    candidate_worktree.mkdir(parents=True)
    attempt = tmp_path / "attempt.md"
    attempt.write_text(
        "# Optimization Attempt\n\n"
        "## Frontier\n\n"
        "- Frontier target: `main`\n"
        "- Candidate id: `feat/example`\n"
        f"- Candidate worktree: `{candidate_worktree}`\n\n"
        "## Hypothesis\n\n"
        "One hypothesis.\n\n"
        "Change summary: one change\n",
        encoding="utf-8",
    )
    report_path = benchmark_dir / "compare.json"
    report_path.write_text(
        json.dumps(_benchmark_report("main", str(candidate_worktree))),
        encoding="utf-8",
    )

    _ = reduce_attempt(
        attempt_markdown_path=attempt,
        benchmark_report_path=report_path,
        artifacts_dir=artifacts_dir,
        allow_recovered_oom_promotion=True,
    )

    decision_files = list((artifacts_dir / "decisions").glob("*.json"))
    assert len(decision_files) == 1
    assert decision_files[0].name.endswith("feat-example.json")
