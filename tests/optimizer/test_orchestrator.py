import json
from pathlib import Path
from typing import cast

from optimizer.frontier import Frontier
from optimizer.orchestrator import (
    BenchmarkCampaign,
    CandidateSpec,
    SequentialOptimizerOrchestrator,
)
from optimizer.records import BenchmarkComparisonRecord, BenchmarkRunRecord, JsonValue


class StubCampaign:
    name: str = "generation_hf"

    def __init__(self, comparison: BenchmarkComparisonRecord) -> None:
        self.comparison: BenchmarkComparisonRecord = comparison
        self.calls: list[tuple[str, str]] = []

    def run(
        self, *, frontier_target: str, candidate_target: str
    ) -> BenchmarkComparisonRecord:
        self.calls.append((frontier_target, candidate_target))
        return self.comparison


def _run(name: str, **overrides: JsonValue) -> BenchmarkRunRecord:
    payload: dict[str, JsonValue] = {
        "input": name,
        "label": name,
        "commit": "abc123",
        "status": "ok",
        "valid": True,
        "steps_requested": 5,
        "steps_observed": 5,
        "tokens_per_sec": 100.0,
        "reward_avg": 0.4,
        "loss_avg": 0.9,
        "effective_batch": 16,
        "oom_events": 0,
    }
    payload.update(overrides)
    return BenchmarkRunRecord.from_dict(payload)


def test_orchestrator_runs_single_candidate_and_updates_frontier(
    tmp_path: Path,
) -> None:
    baseline_run = _run("main", reward_avg=0.4, loss_avg=0.9)
    candidate_run = _run(
        "feat/candidate",
        status="oom_recovered",
        valid=True,
        reward_avg=0.5,
        loss_avg=0.95,
        oom_events=1,
    )
    comparison = BenchmarkComparisonRecord(
        schema_version=1,
        generated_at="20260312T120000Z",
        runs=[baseline_run, candidate_run],
    )
    campaign_stub = StubCampaign(comparison)
    orchestrator = SequentialOptimizerOrchestrator(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
        campaign=cast(BenchmarkCampaign, campaign_stub),
        frontier=Frontier(),
        allow_recovered_oom_promotion=True,
    )

    _ = orchestrator.seed_frontier(
        frontier_id="baseline",
        frontier_target="main",
        frontier_run=baseline_run,
    )
    decision = orchestrator.evaluate_candidate(
        CandidateSpec(
            candidate_id="candidate-1",
            target="feat/candidate",
            change_summary="tighten decode path",
        )
    )

    assert campaign_stub.calls == [("main", "feat/candidate")]
    assert decision.accepted is True
    assert decision.candidate_comparability == "recovered_comparable"
    assert orchestrator.frontier.current is not None
    assert orchestrator.frontier.current.candidate_id == "candidate-1"

    decision_files = list(
        (tmp_path / "optimizer-artifacts" / "decisions").glob("*.json")
    )
    assert len(decision_files) == 1
    payload = cast(
        dict[str, JsonValue],
        json.loads(decision_files[0].read_text(encoding="utf-8")),
    )
    diagnostics = cast(dict[str, JsonValue], payload["diagnostics"])
    assert diagnostics["sequential_only"] is True

    ledger_path = (
        tmp_path / "optimizer-artifacts" / "experiments" / "experiment_ledger.jsonl"
    )
    lines = [
        line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 1
    ledger_payload = cast(dict[str, JsonValue], json.loads(lines[0]))
    assert ledger_payload["experiment_number"] == 1
    assert ledger_payload["outcome"] == "kept"
    assert ledger_payload["primary_metric"] == "tokens_per_sec"
    assert ledger_payload["change_summary"] == "tighten decode path"
    assert ledger_payload["running_best_tokens_per_sec"] == 100.0
    assert ledger_payload["running_best_experiment_number"] == 1
    assert ledger_payload["incumbent_tokens_per_sec_after_decision"] == 100.0
    assert ledger_payload["sequential_only"] is True
    baseline_snapshot = cast(dict[str, JsonValue], ledger_payload["baseline_snapshot"])
    candidate_snapshot = cast(
        dict[str, JsonValue], ledger_payload["candidate_snapshot"]
    )
    assert baseline_snapshot["tokens_per_sec"] == 100.0
    assert candidate_snapshot["tokens_per_sec"] == 100.0
    assert Path(cast(str, ledger_payload["benchmark_report_path"])).exists()
    assert Path(cast(str, ledger_payload["frontier_state_path"])).exists()
    assert Path(cast(str, ledger_payload["decision_path"])).exists()


def test_orchestrator_rejects_non_sequential_report(tmp_path: Path) -> None:
    baseline_run = _run("main")
    candidate_run = _run("feat/candidate", reward_avg=0.5)
    extra_run = _run("feat/extra", reward_avg=0.6)
    campaign_stub = StubCampaign(
        BenchmarkComparisonRecord(
            schema_version=1,
            generated_at="20260312T120000Z",
            runs=[baseline_run, candidate_run, extra_run],
        )
    )
    campaign = cast(
        BenchmarkCampaign,
        campaign_stub,
    )
    orchestrator = SequentialOptimizerOrchestrator(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
        campaign=campaign,
        frontier=Frontier(),
        allow_recovered_oom_promotion=True,
    )

    _ = orchestrator.seed_frontier(
        frontier_id="baseline",
        frontier_target="main",
        frontier_run=baseline_run,
    )

    try:
        _ = orchestrator.evaluate_candidate(
            CandidateSpec(
                candidate_id="candidate-1",
                target="feat/candidate",
                change_summary="bad extra run",
            )
        )
    except ValueError as error:
        assert "exactly one frontier and one candidate run" in str(error)
    else:
        raise AssertionError(
            "Expected sequential-only enforcement to reject extra runs."
        )


def test_orchestrator_rejects_reentrant_evaluation(tmp_path: Path) -> None:
    baseline_run = _run("main", reward_avg=0.4, loss_avg=0.9)
    candidate_run = _run("feat/candidate", reward_avg=0.5, loss_avg=0.8)
    comparison = BenchmarkComparisonRecord(
        schema_version=1,
        generated_at="20260312T120000Z",
        runs=[baseline_run, candidate_run],
    )
    campaign_stub = StubCampaign(comparison)
    orchestrator = SequentialOptimizerOrchestrator(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
        campaign=cast(BenchmarkCampaign, campaign_stub),
        frontier=Frontier(),
        allow_recovered_oom_promotion=True,
    )
    _ = orchestrator.seed_frontier(
        frontier_id="baseline",
        frontier_target="main",
        frontier_run=baseline_run,
    )
    setattr(orchestrator, "_evaluation_active", True)

    try:
        _ = orchestrator.evaluate_candidate(
            CandidateSpec(
                candidate_id="candidate-1",
                target="feat/candidate",
                change_summary="reentrant attempt",
            )
        )
    except RuntimeError as error:
        assert "cannot evaluate multiple candidates at once" in str(error)
    else:
        raise AssertionError("Expected reentrant evaluation to be rejected.")


def test_orchestrator_promotes_recovered_comparable_candidate(tmp_path: Path) -> None:
    baseline_run = _run("main", reward_avg=0.4, loss_avg=0.9)
    candidate_run = _run(
        "feat/recovered",
        status="oom_recovered",
        valid=True,
        reward_avg=0.45,
        loss_avg=1.0,
        oom_events=2,
    )
    comparison = BenchmarkComparisonRecord(
        schema_version=1,
        generated_at="20260312T120000Z",
        runs=[baseline_run, candidate_run],
    )
    campaign_stub = StubCampaign(comparison)
    orchestrator = SequentialOptimizerOrchestrator(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
        campaign=cast(BenchmarkCampaign, campaign_stub),
        frontier=Frontier(),
        allow_recovered_oom_promotion=True,
    )

    _ = orchestrator.seed_frontier(
        frontier_id="baseline",
        frontier_target="main",
        frontier_run=baseline_run,
    )
    decision = orchestrator.evaluate_candidate(
        CandidateSpec(
            candidate_id="candidate-recovered",
            target="feat/recovered",
            change_summary="recover with same batch",
        )
    )

    assert decision.accepted is True
    assert decision.reason == "reward_improved"
    assert decision.candidate_status == "oom_recovered"
    assert decision.candidate_comparability == "recovered_comparable"
    assert orchestrator.frontier.current is not None
    assert orchestrator.frontier.current.candidate_id == "candidate-recovered"


def test_orchestrator_can_disable_recovered_oom_promotion(tmp_path: Path) -> None:
    baseline_run = _run("main", reward_avg=0.4, loss_avg=0.9)
    candidate_run = _run(
        "feat/recovered",
        status="oom_recovered",
        valid=False,
        reward_avg=0.45,
        loss_avg=1.0,
        oom_events=2,
    )
    comparison = BenchmarkComparisonRecord(
        schema_version=1,
        generated_at="20260312T120000Z",
        runs=[baseline_run, candidate_run],
    )
    campaign_stub = StubCampaign(comparison)
    orchestrator = SequentialOptimizerOrchestrator(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
        campaign=cast(BenchmarkCampaign, campaign_stub),
        frontier=Frontier(),
        allow_recovered_oom_promotion=False,
    )

    _ = orchestrator.seed_frontier(
        frontier_id="baseline",
        frontier_target="main",
        frontier_run=baseline_run,
    )
    decision = orchestrator.evaluate_candidate(
        CandidateSpec(
            candidate_id="candidate-recovered",
            target="feat/recovered",
            change_summary="disable recovered promotion",
        )
    )

    assert decision.accepted is False
    assert decision.reason == "candidate_not_comparable"


def test_orchestrator_appends_discarded_experiment_and_preserves_running_best(
    tmp_path: Path,
) -> None:
    baseline_run = _run("main", reward_avg=0.4, loss_avg=0.9, tokens_per_sec=90.0)
    kept_candidate_run = _run(
        "feat/kept",
        reward_avg=0.5,
        loss_avg=0.85,
        tokens_per_sec=120.0,
    )
    discarded_candidate_run = _run(
        "feat/discarded",
        reward_avg=0.3,
        loss_avg=0.95,
        tokens_per_sec=180.0,
    )
    first_campaign = StubCampaign(
        BenchmarkComparisonRecord(
            schema_version=1,
            generated_at="20260312T120000Z",
            runs=[baseline_run, kept_candidate_run],
        )
    )
    orchestrator = SequentialOptimizerOrchestrator(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
        campaign=cast(BenchmarkCampaign, first_campaign),
        frontier=Frontier(),
        allow_recovered_oom_promotion=True,
    )
    _ = orchestrator.seed_frontier(
        frontier_id="baseline",
        frontier_target="main",
        frontier_run=baseline_run,
    )
    _ = orchestrator.evaluate_candidate(
        CandidateSpec(
            candidate_id="candidate-kept",
            target="feat/kept",
            change_summary="improve kept throughput",
        )
    )

    orchestrator.campaign = cast(
        BenchmarkCampaign,
        StubCampaign(
            BenchmarkComparisonRecord(
                schema_version=1,
                generated_at="20260312T120100Z",
                runs=[kept_candidate_run, discarded_candidate_run],
            )
        ),
    )
    decision = orchestrator.evaluate_candidate(
        CandidateSpec(
            candidate_id="candidate-discarded",
            target="feat/discarded",
            change_summary="overfit throughput only",
        )
    )

    assert decision.accepted is False
    ledger_path = (
        tmp_path / "optimizer-artifacts" / "experiments" / "experiment_ledger.jsonl"
    )
    lines = [
        line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 2
    second_payload = cast(dict[str, JsonValue], json.loads(lines[1]))
    assert second_payload["experiment_number"] == 2
    assert second_payload["outcome"] == "discarded"
    assert second_payload["change_summary"] == "overfit throughput only"
    assert second_payload["incumbent_tokens_per_sec_after_decision"] == 120.0
    assert second_payload["running_best_tokens_per_sec"] == 120.0
    assert second_payload["running_best_experiment_number"] == 1
    baseline_snapshot = cast(dict[str, JsonValue], second_payload["baseline_snapshot"])
    candidate_snapshot = cast(
        dict[str, JsonValue], second_payload["candidate_snapshot"]
    )
    assert baseline_snapshot["tokens_per_sec"] == 120.0
    assert candidate_snapshot["tokens_per_sec"] == 180.0
