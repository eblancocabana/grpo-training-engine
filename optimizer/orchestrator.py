from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass
from pathlib import Path

from optimizer.campaigns.generation_hf import GenerationHFCampaign
from optimizer.evaluation.acceptance import decide_acceptance
from optimizer.frontier import Frontier, FrontierEntry, FrontierTransition
from optimizer.records import (
    BenchmarkComparisonRecord,
    BenchmarkRunRecord,
    DecisionRecord,
    JsonValue,
    utc_timestamp,
    write_json_record,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target: str


class BenchmarkCampaign(Protocol):
    name: str

    def run(
        self, *, frontier_target: str, candidate_target: str
    ) -> BenchmarkComparisonRecord: ...


class SequentialOptimizerOrchestrator:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        artifacts_dir: str | Path | None = None,
        campaign: BenchmarkCampaign | None = None,
        frontier: Frontier | None = None,
        allow_recovered_oom_promotion: bool = True,
    ) -> None:
        self.repo_root: Path = Path(repo_root).resolve()
        self.artifacts_dir: Path = (
            Path(artifacts_dir).resolve()
            if artifacts_dir is not None
            else (self.repo_root / "optimizer" / "artifacts" / "records").resolve()
        )
        self.campaign: BenchmarkCampaign = campaign or GenerationHFCampaign(
            repo_root=self.repo_root
        )
        self.frontier: Frontier = frontier or Frontier()
        self._evaluation_active: bool = False
        self.allow_recovered_oom_promotion = allow_recovered_oom_promotion

    def seed_frontier(
        self,
        *,
        frontier_id: str,
        frontier_target: str,
        frontier_run: BenchmarkRunRecord,
    ) -> FrontierTransition:
        entry = FrontierEntry.from_run(
            candidate_id=frontier_id,
            target=frontier_target,
            benchmark=frontier_run,
            allow_recovered_oom_promotion=self.allow_recovered_oom_promotion,
        )
        return self.frontier.seed(entry)

    def evaluate_candidate(self, candidate: CandidateSpec) -> DecisionRecord:
        if self.frontier.current is None:
            raise ValueError("Frontier must be seeded before evaluating a candidate.")
        if self._evaluation_active:
            raise RuntimeError(
                "Sequential optimizer cannot evaluate multiple candidates at once."
            )

        self._evaluation_active = True
        try:
            frontier_entry = self.frontier.current
            benchmark_comparison = self.campaign.run(
                frontier_target=frontier_entry.target,
                candidate_target=candidate.target,
            )
            self._enforce_sequential_report(
                benchmark_comparison, frontier_entry.target, candidate.target
            )

            frontier_run = benchmark_comparison.require_run(frontier_entry.target)
            candidate_run = benchmark_comparison.require_run(candidate.target)
            acceptance = decide_acceptance(
                candidate_run,
                frontier_run,
                allow_recovered_oom_promotion=self.allow_recovered_oom_promotion,
            )
            transition = self.frontier.apply_decision(
                candidate_id=candidate.candidate_id,
                target=candidate.target,
                benchmark=candidate_run,
                decision=acceptance,
            )

            decision_id = utc_timestamp()
            benchmark_report_path = write_json_record(
                self.artifacts_dir
                / "benchmarks"
                / f"{decision_id}_{self.campaign.name}.json",
                benchmark_comparison,
            )
            frontier_state_path = write_json_record(
                self.artifacts_dir / "frontier" / "frontier_state.json",
                self._frontier_payload(),
            )

            decision = DecisionRecord(
                decision_id=decision_id,
                campaign=self.campaign.name,
                candidate_id=candidate.candidate_id,
                candidate_target=candidate.target,
                frontier_id=frontier_entry.candidate_id,
                frontier_target=frontier_entry.target,
                accepted=acceptance.accepted,
                reason=acceptance.reason,
                candidate_status=candidate_run.status,
                frontier_status=frontier_run.status,
                candidate_comparability=acceptance.candidate_classification.comparability,
                frontier_comparability=acceptance.frontier_classification.comparability,
                reward_delta=acceptance.reward_delta,
                loss_delta=acceptance.loss_delta,
                benchmark_report_path=str(benchmark_report_path),
                frontier_state_path=str(frontier_state_path),
                diagnostics={
                    **acceptance.diagnostics,
                    "frontier_transition_reason": transition.reason,
                    "sequential_only": True,
                },
            )
            _ = write_json_record(
                self.artifacts_dir
                / "decisions"
                / f"{decision_id}_{candidate.candidate_id}.json",
                decision,
            )
            return decision
        finally:
            self._evaluation_active = False

    def _enforce_sequential_report(
        self,
        comparison: BenchmarkComparisonRecord,
        frontier_target: str,
        candidate_target: str,
    ) -> None:
        targets = {run.input for run in comparison.runs}
        expected_targets = {frontier_target, candidate_target}
        if targets != expected_targets or len(comparison.runs) != 2:
            message = (
                "Sequential optimizer expects exactly one frontier and one candidate run. "
                f"Expected {expected_targets}, got {targets}."
            )
            raise ValueError(message)

    def _frontier_payload(self) -> dict[str, JsonValue]:
        current = self.frontier.current
        return {
            "sequential_only": True,
            "campaign": self.campaign.name,
            "current": None
            if current is None
            else {
                "candidate_id": current.candidate_id,
                "target": current.target,
                "comparability": current.comparability,
                "benchmark": current.benchmark.to_dict(),
            },
            "history": [
                {
                    "accepted": transition.accepted,
                    "previous_candidate_id": transition.previous_candidate_id,
                    "current_candidate_id": transition.current_candidate_id,
                    "reason": transition.reason,
                }
                for transition in self.frontier.history
            ],
        }
