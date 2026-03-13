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
    ExperimentLedgerRecord,
    ExperimentSnapshotRecord,
    JsonValue,
    append_jsonl_record,
    load_json_record,
    load_last_jsonl_record,
    utc_timestamp,
    write_json_record,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target: str
    change_summary: str


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
        self.allow_recovered_oom_promotion: bool = allow_recovered_oom_promotion

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

    def restore_frontier(self, path: str | Path) -> FrontierTransition:
        payload = load_json_record(path)
        current_obj = payload.get("current")
        if not isinstance(current_obj, dict):
            raise ValueError(
                "Frontier state does not contain a current frontier entry."
            )
        benchmark_obj = current_obj.get("benchmark")
        if isinstance(benchmark_obj, dict):
            benchmark = BenchmarkRunRecord.from_dict(
                {str(key): item for key, item in benchmark_obj.items()}
            )
        else:
            # Legacy format: construct benchmark from direct fields
            tokens_per_sec = current_obj.get("tokens_per_sec")
            vram_peak_gb = current_obj.get("vram_peak_gb")
            if tokens_per_sec is None or vram_peak_gb is None:
                raise ValueError(
                    "Frontier state current entry must include either 'benchmark' or 'tokens_per_sec' and 'vram_peak_gb' fields."
                )
            benchmark = BenchmarkRunRecord(
                input=target,
                label=target,
                commit="unknown",
                triton_mode="auto",
                status="ok",
                valid=True,
                steps_requested=5,
                steps_observed=5,
                tokens_per_sec=float(tokens_per_sec),
                vram_peak_gb=float(vram_peak_gb),
            )
        candidate_id = current_obj.get("candidate_id")
        target = current_obj.get("target")
        comparability = current_obj.get("comparability")
        if not isinstance(candidate_id, str):
            raise ValueError("Frontier state current candidate_id must be a string.")
        if not isinstance(target, str):
            raise ValueError("Frontier state current target must be a string.")
        if not isinstance(comparability, str):
            raise ValueError("Frontier state current comparability must be a string.")
        entry = FrontierEntry(
            candidate_id=candidate_id,
            target=target,
            benchmark=benchmark,
            comparability=comparability,
        )
        transition = FrontierTransition(
            accepted=True,
            previous_candidate_id=None,
            current_candidate_id=entry.candidate_id,
            reason="frontier_restored",
        )
        self.frontier.current = entry
        self.frontier.history = [transition]
        return transition

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
                change_summary=candidate.change_summary,
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
            decision_path = write_json_record(
                self.artifacts_dir
                / "decisions"
                / f"{decision_id}_{candidate.candidate_id}.json",
                decision,
            )
            experiment_ledger = self._build_experiment_ledger(
                decision=decision,
                decision_path=decision_path,
                frontier_entry=frontier_entry,
                frontier_run=frontier_run,
                candidate_run=candidate_run,
            )
            _ = append_jsonl_record(
                self.artifacts_dir / "experiments" / "experiment_ledger.jsonl",
                experiment_ledger,
            )
            return decision
        finally:
            self._evaluation_active = False

    def _build_experiment_ledger(
        self,
        *,
        decision: DecisionRecord,
        decision_path: Path,
        frontier_entry: FrontierEntry,
        frontier_run: BenchmarkRunRecord,
        candidate_run: BenchmarkRunRecord,
    ) -> ExperimentLedgerRecord:
        ledger_path = self.artifacts_dir / "experiments" / "experiment_ledger.jsonl"
        last_record = load_last_jsonl_record(ledger_path)
        previous_experiment_number = 0
        running_best_tokens_per_sec = None
        running_best_experiment_number = None
        if last_record is not None:
            previous_experiment_number = (
                _maybe_int(last_record.get("experiment_number")) or 0
            )
            running_best_tokens_per_sec = _maybe_float(
                last_record.get("running_best_tokens_per_sec")
            )
            running_best_experiment_number = _maybe_int(
                last_record.get("running_best_experiment_number")
            )

        experiment_number = previous_experiment_number + 1
        outcome = "kept" if decision.accepted else "discarded"
        baseline_snapshot = self._build_snapshot(
            target=frontier_entry.target,
            benchmark=frontier_run,
            comparability=decision.frontier_comparability,
        )
        candidate_snapshot = self._build_snapshot(
            target=decision.candidate_target,
            benchmark=candidate_run,
            comparability=decision.candidate_comparability,
        )
        incumbent_tokens_per_sec_after_decision = (
            candidate_run.tokens_per_sec
            if decision.accepted
            else frontier_run.tokens_per_sec
        )
        if incumbent_tokens_per_sec_after_decision is not None and (
            running_best_tokens_per_sec is None
            or incumbent_tokens_per_sec_after_decision > running_best_tokens_per_sec
        ):
            running_best_tokens_per_sec = incumbent_tokens_per_sec_after_decision
            running_best_experiment_number = experiment_number

        return ExperimentLedgerRecord(
            experiment_number=experiment_number,
            decision_id=decision.decision_id,
            campaign=decision.campaign,
            change_summary=decision.change_summary,
            candidate_id=decision.candidate_id,
            candidate_target=decision.candidate_target,
            frontier_id=decision.frontier_id,
            frontier_target=decision.frontier_target,
            outcome=outcome,
            reason=decision.reason,
            primary_metric="tokens_per_sec",
            baseline_snapshot=baseline_snapshot,
            candidate_snapshot=candidate_snapshot,
            incumbent_tokens_per_sec_after_decision=incumbent_tokens_per_sec_after_decision,
            running_best_tokens_per_sec=running_best_tokens_per_sec,
            running_best_experiment_number=running_best_experiment_number,
            benchmark_report_path=decision.benchmark_report_path,
            frontier_state_path=decision.frontier_state_path,
            decision_path=str(decision_path),
            sequential_only=True,
        )

    def _build_snapshot(
        self,
        *,
        target: str,
        benchmark: BenchmarkRunRecord,
        comparability: str,
    ) -> ExperimentSnapshotRecord:
        return ExperimentSnapshotRecord(
            target=target,
            status=benchmark.status,
            comparability=comparability,
            tokens_per_sec=benchmark.tokens_per_sec,
            reward_avg=benchmark.reward_avg,
            loss_avg=benchmark.loss_avg,
            vram_peak_gb=benchmark.vram_peak_gb,
            oom_events=benchmark.oom_events,
            effective_batch=benchmark.effective_batch,
        )

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


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"Expected float-compatible value, got {type(value)!r}.")


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected int-compatible value, got {type(value)!r}.")
