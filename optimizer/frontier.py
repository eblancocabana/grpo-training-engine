from __future__ import annotations

from dataclasses import dataclass, field

from optimizer.evaluation.acceptance import AcceptanceDecision
from optimizer.evaluation.benchmark_gate import classify_run
from optimizer.records import BenchmarkRunRecord


@dataclass(frozen=True)
class FrontierEntry:
    candidate_id: str
    target: str
    benchmark: BenchmarkRunRecord
    comparability: str

    @classmethod
    def from_run(
        cls,
        *,
        candidate_id: str,
        target: str,
        benchmark: BenchmarkRunRecord,
        allow_recovered_oom_promotion: bool = True,
    ) -> "FrontierEntry":
        return cls(
            candidate_id=candidate_id,
            target=target,
            benchmark=benchmark,
            comparability=classify_run(
                benchmark,
                allow_recovered_oom_promotion=allow_recovered_oom_promotion,
            ).comparability,
        )


@dataclass(frozen=True)
class FrontierTransition:
    accepted: bool
    previous_candidate_id: str | None
    current_candidate_id: str
    reason: str


@dataclass
class Frontier:
    current: FrontierEntry | None = None
    history: list[FrontierTransition] = field(default_factory=list)

    def seed(self, entry: FrontierEntry) -> FrontierTransition:
        self.current = entry
        transition = FrontierTransition(
            accepted=True,
            previous_candidate_id=None,
            current_candidate_id=entry.candidate_id,
            reason="frontier_seeded",
        )
        self.history.append(transition)
        return transition

    def apply_decision(
        self,
        *,
        candidate_id: str,
        target: str,
        benchmark: BenchmarkRunRecord,
        decision: AcceptanceDecision,
    ) -> FrontierTransition:
        if self.current is None:
            raise ValueError("Frontier must be seeded before applying decisions.")

        previous_candidate_id = self.current.candidate_id
        current_candidate_id = previous_candidate_id
        if decision.accepted:
            self.current = FrontierEntry.from_run(
                candidate_id=candidate_id,
                target=target,
                benchmark=benchmark,
            )
            current_candidate_id = candidate_id

        transition = FrontierTransition(
            accepted=decision.accepted,
            previous_candidate_id=previous_candidate_id,
            current_candidate_id=current_candidate_id,
            reason=decision.reason,
        )
        self.history.append(transition)
        return transition
