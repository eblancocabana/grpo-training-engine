from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optimizer.records import (
    MutationPlanRecord,
    ObservationRecord,
    WorktreePlanRecord,
    utc_timestamp,
)


@dataclass(frozen=True)
class AIAgentMutationPlanner:
    repo_root: Path

    def build_plan(
        self,
        *,
        frontier_target: str,
        observation: ObservationRecord,
        worktree_plan: WorktreePlanRecord,
        recent_decision_reasons: list[str],
    ) -> MutationPlanRecord:
        files_of_interest = observation.diagnostics.get("files_of_interest")
        candidate_files = (
            [str(item) for item in files_of_interest if isinstance(item, str)]
            if isinstance(files_of_interest, list)
            else []
        )
        constraints = [
            "Work only inside the candidate worktree.",
            "Do not run benchmarks in parallel.",
            "Benchmark authority remains compare_bench.sh.",
            "Profiler evidence is diagnostic, not decisive.",
            "Preserve sequential-only behavior.",
            "Recovered oom_recovered runs may still be comparable if benchmark output remains complete.",
        ]
        prompt = self._build_prompt(
            frontier_target=frontier_target,
            observation=observation,
            worktree_plan=worktree_plan,
            recent_decision_reasons=recent_decision_reasons,
            files_of_interest=candidate_files,
            constraints=constraints,
        )
        return MutationPlanRecord(
            mutation_id=utc_timestamp(),
            candidate_id=worktree_plan.candidate_id,
            selected_target=observation.selected_target,
            target_family=observation.target_family,
            worktree_path=worktree_plan.worktree_path,
            frontier_target=frontier_target,
            prompt=prompt,
            files_of_interest=candidate_files,
            constraints=constraints,
            diagnostics={
                "recent_decision_reasons": list(recent_decision_reasons),
                "observation_rationale": observation.diagnostics.get("rationale"),
            },
        )

    def _build_prompt(
        self,
        *,
        frontier_target: str,
        observation: ObservationRecord,
        worktree_plan: WorktreePlanRecord,
        recent_decision_reasons: list[str],
        files_of_interest: list[str],
        constraints: list[str],
    ) -> str:
        files_block = (
            "\n".join(f"- {path}" for path in files_of_interest)
            or "- src/grpo/trainer.py"
        )
        constraints_block = "\n".join(f"- {item}" for item in constraints)
        decisions_block = (
            "\n".join(f"- {reason}" for reason in recent_decision_reasons) or "- none"
        )
        rationale = observation.diagnostics.get(
            "rationale", "generation-first fallback"
        )
        return (
            "TASK: Optimize the selected hotspot in this isolated candidate worktree.\n"
            f"Frontier target: {frontier_target}\n"
            f"Candidate worktree: {worktree_plan.worktree_path}\n"
            f"Selected target: {observation.selected_target}\n"
            f"Target family: {observation.target_family}\n"
            f"Profiler rationale: {rationale}\n"
            "Files of interest:\n"
            f"{files_block}\n"
            "Recent decision history:\n"
            f"{decisions_block}\n"
            "Constraints:\n"
            f"{constraints_block}\n"
            "Expected outcome: produce a bounded code change in the candidate worktree that can later be benchmarked sequentially against the frontier using compare_bench.sh."
        )
