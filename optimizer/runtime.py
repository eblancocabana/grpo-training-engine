from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optimizer.mutator import AIAgentMutationPlanner
from optimizer.observation import DeepTraceTargetSelector
from optimizer.records import (
    LoopIterationRecord,
    OptimizerSessionRecord,
    utc_timestamp,
    write_json_record,
)
from optimizer.worktree import WorktreePlanner


@dataclass(frozen=True)
class RuntimeConfig:
    baseline: str
    iteration_budget: int | None = 1
    open_ended: bool = False
    trace_summary_path: str | None = None
    candidate_targets: list[str] | None = None
    allow_recovered_oom_promotion: bool = True
    benchmark_steps: int = 5
    benchmark_triton_mode: str = "auto"


class OptimizerRuntime:
    def __init__(self, *, repo_root: str | Path, artifacts_dir: str | Path) -> None:
        self.repo_root: Path = Path(repo_root).resolve()
        self.artifacts_dir: Path = Path(artifacts_dir).resolve()
        self.selector: DeepTraceTargetSelector = DeepTraceTargetSelector()
        self.worktree_planner: WorktreePlanner = WorktreePlanner(
            repo_root=self.repo_root
        )
        self.mutator: AIAgentMutationPlanner = AIAgentMutationPlanner(
            repo_root=self.repo_root
        )

    def run_control_loop(
        self, config: RuntimeConfig
    ) -> tuple[OptimizerSessionRecord, list[LoopIterationRecord]]:
        planned_candidates = self._planned_candidates(config)
        active_candidate_id = planned_candidates[0]
        remaining_candidates = planned_candidates[1:]
        session = OptimizerSessionRecord(
            session_id=utc_timestamp(),
            baseline=config.baseline,
            iteration_budget=config.iteration_budget,
            open_ended=config.open_ended,
            iterations_completed=0,
            latest_frontier_target=config.baseline,
            latest_candidate_id=None,
            status="planning",
            notes={
                "mode": "ai_agent_handoff_only",
                "sequential_only": True,
                "allow_recovered_oom_promotion": config.allow_recovered_oom_promotion,
                "benchmark_steps": config.benchmark_steps,
                "benchmark_triton_mode": config.benchmark_triton_mode,
                "planned_candidates": list(remaining_candidates),
            },
        )
        summary = (
            self.selector.load_summary_file(config.trace_summary_path)
            if config.trace_summary_path
            else {}
        )
        observation = self.selector.select_from_summary(
            frontier_target=config.baseline,
            summary=summary,
        )
        observation_path = self._write(
            "observations", observation.observation_id, observation
        )
        worktree_plan = self.worktree_planner.materialize(
            plan=self.worktree_planner.plan(
                candidate_id=active_candidate_id,
                frontier_target=config.baseline,
                selected_target=observation.selected_target,
            )
        )
        mutation_plan = self.mutator.build_plan(
            frontier_target=config.baseline,
            observation=observation,
            worktree_plan=worktree_plan,
            recent_decision_reasons=[],
        )
        worktree_path = self._write("worktrees", active_candidate_id, worktree_plan)
        mutation_path = self._write(
            "mutation_plans", mutation_plan.mutation_id, mutation_plan
        )
        latest_iteration = LoopIterationRecord(
            iteration_id=utc_timestamp(),
            iteration_index=1,
            frontier_target=config.baseline,
            selected_target=observation.selected_target,
            candidate_id=active_candidate_id,
            worktree_path=worktree_plan.worktree_path,
            observation_path=str(observation_path),
            mutation_plan_path=str(mutation_path),
            status="awaiting_ai_agent",
            notes={
                "benchmark_steps": config.benchmark_steps,
                "benchmark_triton_mode": config.benchmark_triton_mode,
                "worktree_plan_path": str(worktree_path),
            },
        )
        session = OptimizerSessionRecord(
            session_id=session.session_id,
            baseline=session.baseline,
            iteration_budget=session.iteration_budget,
            open_ended=session.open_ended,
            iterations_completed=1,
            latest_frontier_target=config.baseline,
            latest_candidate_id=latest_iteration.candidate_id,
            status="awaiting_ai_agent",
            notes={
                **session.notes,
                "observation_path": str(observation_path),
                "latest_worktree_plan_path": latest_iteration.notes[
                    "worktree_plan_path"
                ],
                "latest_mutation_plan_path": latest_iteration.mutation_plan_path,
            },
        )
        session_path = self._write("sessions", session.session_id, session)
        notes = dict(latest_iteration.notes)
        notes["session_path"] = str(session_path)
        persisted_iteration = LoopIterationRecord(
            iteration_id=latest_iteration.iteration_id,
            iteration_index=latest_iteration.iteration_index,
            frontier_target=latest_iteration.frontier_target,
            selected_target=latest_iteration.selected_target,
            candidate_id=latest_iteration.candidate_id,
            worktree_path=latest_iteration.worktree_path,
            observation_path=latest_iteration.observation_path,
            mutation_plan_path=latest_iteration.mutation_plan_path,
            status=latest_iteration.status,
            notes=notes,
        )
        _ = self._write(
            "iterations", persisted_iteration.iteration_id, persisted_iteration
        )
        return session, [persisted_iteration]

    def _planned_candidates(self, config: RuntimeConfig) -> list[str]:
        raw_targets = list(config.candidate_targets or [])
        budget = config.iteration_budget
        if budget is None:
            budget = len(raw_targets) if raw_targets else 1
        budget = max(1, budget)

        if not raw_targets:
            return [f"cand-{utc_timestamp().lower()}" for _ in range(budget)]

        planned: list[str] = []
        while len(planned) < budget:
            for target in raw_targets:
                if len(planned) >= budget:
                    break
                planned.append(f"{target}-{utc_timestamp().lower()}")
            if not config.open_ended and len(planned) >= len(raw_targets):
                break
        return planned[:budget]

    def _write(self, category: str, record_id: str, payload: object) -> Path:
        return write_json_record(
            self.artifacts_dir / category / f"{record_id}.json",
            payload,
        )
