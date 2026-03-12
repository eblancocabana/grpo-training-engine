from pathlib import Path

from optimizer.mutator import AIAgentMutationPlanner
from optimizer.observation import DeepTraceTargetSelector
from optimizer.worktree import WorktreePlanner


def test_mutation_plan_contains_worktree_and_constraints(tmp_path: Path) -> None:
    selector = DeepTraceTargetSelector()
    observation = selector.select_from_summary(
        frontier_target="main",
        summary={"key_averages": [{"name": "_paged_attention_decode_kernel"}]},
    )
    planner = WorktreePlanner(repo_root=tmp_path)
    worktree_plan = planner.plan(
        candidate_id="cand-1",
        frontier_target="main",
        selected_target=observation.selected_target,
    )
    mutator = AIAgentMutationPlanner(repo_root=tmp_path)

    plan = mutator.build_plan(
        frontier_target="main",
        observation=observation,
        worktree_plan=worktree_plan,
        recent_decision_reasons=["reward_improved"],
    )

    assert plan.candidate_id == "cand-1"
    assert "compare_bench.sh" in plan.prompt
    assert any(
        "Work only inside the candidate worktree." == item for item in plan.constraints
    )
    assert worktree_plan.worktree_path in plan.prompt
