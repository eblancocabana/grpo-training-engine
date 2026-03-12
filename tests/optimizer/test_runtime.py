from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from optimizer.runtime import OptimizerRuntime, RuntimeConfig
from optimizer.records import JsonValue


def test_runtime_generates_ai_agent_handoff_artifacts(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _ = summary_path.write_text(
        json.dumps(
            {
                "key_averages": [
                    {"name": "_paged_attention_decode_kernel"},
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = OptimizerRuntime(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
    )

    session, iterations = runtime.run_control_loop(
        RuntimeConfig(
            baseline="main",
            iteration_budget=1,
            open_ended=False,
            candidate_targets=["feat/example-candidate"],
            trace_summary_path=str(summary_path),
        )
    )
    iteration = iterations[0]

    assert session.status == "awaiting_ai_agent"
    assert session.iterations_completed == 1
    assert iteration.candidate_id.startswith("feat/example-candidate-")
    assert iteration.selected_target == "generation_triton_decode"
    assert Path(iteration.observation_path).exists()
    assert Path(iteration.mutation_plan_path).exists()
    assert Path(iteration.worktree_path).exists()

    worktree_payload = cast(
        dict[str, JsonValue],
        json.loads(
            (
                tmp_path
                / "optimizer-artifacts"
                / "worktrees"
                / f"{iteration.candidate_id}.json"
            ).read_text(encoding="utf-8")
        ),
    )
    notes = cast(dict[str, JsonValue], worktree_payload["notes"])
    assert notes["fresh_workspace"] is True


def test_runtime_candidate_target_gets_unique_attempt_suffix(tmp_path: Path) -> None:
    runtime = OptimizerRuntime(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
    )

    _, iterations = runtime.run_control_loop(
        RuntimeConfig(
            baseline="main",
            iteration_budget=1,
            open_ended=False,
            candidate_targets=["feat/example-candidate"],
        )
    )
    iteration = iterations[0]

    assert iteration.candidate_id.startswith("feat/example-candidate-")


def test_runtime_materializes_only_one_active_iteration_from_candidates_and_budget(
    tmp_path: Path,
) -> None:
    runtime = OptimizerRuntime(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "optimizer-artifacts",
    )

    session, iterations = runtime.run_control_loop(
        RuntimeConfig(
            baseline="main",
            iteration_budget=3,
            open_ended=True,
            candidate_targets=["feat/a", "feat/b"],
            benchmark_steps=7,
            benchmark_triton_mode="off",
        )
    )

    assert session.iterations_completed == 1
    assert len(iterations) == 1
    assert iterations[0].iteration_index == 1
    assert iterations[0].candidate_id.startswith("feat/a-")
    assert iterations[0].notes["benchmark_steps"] == 7
    assert iterations[0].notes["benchmark_triton_mode"] == "off"
    planned_candidates = session.notes["planned_candidates"]
    assert isinstance(planned_candidates, list)
    assert len(planned_candidates) == 2
    assert str(planned_candidates[0]).startswith("feat/b-")
    assert str(planned_candidates[1]).startswith("feat/a-")

    worktree_dir = tmp_path / "optimizer_worktrees"
    active_worktrees = [path for path in worktree_dir.iterdir() if path.is_dir()]
    assert len(active_worktrees) == 1

    worktree_records = list(
        (tmp_path / "optimizer-artifacts" / "worktrees").rglob("*.json")
    )
    mutation_records = list(
        (tmp_path / "optimizer-artifacts" / "mutation_plans").glob("*.json")
    )
    iteration_records = list(
        (tmp_path / "optimizer-artifacts" / "iterations").glob("*.json")
    )
    assert len(worktree_records) == 1
    assert len(mutation_records) == 1
    assert len(iteration_records) == 1
