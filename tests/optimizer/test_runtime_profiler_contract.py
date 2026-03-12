from pathlib import Path

from optimizer.adapters.profiler import (
    build_profiler_command,
    build_profiler_invocation,
)
from optimizer.runtime import OptimizerRuntime, RuntimeConfig


def test_runtime_generated_candidate_is_resolved_by_profiler_adapter(
    tmp_path: Path,
) -> None:
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
    worktree_root = Path(iteration.worktree_path)
    _ = (worktree_root / "train.py").write_text(
        "print('candidate')\n", encoding="utf-8"
    )

    profiler_command = build_profiler_command(
        repo_root=tmp_path,
        target="feat/example-candidate",
        max_steps=3,
    )

    assert profiler_command[1] == str(worktree_root / "train.py")


def test_runtime_generated_candidate_exposes_profiler_cwd(tmp_path: Path) -> None:
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
    worktree_root = Path(iteration.worktree_path)
    _ = (worktree_root / "train.py").write_text(
        "print('candidate')\n", encoding="utf-8"
    )

    invocation = build_profiler_invocation(
        repo_root=tmp_path,
        target="feat/example-candidate",
        max_steps=3,
    )

    assert invocation.command[1] == str(worktree_root / "train.py")
    assert invocation.cwd == worktree_root


def test_profiler_invocation_falls_back_to_repo_root_cwd(tmp_path: Path) -> None:
    invocation = build_profiler_invocation(
        repo_root=tmp_path,
        target="missing-candidate",
        max_steps=3,
    )

    assert invocation.command[1] == str(tmp_path / "train.py")
    assert invocation.cwd == tmp_path
