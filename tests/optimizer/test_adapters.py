from pathlib import Path

from optimizer.adapters.benchmark import CompareBenchAdapter
from optimizer.adapters.profiler import (
    build_profiler_command,
    build_profiler_invocation,
)
from optimizer.adapters.triton import build_triton_command


def test_builds_compare_bench_command_with_expected_shape(tmp_path: Path) -> None:
    adapter = CompareBenchAdapter(repo_root=tmp_path, steps=7, triton_mode="off")

    command = adapter.build_command(frontier_target="main", candidate_target="feat/x")

    assert command[:2] == ["bash", str(tmp_path / "tools" / "compare_bench.sh")]
    assert "--steps" in command
    assert "7" in command
    assert command[-2:] == ["main", "feat/x"]


def test_builds_diagnostic_commands() -> None:
    triton_command = build_triton_command(repo_root=".", max_steps=5)
    profiler_command = build_profiler_command(
        repo_root=".",
        target="feat/x",
        max_steps=5,
        no_triton=True,
    )

    assert any("compare_triton.py" in part for part in triton_command)
    assert "--max-steps" in triton_command
    assert any("train.py" in part for part in profiler_command)
    assert "--profile" in profiler_command
    assert "--no-triton" in profiler_command


def test_profiler_command_prefers_existing_candidate_worktree(tmp_path: Path) -> None:
    candidate_root = tmp_path / "optimizer_worktrees" / "feat-x"
    candidate_root.mkdir(parents=True)
    _ = (candidate_root / "train.py").write_text("print('x')\n", encoding="utf-8")

    profiler_command = build_profiler_command(
        repo_root=tmp_path,
        target="feat-x",
        max_steps=3,
    )

    assert profiler_command[1] == str(candidate_root / "train.py")


def test_profiler_command_resolves_latest_runtime_generated_candidate(
    tmp_path: Path,
) -> None:
    older = tmp_path / "optimizer_worktrees" / "feat-x-20260312t100000000000z"
    newer = tmp_path / "optimizer_worktrees" / "feat-x-20260312t100000000001z"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    _ = (older / "train.py").write_text("print('old')\n", encoding="utf-8")
    _ = (newer / "train.py").write_text("print('new')\n", encoding="utf-8")

    profiler_command = build_profiler_command(
        repo_root=tmp_path,
        target="feat-x",
        max_steps=3,
    )

    assert profiler_command[1] == str(newer / "train.py")


def test_profiler_invocation_exposes_candidate_cwd(tmp_path: Path) -> None:
    candidate_root = tmp_path / "optimizer_worktrees" / "feat-x"
    candidate_root.mkdir(parents=True)
    _ = (candidate_root / "train.py").write_text("print('x')\n", encoding="utf-8")

    invocation = build_profiler_invocation(
        repo_root=tmp_path,
        target="feat-x",
        max_steps=3,
    )

    assert invocation.command[1] == str(candidate_root / "train.py")
    assert invocation.cwd == candidate_root
