from pathlib import Path
import subprocess

from optimizer.worktree import WorktreePlanner


def test_materialize_replaces_stale_directory_in_non_git_mode(tmp_path: Path) -> None:
    planner = WorktreePlanner(repo_root=tmp_path)
    plan = planner.plan(
        candidate_id="cand-1",
        frontier_target="main",
        selected_target="generation_general",
    )
    stale_path = Path(plan.worktree_path)
    stale_path.mkdir(parents=True)
    (stale_path / "stale.txt").write_text("old", encoding="utf-8")

    materialized = planner.materialize(plan=plan)

    assert Path(materialized.worktree_path).exists()
    assert not (Path(materialized.worktree_path) / "stale.txt").exists()
    assert materialized.notes["fresh_workspace"] is True


def test_materialize_removes_existing_branch_in_git_mode(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "branch", "optimizer/cand-1"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    planner = WorktreePlanner(repo_root=tmp_path)
    plan = planner.plan(
        candidate_id="cand-1",
        frontier_target="HEAD",
        selected_target="generation_general",
    )

    materialized = planner.materialize(plan=plan)

    result = subprocess.run(
        ["git", "branch", "--list", "optimizer/cand-1"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("optimizer/cand-1")
    assert Path(materialized.worktree_path).exists()


def test_materialize_falls_back_when_path_is_stale_unregistered_directory_in_git_mode(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    planner = WorktreePlanner(repo_root=tmp_path)
    plan = planner.plan(
        candidate_id="cand-2",
        frontier_target="HEAD",
        selected_target="generation_general",
    )
    stale_path = Path(plan.worktree_path)
    stale_path.mkdir(parents=True)
    (stale_path / "stale.txt").write_text("old", encoding="utf-8")

    materialized = planner.materialize(plan=plan)

    assert Path(materialized.worktree_path).exists()
    assert not (Path(materialized.worktree_path) / "stale.txt").exists()
