from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from optimizer.records import WorktreePlanRecord


@dataclass(frozen=True)
class WorktreePlanner:
    repo_root: Path
    base_dir_name: str = "optimizer_worktrees"

    def plan(
        self,
        *,
        candidate_id: str,
        frontier_target: str,
        selected_target: str,
    ) -> WorktreePlanRecord:
        worktree_root = self.repo_root / self.base_dir_name
        worktree_path = worktree_root / candidate_id
        branch_name = f"optimizer/{candidate_id}"
        return WorktreePlanRecord(
            candidate_id=candidate_id,
            branch_name=branch_name,
            worktree_path=str(worktree_path),
            frontier_target=frontier_target,
            selected_target=selected_target,
            notes={
                "sequential_only": True,
                "requires_external_ai_agent": True,
            },
        )

    def materialize(self, *, plan: WorktreePlanRecord) -> WorktreePlanRecord:
        worktree_path = Path(plan.worktree_path)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_existing_worktree(plan)
        if (self.repo_root / ".git").exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "worktree",
                    "add",
                    "-b",
                    plan.branch_name,
                    str(worktree_path),
                    plan.frontier_target,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            worktree_path.mkdir(parents=True, exist_ok=True)
        notes = dict(plan.notes)
        notes["materialized"] = True
        notes["materialization_mode"] = (
            "git_worktree" if (self.repo_root / ".git").exists() else "directory_stub"
        )
        notes["fresh_workspace"] = True
        return WorktreePlanRecord(
            candidate_id=plan.candidate_id,
            branch_name=plan.branch_name,
            worktree_path=plan.worktree_path,
            frontier_target=plan.frontier_target,
            selected_target=plan.selected_target,
            notes=notes,
        )

    def _remove_existing_worktree(self, plan: WorktreePlanRecord) -> None:
        worktree_path = Path(plan.worktree_path)
        if not worktree_path.exists():
            if (self.repo_root / ".git").exists():
                self._remove_existing_branch(plan.branch_name)
            return
        if (self.repo_root / ".git").exists():
            try:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.repo_root),
                        "worktree",
                        "remove",
                        "--force",
                        str(worktree_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                shutil.rmtree(worktree_path)
            self._remove_existing_branch(plan.branch_name)
            return
        shutil.rmtree(worktree_path)

    def _remove_existing_branch(self, branch_name: str) -> None:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(self.repo_root),
                "branch",
                "--list",
                branch_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "branch",
                    "-D",
                    branch_name,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
