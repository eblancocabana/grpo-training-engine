from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfilerInvocation:
    command: list[str]
    cwd: Path


def build_profiler_command(
    *,
    repo_root: str | Path,
    target: str | None = None,
    max_steps: int | None = None,
    no_triton: bool = False,
) -> list[str]:
    return build_profiler_invocation(
        repo_root=repo_root,
        target=target,
        max_steps=max_steps,
        no_triton=no_triton,
    ).command


def build_profiler_invocation(
    *,
    repo_root: str | Path,
    target: str | None = None,
    max_steps: int | None = None,
    no_triton: bool = False,
) -> ProfilerInvocation:
    train_entry = _resolve_train_entry(repo_root=repo_root, target=target)
    command = [
        "python",
        str(train_entry),
        "--profile",
        "--no-wandb",
    ]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if no_triton:
        command.append("--no-triton")
    return ProfilerInvocation(command=command, cwd=train_entry.parent)


def _resolve_train_entry(*, repo_root: str | Path, target: str | None) -> Path:
    root = Path(repo_root)
    if target:
        direct_candidate_root = root / "optimizer_worktrees" / target
        direct_candidate_train = direct_candidate_root / "train.py"
        if direct_candidate_train.exists():
            return direct_candidate_train

        matching_roots = sorted(
            (root / "optimizer_worktrees").glob(f"{target}-*"),
            key=lambda path: path.name,
        )
        for candidate_root in reversed(matching_roots):
            candidate_train = candidate_root / "train.py"
            if candidate_train.exists():
                return candidate_train
    return root / "train.py"
