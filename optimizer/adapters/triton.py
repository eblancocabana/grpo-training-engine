from __future__ import annotations

from pathlib import Path


def build_triton_command(
    *,
    repo_root: str | Path,
    mode: str = "both",
    max_steps: int | None = None,
    group_size: int | None = None,
    sample_interval: float | None = None,
) -> list[str]:
    command = [
        "python",
        str(Path(repo_root) / "tools" / "compare_triton.py"),
        "--mode",
        mode,
    ]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if group_size is not None:
        command.extend(["--group-size", str(group_size)])
    if sample_interval is not None:
        command.extend(["--sample-interval", str(sample_interval)])
    return command
