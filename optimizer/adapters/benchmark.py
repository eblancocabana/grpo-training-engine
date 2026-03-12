from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from optimizer.records import BenchmarkComparisonRecord, load_benchmark_report

_JSON_OUTPUT_PATTERN = re.compile(r"Wrote\s+(?P<path>.+compare_bench_[^\s]+\.json)")


@dataclass
class CompareBenchAdapter:
    repo_root: Path
    steps: int = 5
    triton_mode: str = "auto"
    script_path: Path | None = None

    def __post_init__(self) -> None:
        if self.script_path is None:
            self.script_path = self.repo_root / "tools" / "compare_bench.sh"

    def build_command(
        self, *, frontier_target: str, candidate_target: str
    ) -> list[str]:
        if frontier_target == candidate_target:
            raise ValueError(
                "Sequential optimizer requires distinct frontier and candidate targets."
            )

        return [
            "bash",
            str(self.script_path),
            "--steps",
            str(self.steps),
            "--triton",
            self.triton_mode,
            frontier_target,
            candidate_target,
        ]

    def run(
        self, *, frontier_target: str, candidate_target: str
    ) -> BenchmarkComparisonRecord:
        command = self.build_command(
            frontier_target=frontier_target, candidate_target=candidate_target
        )
        result = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (
                f"compare_bench.sh failed with exit code {result.returncode}:"
                f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            raise RuntimeError(message)

        output = f"{result.stdout}\n{result.stderr}"
        json_path = self._extract_json_path(output)
        return load_benchmark_report(json_path)

    def _extract_json_path(self, output: str) -> Path:
        match = _JSON_OUTPUT_PATTERN.search(output)
        if not match:
            raise RuntimeError("compare_bench.sh did not report a JSON output path.")
        return Path(match.group("path")).expanduser().resolve()
