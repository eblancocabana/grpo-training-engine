from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optimizer.adapters.benchmark import CompareBenchAdapter
from optimizer.records import BenchmarkComparisonRecord


@dataclass
class GenerationHFCampaign:
    repo_root: Path
    steps: int = 5
    triton_mode: str = "auto"
    script_path: Path | None = None
    name: str = "generation_hf"
    adapter: CompareBenchAdapter | None = None

    def __post_init__(self) -> None:
        if self.adapter is None:
            self.adapter = CompareBenchAdapter(
                repo_root=self.repo_root,
                steps=self.steps,
                triton_mode=self.triton_mode,
                script_path=self.script_path,
            )

    def run(
        self, *, frontier_target: str, candidate_target: str
    ) -> BenchmarkComparisonRecord:
        if self.adapter is None:
            raise RuntimeError("GenerationHFCampaign adapter was not initialized.")
        return self.adapter.run(
            frontier_target=frontier_target, candidate_target=candidate_target
        )
