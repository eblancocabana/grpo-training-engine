from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from optimizer.records import JsonValue, ObservationRecord, utc_timestamp


@dataclass(frozen=True)
class TargetSelection:
    selected_target: str
    target_family: str
    rationale: str
    files_of_interest: list[str]
    diagnostics: dict[str, JsonValue]


class DeepTraceTargetSelector:
    def select_from_summary(
        self,
        *,
        frontier_target: str,
        summary: dict[str, JsonValue] | None,
    ) -> ObservationRecord:
        selection = self._select_target(summary or {})
        return ObservationRecord(
            observation_id=utc_timestamp(),
            source="deep_trace_generation_safe",
            frontier_target=frontier_target,
            trace_phase="generation",
            selected_target=selection.selected_target,
            target_family=selection.target_family,
            summary=summary or {},
            diagnostics={
                "rationale": selection.rationale,
                "files_of_interest": selection.files_of_interest,
                **selection.diagnostics,
            },
        )

    def load_summary_file(self, path: str | Path) -> dict[str, JsonValue]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Observation summary root must be a JSON object.")
        return cast(dict[str, JsonValue], payload)

    def _select_target(self, summary: dict[str, JsonValue]) -> TargetSelection:
        key_averages = summary.get("key_averages")
        if isinstance(key_averages, list):
            names = [
                str(item.get("name"))
                for item in key_averages
                if isinstance(item, dict) and item.get("name") is not None
            ]
            joined = " ".join(names).lower()
            if any(token in joined for token in ["paged", "decode", "attention"]):
                return TargetSelection(
                    selected_target="generation_triton_decode",
                    target_family="generation",
                    rationale="deep-trace kernel names point to paged/decode/attention hot path",
                    files_of_interest=[
                        "src/triton_kernels/paged_kv.py",
                        "src/grpo/trainer.py",
                    ],
                    diagnostics={"matched_kernels": names[:10]},
                )
            if any(token in joined for token in ["sample", "logits"]):
                return TargetSelection(
                    selected_target="generation_sampling",
                    target_family="generation",
                    rationale="deep-trace names point to sampling/logits hot path",
                    files_of_interest=["src/grpo/trainer.py"],
                    diagnostics={"matched_kernels": names[:10]},
                )

        return TargetSelection(
            selected_target="generation_general",
            target_family="generation",
            rationale="safe fallback to generation-first target when trace is sparse or generic",
            files_of_interest=["src/grpo/trainer.py"],
            diagnostics={"fallback": True},
        )
