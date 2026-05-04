"""Shared dataclasses for reasoning evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TaskType = Literal["math", "multiple_choice", "drop", "code"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    task_type: TaskType
    hf_candidates: tuple[dict[str, Any], ...] = ()
    split: str = "test"
    prompt_field: tuple[str, ...] = ("question", "problem", "prompt", "input")
    answer_field: tuple[str, ...] = ("answer", "final_answer", "target", "label")
    local_path_env: str | None = None
    default_limit: int | None = None
    requires_local_file: bool = False


@dataclass
class EvalExample:
    dataset: str
    family: str
    task_type: TaskType
    example_id: str
    question: str
    answer: Any
    choices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    model_key: str
    dataset: str
    protocol: str
    example_id: str
    sample_index: int
    prompt: str
    text: str
    token_count: int
    finish_reason: str | None
    latency_s: float | None = None

