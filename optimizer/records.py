from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


@dataclass(frozen=True)
class BenchmarkRunRecord:
    input: str
    label: str
    commit: str | None
    status: str
    valid: bool
    steps_requested: int | None = None
    steps_observed: int | None = None
    tokens_per_sec: float | None = None
    time_avg_s: float | None = None
    time_min_s: float | None = None
    time_max_s: float | None = None
    vram_avg_gb: float | None = None
    vram_peak_gb: float | None = None
    loss_avg: float | None = None
    reward_avg: float | None = None
    effective_batch: int | None = None
    oom_events: int = 0
    failure_phase: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    log_path: str | None = None
    triton_mode: str | None = None
    triton_arg: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BenchmarkRunRecord":
        return cls(
            input=_require_str(data, "input"),
            label=_coerce_str(data.get("label", data["input"])),
            commit=_optional_str(data.get("commit")),
            status=_require_str(data, "status"),
            valid=bool(data.get("valid", False)),
            steps_requested=_maybe_int(data.get("steps_requested")),
            steps_observed=_maybe_int(data.get("steps_observed")),
            tokens_per_sec=_maybe_float(data.get("tokens_per_sec")),
            time_avg_s=_maybe_float(data.get("time_avg_s")),
            time_min_s=_maybe_float(data.get("time_min_s")),
            time_max_s=_maybe_float(data.get("time_max_s")),
            vram_avg_gb=_maybe_float(data.get("vram_avg_gb")),
            vram_peak_gb=_maybe_float(data.get("vram_peak_gb")),
            loss_avg=_maybe_float(data.get("loss_avg")),
            reward_avg=_maybe_float(data.get("reward_avg")),
            effective_batch=_maybe_int(data.get("effective_batch")),
            oom_events=_maybe_int(data.get("oom_events")) or 0,
            failure_phase=_optional_str(data.get("failure_phase")),
            error_type=_optional_str(data.get("error_type")),
            error_message=_optional_str(data.get("error_message")),
            log_path=_optional_str(data.get("log_path")),
            triton_mode=_optional_str(data.get("triton_mode")),
            triton_arg=_optional_str(data.get("triton_arg")),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkComparisonRecord:
    schema_version: int
    generated_at: str
    runs: list[BenchmarkRunRecord]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BenchmarkComparisonRecord":
        raw_runs_value: object = data.get("runs", [])
        if not isinstance(raw_runs_value, list):
            raise ValueError("Benchmark report 'runs' must be a list.")
        raw_runs = cast(list[object], raw_runs_value)
        return cls(
            schema_version=_maybe_int(data.get("schema_version")) or 1,
            generated_at=str(data.get("generated_at", "unknown")),
            runs=[
                BenchmarkRunRecord.from_dict(_require_mapping(item, "runs[]"))
                for item in raw_runs
            ],
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "runs": [run.to_dict() for run in self.runs],
        }

    def require_run(self, target: str) -> BenchmarkRunRecord:
        for run in self.runs:
            if run.input == target:
                return run
        raise KeyError(f"Run for target '{target}' not found in benchmark report.")


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    campaign: str
    candidate_id: str
    candidate_target: str
    frontier_id: str
    frontier_target: str
    change_summary: str
    accepted: bool
    reason: str
    candidate_status: str
    frontier_status: str
    candidate_comparability: str
    frontier_comparability: str
    reward_delta: float | None
    loss_delta: float | None
    benchmark_report_path: str
    frontier_state_path: str
    diagnostics: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSnapshotRecord:
    target: str
    status: str
    comparability: str
    tokens_per_sec: float | None
    time_avg_s: float | None
    reward_avg: float | None
    loss_avg: float | None
    vram_peak_gb: float | None
    oom_events: int
    effective_batch: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExperimentSnapshotRecord":
        return cls(
            target=_require_str(data, "target"),
            status=_require_str(data, "status"),
            comparability=str(data.get("comparability", "unknown")),
            tokens_per_sec=_maybe_float(data.get("tokens_per_sec")),
            time_avg_s=_maybe_float(data.get("time_avg_s")),
            reward_avg=_maybe_float(data.get("reward_avg")),
            loss_avg=_maybe_float(data.get("loss_avg")),
            vram_peak_gb=_maybe_float(data.get("vram_peak_gb")),
            oom_events=_maybe_int(data.get("oom_events")) or 0,
            effective_batch=_maybe_int(data.get("effective_batch")),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentLedgerRecord:
    experiment_number: int
    decision_id: str
    campaign: str
    change_summary: str
    candidate_id: str
    candidate_target: str
    frontier_id: str
    frontier_target: str
    outcome: str
    reason: str
    primary_metric: str
    baseline_snapshot: ExperimentSnapshotRecord
    candidate_snapshot: ExperimentSnapshotRecord
    incumbent_tokens_per_sec_after_decision: float | None
    running_best_tokens_per_sec: float | None
    running_best_experiment_number: int | None
    benchmark_report_path: str
    frontier_state_path: str
    decision_path: str | None = None
    timestamp: str | None = None
    attempt_number: int | None = None
    selected_target: str | None = None
    hypothesis: str | None = None
    worktree_path: str | None = None
    attempt_markdown_path: str | None = None
    tokens_per_sec_delta: float | None = None
    tokens_per_sec_pct_change: float | None = None
    sequential_only: bool = True
    generation_only: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExperimentLedgerRecord":
        experiment_number = _maybe_int(data.get("experiment_number")) or 0
        baseline_snapshot = ExperimentSnapshotRecord.from_dict(
            _require_mapping(data.get("baseline_snapshot"), "baseline_snapshot")
        )
        candidate_snapshot = ExperimentSnapshotRecord.from_dict(
            _require_mapping(data.get("candidate_snapshot"), "candidate_snapshot")
        )
        frontier_target = data.get("frontier_target")
        if not isinstance(frontier_target, str):
            frontier_target = baseline_snapshot.target
        candidate_target = data.get("candidate_target")
        if not isinstance(candidate_target, str):
            candidate_target = candidate_snapshot.target
        frontier_id = data.get("frontier_id")
        if not isinstance(frontier_id, str):
            frontier_id = str(frontier_target)
        decision_id = data.get("decision_id")
        if not isinstance(decision_id, str):
            decision_id = str(data.get("timestamp", f"legacy-{experiment_number}"))
        benchmark_report_path = data.get("benchmark_report_path")
        if not isinstance(benchmark_report_path, str):
            benchmark_report_path = ""
        return cls(
            experiment_number=experiment_number,
            decision_id=decision_id,
            campaign=str(data.get("campaign", "generation_hf")),
            change_summary=str(data.get("change_summary", "")),
            candidate_id=str(data.get("candidate_id", candidate_target)),
            candidate_target=str(candidate_target),
            frontier_id=str(frontier_id),
            frontier_target=str(frontier_target),
            outcome=str(data.get("outcome", "unknown")),
            reason=str(data.get("reason", "unknown")),
            primary_metric=str(data.get("primary_metric", "tokens_per_sec")),
            baseline_snapshot=baseline_snapshot,
            candidate_snapshot=candidate_snapshot,
            incumbent_tokens_per_sec_after_decision=_maybe_float(
                data.get("incumbent_tokens_per_sec_after_decision")
            ),
            running_best_tokens_per_sec=_maybe_float(
                data.get("running_best_tokens_per_sec")
            ),
            running_best_experiment_number=_maybe_int(
                data.get("running_best_experiment_number")
            ),
            benchmark_report_path=benchmark_report_path,
            frontier_state_path=str(data.get("frontier_state_path", "")),
            decision_path=_optional_str(data.get("decision_path")),
            timestamp=_optional_str(data.get("timestamp")),
            attempt_number=_maybe_int(data.get("attempt_number")),
            selected_target=_optional_str(data.get("selected_target")),
            hypothesis=_optional_str(data.get("hypothesis")),
            worktree_path=_optional_str(data.get("worktree_path")),
            attempt_markdown_path=_optional_str(data.get("attempt_markdown_path")),
            tokens_per_sec_delta=_maybe_float(data.get("tokens_per_sec_delta")),
            tokens_per_sec_pct_change=_maybe_float(
                data.get("tokens_per_sec_pct_change")
            ),
            sequential_only=bool(data.get("sequential_only", True)),
            generation_only=bool(data.get("generation_only", True)),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)


def load_benchmark_report(path: str | Path) -> BenchmarkComparisonRecord:
    json_loads = cast(Callable[[str], object], json.loads)
    payload_obj = json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError("Benchmark report root must be a JSON object.")
    payload_mapping = cast(Mapping[object, object], payload_obj)
    payload = {str(key): item for key, item in payload_mapping.items()}
    return BenchmarkComparisonRecord.from_dict(payload)


def load_json_record(path: str | Path) -> dict[str, object]:
    json_loads = cast(Callable[[str], object], json.loads)
    payload_obj = json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError("JSON record root must be a JSON object.")
    payload_mapping = cast(Mapping[object, object], payload_obj)
    return {str(key): item for key, item in payload_mapping.items()}


def write_json_record(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ = destination.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def append_jsonl_record(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        _ = handle.write(json.dumps(_to_jsonable(payload), sort_keys=True) + "\n")
    return destination


def load_last_jsonl_record(path: str | Path) -> dict[str, object] | None:
    destination = Path(path)
    if not destination.exists():
        return None
    last_line = ""
    with destination.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    payload_obj = cast(Callable[[str], object], json.loads)(last_line)
    if not isinstance(payload_obj, dict):
        raise ValueError("JSONL record root must be a JSON object.")
    payload_mapping = cast(Mapping[object, object], payload_obj)
    return {str(key): item for key, item in payload_mapping.items()}


def _to_jsonable(value: object) -> JsonValue:
    if isinstance(
        value,
        (
            BenchmarkRunRecord,
            BenchmarkComparisonRecord,
            DecisionRecord,
            ExperimentSnapshotRecord,
            ExperimentLedgerRecord,
        ),
    ):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        return {str(key): _to_jsonable(item) for key, item in mapping_value.items()}
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return [_to_jsonable(item) for item in list_value]
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        return [_to_jsonable(item) for item in tuple_value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON payload type: {type(value)!r}")


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Field '{field_name}' must be a mapping.")
    mapping_value = cast(Mapping[object, object], value)
    return {str(key): item for key, item in mapping_value.items()}


def _require_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string.")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected string or None, got {type(value)!r}.")


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    raise ValueError(f"Cannot convert value of type {type(value)!r} to string.")


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"Expected float-compatible value, got {type(value)!r}.")


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected int-compatible value, got {type(value)!r}.")
