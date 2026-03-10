from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false, reportUnnecessaryComparison=false, reportUnusedFunction=false, reportImplicitOverride=false

import gzip
import importlib
import io
import json
import os
import threading
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response, StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp

from src.core.memory_manager import MemoryManager
from src.utils.logging_utils import get_logger
from tools.vram_profiler.profiler_hooks import ProfilerState, estimate_trace_size
from src.utils.config import get_8gb_vram_config

logger = get_logger(__name__)
_TRACE_MAX_BYTES = 100 * 1024 * 1024
_MIN_FREE_VRAM_GB = 0.8
_SAFE_MIN_FREE_VRAM_GB = 1.5
_SAFE_MAX_TRACE_BYTES = 20 * 1024 * 1024
_SAFE_DEFAULTS = {
    "steps": 1,
    "wait": 1,
    "warmup": 1,
    "profile_memory": False,
    "record_shapes": False,
    "with_stack": False,
    "sync": True,
    "max_duration_s": 20.0,
    "estimated_events_per_step": 1000,
    "bytes_per_event": 120,
}
_SAFE_MAX_STEPS = 1
_SAFE_MAX_WARMUP = 1
_SAFE_MAX_WAIT = 1
_GUARDRAIL_ALTERNATIVES = ["snapshot", "nvtx", "tier1"]


class NonStreamingGZipMiddleware(BaseHTTPMiddleware):
    _minimum_size: int
    _compresslevel: int

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = 500,
        compresslevel: int = 9,
    ) -> None:
        super().__init__(app)
        self._minimum_size = minimum_size
        self._compresslevel = compresslevel

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        if isinstance(response, StreamingResponse):
            return response

        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response

        if response.headers.get("Content-Encoding"):
            return response

        body = getattr(response, "body", None) or b""
        if len(body) < self._minimum_size:
            return response

        compressed = gzip.compress(body, compresslevel=self._compresslevel)
        headers = dict(response.headers)
        _ = headers.pop("Content-Length", None)
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )


def _schema_metadata(
    state: ProfilerState,
    *,
    window: Mapping[str, object] | None = None,
    sampling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    window_payload: dict[str, object] = {
        "since_step": None,
        "since_ts": None,
        "until_ts": None,
    }
    if window:
        window_payload.update(window)

    sampling_payload: dict[str, object] = {
        "maxlen": state.maxlen,
    }
    if sampling:
        sampling_payload.update(sampling)

    return {
        "schema_version": state.schema_version,
        "units": {
            "time": "s",
            "duration": "ms",
            "memory": "bytes",
        },
        "sampling": sampling_payload,
        "window": window_payload,
        "source": {
            "component": "profiler_server",
            "pid": os.getpid(),
        },
    }


class _TorchCuda(Protocol):
    def is_available(self) -> bool: ...

    def get_device_name(self, device: int) -> str: ...


class _TorchModule(Protocol):
    cuda: _TorchCuda


def _get_torch_module() -> _TorchModule | None:
    try:
        return cast(_TorchModule, cast(object, importlib.import_module("torch")))
    except Exception:
        return None


def _with_schema(
    payload: Mapping[str, object],
    state: ProfilerState,
    *,
    window: Mapping[str, object] | None = None,
    sampling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    enriched = dict(payload)
    enriched.update(_schema_metadata(state, window=window, sampling=sampling))
    return enriched


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_window_step(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_fields(value: str | None) -> list[list[str]] | None:
    if not value:
        return None
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not fields:
        return None
    return [field.split(".") for field in fields]


def _parse_window_size(value: str | None, *, default: int = 200) -> int:
    if not value:
        return default
    value = value.strip().lower()
    if value.startswith("last_"):
        value = value.replace("last_", "")
    try:
        parsed = int(value)
        return max(parsed, 1)
    except ValueError:
        return default


def _parse_list_filters(request: Request) -> dict[str, object]:
    params = request.query_params
    since_step = _parse_int(params.get("since_step"))
    until_step = _parse_window_step(params.get("until_step"))
    since_ts = _parse_float(params.get("since_ts"))
    until_ts = _parse_float(params.get("until_ts"))
    limit = _parse_int(params.get("limit"))
    offset = _parse_int(params.get("offset"))
    fields = _parse_fields(params.get("fields"))
    return {
        "since_step": since_step,
        "until_step": until_step,
        "since_ts": since_ts,
        "until_ts": until_ts,
        "limit": limit,
        "offset": offset or 0,
        "fields": fields,
    }


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values_sorted = sorted(values)
    count = len(values_sorted)

    def _pct(p: float) -> float:
        index = max(min(int(round((count - 1) * p)), count - 1), 0)
        return values_sorted[index]

    mean = sum(values_sorted) / count
    variance = sum((v - mean) ** 2 for v in values_sorted) / count
    std = (variance**0.5) if isinstance(variance, (int, float)) else 0.0
    return {
        "count": float(count),
        "mean": mean,
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "std": std,
    }


def _metric_values(
    state: ProfilerState, metric: str, filters: Mapping[str, object]
) -> list[float]:
    values: list[float] = []
    for entry in state.metrics:
        if not _filter_entry(entry, filters):
            continue
        if entry.get("name") != metric:
            continue
        value = entry.get("value")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _latest_metric_entry(
    state: ProfilerState, metric: str
) -> Mapping[str, object] | None:
    for entry in reversed(list(state.metrics)):
        if entry.get("name") == metric:
            return entry
    return None


def _latest_oom_backoff(state: ProfilerState) -> dict[str, object] | None:
    entry = _latest_metric_entry(state, "oom_backoff_count")
    if entry is not None:
        value = entry.get("value")
        if isinstance(value, (int, float)) and value > 0:
            return {
                "oom_backoff_count": int(value),
                "step": entry.get("step"),
                "ts": entry.get("ts"),
            }
    for phase in reversed(list(state.phases)):
        if phase.get("name") != "micro_batching":
            continue
        meta = phase.get("meta")
        if not isinstance(meta, Mapping):
            continue
        backoff = meta.get("oom_backoff_count")
        if isinstance(backoff, (int, float)) and backoff > 0:
            return {
                "oom_backoff_count": int(backoff),
                "step": phase.get("step"),
                "ts": phase.get("ts"),
            }
    return None


def _window_metrics_summary(
    state: ProfilerState, filters: Mapping[str, object]
) -> dict[str, dict[str, float] | None]:
    summary: dict[str, dict[str, float] | None] = {}
    names = state.get_metric_names()
    for name in names:
        values = _metric_values(state, name, filters)
        summary[name] = _summary_stats(values)
    return summary


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if value_normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _filter_entry(entry: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    since_step = cast(int | None, filters.get("since_step"))
    until_step = cast(int | None, filters.get("until_step"))
    since_ts = cast(float | None, filters.get("since_ts"))
    until_ts = cast(float | None, filters.get("until_ts"))

    if since_step is not None:
        step = entry.get("step")
        if step is None or not isinstance(step, int) or step < since_step:
            return False
    if until_step is not None:
        step = entry.get("step")
        if step is None or not isinstance(step, int) or step > until_step:
            return False
    if since_ts is not None:
        ts = entry.get("ts")
        if ts is None or not isinstance(ts, (int, float)) or float(ts) < since_ts:
            return False
    if until_ts is not None:
        ts = entry.get("ts")
        if ts is None or not isinstance(ts, (int, float)) or float(ts) > until_ts:
            return False
    return True


def _get_nested_value(data: Mapping[str, object], path: list[str]) -> object:
    current: Mapping[str, object] = data
    for index, key in enumerate(path):
        if key not in current:
            return _MISSING
        value = current[key]
        if index == len(path) - 1:
            return value
        if not isinstance(value, Mapping):
            return _MISSING
        current = cast(Mapping[str, object], value)
    return _MISSING


def _set_nested_value(
    target: dict[str, object], path: list[str], value: object
) -> None:
    current: dict[str, object] = target
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = cast(dict[str, object], next_value)
    current[path[-1]] = value


def _select_fields(
    entry: Mapping[str, object], fields: list[list[str]] | None
) -> dict[str, object]:
    if not fields:
        return dict(entry)
    selected: dict[str, object] = {}
    for path in fields:
        value = _get_nested_value(entry, path)
        if value is _MISSING:
            continue
        _set_nested_value(selected, path, value)
    return selected


def _iter_filtered_entries(
    entries: Iterable[Mapping[str, object]],
    filters: Mapping[str, object],
    *,
    predicate: Callable[[Mapping[str, object]], bool] | None = None,
    every: int = 1,
) -> Iterable[dict[str, object]]:
    fields = cast(list[list[str]] | None, filters.get("fields"))
    limit = cast(int | None, filters.get("limit"))
    offset = cast(int, filters.get("offset") or 0)

    emitted = 0
    matched = 0
    for entry in entries:
        if not _filter_entry(entry, filters):
            continue
        if predicate is not None and not predicate(entry):
            continue
        if every > 1 and matched % every != 0:
            matched += 1
            continue
        if matched < offset:
            matched += 1
            continue
        matched += 1
        yield _select_fields(entry, fields)
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def _stream_json_list(
    items: Iterable[dict[str, object]],
    state: ProfilerState,
    *,
    payload: Mapping[str, object] | None = None,
    window: Mapping[str, object] | None = None,
    sampling: Mapping[str, object] | None = None,
) -> StreamingResponse:
    header: dict[str, object] = {
        "status": "ok",
    }
    if payload:
        for key, value in payload.items():
            header[key] = value
    for key, value in _schema_metadata(state, window=window, sampling=sampling).items():
        header[key] = value
    header_json = json.dumps(header, separators=(",", ":"))
    preamble = f'{header_json[:-1]},"data":['

    def _iterator() -> Iterable[bytes]:
        yield preamble.encode("utf-8")
        first = True
        for item in items:
            if first:
                first = False
            else:
                yield b","
            yield json.dumps(item, separators=(",", ":")).encode("utf-8")
        yield b"]}"

    return StreamingResponse(_iterator(), media_type="application/json")


def _window_from_filters(filters: Mapping[str, object]) -> dict[str, object]:
    return {
        "since_step": filters.get("since_step"),
        "until_step": filters.get("until_step"),
        "since_ts": filters.get("since_ts"),
        "until_ts": filters.get("until_ts"),
    }


def _parse_names(value: str | None) -> set[str] | None:
    if not value:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def _stat_value(stats: Mapping[str, object], keys: list[str]) -> float | None:
    for key in keys:
        path = key.split(".")
        value = _get_nested_value(stats, path)
        if value is _MISSING:
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _fragmentation_ratio(reserved: float | None, active: float | None) -> float | None:
    if reserved is None or active is None or reserved <= 0:
        return None
    return max((reserved - active) / reserved, 0.0)


class _Missing:
    pass


_MISSING = _Missing()


def _sampling_from_filters(
    filters: Mapping[str, object], *, every: int = 1
) -> dict[str, object]:
    return {
        "maxlen": ProfilerState.get_instance().maxlen,
        "limit": filters.get("limit"),
        "offset": filters.get("offset"),
        "every": every,
        "fields": filters.get("fields"),
    }


def _latest_trace_key_averages(state: ProfilerState) -> list[Mapping[str, object]]:
    summary = _latest_trace_summary(state) or {}
    key_averages = summary.get("key_averages")
    if isinstance(key_averages, list):
        return [entry for entry in key_averages if isinstance(entry, Mapping)]
    return []


def _trace_phase_intervals(
    trace: Mapping[str, object],
) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    start_by_key: dict[tuple[str, int | None, int | None], float] = {}
    for event in _trace_events(trace):
        name = event.get("name")
        if not isinstance(name, str) or not name.startswith("phase:"):
            continue
        ts = event.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        tid = cast(
            int | None, event.get("tid") if isinstance(event.get("tid"), int) else None
        )
        pid = cast(
            int | None, event.get("pid") if isinstance(event.get("pid"), int) else None
        )
        parts = name.split(":")
        phase = parts[1] if len(parts) > 1 else name
        ph = event.get("ph")
        if ph == "X":
            dur = event.get("dur")
            if isinstance(dur, (int, float)):
                intervals.append((float(ts), float(ts) + float(dur), phase))
        elif ph == "B":
            start_by_key[(phase, pid, tid)] = float(ts)
        elif ph == "E":
            key = (phase, pid, tid)
            start = start_by_key.pop(key, None)
            if start is not None:
                intervals.append((start, float(ts), phase))
    return intervals


def _error_response(
    state: ProfilerState,
    message: str,
    *,
    status_code: int = 500,
    window: Mapping[str, object] | None = None,
    sampling: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> JSONResponse:
    payload: dict[str, object] = {
        "status": "error",
        "error": message,
    }
    if extra:
        payload.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=_with_schema(payload, state, window=window, sampling=sampling),
    )


def _guardrail_payload(
    *,
    reason: str,
    alternatives: list[str] | None = None,
    force_export: bool | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "reason": reason,
        "alternatives": alternatives or list(_GUARDRAIL_ALTERNATIVES),
    }
    if force_export is not None:
        payload["force_export"] = force_export
    if details:
        payload.update(details)
    return payload


def _estimate_trace_size_bytes(
    *,
    active_steps: int,
    estimated_events_per_step: int,
    bytes_per_event: int,
    record_shapes: bool,
    with_stack: bool,
    profile_memory: bool,
) -> int:
    multiplier = 1.0
    if record_shapes:
        multiplier += 0.5
    if with_stack:
        multiplier += 1.0
    if profile_memory:
        multiplier += 0.3
    return int(active_steps * estimated_events_per_step * bytes_per_event * multiplier)


def _deep_profile_config(state: ProfilerState) -> dict[str, object]:
    config_obj = state.get_deep_profile_config()
    if config_obj is None:
        return {}
    return {
        "steps": config_obj.steps,
        "wait": config_obj.wait,
        "warmup": config_obj.warmup,
        "profile_memory": config_obj.profile_memory,
        "record_shapes": config_obj.record_shapes,
        "with_stack": config_obj.with_stack,
        "sync": config_obj.sync,
        "output_dir": config_obj.output_dir,
        "trace_name": config_obj.trace_name,
        "force_export": config_obj.force_export,
        "estimated_events_per_step": config_obj.estimated_events_per_step,
        "bytes_per_event": config_obj.bytes_per_event,
        "trace_phase": config_obj.trace_phase,
        "max_duration_s": config_obj.max_duration_s,
    }


def _latest_trace_summary(state: ProfilerState) -> dict[str, object] | None:
    for entry in reversed(list(state.trace_summaries)):
        summary = entry.get("summary")
        if isinstance(summary, Mapping):
            return dict(cast(Mapping[str, object], summary))
    return None


def _resolve_trace_path(state: ProfilerState, request: Request) -> str | None:
    params = request.query_params
    path_override = params.get("path") or params.get("trace_path")
    if path_override:
        return path_override
    summary = _latest_trace_summary(state)
    if summary:
        trace_path = summary.get("trace_path")
        if isinstance(trace_path, str):
            return trace_path
    last_trace = state.get_last_trace_path()
    if isinstance(last_trace, str) and last_trace:
        return last_trace
    config = _deep_profile_config(state)
    trace_name = config.get("trace_name")
    output_dir = config.get("output_dir")
    if isinstance(trace_name, str):
        candidate = trace_name
        if not Path(candidate).suffix:
            candidate = f"{candidate}.json"
        if isinstance(output_dir, str) and output_dir:
            return str(Path(output_dir) / candidate)
        return candidate
    return None


def _load_json_file(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return cast(object, json.load(handle))


def _load_trace(
    path: str,
    *,
    force: bool,
    max_bytes: int = _TRACE_MAX_BYTES,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return None, {"reason": "not_found"}
    size_bytes = trace_path.stat().st_size
    if size_bytes > max_bytes and not force:
        return None, {
            "reason": "too_large",
            "size_bytes": size_bytes,
            "max_bytes": max_bytes,
        }
    try:
        raw = _load_json_file(trace_path)
    except Exception as exc:
        logger.exception("Failed to load trace %s", trace_path)
        return None, {"reason": "load_failed", "detail": str(exc)}
    if isinstance(raw, Mapping):
        return dict(cast(Mapping[str, object], raw)), {"size_bytes": size_bytes}
    return None, {"reason": "invalid_format", "size_bytes": size_bytes}


def _trace_metadata(trace: Mapping[str, object]) -> dict[str, object]:
    events = trace.get("traceEvents")
    last_valid_ts: float | None = None
    events_list: list[Mapping[str, object]] = []
    if isinstance(events, list):
        for event_raw in cast(list[object], events):
            if isinstance(event_raw, Mapping):
                events_list.append(cast(Mapping[str, object], event_raw))
        for event in events_list:
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                last_valid_ts = max(last_valid_ts or float("-inf"), float(ts))
        if last_valid_ts == float("-inf"):
            last_valid_ts = None
    truncated = bool(trace.get("truncated", False))
    return {
        "event_count": len(events_list),
        "truncated": truncated,
        "last_valid_ts": last_valid_ts,
    }


def _get_snapshot_dir(request: Request) -> Path:
    override = request.query_params.get("dir")
    if override:
        return Path(override)
    env_override = os.getenv("VRAM_PROFILER_SNAPSHOT_DIR")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent / "snapshots"


def _iter_snapshot_files(snapshot_dir: Path) -> list[Path]:
    if not snapshot_dir.exists():
        return []
    return sorted(
        snapshot_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )


def _load_snapshot(snapshot_path: Path) -> dict[str, object] | None:
    if not snapshot_path.exists():
        return None
    try:
        raw = _load_json_file(snapshot_path)
    except Exception as exc:
        logger.exception("Failed to load snapshot %s: %s", snapshot_path, exc)
        return None
    if isinstance(raw, Mapping):
        return dict(cast(Mapping[str, object], raw))
    return None


def _snapshot_by_id(
    request: Request, snapshot_id: str
) -> tuple[Path | None, dict[str, object] | None]:
    snapshot_dir = _get_snapshot_dir(request)
    snapshot_path = snapshot_dir / f"{snapshot_id}.json"
    if not snapshot_path.exists():
        return None, None
    return snapshot_path, _load_snapshot(snapshot_path)


def _parse_compare_window(request: Request, prefix: str) -> dict[str, object]:
    params = request.query_params
    return {
        "since_step": _parse_int(params.get(f"{prefix}_since_step")),
        "since_ts": _parse_float(params.get(f"{prefix}_since_ts")),
        "until_ts": _parse_float(params.get(f"{prefix}_until_ts")),
        "until_step": _parse_int(params.get(f"{prefix}_until_step")),
    }


def _normalize_memory_stats(
    stats: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    stats_bytes: dict[str, object] = {}
    for raw_key, value in stats.items():
        key = str(raw_key)
        if key.endswith("_gb") and isinstance(value, (int, float)):
            stats_bytes[key.replace("_gb", "_bytes")] = float(value) * (1024**3)
        elif key.endswith("_bytes") and isinstance(value, (int, float)):
            stats_bytes[key] = float(value)
        else:
            stats_bytes[key] = value

    if stats_bytes == stats:
        return dict(stats), None
    return stats_bytes, {str(k): v for k, v in stats.items()}


def _memory_entry_payload(entry: Mapping[str, object]) -> dict[str, object]:
    payload = dict(entry)
    stats = entry.get("stats")
    if isinstance(stats, Mapping):
        normalized, raw = _normalize_memory_stats(cast(Mapping[str, object], stats))
        payload["stats"] = normalized
        if raw is not None:
            payload["stats_raw"] = raw
    return payload


def _timeline_entry(entry: Mapping[str, object]) -> dict[str, object] | None:
    stats = entry.get("stats")
    if not isinstance(stats, Mapping):
        return None
    normalized, _ = _normalize_memory_stats(cast(Mapping[str, object], stats))
    reserved = _stat_value(
        normalized,
        [
            "reserved_bytes",
            "reserved",
            "total_reserved_bytes",
            "total_reserved",
        ],
    )
    allocated = _stat_value(
        normalized,
        [
            "allocated_bytes",
            "allocated",
            "active_bytes",
            "active",
        ],
    )
    ts = entry.get("ts")
    if not isinstance(ts, (int, float)):
        return None
    return {
        "t": float(ts),
        "active": allocated,
        "total": reserved,
        "active_bytes": allocated,
        "total_bytes": reserved,
        "step": entry.get("step"),
        "context": entry.get("context"),
    }


def _iter_timeline_entries(
    entries: Iterable[Mapping[str, object]],
    filters: Mapping[str, object],
    *,
    every: int = 1,
) -> Iterable[dict[str, object]]:
    fields = cast(list[list[str]] | None, filters.get("fields"))
    limit = cast(int | None, filters.get("limit"))
    offset = cast(int, filters.get("offset") or 0)
    emitted = 0
    matched = 0
    for entry in entries:
        if not _filter_entry(entry, filters):
            continue
        timeline = _timeline_entry(entry)
        if timeline is None:
            continue
        if every > 1 and matched % every != 0:
            matched += 1
            continue
        if matched < offset:
            matched += 1
            continue
        matched += 1
        payload = _select_fields(timeline, fields)
        yield payload
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def _phase_duration_stats(
    state: ProfilerState,
    filters: Mapping[str, object],
) -> list[dict[str, object]]:
    totals: dict[str, list[float]] = {}
    for entry in state.phases:
        if not _filter_entry(entry, filters):
            continue
        name = entry.get("name")
        duration = entry.get("duration_ms")
        if not isinstance(name, str) or not isinstance(duration, (int, float)):
            continue
        totals.setdefault(name, []).append(float(duration))
    payload: list[dict[str, object]] = []
    for name, durations in totals.items():
        if not durations:
            continue
        payload.append(
            {
                "name": name,
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
            }
        )
    payload.sort(key=lambda item: cast(float, item.get("total_ms", 0.0)), reverse=True)
    return payload


def _trace_events(trace: Mapping[str, object]) -> list[Mapping[str, object]]:
    events = trace.get("traceEvents")
    if isinstance(events, list):
        payload: list[Mapping[str, object]] = []
        for event_raw in cast(list[object], events):
            if isinstance(event_raw, Mapping):
                payload.append(cast(Mapping[str, object], event_raw))
        return payload
    return []


def _trace_kernel_stats(trace: Mapping[str, object]) -> list[dict[str, object]]:
    kernels: dict[tuple[str, str | None], dict[str, object]] = {}
    phase_intervals = _trace_phase_intervals(trace)
    for event in _trace_events(trace):
        name = event.get("name")
        if not isinstance(name, str):
            continue
        cat = event.get("cat")
        if (
            isinstance(cat, str)
            and "kernel" not in cat.lower()
            and "cuda" not in cat.lower()
        ):
            continue
        duration = event.get("dur")
        if not isinstance(duration, (int, float)):
            continue
        phase = None
        ts = event.get("ts")
        if isinstance(ts, (int, float)):
            for start, end, phase_name in phase_intervals:
                if start <= float(ts) <= end:
                    phase = phase_name
                    break
        args = event.get("args")
        if phase is None and isinstance(args, Mapping):
            phase_value = args.get("phase")
            if isinstance(phase_value, str):
                phase = phase_value
        entry = kernels.setdefault(
            (name, phase),
            {"name": name, "count": 0, "total_us": 0.0, "max_us": 0.0, "phase": phase},
        )
        entry["count"] = cast(int, entry["count"]) + 1
        entry["total_us"] = cast(float, entry["total_us"]) + float(duration)
        entry["max_us"] = max(cast(float, entry["max_us"]), float(duration))
    payload = list(kernels.values())
    payload.sort(key=lambda item: cast(float, item.get("total_us", 0.0)), reverse=True)
    return payload


def _detect_anomalies(
    entries: Iterable[Mapping[str, object]],
    *,
    metric: str,
    window_size: int,
    z_threshold: float = 2.0,
) -> list[dict[str, object]]:
    values: list[tuple[float, Mapping[str, object]]] = []
    for entry in entries:
        if entry.get("name") != metric:
            continue
        value = entry.get("value")
        if isinstance(value, (int, float)):
            values.append((float(value), entry))
    if not values:
        return []
    values = values[-window_size:]
    series = [item[0] for item in values]
    if len(series) < 2:
        return []
    mean = sum(series) / len(series)
    variance = sum((val - mean) ** 2 for val in series) / len(series)
    std = (variance**0.5) if isinstance(variance, (int, float)) else 0.0
    if std == 0:
        return []
    anomalies: list[dict[str, object]] = []
    for value, entry in values:
        z = float((value - mean) / std)
        if abs(z) < z_threshold:
            continue
        anomalies.append(
            {
                "metric": metric,
                "value": value,
                "z_score": z,
                "severity": abs(z),
                "step": entry.get("step"),
                "ts": entry.get("ts"),
                "tags": entry.get("tags"),
            }
        )
    return anomalies


def _trace_stack_stats(trace: Mapping[str, object]) -> list[dict[str, object]]:
    stacks: dict[str, dict[str, object]] = {}
    for event in _trace_events(trace):
        stack = event.get("stack") or event.get("stackTrace")
        if stack is None:
            args = event.get("args")
            if isinstance(args, Mapping):
                args_map = cast(Mapping[str, object], args)
                stack = args_map.get("stack") or args_map.get("stackTrace")
        if isinstance(stack, str):
            frames = [line for line in stack.split("\n") if line]
        elif isinstance(stack, list):
            frames: list[str] = []
            for frame_raw in cast(list[object], stack):
                if frame_raw is None:
                    continue
                frames.append(str(frame_raw))
        else:
            continue
        if not frames:
            continue
        key = "|".join(frames)
        entry = stacks.setdefault(key, {"frames": frames, "count": 0})
        entry["count"] = cast(int, entry["count"]) + 1
    payload: list[dict[str, object]] = []
    for index, entry in enumerate(stacks.values()):
        payload.append({"id": index, **entry})
    payload.sort(key=lambda item: cast(int, item.get("count", 0)), reverse=True)
    return payload


def _trace_threads(trace: Mapping[str, object]) -> list[dict[str, object]]:
    threads: dict[tuple[int, int], dict[str, object]] = {}
    names: dict[tuple[int, int], str] = {}
    for event in _trace_events(trace):
        tid = event.get("tid")
        pid = event.get("pid")
        if not isinstance(tid, int) or not isinstance(pid, int):
            continue
        key = (pid, tid)
        entry = threads.setdefault(key, {"pid": pid, "tid": tid, "count": 0})
        entry["count"] = cast(int, entry["count"]) + 1
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            args = event.get("args")
            if isinstance(args, Mapping):
                args_map = cast(Mapping[str, object], args)
                thread_name = args_map.get("name")
                if isinstance(thread_name, str):
                    names[key] = thread_name
    payload: list[dict[str, object]] = []
    for key, entry in threads.items():
        thread_name = names.get(key)
        if thread_name:
            entry["name"] = thread_name
        payload.append(entry)
    payload.sort(key=lambda item: cast(int, item.get("count", 0)), reverse=True)
    return payload


def _snapshot_events(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    device_traces = snapshot.get("device_traces")
    events: list[dict[str, object]] = []
    if isinstance(device_traces, list):
        for trace_raw in cast(list[object], device_traces):
            if isinstance(trace_raw, list):
                trace_list: list[Mapping[str, object]] = []
                for event_raw in cast(list[object], trace_raw):
                    if isinstance(event_raw, Mapping):
                        trace_list.append(cast(Mapping[str, object], event_raw))
                for event in trace_list:
                    events.append(dict(event))
    return events


def _snapshot_segments(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    segments = snapshot.get("segments")
    if isinstance(segments, list):
        segment_list: list[Mapping[str, object]] = []
        for segment_raw in cast(list[object], segments):
            if isinstance(segment_raw, Mapping):
                segment_list.append(cast(Mapping[str, object], segment_raw))
        return [dict(segment) for segment in segment_list]
    return []


def _snapshot_timeline(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    for event in _snapshot_events(snapshot):
        ts = event.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        timeline.append(
            {
                "t": float(ts),
                "action": event.get("action"),
                "size": event.get("size"),
                "addr": event.get("addr"),
                "stream": event.get("stream"),
            }
        )
    timeline.sort(key=lambda item: cast(float, item.get("t", 0.0)))
    return timeline


def _snapshot_bottlenecks(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    events = _snapshot_events(snapshot)
    large_events = [
        event for event in events if isinstance(event.get("size"), (int, float))
    ]
    large_events.sort(key=lambda item: cast(float, item.get("size", 0.0)), reverse=True)
    payload: list[dict[str, object]] = []
    for event in large_events[:50]:
        payload.append(
            {
                "size": event.get("size"),
                "action": event.get("action"),
                "addr": event.get("addr"),
                "segment": event.get("segment"),
                "stream": event.get("stream"),
            }
        )
    return payload


def _metrics_summary_for_window(
    state: ProfilerState, filters: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for entry in state.metrics:
        if not _filter_entry(entry, filters):
            continue
        name = entry.get("name")
        value = entry.get("value")
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            continue
        payload = summary.setdefault(name, {"count": 0, "sum": 0.0, "last": value})
        payload["count"] = cast(int, payload["count"]) + 1
        payload["sum"] = cast(float, payload["sum"]) + float(value)
        payload["last"] = value
    for payload in summary.values():
        count = cast(int, payload.get("count", 0))
        if count:
            payload["avg"] = cast(float, payload.get("sum", 0.0)) / count
    return summary


def _memory_summary_for_window(
    state: ProfilerState, filters: Mapping[str, object]
) -> dict[str, object]:
    peak_total: float | None = None
    peak_active: float | None = None
    last: dict[str, object] | None = None
    for entry in state.memory:
        if not _filter_entry(entry, filters):
            continue
        timeline = _timeline_entry(entry)
        if timeline is None:
            continue
        total = timeline.get("total")
        active = timeline.get("active")
        if isinstance(total, (int, float)):
            peak_total = max(peak_total or float("-inf"), float(total))
        if isinstance(active, (int, float)):
            peak_active = max(peak_active or float("-inf"), float(active))
        last = timeline
    return {
        "peak_total": None if peak_total == float("-inf") else peak_total,
        "peak_active": None if peak_active == float("-inf") else peak_active,
        "last": last,
    }


def _diff_maps(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    keys = set(left) | set(right)
    payload: dict[str, object] = {}
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        delta = None
        if isinstance(left_value, (int, float)) and isinstance(
            right_value, (int, float)
        ):
            delta = right_value - left_value
        payload[key] = {
            "left": left_value,
            "right": right_value,
            "delta": delta,
        }
    return payload


api_router = APIRouter(prefix="/api")


@api_router.get("/status")
def get_status() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        memory_manager = MemoryManager()
        last_step = state.get_last_step_info()
        gpu_name = None
        torch_module = _get_torch_module()
        if torch_module is not None:
            if torch_module.cuda.is_available():
                try:
                    gpu_name = torch_module.cuda.get_device_name(0)
                except Exception:
                    gpu_name = None
        vram_stats = memory_manager.get_memory_stats()
        buffers = {
            "metrics": state.metrics,
            "phases": state.phases,
            "memory": state.memory,
            "anomalies": state.anomalies,
            "trace_summaries": state.trace_summaries,
        }
        buffer_stats: dict[str, dict[str, object]] = {}
        for name, buffer in buffers.items():
            size = len(buffer)
            maxlen = buffer.maxlen or 0
            fill_pct = (float(size) / float(maxlen) * 100.0) if maxlen else None
            buffer_stats[name] = {
                "size": size,
                "maxlen": maxlen,
                "fill_pct": fill_pct,
            }
        payload: dict[str, object] = {
            "status": "ok",
            "timestamp": time.time(),
            "training_active": state.is_training_active(),
            "global_step": last_step.get("step"),
            "epoch": last_step.get("epoch"),
            "gpu": gpu_name,
            "vram": vram_stats,
            "deep_profile_active": state.deep_profile_active,
            "tiers": {
                "tier1": {
                    "active": state.is_training_active(),
                    "label": "always-on",
                },
                "tier2": {
                    "active": state.deep_profile_active,
                    "label": "hot-attach",
                },
            },
            "buffers": buffer_stats,
            "drop_counts": dict(state.drop_counts),
            "buffer_meta": state.get_buffer_meta(),
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build status payload")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(status_code=500, content=_with_schema(payload, state))


@api_router.get("/summary")
def get_summary() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        step_times: list[float] = []
        avg_rewards: list[float] = []
        data_latencies: list[float] = []
        phase_durations: dict[str, list[float]] = {}
        for entry in state.metrics:
            name = entry.get("name")
            value = entry.get("value")
            if name == "step_time_s" and isinstance(value, (int, float)):
                step_times.append(float(value))
            if name == "avg_reward" and isinstance(value, (int, float)):
                avg_rewards.append(float(value))
        for entry in state.phases:
            name = entry.get("name")
            duration = entry.get("duration_ms")
            if isinstance(name, str) and isinstance(duration, (int, float)):
                phase_durations.setdefault(name, []).append(float(duration))
                if name == "data":
                    data_latencies.append(float(duration))

        bottleneck_phase = None
        if phase_durations:
            totals = {name: sum(values) for name, values in phase_durations.items()}
            bottleneck_phase = max(totals.items(), key=lambda item: item[1])[0]

        step_stats = _summary_stats(step_times)
        reward_stats = _summary_stats(avg_rewards)
        data_stats = _summary_stats(data_latencies)
        vram_latest = None
        if state.memory:
            vram_latest = _memory_entry_payload(state.memory[-1]).get("stats")

        top_bottlenecks = _phase_duration_stats(state, {})[:3]
        payload = {
            "status": "ok",
            "counts": {
                "metrics": len(state.metrics),
                "phases": len(state.phases),
                "memory": len(state.memory),
                "anomalies": len(state.anomalies),
                "trace_summaries": len(state.trace_summaries),
            },
            "drop_counts": dict(state.drop_counts),
            "step_time": step_stats,
            "avg_reward": reward_stats,
            "data_latency": data_stats,
            "vram_latest": vram_latest,
            "top_phase": bottleneck_phase,
            "bottleneck_category": "phase" if bottleneck_phase else None,
            "top_bottlenecks": top_bottlenecks,
            "links": {
                "phases": "/api/phases/summary",
                "bottlenecks": "/api/bottlenecks",
                "metrics": "/api/metrics/summary",
                "timeline": "/api/timeline/summary",
            },
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build summary payload")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(status_code=500, content=_with_schema(payload, state))


@api_router.get("/config")
def get_config() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        config_snapshot = state.get_config_snapshot()
        if config_snapshot is None:
            config_snapshot = get_8gb_vram_config().to_dict()
        payload: dict[str, object] = {
            "status": "ok",
            "buffers": {
                "maxlen": state.maxlen,
            },
            "deep_profile": {
                "active": state.deep_profile_active,
            },
            "config": config_snapshot,
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build config payload")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(status_code=500, content=_with_schema(payload, state))


@api_router.get("/capabilities")
def get_capabilities() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        experimental_config_available = False
        torch_module = _get_torch_module()
        if torch_module is not None:
            profiler_module = getattr(torch_module, "profiler", None)
            experimental_config_available = bool(
                profiler_module
                and (
                    getattr(profiler_module, "ExperimentalConfig", None)
                    or getattr(profiler_module, "_ExperimentalConfig", None)
                )
            )
        payload = {
            "status": "ok",
            "features": {
                "streaming": True,
                "sse": False,
                "deep_profile": True,
                "experimental_config": experimental_config_available,
                "static_gui": True,
                "tiers": ["tier1", "tier2"],
            },
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build capabilities payload")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(status_code=500, content=_with_schema(payload, state))


@api_router.get("/metrics/latest")
def get_metrics_latest(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    name_filter = request.query_params.get("name")
    window = _window_from_filters(filters)
    try:
        latest: dict[str, object] | None = None
        for entry in reversed(list(state.metrics)):
            if _filter_entry(entry, filters):
                if name_filter and entry.get("name") != name_filter:
                    continue
                latest = _select_fields(
                    entry, cast(list[list[str]] | None, filters.get("fields"))
                )
                break
        payload = {"status": "ok", "metric": latest}
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build metrics latest payload")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/metrics/history")
def get_metrics_history(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        items = _iter_filtered_entries(state.metrics, filters)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream metrics history")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/metrics/sample")
def get_metrics_sample(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    every = _parse_int(request.query_params.get("every")) or 1
    window = _window_from_filters(filters)
    try:
        items = _iter_filtered_entries(state.metrics, filters, every=max(every, 1))
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters, every=max(every, 1)),
        )
    except Exception as exc:
        logger.exception("Failed to stream metrics sample")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload,
                state,
                window=window,
                sampling=_sampling_from_filters(filters, every=max(every, 1)),
            ),
        )


@api_router.get("/metrics/summary")
def get_metrics_summary(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        counts: dict[str, int] = {}
        total = 0
        last_ts: float | None = None
        for entry in state.metrics:
            if not _filter_entry(entry, filters):
                continue
            total += 1
            name = entry.get("name")
            if isinstance(name, str):
                counts[name] = counts.get(name, 0) + 1
            ts = entry.get("ts")
            if isinstance(ts, (int, float)):
                last_ts = ts
        summary = _window_metrics_summary(state, filters)
        payload = {
            "status": "ok",
            "total": total,
            "by_name": counts,
            "last_ts": last_ts,
            "summary": summary,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build metrics summary")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/metrics/diff")
def get_metrics_diff(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    names = _parse_names(request.query_params.get("names"))
    from_step = _parse_int(request.query_params.get("from_step"))
    to_step = _parse_int(request.query_params.get("to_step"))
    window = _window_from_filters(filters)
    try:
        if from_step is not None or to_step is not None:
            window_filters = dict(filters)
            if from_step is not None:
                window_filters["since_step"] = from_step
            if to_step is not None:
                window_filters["until_step"] = to_step
            diffs: dict[str, object] = {}
            metric_names = names or set(state.get_metric_names())
            for metric_name in metric_names:
                if not isinstance(metric_name, str):
                    continue
                values = _metric_values(state, metric_name, window_filters)
                if not values:
                    diffs[metric_name] = {
                        "latest": None,
                        "previous": None,
                        "delta": None,
                    }
                    continue
                latest_value = values[-1]
                previous_value = values[0]
                diffs[metric_name] = {
                    "latest": latest_value,
                    "previous": previous_value,
                    "delta": latest_value - previous_value,
                }
        else:
            latest_by_name: dict[str, object] = {}
            prev_by_name: dict[str, object] = {}
            for entry in reversed(list(state.metrics)):
                if not _filter_entry(entry, filters):
                    continue
                name = entry.get("name")
                if not isinstance(name, str):
                    continue
                if names and name not in names:
                    continue
                if name not in latest_by_name:
                    latest_by_name[name] = entry
                elif name not in prev_by_name:
                    prev_by_name[name] = entry
                if name in latest_by_name and name in prev_by_name:
                    if names and len(prev_by_name) == len(names):
                        break
            diffs = {}
            for name, latest in latest_by_name.items():
                if not isinstance(latest, Mapping):
                    continue
                latest_map = cast(Mapping[str, object], latest)
                prev = prev_by_name.get(name)
                latest_value = latest_map.get("value")
                prev_value = None
                if isinstance(prev, Mapping):
                    prev_map = cast(Mapping[str, object], prev)
                    prev_value = prev_map.get("value")
                delta = None
                if isinstance(latest_value, (int, float)) and isinstance(
                    prev_value, (int, float)
                ):
                    delta = latest_value - prev_value
                diffs[name] = {
                    "latest": latest_value,
                    "previous": prev_value,
                    "delta": delta,
                }
        payload = {
            "status": "ok",
            "diffs": diffs,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build metrics diff")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/phases")
def get_phases(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        items = _iter_filtered_entries(state.phases, filters)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream phases")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/phases/summary")
def get_phases_summary(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        totals: dict[str, int] = {}
        durations: dict[str, list[float]] = {}
        for entry in state.phases:
            if not _filter_entry(entry, filters):
                continue
            name = entry.get("name")
            if isinstance(name, str):
                totals[name] = totals.get(name, 0) + 1
                duration = entry.get("duration_ms")
                if isinstance(duration, (int, float)):
                    durations.setdefault(name, []).append(float(duration))
        duration_summary: dict[str, object] = {}
        total_all = sum(sum(values) for values in durations.values()) or 0.0
        for name, values in durations.items():
            stats = _summary_stats(values)
            if stats is None:
                continue
            share = (sum(values) / total_all) if total_all else None
            duration_summary[name] = {
                **stats,
                "share": share,
            }
        payload = {
            "status": "ok",
            "counts": totals,
            "durations": duration_summary,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build phases summary")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/phases/overlap")
def get_phases_overlap(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        windows: list[tuple[float, float, str, int | None]] = []
        start_times: dict[tuple[int | None, str], float] = {}
        for entry in state.phases:
            if not _filter_entry(entry, filters):
                continue
            name = entry.get("name")
            status = entry.get("status")
            step = entry.get("step")
            ts = entry.get("ts")
            if not isinstance(name, str) or not isinstance(status, str):
                continue
            if not isinstance(ts, (int, float)):
                continue
            key = (step if isinstance(step, int) else None, name)
            if status == "start":
                start_times[key] = float(ts)
            elif status == "end":
                start = start_times.pop(key, None)
                duration_ms = entry.get("duration_ms")
                if start is not None:
                    end = start + (
                        float(duration_ms) / 1000.0
                        if isinstance(duration_ms, (int, float))
                        else 0.0
                    )
                    windows.append(
                        (start, end, name, step if isinstance(step, int) else None)
                    )

        overlaps = 0
        max_concurrent = 0
        events: list[tuple[float, int]] = []
        for start, end, _, _ in windows:
            events.append((start, 1))
            events.append((end, -1))
        events.sort()
        current = 0
        for _, delta in events:
            current += delta
            if current > 1:
                overlaps += 1
            max_concurrent = max(max_concurrent, current)
        cpu_ms = None
        gpu_ms = None
        key_averages = _latest_trace_key_averages(state)
        if key_averages:
            cpu_total = sum(
                cast(float, entry.get("cpu_time_total"))
                for entry in key_averages
                if isinstance(entry.get("cpu_time_total"), (int, float))
            )
            cuda_total = sum(
                cast(float, entry.get("cuda_time_total"))
                for entry in key_averages
                if isinstance(entry.get("cuda_time_total"), (int, float))
            )
            cpu_ms = round(cpu_total / 1000.0, 3) if cpu_total else None
            gpu_ms = round(cuda_total / 1000.0, 3) if cuda_total else None

        total_ms = 0.0
        total_samples = 0
        for entry in state.phases:
            if not _filter_entry(entry, filters):
                continue
            duration = entry.get("duration_ms")
            if isinstance(duration, (int, float)):
                total_ms += float(duration)
                total_samples += 1
        idle_ms = None
        if total_samples and (cpu_ms is not None or gpu_ms is not None):
            busy_ms = max(cpu_ms or 0.0, gpu_ms or 0.0)
            idle_ms = round(max(total_ms - busy_ms, 0.0), 3)

        payload = {
            "status": "ok",
            "total_windows": len(windows),
            "overlap_events": overlaps,
            "max_concurrent": max_concurrent,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "idle_ms": idle_ms,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build phases overlap")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/micro_batching/history")
def get_micro_batching_history(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)

    def _predicate(entry: Mapping[str, object]) -> bool:
        name = entry.get("name")
        if isinstance(name, str) and "micro" in name:
            return True
        meta = entry.get("meta")
        if isinstance(meta, Mapping):
            meta_map = cast(Mapping[str, object], meta)
            return any(str(key).startswith("micro") for key in meta_map.keys())
        return False

    try:
        items = _iter_filtered_entries(state.phases, filters, predicate=_predicate)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream micro batching history")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/memory/latest")
def get_memory_latest(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        latest: dict[str, object] | None = None
        for entry in reversed(list(state.memory)):
            if _filter_entry(entry, filters):
                latest = _memory_entry_payload(entry)
                break
        payload = {"status": "ok", "memory": latest}
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build memory latest")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/memory/history")
def get_memory_history(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        items = (
            _memory_entry_payload(entry)
            for entry in _iter_filtered_entries(state.memory, filters)
        )
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream memory history")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/memory/summary")
def get_memory_summary(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        total = 0
        last_ts: float | None = None
        latest_stats: dict[str, object] | None = None
        for entry in state.memory:
            if not _filter_entry(entry, filters):
                continue
            total += 1
            ts = entry.get("ts")
            if isinstance(ts, (int, float)):
                last_ts = ts
            payload = _memory_entry_payload(entry)
            latest_stats = (
                cast(dict[str, object], payload.get("stats"))
                if payload.get("stats")
                else latest_stats
            )
        payload = {
            "status": "ok",
            "count": total,
            "last_ts": last_ts,
            "latest_stats": latest_stats,
            "summary": _memory_summary_for_window(state, filters),
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build memory summary")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/memory/fragmentation")
def get_memory_fragmentation(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        latest_ratio: float | None = None
        peak_ratio: float | None = None
        for entry in state.memory:
            if not _filter_entry(entry, filters):
                continue
            stats = entry.get("stats")
            if not isinstance(stats, Mapping):
                continue
            normalized, _ = _normalize_memory_stats(cast(Mapping[str, object], stats))
            reserved = _stat_value(
                normalized,
                [
                    "reserved_bytes",
                    "reserved",
                    "total_reserved_bytes",
                    "total_reserved",
                ],
            )
            allocated = _stat_value(
                normalized, ["allocated_bytes", "allocated", "active_bytes", "active"]
            )
            ratio = _fragmentation_ratio(reserved, allocated)
            if ratio is None:
                continue
            latest_ratio = ratio
            if peak_ratio is None or ratio > peak_ratio:
                peak_ratio = ratio
        payload = {
            "status": "ok",
            "latest": latest_ratio,
            "peak": peak_ratio,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build memory fragmentation")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/memory/ooms")
def get_memory_ooms(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)

    def _predicate(entry: Mapping[str, object]) -> bool:
        return entry.get("kind") == "oom"

    try:
        items = _iter_filtered_entries(state.anomalies, filters, predicate=_predicate)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream memory ooms")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/timeline")
def get_timeline(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        items = _iter_timeline_entries(state.memory, filters)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream timeline")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/timeline/downsample")
def get_timeline_downsample(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    every = _parse_int(request.query_params.get("every")) or 10
    window = _window_from_filters(filters)
    try:
        items = _iter_timeline_entries(state.memory, filters, every=max(every, 1))
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters, every=max(every, 1)),
        )
    except Exception as exc:
        logger.exception("Failed to stream timeline downsample")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload,
                state,
                window=window,
                sampling=_sampling_from_filters(filters, every=max(every, 1)),
            ),
        )


@api_router.get("/timeline/summary")
def get_timeline_summary(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        points = 0
        peak_total: float | None = None
        peak_active: float | None = None
        last_ts: float | None = None
        for entry in state.memory:
            if not _filter_entry(entry, filters):
                continue
            timeline = _timeline_entry(entry)
            if timeline is None:
                continue
            points += 1
            total = timeline.get("total")
            active = timeline.get("active")
            if isinstance(total, (int, float)):
                peak_total = max(peak_total or float("-inf"), float(total))
            if isinstance(active, (int, float)):
                peak_active = max(peak_active or float("-inf"), float(active))
            ts = timeline.get("t")
            if isinstance(ts, (int, float)):
                last_ts = ts
        payload = {
            "status": "ok",
            "points": points,
            "peak_total": None if peak_total == float("-inf") else peak_total,
            "peak_active": None if peak_active == float("-inf") else peak_active,
            "last_ts": last_ts,
            "event_count": points,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build timeline summary")
        payload = {"status": "error", "error": str(exc)}
        return JSONResponse(
            status_code=500,
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            ),
        )


@api_router.get("/bottlenecks")
def get_bottlenecks(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        phases = _phase_duration_stats(state, filters)
        trace_path = _resolve_trace_path(state, request)
        trace_info: dict[str, object] = {"available": False}
        kernels: list[dict[str, object]] = []
        stacks: list[dict[str, object]] = []
        if trace_path:
            force = _parse_bool(request.query_params.get("force")) or False
            trace, info = _load_trace(trace_path, force=force)
            trace_info = {"available": trace is not None, **info, "path": trace_path}
            if trace:
                kernels = _trace_kernel_stats(trace)[:20]
                stacks = _trace_stack_stats(trace)[:20]
                trace_info.update(_trace_metadata(trace))
        top = _parse_int(request.query_params.get("top")) or 20
        payload = {
            "status": "ok",
            "phases": phases[:top],
            "kernels": kernels[:top],
            "stacks": stacks[:top],
            "trace": trace_info,
            "top": top,
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build bottlenecks payload")
        return _error_response(
            state, str(exc), window=window, sampling=_sampling_from_filters(filters)
        )


@api_router.get("/bottlenecks/by_phase")
def get_bottlenecks_by_phase(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        payload = {
            "status": "ok",
            "phases": _phase_duration_stats(state, filters),
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to build bottlenecks by phase")
        return _error_response(
            state, str(exc), window=window, sampling=_sampling_from_filters(filters)
        )


@api_router.get("/bottlenecks/by_kernel")
def get_bottlenecks_by_kernel(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        return _error_response(
            state,
            "Trace not found",
            status_code=404,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        status = 413 if info.get("reason") == "too_large" else 404
        return _error_response(
            state,
            "Trace unavailable",
            status_code=status,
            window=window,
            sampling=_sampling_from_filters(filters),
            extra=info,
        )
    name_filter = request.query_params.get("name")
    phase_filter = request.query_params.get("phase")
    sort_key = request.query_params.get("sort") or "total_us"
    kernels = _trace_kernel_stats(trace)
    if name_filter:
        kernels = [
            entry for entry in kernels if name_filter in str(entry.get("name", ""))
        ]
    if phase_filter:
        kernels = [entry for entry in kernels if entry.get("phase") == phase_filter]
    if sort_key in {"total_us", "max_us", "count"}:
        kernels.sort(
            key=lambda item: cast(float, item.get(sort_key, 0.0)), reverse=True
        )
    payload = {
        "status": "ok",
        "kernels": kernels,
        "trace": {"path": trace_path, **info, **_trace_metadata(trace)},
    }
    return JSONResponse(
        content=_with_schema(
            payload, state, window=window, sampling=_sampling_from_filters(filters)
        )
    )


@api_router.get("/bottlenecks/by_stack")
def get_bottlenecks_by_stack(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        return _error_response(
            state,
            "Trace not found",
            status_code=404,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        status = 413 if info.get("reason") == "too_large" else 404
        return _error_response(
            state,
            "Trace unavailable",
            status_code=status,
            window=window,
            sampling=_sampling_from_filters(filters),
            extra=info,
        )
    payload = {
        "status": "ok",
        "stacks": _trace_stack_stats(trace),
        "trace": {"path": trace_path, **info, **_trace_metadata(trace)},
    }
    return JSONResponse(
        content=_with_schema(
            payload, state, window=window, sampling=_sampling_from_filters(filters)
        )
    )


@api_router.post("/profile/start")
def start_profile(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    params = request.query_params
    try:
        if state.deep_profile_active:
            return _error_response(
                state, "Deep profiling already active", status_code=409
            )
        safe_mode = _parse_bool(params.get("safe"))
        if safe_mode is None:
            safe_mode = True
        warnings: list[str] = []
        max_duration_s: float | None = None
        memory_manager = MemoryManager()
        free_gb = memory_manager.get_available_memory_gb()
        min_free_gb = _SAFE_MIN_FREE_VRAM_GB if safe_mode else _MIN_FREE_VRAM_GB
        if free_gb < min_free_gb:
            return _error_response(
                state,
                "Insufficient free VRAM",
                status_code=409,
                extra={
                    "guardrail": _guardrail_payload(
                        reason="insufficient_vram",
                        details={
                            "free_gb": free_gb,
                            "min_free_gb": min_free_gb,
                            "safe_mode": safe_mode,
                        },
                    ),
                },
            )
        oom_backoff = _latest_oom_backoff(state)
        if oom_backoff is not None:
            return _error_response(
                state,
                "Recent OOM backoff detected",
                status_code=409,
                extra={
                    "guardrail": _guardrail_payload(
                        reason="recent_oom_backoff",
                        details={"oom_backoff": oom_backoff},
                    ),
                },
            )
        steps = _parse_int(params.get("steps") or params.get("active"))
        wait = _parse_int(params.get("wait"))
        warmup = _parse_int(params.get("warmup"))
        profile_memory = _parse_bool(params.get("profile_memory"))
        record_shapes = _parse_bool(params.get("record_shapes"))
        with_stack = _parse_bool(params.get("with_stack"))
        sync = _parse_bool(params.get("sync"))
        output_dir = params.get("output_dir") or params.get("output")
        trace_name = params.get("trace_name") or f"deep_profile_{int(time.time())}"
        trace_phase = params.get("trace_phase") or params.get("phase")
        if safe_mode and trace_phase is None:
            trace_phase = "generation"
        trace_path = None
        if isinstance(trace_name, str):
            candidate = trace_name
            if not Path(candidate).suffix:
                candidate = f"{candidate}.json"
            if output_dir:
                trace_path = str(Path(output_dir) / candidate)
            else:
                trace_path = candidate
        force_export = _parse_bool(params.get("force")) or False
        if safe_mode:
            steps = steps if steps is not None else _SAFE_DEFAULTS["steps"]
            wait = wait if wait is not None else _SAFE_DEFAULTS["wait"]
            warmup = warmup if warmup is not None else _SAFE_DEFAULTS["warmup"]
            steps = max(min(steps, _SAFE_MAX_STEPS), _SAFE_DEFAULTS["steps"])
            wait = max(min(wait, _SAFE_MAX_WAIT), _SAFE_DEFAULTS["wait"])
            warmup = max(min(warmup, _SAFE_MAX_WARMUP), _SAFE_DEFAULTS["warmup"])
            requested_steps = _parse_int(params.get("steps") or params.get("active"))
            requested_wait = _parse_int(params.get("wait"))
            requested_warmup = _parse_int(params.get("warmup"))
            if requested_steps is not None and steps < requested_steps:
                warnings.append("steps_clamped")
            if requested_wait is not None and wait < requested_wait:
                warnings.append("wait_clamped")
            if requested_warmup is not None and warmup < requested_warmup:
                warnings.append("warmup_clamped")
            if profile_memory is None:
                profile_memory = _SAFE_DEFAULTS["profile_memory"]
            elif profile_memory:
                profile_memory = _SAFE_DEFAULTS["profile_memory"]
                warnings.append("profile_memory_disabled")
            if record_shapes is None:
                record_shapes = _SAFE_DEFAULTS["record_shapes"]
            elif record_shapes:
                record_shapes = _SAFE_DEFAULTS["record_shapes"]
                warnings.append("record_shapes_disabled")
            if with_stack is None:
                with_stack = _SAFE_DEFAULTS["with_stack"]
            elif with_stack:
                with_stack = _SAFE_DEFAULTS["with_stack"]
                warnings.append("with_stack_disabled")
            if sync is None:
                sync = _SAFE_DEFAULTS["sync"]
            elif not sync:
                sync = _SAFE_DEFAULTS["sync"]
                warnings.append("sync_enabled")
            if force_export:
                force_export = False
                warnings.append("force_export_disabled")
            max_duration_s = _parse_float(params.get("max_duration_s"))
            if max_duration_s is None:
                max_duration_s = _SAFE_DEFAULTS["max_duration_s"]
            elif max_duration_s > _SAFE_DEFAULTS["max_duration_s"]:
                max_duration_s = _SAFE_DEFAULTS["max_duration_s"]
                warnings.append("max_duration_clamped")
            elif max_duration_s <= 0:
                max_duration_s = _SAFE_DEFAULTS["max_duration_s"]
                warnings.append("max_duration_reset")
            estimated_events_per_step = (
                _parse_int(params.get("estimated_events_per_step"))
                or _SAFE_DEFAULTS["estimated_events_per_step"]
            )
            bytes_per_event = (
                _parse_int(params.get("bytes_per_event"))
                or _SAFE_DEFAULTS["bytes_per_event"]
            )
        else:
            steps = steps or 1
            wait = wait or 0
            warmup = warmup or 0
            profile_memory = (
                bool(profile_memory) if profile_memory is not None else True
            )
            record_shapes = bool(record_shapes) if record_shapes is not None else False
            with_stack = bool(with_stack) if with_stack is not None else False
            sync = bool(sync) if sync is not None else False
            estimated_events_per_step = (
                _parse_int(params.get("estimated_events_per_step")) or 5000
            )
            bytes_per_event = _parse_int(params.get("bytes_per_event")) or 200
        steps_int = int(steps)
        wait_int = int(wait)
        warmup_int = int(warmup)
        estimated_events_int = int(estimated_events_per_step)
        bytes_per_event_int = int(bytes_per_event)
        estimated_size = _estimate_trace_size_bytes(
            active_steps=steps_int,
            estimated_events_per_step=estimated_events_int,
            bytes_per_event=bytes_per_event_int,
            record_shapes=bool(record_shapes) if record_shapes is not None else False,
            with_stack=bool(with_stack) if with_stack is not None else False,
            profile_memory=bool(profile_memory) if profile_memory is not None else True,
        )
        max_bytes = _SAFE_MAX_TRACE_BYTES if safe_mode else _TRACE_MAX_BYTES
        if estimated_size > max_bytes and not force_export:
            return _error_response(
                state,
                "Estimated trace too large",
                status_code=413,
                extra={
                    "guardrail": _guardrail_payload(
                        reason="trace_too_large",
                        alternatives=[
                            *_GUARDRAIL_ALTERNATIVES,
                            "force_export",
                        ],
                        force_export=True,
                        details={
                            "estimated_size_bytes": estimated_size,
                            "max_bytes": max_bytes,
                            "safe_mode": safe_mode,
                        },
                    ),
                },
            )
        state.request_deep_profile(
            steps=steps_int,
            wait=wait_int,
            warmup=warmup_int,
            profile_memory=bool(profile_memory),
            record_shapes=bool(record_shapes),
            with_stack=bool(with_stack),
            sync=bool(sync),
            output_dir=output_dir,
            trace_name=trace_name,
            force_export=force_export,
            estimated_events_per_step=estimated_events_int,
            bytes_per_event=bytes_per_event_int,
            trace_phase=trace_phase,
            max_duration_s=max_duration_s,
        )
        if trace_path:
            state.set_last_trace_path(trace_path)
        config_payload = _deep_profile_config(state)
        config_payload["max_duration_s"] = max_duration_s
        payload = {
            "status": "ok",
            "active": state.deep_profile_active,
            "config": config_payload,
            "estimated_size_bytes": estimated_size,
            "safe_mode": safe_mode,
            "warnings": warnings,
            "max_duration_s": max_duration_s,
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to start deep profile")
        return _error_response(state, str(exc))


@api_router.post("/profile/stop")
def stop_profile() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        state.request_stop()
        payload = {
            "status": "ok",
            "active": state.deep_profile_active,
            "config": _deep_profile_config(state),
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to stop deep profile")
        return _error_response(state, str(exc))


@api_router.get("/profile/status")
def get_profile_status() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        step_index = getattr(state, "_deep_profile_step_index", 0)
        total_steps = getattr(state, "_deep_profile_total_steps", 0)
        estimated_size = estimate_trace_size(state=state)
        payload = {
            "status": "ok",
            "active": state.deep_profile_active,
            "step_index": step_index,
            "total_steps": total_steps,
            "remaining_steps": max(total_steps - step_index, 0)
            if total_steps
            else None,
            "estimated_size_bytes": estimated_size,
            "config": _deep_profile_config(state),
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build profile status")
        return _error_response(state, str(exc))


@api_router.get("/profile/trace")
def get_profile_trace(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        # Return 200 with empty trace metadata when trace path is missing
        payload = {
            "status": "ok",
            "trace": None,
            "trace_meta": {"reason": "not_found", "path": None},
        }
        return JSONResponse(content=_with_schema(payload, state))
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        # Return 413 when size cap exceeded (force not set), otherwise 200 with trace_meta
        if info.get("reason") == "too_large" and not force:
            return _error_response(
                state, "Trace size exceeds limit", status_code=413, extra=info
            )
        # For other load failures, return 200 with trace_meta
        payload = {
            "status": "ok",
            "trace": None,
            "trace_meta": {"path": trace_path, **info},
        }
        return JSONResponse(content=_with_schema(payload, state))
    payload = {
        "status": "ok",
        "trace": trace,
        "trace_meta": {"path": trace_path, **info, **_trace_metadata(trace)},
    }
    fmt = request.query_params.get("format")
    if fmt and fmt.lower() in {"perfetto", "chrome"}:
        trace_meta = dict(cast(dict[str, object], payload.get("trace_meta", {})))
        trace_meta["format"] = fmt.lower()
        payload["trace_meta"] = trace_meta
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/profile/summary")
def get_profile_summary(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        latest = _latest_trace_summary(state)
        summary_only = _parse_bool(request.query_params.get("summary_only")) or False
        payload: dict[str, object] = {
            "status": "ok",
            "latest": latest,
        }
        if not summary_only:
            payload["history"] = list(state.trace_summaries)
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build profile summary")
        return _error_response(state, str(exc))


@api_router.get("/trace/events")
def get_trace_events(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        return _error_response(
            state,
            "Trace not found",
            status_code=404,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        status = 413 if info.get("reason") == "too_large" else 404
        return _error_response(
            state,
            "Trace unavailable",
            status_code=status,
            window=window,
            sampling=_sampling_from_filters(filters),
            extra=info,
        )

    every = _parse_int(request.query_params.get("every")) or 1

    def _items() -> Iterable[dict[str, object]]:
        fields = cast(list[list[str]] | None, filters.get("fields"))
        limit = cast(int | None, filters.get("limit"))
        offset = cast(int, filters.get("offset") or 0)
        emitted = 0
        matched = 0
        name_filter = request.query_params.get("name")
        for event in _trace_events(trace):
            if name_filter and name_filter not in str(event.get("name", "")):
                continue
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                if filters.get("since_ts") is not None and float(ts) < cast(
                    float, filters.get("since_ts")
                ):
                    continue
                if filters.get("until_ts") is not None and float(ts) > cast(
                    float, filters.get("until_ts")
                ):
                    continue
            if every > 1 and matched % every != 0:
                matched += 1
                continue
            if matched < offset:
                matched += 1
                continue
            matched += 1
            payload = _select_fields(event, fields)
            yield payload
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    return _stream_json_list(
        _items(),
        state,
        payload={"trace": {"path": trace_path, **info, **_trace_metadata(trace)}},
        window=window,
        sampling=_sampling_from_filters(filters, every=max(every, 1)),
    )


@api_router.get("/trace/stacks")
def get_trace_stacks(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        return _error_response(state, "Trace not found", status_code=404)
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        status = 413 if info.get("reason") == "too_large" else 404
        return _error_response(
            state, "Trace unavailable", status_code=status, extra=info
        )
    payload = {
        "status": "ok",
        "stacks": _trace_stack_stats(trace),
        "trace": {"path": trace_path, **info, **_trace_metadata(trace)},
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/trace/threads")
def get_trace_threads(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        return _error_response(state, "Trace not found", status_code=404)
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        status = 413 if info.get("reason") == "too_large" else 404
        return _error_response(
            state, "Trace unavailable", status_code=status, extra=info
        )
    payload = {
        "status": "ok",
        "threads": _trace_threads(trace),
        "trace": {"path": trace_path, **info, **_trace_metadata(trace)},
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/trace/metadata")
def get_trace_metadata(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    trace_path = _resolve_trace_path(state, request)
    if not trace_path:
        return _error_response(state, "Trace not found", status_code=404)
    force = _parse_bool(request.query_params.get("force")) or False
    trace, info = _load_trace(trace_path, force=force)
    if trace is None:
        status = 413 if info.get("reason") == "too_large" else 404
        return _error_response(
            state, "Trace unavailable", status_code=status, extra=info
        )
    payload = {
        "status": "ok",
        "metadata": _trace_metadata(trace),
        "trace": {"path": trace_path, **info},
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/snapshots")
def get_snapshots(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    snapshot_dir = _get_snapshot_dir(request)
    try:
        files = _iter_snapshot_files(snapshot_dir)
        limit = _parse_int(request.query_params.get("limit"))
        offset = _parse_int(request.query_params.get("offset")) or 0
        entries: list[dict[str, object]] = []
        for index, path in enumerate(files):
            if index < offset:
                continue
            if limit is not None and len(entries) >= limit:
                break
            stat = path.stat()
            entries.append(
                {
                    "id": path.stem,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_ts": stat.st_mtime,
                }
            )
        payload = {
            "status": "ok",
            "snapshots": entries,
            "total": len(files),
            "snapshot_dir": str(snapshot_dir),
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to list snapshots")
        return _error_response(state, str(exc))


@api_router.get("/snapshots/{snapshot_id}/meta")
def get_snapshot_meta(snapshot_id: str, request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    path, snapshot = _snapshot_by_id(request, snapshot_id)
    if snapshot is None or path is None:
        return _error_response(state, "Snapshot not found", status_code=404)
    stat = path.stat()
    payload = {
        "status": "ok",
        "snapshot": {
            "id": snapshot_id,
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_ts": stat.st_mtime,
            "compressed": snapshot.get("compressed", False),
            "user_metadata": snapshot.get("user_metadata"),
            "device_names": snapshot.get("device_names"),
            "device_properties": snapshot.get("device_properties"),
            "trace_name": snapshot.get("traceName"),
        },
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/snapshots/{snapshot_id}/timeline")
def get_snapshot_timeline(snapshot_id: str, request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    _, snapshot = _snapshot_by_id(request, snapshot_id)
    if snapshot is None:
        return _error_response(
            state,
            "Snapshot not found",
            status_code=404,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    every = _parse_int(request.query_params.get("every")) or 1

    def _items() -> Iterable[dict[str, object]]:
        entries = _snapshot_timeline(snapshot)
        fields = cast(list[list[str]] | None, filters.get("fields"))
        limit = cast(int | None, filters.get("limit"))
        offset = cast(int, filters.get("offset") or 0)
        emitted = 0
        matched = 0
        for entry in entries:
            if not _filter_entry(entry, filters):
                continue
            if every > 1 and matched % every != 0:
                matched += 1
                continue
            if matched < offset:
                matched += 1
                continue
            matched += 1
            yield _select_fields(entry, fields)
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    return _stream_json_list(
        _items(),
        state,
        payload={"snapshot_id": snapshot_id},
        window=window,
        sampling=_sampling_from_filters(filters, every=max(every, 1)),
    )


@api_router.get("/snapshots/{snapshot_id}/events")
def get_snapshot_events(snapshot_id: str, request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    _, snapshot = _snapshot_by_id(request, snapshot_id)
    if snapshot is None:
        return _error_response(
            state,
            "Snapshot not found",
            status_code=404,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    every = _parse_int(request.query_params.get("every")) or 1

    def _items() -> Iterable[dict[str, object]]:
        entries = _snapshot_events(snapshot)
        fields = cast(list[list[str]] | None, filters.get("fields"))
        limit = cast(int | None, filters.get("limit"))
        offset = cast(int, filters.get("offset") or 0)
        emitted = 0
        matched = 0
        for entry in entries:
            if not _filter_entry(entry, filters):
                continue
            if every > 1 and matched % every != 0:
                matched += 1
                continue
            if matched < offset:
                matched += 1
                continue
            matched += 1
            yield _select_fields(entry, fields)
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    return _stream_json_list(
        _items(),
        state,
        payload={"snapshot_id": snapshot_id},
        window=window,
        sampling=_sampling_from_filters(filters, every=max(every, 1)),
    )


@api_router.get("/snapshots/{snapshot_id}/segments")
def get_snapshot_segments(snapshot_id: str, request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    _, snapshot = _snapshot_by_id(request, snapshot_id)
    if snapshot is None:
        return _error_response(state, "Snapshot not found", status_code=404)
    payload = {
        "status": "ok",
        "snapshot_id": snapshot_id,
        "segments": _snapshot_segments(snapshot),
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/snapshots/{snapshot_id}/bottlenecks")
def get_snapshot_bottlenecks(snapshot_id: str, request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    _, snapshot = _snapshot_by_id(request, snapshot_id)
    if snapshot is None:
        return _error_response(state, "Snapshot not found", status_code=404)
    payload = {
        "status": "ok",
        "snapshot_id": snapshot_id,
        "bottlenecks": _snapshot_bottlenecks(snapshot),
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/anomalies")
def get_anomalies(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    metric = request.query_params.get("metric")
    if metric:
        window_size = _parse_window_size(
            request.query_params.get("window"), default=200
        )
        z_threshold = _parse_float(request.query_params.get("z")) or 2.0
        try:
            entries = list(state.metrics)
            anomalies = _detect_anomalies(
                entries,
                metric=str(metric),
                window_size=window_size,
                z_threshold=float(z_threshold),
            )
            payload = {
                "status": "ok",
                "metric": metric,
                "window": window_size,
                "z_threshold": z_threshold,
                "anomalies": anomalies,
            }
            return JSONResponse(content=_with_schema(payload, state, window=window))
        except Exception as exc:
            logger.exception("Failed to detect anomalies")
            return _error_response(state, str(exc), window=window)
    try:
        items = _iter_filtered_entries(state.anomalies, filters)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream anomalies")
        return _error_response(
            state, str(exc), window=window, sampling=_sampling_from_filters(filters)
        )


@api_router.get("/anomalies/detect")
def get_anomalies_detect(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    metric = request.query_params.get("metric") or "step_time_s"
    window_size = _parse_window_size(request.query_params.get("window"), default=200)
    z_threshold = _parse_float(request.query_params.get("z")) or 2.0
    try:
        entries = list(state.metrics)
        anomalies = _detect_anomalies(
            entries,
            metric=str(metric),
            window_size=window_size,
            z_threshold=float(z_threshold),
        )
        payload = {
            "status": "ok",
            "metric": metric,
            "window": window_size,
            "z_threshold": z_threshold,
            "anomalies": anomalies,
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to detect anomalies")
        return _error_response(state, str(exc))


@api_router.get("/regressions")
def get_regressions(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    baseline = request.query_params.get("baseline")
    current = request.query_params.get("current")
    metric = request.query_params.get("metric") or "step_time_s"
    if baseline or current:
        baseline_size = _parse_window_size(baseline, default=200)
        current_size = _parse_window_size(current, default=50)
        try:
            series = [
                cast(float, entry.get("value"))
                for entry in state.metrics
                if entry.get("name") == metric
                and isinstance(entry.get("value"), (int, float))
            ]
            baseline_values = series[
                -(baseline_size + current_size) : -current_size or None
            ]
            current_values = series[-current_size:]
            baseline_mean = (
                sum(baseline_values) / len(baseline_values) if baseline_values else None
            )
            current_mean = (
                sum(current_values) / len(current_values) if current_values else None
            )
            delta = None
            percent = None
            if baseline_mean is not None and current_mean is not None:
                delta = current_mean - baseline_mean
                percent = (delta / baseline_mean) * 100 if baseline_mean else None
            payload = {
                "status": "ok",
                "metric": metric,
                "baseline": {
                    "window": baseline_size,
                    "mean": baseline_mean,
                    "count": len(baseline_values),
                },
                "current": {
                    "window": current_size,
                    "mean": current_mean,
                    "count": len(current_values),
                },
                "delta": delta,
                "percent": percent,
            }
            return JSONResponse(content=_with_schema(payload, state, window=window))
        except Exception as exc:
            logger.exception("Failed to compute regressions")
            return _error_response(state, str(exc), window=window)

    def _predicate(entry: Mapping[str, object]) -> bool:
        kind = entry.get("kind")
        if isinstance(kind, str) and kind.lower() == "regression":
            return True
        message = entry.get("message")
        if isinstance(message, str) and "regression" in message.lower():
            return True
        return False

    try:
        items = _iter_filtered_entries(state.anomalies, filters, predicate=_predicate)
        return _stream_json_list(
            items,
            state,
            window=window,
            sampling=_sampling_from_filters(filters),
        )
    except Exception as exc:
        logger.exception("Failed to stream regressions")
        return _error_response(
            state, str(exc), window=window, sampling=_sampling_from_filters(filters)
        )


@api_router.get("/regressions/detect")
def get_regressions_detect(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    metric = request.query_params.get("metric") or "step_time_s"
    window_size = _parse_window_size(request.query_params.get("window"), default=200)
    z_threshold = _parse_float(request.query_params.get("z")) or 2.0
    try:
        entries = list(state.metrics)
        anomalies = _detect_anomalies(
            entries,
            metric=str(metric),
            window_size=window_size,
            z_threshold=float(z_threshold),
        )
        regressions: list[dict[str, object]] = []
        for item in anomalies:
            z_score = item.get("z_score")
            if isinstance(z_score, (int, float)) and float(z_score) > 0:
                regressions.append(item)
        payload = {
            "status": "ok",
            "metric": metric,
            "window": window_size,
            "z_threshold": z_threshold,
            "regressions": regressions,
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to detect regressions")
        return _error_response(state, str(exc))


@api_router.get("/compare/metrics")
def compare_metrics(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    left = _parse_compare_window(request, "left")
    right = _parse_compare_window(request, "right")
    try:
        left_summary = _metrics_summary_for_window(state, left)
        right_summary = _metrics_summary_for_window(state, right)
        payload = {
            "status": "ok",
            "left": left_summary,
            "right": right_summary,
            "diff": _diff_maps(left_summary, right_summary),
        }
        return JSONResponse(
            content=_with_schema(payload, state, window={"left": left, "right": right})
        )
    except Exception as exc:
        logger.exception("Failed to compare metrics")
        return _error_response(state, str(exc), window={"left": left, "right": right})


@api_router.get("/compare/phases")
def compare_phases(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    left = _parse_compare_window(request, "left")
    right = _parse_compare_window(request, "right")
    try:
        left_stats = _phase_duration_stats(state, left)
        right_stats = _phase_duration_stats(state, right)
        left_map: dict[str, object] = {
            cast(str, entry["name"]): entry.get("total_ms")
            for entry in left_stats
            if isinstance(entry.get("name"), str)
        }
        right_map: dict[str, object] = {
            cast(str, entry["name"]): entry.get("total_ms")
            for entry in right_stats
            if isinstance(entry.get("name"), str)
        }
        payload = {
            "status": "ok",
            "left": left_stats,
            "right": right_stats,
            "diff": _diff_maps(left_map, right_map),
        }
        return JSONResponse(
            content=_with_schema(payload, state, window={"left": left, "right": right})
        )
    except Exception as exc:
        logger.exception("Failed to compare phases")
        return _error_response(state, str(exc), window={"left": left, "right": right})


@api_router.get("/compare/kernels")
def compare_kernels(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    left_path = request.query_params.get("left_path") or request.query_params.get(
        "left_trace"
    )
    right_path = request.query_params.get("right_path") or request.query_params.get(
        "right_trace"
    )
    if not left_path or not right_path:
        return _error_response(
            state, "Both left_path and right_path are required", status_code=400
        )
    force = _parse_bool(request.query_params.get("force")) or False
    left_trace, left_info = _load_trace(left_path, force=force)
    right_trace, right_info = _load_trace(right_path, force=force)
    if left_trace is None or right_trace is None:
        status = (
            413
            if (
                left_info.get("reason") == "too_large"
                or right_info.get("reason") == "too_large"
            )
            else 404
        )
        return _error_response(
            state,
            "Trace unavailable",
            status_code=status,
            extra={"left": left_info, "right": right_info},
        )
    left_stats = _trace_kernel_stats(left_trace)
    right_stats = _trace_kernel_stats(right_trace)
    left_map: dict[str, object] = {
        cast(str, entry["name"]): entry.get("total_us")
        for entry in left_stats
        if isinstance(entry.get("name"), str)
    }
    right_map: dict[str, object] = {
        cast(str, entry["name"]): entry.get("total_us")
        for entry in right_stats
        if isinstance(entry.get("name"), str)
    }
    payload = {
        "status": "ok",
        "left": left_stats,
        "right": right_stats,
        "diff": _diff_maps(left_map, right_map),
        "trace": {"left": left_path, "right": right_path},
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/compare/memory")
def compare_memory(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    left = _parse_compare_window(request, "left")
    right = _parse_compare_window(request, "right")
    try:
        left_summary = _memory_summary_for_window(state, left)
        right_summary = _memory_summary_for_window(state, right)
        left_map: dict[str, object] = {
            "peak_total": left_summary.get("peak_total"),
            "peak_active": left_summary.get("peak_active"),
        }
        right_map: dict[str, object] = {
            "peak_total": right_summary.get("peak_total"),
            "peak_active": right_summary.get("peak_active"),
        }
        diff = _diff_maps(left_map, right_map)
        payload = {
            "status": "ok",
            "left": left_summary,
            "right": right_summary,
            "diff": diff,
        }
        return JSONResponse(
            content=_with_schema(payload, state, window={"left": left, "right": right})
        )
    except Exception as exc:
        logger.exception("Failed to compare memory")
        return _error_response(state, str(exc), window={"left": left, "right": right})


@api_router.get("/export.json")
def export_json(request: Request) -> JSONResponse:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        payload = {
            "status": "ok",
            "metrics": list(_iter_filtered_entries(state.metrics, filters)),
            "phases": list(_iter_filtered_entries(state.phases, filters)),
            "memory": list(_iter_filtered_entries(state.memory, filters)),
            "anomalies": list(_iter_filtered_entries(state.anomalies, filters)),
            "trace_summaries": list(state.trace_summaries),
            "buffer_meta": state.get_buffer_meta(),
        }
        return JSONResponse(
            content=_with_schema(
                payload, state, window=window, sampling=_sampling_from_filters(filters)
            )
        )
    except Exception as exc:
        logger.exception("Failed to export json")
        return _error_response(
            state, str(exc), window=window, sampling=_sampling_from_filters(filters)
        )


@api_router.get("/export.zip")
def export_zip(request: Request) -> Response:
    state = ProfilerState.get_instance()
    filters = _parse_list_filters(request)
    window = _window_from_filters(filters)
    try:
        export_payload = {
            "status": "ok",
            "metrics": list(_iter_filtered_entries(state.metrics, filters)),
            "phases": list(_iter_filtered_entries(state.phases, filters)),
            "memory": list(_iter_filtered_entries(state.memory, filters)),
            "anomalies": list(_iter_filtered_entries(state.anomalies, filters)),
            "trace_summaries": list(state.trace_summaries),
            "buffer_meta": state.get_buffer_meta(),
        }
        meta_payload = _schema_metadata(
            state, window=window, sampling=_sampling_from_filters(filters)
        )
        trace_path = _resolve_trace_path(state, request)
        force = _parse_bool(request.query_params.get("force")) or False
        trace, _ = _load_trace(trace_path, force=force) if trace_path else (None, {})

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("export.json", json.dumps(export_payload))
            archive.writestr("metadata.json", json.dumps(meta_payload))
            if trace is not None:
                archive.writestr("trace.json", json.dumps(trace))
        _ = buffer.seek(0)
        filename = request.query_params.get("filename") or "vram_profiler_export.zip"
        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        return StreamingResponse(
            iter([buffer.getvalue()]), media_type="application/zip", headers=headers
        )
    except Exception as exc:
        logger.exception("Failed to export zip")
        return _error_response(
            state, str(exc), window=window, sampling=_sampling_from_filters(filters)
        )


@api_router.get("/navigation/summary")
def get_navigation_summary() -> JSONResponse:
    state = ProfilerState.get_instance()
    try:
        payload = {
            "status": "ok",
            "counts": {
                "metrics": len(state.metrics),
                "phases": len(state.phases),
                "memory": len(state.memory),
                "anomalies": len(state.anomalies),
                "trace_summaries": len(state.trace_summaries),
            },
            "deep_profile_active": state.deep_profile_active,
        }
        return JSONResponse(content=_with_schema(payload, state))
    except Exception as exc:
        logger.exception("Failed to build navigation summary")
        return _error_response(state, str(exc))


@api_router.get("/navigation/links")
def get_navigation_links() -> JSONResponse:
    state = ProfilerState.get_instance()
    links = [
        "/api/status",
        "/api/summary",
        "/api/metrics/latest",
        "/api/metrics/history",
        "/api/metrics/summary",
        "/api/phases",
        "/api/phases/summary",
        "/api/memory/latest",
        "/api/memory/history",
        "/api/timeline",
        "/api/bottlenecks",
        "/api/profile/status",
        "/api/trace/metadata",
        "/api/snapshots",
        "/api/anomalies",
        "/api/export.json",
        "/api/export.zip",
    ]
    payload = {
        "status": "ok",
        "links": links,
    }
    return JSONResponse(content=_with_schema(payload, state))


@api_router.get("/")
def get_api_root() -> JSONResponse:
    state = ProfilerState.get_instance()
    payload = {
        "status": "ok",
        "message": "Profiler API root",
        "links": ["/api/status", "/api/summary", "/api/metrics/latest"],
    }
    return JSONResponse(content=_with_schema(payload, state))


app = FastAPI(title="VRAM Profiler", version="0.2.0")
app.add_middleware(NonStreamingGZipMiddleware)
app.include_router(api_router)

static_dir = Path(__file__).resolve().parent / "app"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="vram-profiler-gui")


@app.get("/gui")
def redirect_gui() -> Response:
    return Response(status_code=307, headers={"Location": "/"})


_server_lock = threading.Lock()
_server_thread: threading.Thread | None = None


class _UvicornConfig(Protocol): ...


class _UvicornServer(Protocol):
    def run(self) -> None: ...


class _UvicornModule(Protocol):
    def Config(
        self, app: ASGIApp, *, host: str, port: int, log_level: str
    ) -> _UvicornConfig: ...

    def Server(self, config: _UvicornConfig) -> _UvicornServer: ...


def _run_server(port: int) -> None:
    logger.info("Starting profiler server on port %s", port)
    uvicorn_module = cast(
        _UvicornModule, cast(object, importlib.import_module("uvicorn"))
    )
    config = uvicorn_module.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn_module.Server(config)
    server.run()


def start_server(port: int = 8550) -> threading.Thread:
    global _server_thread
    with _server_lock:
        if _server_thread and _server_thread.is_alive():
            return _server_thread
        _server_thread = threading.Thread(
            target=_run_server,
            args=(port,),
            daemon=True,
            name="vram-profiler-server",
        )
        _server_thread.start()
        return _server_thread
