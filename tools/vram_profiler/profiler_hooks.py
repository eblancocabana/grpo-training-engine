from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity

from src.core.memory_manager import MemoryManager
from src.utils.logging_utils import get_logger

logger = get_logger("profiler.hooks")

_SAFE_MAX_TRACE_BYTES = 20 * 1024 * 1024


@dataclass
class DeepProfileConfig:
    steps: int
    wait: int
    warmup: int
    profile_memory: bool
    record_shapes: bool
    with_stack: bool
    sync: bool
    output_dir: str | None
    trace_name: str
    force_export: bool
    estimated_events_per_step: int
    bytes_per_event: int
    trace_phase: str | None

    @property
    def total_steps(self) -> int:
        return max(self.wait + self.warmup + self.steps, 0)


class ProfilerState:
    _instance: "ProfilerState | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, maxlen: int = 10000, schema_version: str = "1.0") -> None:
        self.maxlen = maxlen
        self.schema_version = schema_version
        self.metrics: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.phases: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.memory: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.anomalies: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.trace_summaries: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.drop_counts: dict[str, int] = defaultdict(int)
        self.deep_profile_active = False
        self._deep_profile_config: DeepProfileConfig | None = None
        self._deep_profile_step_index = 0
        self._deep_profile_total_steps = 0
        self._deep_profile_stop_requested = False
        self._last_trace_path: str | None = None
        self._training_active = False
        self._config_snapshot: dict[str, Any] | None = None
        self._last_step_info: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls, maxlen: int = 10000) -> "ProfilerState":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(maxlen=maxlen)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def reset(self) -> None:
        with self._lock:
            self.metrics.clear()
            self.phases.clear()
            self.memory.clear()
            self.anomalies.clear()
            self.trace_summaries.clear()
            self.drop_counts.clear()
            self.deep_profile_active = False
            self._deep_profile_config = None
            self._deep_profile_step_index = 0
            self._deep_profile_total_steps = 0
            self._deep_profile_stop_requested = False
            self._last_trace_path = None
            self._training_active = False
            self._last_step_info = {}

    def set_training_active(self, active: bool) -> None:
        with self._lock:
            self._training_active = active

    def is_training_active(self) -> bool:
        with self._lock:
            return self._training_active

    def set_config_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        with self._lock:
            self._config_snapshot = dict(snapshot)

    def get_config_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._config_snapshot is None:
                return None
            return dict(self._config_snapshot)

    def get_last_step_info(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_step_info)

    def update_last_step_info(self, step: int, epoch: int | None, ts: float) -> None:
        with self._lock:
            self._last_step_info = {"step": step, "epoch": epoch, "ts": ts}

    def get_metric_names(self) -> list[str]:
        with self._lock:
            names = {
                entry.get("name")
                for entry in self.metrics
                if isinstance(entry.get("name"), str)
            }
        return sorted({str(name) for name in names if name is not None})

    def get_buffer_meta(self) -> dict[str, dict[str, Any]]:
        buffers: dict[str, deque[dict[str, Any]]] = {
            "metrics": self.metrics,
            "phases": self.phases,
            "memory": self.memory,
            "anomalies": self.anomalies,
            "trace_summaries": self.trace_summaries,
        }
        meta: dict[str, dict[str, Any]] = {}
        with self._lock:
            for name, buffer in buffers.items():
                oldest = buffer[0] if buffer else None
                latest = buffer[-1] if buffer else None
                meta[name] = {
                    "oldest_step": oldest.get("step")
                    if isinstance(oldest, Mapping)
                    else None,
                    "oldest_ts": oldest.get("ts")
                    if isinstance(oldest, Mapping)
                    else None,
                    "latest_step": latest.get("step")
                    if isinstance(latest, Mapping)
                    else None,
                    "latest_ts": latest.get("ts")
                    if isinstance(latest, Mapping)
                    else None,
                }
        return meta

    def _append(
        self, buffer: deque[dict[str, Any]], buffer_name: str, entry: dict[str, Any]
    ) -> None:
        with self._lock:
            if buffer.maxlen is not None and len(buffer) >= buffer.maxlen:
                self.drop_counts[buffer_name] += 1
            buffer.append(entry)

    def record_metric(
        self,
        name: str,
        value: float | int,
        *,
        step: int,
        ts: float,
        meta: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "name": name,
            "value": float(value),
            "step": step,
            "ts": ts,
        }
        if meta:
            payload["meta"] = dict(meta)
        if tags:
            payload["tags"] = list(tags)
        self._append(self.metrics, "metrics", payload)

    def record_phase(
        self,
        name: str,
        *,
        status: str,
        step: int,
        ts: float,
        duration_ms: float | None = None,
        meta: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "name": name,
            "status": status,
            "step": step,
            "ts": ts,
        }
        if duration_ms is not None:
            payload["duration_ms"] = float(duration_ms)
        if meta:
            payload["meta"] = dict(meta)
        if tags:
            payload["tags"] = list(tags)
        self._append(self.phases, "phases", payload)

    def record_memory(
        self,
        *,
        step: int,
        ts: float,
        stats: Mapping[str, Any] | None,
        context: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "step": step,
            "ts": ts,
            "stats": dict(stats) if stats else None,
            "context": context,
        }
        self._append(self.memory, "memory", payload)

    def record_anomaly(
        self,
        *,
        kind: str,
        step: int,
        ts: float,
        message: str | None = None,
        context: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "kind": kind,
            "step": step,
            "ts": ts,
        }
        if message:
            payload["message"] = message
        if context:
            payload["context"] = dict(context)
        if tags:
            payload["tags"] = list(tags)
        if meta:
            payload["meta"] = dict(meta)
        self._append(self.anomalies, "anomalies", payload)

    def record_trace_summary(self, summary: Mapping[str, Any]) -> None:
        payload = dict(summary)
        if "ts" not in payload:
            payload["ts"] = time.time()
        self._append(self.trace_summaries, "trace_summaries", payload)

    def annotate_step(
        self,
        step: int,
        *,
        tag: str,
        ts: float | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "kind": "annotation",
            "step": step,
            "ts": ts or time.time(),
            "tags": [tag],
        }
        if meta:
            payload["meta"] = dict(meta)
        self._append(self.anomalies, "anomalies", payload)

    def request_deep_profile(
        self,
        *,
        steps: int,
        wait: int,
        warmup: int,
        profile_memory: bool,
        record_shapes: bool,
        with_stack: bool,
        sync: bool,
        output_dir: str | None,
        trace_name: str,
        force_export: bool,
        estimated_events_per_step: int,
        bytes_per_event: int,
        trace_phase: str | None,
    ) -> None:
        with self._lock:
            self._deep_profile_config = DeepProfileConfig(
                steps=steps,
                wait=wait,
                warmup=warmup,
                profile_memory=profile_memory,
                record_shapes=record_shapes,
                with_stack=with_stack,
                sync=sync,
                output_dir=output_dir,
                trace_name=trace_name,
                force_export=force_export,
                estimated_events_per_step=estimated_events_per_step,
                bytes_per_event=bytes_per_event,
                trace_phase=trace_phase,
            )
            self.deep_profile_active = True
            self._deep_profile_step_index = 0
            self._deep_profile_total_steps = self._deep_profile_config.total_steps
            self._deep_profile_stop_requested = False
            self._last_trace_path = None

    def get_deep_profile_config(self) -> DeepProfileConfig | None:
        with self._lock:
            return self._deep_profile_config

    def set_deep_profile_progress(self, step_index: int, total_steps: int) -> None:
        with self._lock:
            self._deep_profile_step_index = step_index
            self._deep_profile_total_steps = total_steps

    def stop_deep_profile(self) -> None:
        with self._lock:
            self.deep_profile_active = False
            self._deep_profile_config = None
            self._deep_profile_step_index = 0
            self._deep_profile_total_steps = 0
            self._deep_profile_stop_requested = False

    def request_stop(self) -> None:
        with self._lock:
            self._deep_profile_stop_requested = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._deep_profile_stop_requested

    def set_last_trace_path(self, trace_path: str) -> None:
        with self._lock:
            self._last_trace_path = trace_path

    def get_last_trace_path(self) -> str | None:
        with self._lock:
            return self._last_trace_path


def estimate_trace_size(state: ProfilerState) -> int:
    config = state.get_deep_profile_config()
    if config is None:
        return 0
    multiplier = 1.0
    if config.record_shapes:
        multiplier += 0.5
    if config.with_stack:
        multiplier += 1.0
    if config.profile_memory:
        multiplier += 0.3
    return int(
        config.steps
        * config.estimated_events_per_step
        * config.bytes_per_event
        * multiplier
    )


class ProfilerHooks:
    def __init__(
        self,
        *,
        state: ProfilerState,
        memory_manager: MemoryManager,
        use_nvtx: bool = True,
    ) -> None:
        self._state = state
        self._memory_manager = memory_manager
        self._use_nvtx = use_nvtx and torch.cuda.is_available()
        self._step_start_perf: float | None = None
        self._step_start_ts: float | None = None
        self._phase_start_perf: dict[str, float] = {}
        self._phase_start_ts: dict[str, float] = {}
        self._phase_recorders: dict[str, list[Any]] = defaultdict(list)
        self._profiler: torch.profiler.profile | None = None
        self._profiler_started_ts: float | None = None
        self._deep_profile_total_steps = 0
        self._deep_profile_step_index = 0

    def on_training_start(self, step: int, epoch: int | None) -> None:
        self._state.set_training_active(True)
        self._state.update_last_step_info(step, epoch, time.time())

    def on_training_end(self, step: int, epoch: int | None) -> None:
        self._state.set_training_active(False)
        self._state.update_last_step_info(step, epoch, time.time())
        self._finalize_profiler_if_needed(force=True)

    def on_step_start(self, step: int, epoch: int | None = None) -> None:
        self._step_start_perf = time.perf_counter()
        self._step_start_ts = time.time()
        self._state.update_last_step_info(step, epoch, self._step_start_ts)
        if self._use_nvtx:
            torch.cuda.nvtx.range_push(f"step:{step}")
        self._maybe_start_profiler()
        config = self._state.get_deep_profile_config()
        if config and config.sync:
            torch.cuda.synchronize()

    def on_step_end(
        self, step: int, metrics: Mapping[str, float], epoch: int | None = None
    ) -> None:
        end_ts = time.time()
        duration_s = None
        if self._step_start_perf is not None:
            duration_s = time.perf_counter() - self._step_start_perf
        if duration_s is not None:
            self._state.record_metric("step_time_s", duration_s, step=step, ts=end_ts)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key == "step_time_s":
                    continue
                self._state.record_metric(str(key), float(value), step=step, ts=end_ts)
        memory_stats = self._memory_manager.get_memory_stats()
        self._state.record_memory(
            step=step, ts=end_ts, stats=memory_stats, context="train"
        )
        micro_meta = {
            "micro_gen": metrics.get("gen_micro_batch"),
            "micro_train": metrics.get("train_micro_batch"),
            "oom_backoff_count": metrics.get("oom_backoff_count"),
        }
        self._state.record_phase(
            "micro_batching",
            status="meta",
            step=step,
            ts=end_ts,
            meta={k: v for k, v in micro_meta.items() if v is not None},
        )
        if self._use_nvtx:
            torch.cuda.nvtx.range_pop()
        self._advance_profiler(step)
        self._state.update_last_step_info(step, epoch, end_ts)

    def on_phase_start(self, name: str, step: int | None = None) -> None:
        ts = time.time()
        self._phase_start_perf[name] = time.perf_counter()
        self._phase_start_ts[name] = ts
        if self._use_nvtx:
            torch.cuda.nvtx.range_push(f"phase:{name}")
        if self._profiler is not None:
            ctx = record_function(f"phase:{name}")
            ctx.__enter__()
            self._phase_recorders[name].append(ctx)
        if step is not None:
            self._state.record_phase(name, status="start", step=step, ts=ts)

    def on_phase_end(self, name: str, step: int | None = None) -> None:
        ts = time.time()
        start_perf = self._phase_start_perf.pop(name, None)
        duration_ms = None
        if start_perf is not None:
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
        if self._use_nvtx:
            torch.cuda.nvtx.range_pop()
        if self._profiler is not None:
            recorders = self._phase_recorders.get(name)
            if recorders:
                ctx = recorders.pop()
                ctx.__exit__(None, None, None)
        if step is not None:
            self._state.record_phase(
                name,
                status="end",
                step=step,
                ts=ts,
                duration_ms=duration_ms,
            )

    def on_data_start(self, step: int | None = None) -> None:
        self.on_phase_start("data", step=step)

    def on_data_end(self, step: int | None = None) -> None:
        self.on_phase_end("data", step=step)

    def on_oom(self, step: int, context: Mapping[str, Any]) -> None:
        if self._use_nvtx and self._step_start_perf is not None:
            torch.cuda.nvtx.range_pop()
            self._step_start_perf = None
        self._state.record_anomaly(
            kind="oom",
            step=step,
            ts=time.time(),
            message="CUDA OOM",
            context=context,
            tags=["oom"],
        )

    def annotate_step(
        self, step: int, tag: str, meta: Mapping[str, Any] | None = None
    ) -> None:
        self._state.annotate_step(step, tag=tag, ts=time.time(), meta=meta)

    def _maybe_start_profiler(self) -> None:
        config = self._state.get_deep_profile_config()
        if config is None:
            return
        if self._profiler is not None:
            return
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.steps,
            repeat=1,
        )
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
            with_stack=config.with_stack,
        )
        if profiler is not None:
            profiler.__enter__()
            self._profiler = profiler
        else:
            self._profiler = None
        self._profiler_started_ts = time.time()
        self._deep_profile_total_steps = config.total_steps
        self._deep_profile_step_index = 0
        self._state.set_deep_profile_progress(0, self._deep_profile_total_steps)

    def _advance_profiler(self, step: int) -> None:
        if self._profiler is None:
            return
        self._profiler.step()
        self._deep_profile_step_index += 1
        self._state.set_deep_profile_progress(
            self._deep_profile_step_index, self._deep_profile_total_steps
        )
        if self._state.should_stop():
            self._finalize_profiler_if_needed(step=step, force=True)
            return
        if self._deep_profile_step_index >= self._deep_profile_total_steps:
            self._finalize_profiler_if_needed(step=step)

    def _finalize_profiler_if_needed(
        self, step: int | None = None, force: bool = False
    ) -> None:
        if self._profiler is None and not force:
            return
        profiler = self._profiler
        self._profiler = None
        if profiler is None:
            return
        try:
            profiler.__exit__(None, None, None)
            summary = self._build_profiler_summary(profiler)
            trace_path = self._export_trace(profiler)
            filtered_path, filtered_size, raw_size = self._postprocess_trace(trace_path)
            if filtered_path:
                self._state.set_last_trace_path(filtered_path)
            summary.update(
                {
                    "trace_path": filtered_path or trace_path,
                    "trace_size_bytes": raw_size,
                    "trace_filtered_size_bytes": filtered_size,
                    "step": step,
                    "started_ts": self._profiler_started_ts,
                    "ended_ts": time.time(),
                }
            )
            self._state.record_trace_summary({"summary": summary, "ts": time.time()})
            if step is not None:
                if raw_size is not None:
                    self._state.record_metric(
                        "trace_size_bytes",
                        float(raw_size),
                        step=step,
                        ts=time.time(),
                    )
                if filtered_size is not None:
                    self._state.record_metric(
                        "trace_filtered_size_bytes",
                        float(filtered_size),
                        step=step,
                        ts=time.time(),
                    )
        except Exception as exc:
            logger.exception("Failed to export deep profile trace")
            self._state.record_anomaly(
                kind="profile_error",
                step=step or 0,
                ts=time.time(),
                message=str(exc),
                context={"phase": "export_trace"},
                tags=["profile"],
            )
        finally:
            self._state.stop_deep_profile()

    def _export_trace(self, profiler: torch.profiler.profile) -> str:
        config = self._state.get_deep_profile_config()
        if config is None:
            return ""
        trace_name = config.trace_name or f"deep_profile_{int(time.time())}"
        if not Path(trace_name).suffix:
            trace_name = f"{trace_name}.json"
        output_dir = Path(config.output_dir) if config.output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / trace_name
        profiler.export_chrome_trace(str(trace_path))
        return str(trace_path)

    def _trace_size_bytes(self, trace_path: str | None) -> int | None:
        if not trace_path:
            return None
        try:
            return Path(trace_path).stat().st_size
        except OSError:
            return None

    def _postprocess_trace(
        self, trace_path: str | None
    ) -> tuple[str | None, int | None, int | None]:
        if not trace_path:
            return None, None, None
        raw_size = self._trace_size_bytes(trace_path)
        config = self._state.get_deep_profile_config()
        if config is None or not config.trace_phase:
            return trace_path, raw_size, raw_size
        try:
            raw = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        except Exception:
            return trace_path, raw_size, raw_size
        if not isinstance(raw, Mapping):
            return trace_path, raw_size, raw_size
        trace_dict: dict[str, Any] = dict(raw)
        events = trace_dict.get("traceEvents")
        if not isinstance(events, list):
            return trace_path, raw_size, raw_size

        phase_intervals = self._trace_phase_intervals_from_events(events)
        if not phase_intervals:
            return trace_path, raw_size, raw_size

        filtered: list[dict[str, Any]] = []
        for event_raw in events:
            if not isinstance(event_raw, Mapping):
                continue
            name = event_raw.get("name")
            if isinstance(name, str) and name.startswith("phase:"):
                filtered.append(dict(event_raw))
                continue
            ts = event_raw.get("ts")
            if ts is None:
                filtered.append(dict(event_raw))
                continue
            if isinstance(ts, (int, float)) and self._ts_in_intervals(
                float(ts), phase_intervals
            ):
                filtered.append(dict(event_raw))

        max_events = int(_SAFE_MAX_TRACE_BYTES // max(config.bytes_per_event, 1))
        truncated = False
        if len(filtered) > max_events:
            filtered = filtered[:max_events]
            truncated = True

        trace_dict["traceEvents"] = filtered
        if truncated:
            trace_dict["truncated"] = True
            trace_dict["truncation_reason"] = "safe_mode_cap"
        filtered_path = Path(trace_path).with_suffix(".filtered.json")
        try:
            filtered_path.write_text(json.dumps(trace_dict), encoding="utf-8")
        except Exception:
            return trace_path, raw_size, raw_size
        filtered_size = self._trace_size_bytes(str(filtered_path))
        return str(filtered_path), filtered_size, raw_size

    def _trace_phase_intervals_from_events(
        self, events: list[object]
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        config = self._state.get_deep_profile_config()
        phase = config.trace_phase if config else None
        if not phase:
            return intervals
        start_by_key: dict[tuple[int | None, int | None], float] = {}
        for event_raw in events:
            if not isinstance(event_raw, Mapping):
                continue
            name = event_raw.get("name")
            if not isinstance(name, str) or name != f"phase:{phase}":
                continue
            ts = event_raw.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            tid = (
                event_raw.get("tid") if isinstance(event_raw.get("tid"), int) else None
            )
            pid = (
                event_raw.get("pid") if isinstance(event_raw.get("pid"), int) else None
            )
            ph = event_raw.get("ph")
            if ph == "X":
                dur = event_raw.get("dur")
                if isinstance(dur, (int, float)):
                    intervals.append((float(ts), float(ts) + float(dur)))
            elif ph == "B":
                start_by_key[(pid, tid)] = float(ts)
            elif ph == "E":
                key = (pid, tid)
                start = start_by_key.pop(key, None)
                if start is not None:
                    intervals.append((start, float(ts)))
        return intervals

    @staticmethod
    def _ts_in_intervals(ts: float, intervals: list[tuple[float, float]]) -> bool:
        for start, end in intervals:
            if start <= ts <= end:
                return True
        return False

    def _build_profiler_summary(
        self, profiler: torch.profiler.profile
    ) -> dict[str, Any]:
        rows = []
        for entry in profiler.key_averages():
            rows.append(
                {
                    "name": entry.key,
                    "count": entry.count,
                    "cpu_time_total": entry.cpu_time_total,
                    "self_cpu_time_total": entry.self_cpu_time_total,
                    "cuda_time_total": getattr(entry, "cuda_time_total", 0.0),
                    "self_cuda_time_total": getattr(entry, "self_cuda_time_total", 0.0),
                    "cpu_memory_usage": entry.cpu_memory_usage,
                    "cuda_memory_usage": getattr(entry, "cuda_memory_usage", 0.0),
                }
            )
        return {
            "key_averages": rows,
            "config": self._state.get_deep_profile_config().__dict__
            if self._state.get_deep_profile_config()
            else {},
        }
