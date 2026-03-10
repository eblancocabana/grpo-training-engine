# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnusedCallResult=false, reportPrivateUsage=false
# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import json
import logging

import torch
from fastapi.testclient import TestClient
from pytest import LogCaptureFixture, MonkeyPatch

from src.core.memory_manager import MemoryManager
from tools.vram_profiler import profiler_hooks
from tools.vram_profiler.profiler_hooks import ProfilerHooks, ProfilerState
from tools.vram_profiler.profiler_server import app


def _seed_state() -> ProfilerState:
    ProfilerState.reset_instance()
    state = ProfilerState.get_instance(maxlen=3)
    state.record_metric("loss", 1.0, step=1, ts=10.0)
    state.record_metric("loss", 0.5, step=2, ts=20.0)
    state.record_metric("avg_reward", 0.2, step=2, ts=20.0)
    state.record_phase("generation", status="start", step=1, ts=10.0)
    state.record_phase("generation", status="end", step=1, ts=10.5, duration_ms=500.0)
    state.record_memory(
        step=1, ts=10.0, stats={"allocated_gb": 1.0, "reserved_gb": 1.2}
    )
    state.record_anomaly(kind="oom", step=2, ts=20.0, message="OOM")
    return state


def _available_memory_gb_low(self: MemoryManager) -> float:
    _ = self
    return 0.1


def _available_memory_gb_high(self: MemoryManager) -> float:
    _ = self
    return 10.0


def test_status_schema_contains_metadata() -> None:
    _seed_state()
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"]
    assert "units" in payload
    assert "sampling" in payload
    assert "window" in payload
    assert "source" in payload


def test_metrics_history_streaming() -> None:
    _seed_state()
    client = TestClient(app)
    response = client.get("/api/metrics/history?limit=2")
    assert response.status_code == 200
    payload = json.loads(response.text)
    assert payload["status"] == "ok"
    assert len(payload["data"]) == 2
    assert payload["data"][0]["name"] in {"loss", "avg_reward"}


def test_filters_by_since_step() -> None:
    _seed_state()
    client = TestClient(app)
    response = client.get("/api/metrics/history?since_step=2")
    assert response.status_code == 200
    payload = json.loads(response.text)
    assert all(entry["step"] >= 2 for entry in payload["data"])


def test_fields_filter() -> None:
    _seed_state()
    client = TestClient(app)
    response = client.get("/api/metrics/history?fields=name,value&limit=1")
    assert response.status_code == 200
    payload = json.loads(response.text)
    entry = payload["data"][0]
    assert set(entry.keys()) <= {"name", "value"}


def test_profile_guardrail_insufficient_vram(monkeypatch: MonkeyPatch) -> None:
    _ = _seed_state()
    monkeypatch.setattr(
        MemoryManager, "get_available_memory_gb", _available_memory_gb_low
    )
    client = TestClient(app)
    response = client.post("/api/profile/start?steps=1")
    payload = response.json()
    assert response.status_code == 409
    guardrail = payload["guardrail"]
    assert guardrail["reason"] == "insufficient_vram"
    assert guardrail["alternatives"] == ["snapshot", "nvtx", "tier1"]
    assert "force_export" not in guardrail


def test_profile_guardrail_recent_oom_backoff(monkeypatch: MonkeyPatch) -> None:
    state = _seed_state()
    state.record_metric("oom_backoff_count", 1, step=3, ts=30.0)
    monkeypatch.setattr(
        MemoryManager, "get_available_memory_gb", _available_memory_gb_high
    )
    client = TestClient(app)
    response = client.post("/api/profile/start?safe=1")
    payload = response.json()
    assert response.status_code == 409
    guardrail = payload["guardrail"]
    assert guardrail["reason"] == "recent_oom_backoff"
    assert guardrail["alternatives"] == ["snapshot", "nvtx", "tier1"]
    assert "force_export" not in guardrail


def test_profile_guardrail_trace_too_large(monkeypatch: MonkeyPatch) -> None:
    _ = _seed_state()
    monkeypatch.setattr(
        MemoryManager, "get_available_memory_gb", _available_memory_gb_high
    )
    client = TestClient(app)
    response = client.post(
        "/api/profile/start?safe=1&steps=1&estimated_events_per_step=1000000"
    )
    payload = response.json()
    assert response.status_code == 413
    guardrail = payload["guardrail"]
    assert guardrail["reason"] == "trace_too_large"
    assert guardrail["alternatives"] == ["snapshot", "nvtx", "tier1", "force_export"]
    assert guardrail["force_export"] is True


def test_profile_safe_defaults_enforced(monkeypatch: MonkeyPatch) -> None:
    _ = _seed_state()
    monkeypatch.setattr(
        MemoryManager, "get_available_memory_gb", _available_memory_gb_high
    )
    client = TestClient(app)
    response = client.post(
        (
            "/api/profile/start?safe=1&steps=5&wait=3&warmup=2&with_stack=1"
            "&record_shapes=1&profile_memory=1&sync=0&max_duration_s=60"
        )
    )
    payload = response.json()
    assert response.status_code == 200
    config = payload["config"]
    assert config["steps"] == 1
    assert config["wait"] == 1
    assert config["warmup"] == 1
    assert config["with_stack"] is False
    assert config["record_shapes"] is False
    assert config["profile_memory"] is False
    assert config["sync"] is True
    assert config["max_duration_s"] <= 20.0


def test_experimentalconfig_fallback_warns_once(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    ProfilerState.reset_instance()
    state = ProfilerState.get_instance()
    state.request_deep_profile(
        steps=1,
        wait=0,
        warmup=0,
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
        sync=True,
        output_dir=None,
        trace_name="test",
        force_export=False,
        estimated_events_per_step=1,
        bytes_per_event=1,
        trace_phase=None,
        max_duration_s=10.0,
    )
    hooks = ProfilerHooks(state=state, memory_manager=MemoryManager())

    monkeypatch.delattr(torch.profiler, "ExperimentalConfig", raising=False)
    monkeypatch.delattr(torch.profiler, "_ExperimentalConfig", raising=False)
    monkeypatch.setattr(profiler_hooks, "_experimental_config_warning_emitted", False)

    captured: dict[str, object] = {}

    class DummyProfiler:
        def __enter__(self) -> "DummyProfiler":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_profile(**kwargs: object) -> DummyProfiler:
        captured.update(kwargs)
        return DummyProfiler()

    monkeypatch.setattr(torch.profiler, "profile", fake_profile)

    with caplog.at_level(logging.WARNING, logger="profiler.hooks"):
        hooks._maybe_start_profiler()

    warnings = [
        record
        for record in caplog.records
        if "experimental_config" in record.getMessage().lower()
    ]
    assert len(warnings) == 1
    assert "experimental_config" not in captured


def test_profile_oom_aborts_deep_profile(monkeypatch: MonkeyPatch) -> None:
    ProfilerState.reset_instance()
    state = ProfilerState.get_instance()
    state.request_deep_profile(
        steps=1,
        wait=0,
        warmup=0,
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
        sync=False,
        output_dir=None,
        trace_name="test",
        force_export=False,
        estimated_events_per_step=1,
        bytes_per_event=1,
        trace_phase=None,
        max_duration_s=None,
    )

    class DummyProfiler:
        def __enter__(self) -> "DummyProfiler":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def step(self) -> None:
            return None

        def key_averages(self):
            return []

        def export_chrome_trace(self, _path: str) -> None:
            return None

    monkeypatch.setattr(torch.profiler, "profile", lambda **_: DummyProfiler())
    hooks = ProfilerHooks(state=state, memory_manager=MemoryManager())
    hooks._maybe_start_profiler()

    hooks.on_oom(step=1, context={"note": "test"})

    assert state.deep_profile_active is False
    last_anomaly = list(state.anomalies)[-1]
    assert last_anomaly["kind"] == "oom"
    context = last_anomaly.get("context")
    assert isinstance(context, dict)
    assert "profiling_aborted" in context


def test_profile_max_duration_finalizes(monkeypatch: MonkeyPatch) -> None:
    ProfilerState.reset_instance()
    state = ProfilerState.get_instance()
    state.request_deep_profile(
        steps=1,
        wait=0,
        warmup=0,
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
        sync=False,
        output_dir=None,
        trace_name="test",
        force_export=False,
        estimated_events_per_step=1,
        bytes_per_event=1,
        trace_phase=None,
        max_duration_s=0.01,
    )

    class DummyProfiler:
        def __enter__(self) -> "DummyProfiler":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def step(self) -> None:
            return None

        def key_averages(self):
            return []

        def export_chrome_trace(self, _path: str) -> None:
            return None

    monkeypatch.setattr(torch.profiler, "profile", lambda **_: DummyProfiler())
    hooks = ProfilerHooks(state=state, memory_manager=MemoryManager())
    hooks._maybe_start_profiler()

    hooks._handle_profiler_timeout()

    assert state.deep_profile_active is False
    last_anomaly = list(state.anomalies)[-1]
    assert last_anomaly["kind"] == "profile_error"
