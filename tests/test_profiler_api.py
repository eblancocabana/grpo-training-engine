import json

from fastapi.testclient import TestClient

from tools.vram_profiler.profiler_server import app
from tools.vram_profiler.profiler_hooks import ProfilerState


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


def test_profile_guardrail_insufficient_vram() -> None:
    _seed_state()
    client = TestClient(app)
    response = client.post("/api/profile/start?steps=1")
    payload = response.json()
    assert "schema_version" in payload
    assert response.status_code in {200, 409}
