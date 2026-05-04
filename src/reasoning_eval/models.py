"""Model configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    adapter: str | None = None
    selection_source: str | None = None
    metadata: dict[str, Any] | None = None


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("models config must be a mapping")
        return loaded
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML models config files") from exc


def load_models_config(path: str | Path, *, only: list[str] | None = None) -> list[ModelSpec]:
    payload = _load_yaml_or_json(Path(path))
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("models config must contain a non-empty 'models' mapping")
    selected = set(only or raw_models.keys())
    specs: list[ModelSpec] = []
    for key, raw in raw_models.items():
        if key not in selected:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"model entry '{key}' must be a mapping")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"model entry '{key}' is missing model_id")
        adapter = raw.get("adapter")
        if adapter in ("", "null"):
            adapter = None
        selection_source = raw.get("selection_source")
        if key.endswith("_best") and selection_source != "in_training_validation":
            raise ValueError(
                f"model '{key}' looks like a best checkpoint; set selection_source: "
                "in_training_validation so benchmark results are not used for selection"
            )
        specs.append(
            ModelSpec(
                key=str(key),
                model_id=model_id,
                adapter=str(adapter) if adapter is not None else None,
                selection_source=str(selection_source) if selection_source is not None else None,
                metadata={k: v for k, v in raw.items() if k not in {"model_id", "adapter", "selection_source"}},
            )
        )
    missing = selected - {spec.key for spec in specs}
    if missing:
        raise ValueError(f"Requested model keys not found in config: {', '.join(sorted(missing))}")
    return specs

