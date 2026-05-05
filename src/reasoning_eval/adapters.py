"""Adapter preparation helpers for vLLM evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from src.reasoning_eval.models import ModelSpec


def _load_lora_tensors(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported LoRA checkpoint payload in {path}: {type(payload).__name__}")

    if isinstance(payload.get("lora_weights"), dict):
        source = payload["lora_weights"]
    elif isinstance(payload.get("model_state_dict"), dict):
        source = payload["model_state_dict"]
    else:
        raise ValueError(
            f"{path} is not a recognized manual LoRA checkpoint. Expected "
            "'lora_weights' or 'model_state_dict'."
        )

    tensors: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if ("lora_A.weight" not in key) and ("lora_B.weight" not in key):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        tensors[str(key)] = value.detach().cpu().contiguous()

    if not tensors:
        raise ValueError(f"No LoRA tensors found in {path}")
    return tensors


def _infer_lora_config(model: ModelSpec, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    metadata = model.metadata or {}
    rank = metadata.get("rank")
    alpha = metadata.get("alpha")
    target_modules = metadata.get("target_modules")

    if rank is None:
        first_a = next((tensor for key, tensor in tensors.items() if "lora_A.weight" in key), None)
        if first_a is None:
            raise ValueError(f"Could not infer LoRA rank for {model.key}")
        rank = int(first_a.shape[0])
    if alpha is None:
        alpha = int(rank) * 2
    if target_modules is None:
        target_modules = sorted(
            {
                key.rsplit(".", 2)[0].rsplit(".", 1)[-1]
                for key in tensors
                if key.endswith((".lora_A.weight", ".lora_B.weight"))
            }
        )

    return {
        "base_model_name_or_path": model.model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": int(alpha),
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": int(rank),
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def prepare_vllm_adapter(model: ModelSpec, output_dir: Path) -> str | None:
    """Return an adapter path that vLLM can load.

    vLLM expects LoRA adapters in PEFT directory form. This project stores
    manual LoRA checkpoints as ``.pt`` files, so those are converted into a
    local PEFT-compatible directory under the evaluation output. Existing
    adapter directories are returned unchanged.
    """
    if model.adapter is None:
        return None

    adapter_path = Path(model.adapter)
    if adapter_path.is_dir():
        return str(adapter_path)
    if adapter_path.suffix != ".pt":
        return str(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found for {model.key}: {adapter_path}")

    converted_root = output_dir / "converted_adapters"
    converted_dir = converted_root / f"{model.key}_{_fingerprint(adapter_path)}"
    adapter_model_path = converted_dir / "adapter_model.safetensors"
    adapter_config_path = converted_dir / "adapter_config.json"
    metadata_path = converted_dir / "conversion_metadata.json"
    if adapter_model_path.exists() and adapter_config_path.exists():
        return str(converted_dir)

    converted_dir.mkdir(parents=True, exist_ok=True)
    tensors = _load_lora_tensors(adapter_path)
    config = _infer_lora_config(model, tensors)
    save_file(tensors, adapter_model_path)
    adapter_config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "source_adapter": str(adapter_path),
                "model_key": model.key,
                "tensor_count": len(tensors),
                "format": "manual_lora_pt_to_peft_safetensors",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(converted_dir)

