"""
Generic math dataset loading and normalization for GRPO training.

This module keeps raw Hugging Face schemas behind a single normalized format:
question, final answer, optional solution/reference text, source metadata, split,
and dataset name. Splits are deterministic and independent of dataloader order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.utils.logging_utils import get_logger

logger = get_logger("data.math")

SUPPORTED_SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.90, 0.05, 0.05)
DEFAULT_SPLIT_SEED = 42
PROMPT_FORMAT_VERSION = "math_grpo_prompt_v1"


class MathDatasetError(ValueError):
    """Raised when a math dataset cannot be loaded or normalized safely."""


@dataclass(frozen=True)
class MathDatasetSpec:
    """Hugging Face dataset registration metadata."""

    name: str
    hf_path: str
    hf_config: Optional[str] = None


DATASET_SPECS: Dict[str, MathDatasetSpec] = {
    "gsm8k": MathDatasetSpec("gsm8k", "gsm8k", "main"),
    "open-rs": MathDatasetSpec("open-rs", "knoveleng/open-rs"),
    "dapo-math-17k": MathDatasetSpec("dapo-math-17k", "OpenRLHF/dapo-math-17k"),
    "open-deepscaler": MathDatasetSpec("open-deepscaler", "knoveleng/open-deepscaler"),
}

FILTERED_DATASET_DIRS: Dict[str, str] = {
    "dapo-open-rs-lenfilter-640": "data/filtered/dapo-open-rs_lenfilter_640",
    "dapo-math-17k-lenfilter-640": "data/filtered/dapo-math-17k_lenfilter_640",
    "open-rs-lenfilter-640": "data/filtered/open-rs_lenfilter_640",
    "open-deepscaler-lenfilter-640": "data/filtered/open-deepscaler_lenfilter_640",
}


def supported_dataset_names() -> list[str]:
    """Return supported normalized dataset names."""
    return list(DATASET_SPECS) + list(FILTERED_DATASET_DIRS)


def canonical_dataset_name(dataset_name: str) -> str:
    """Validate and canonicalize a dataset alias."""
    normalized = str(dataset_name).strip().lower()
    if normalized not in DATASET_SPECS and normalized not in FILTERED_DATASET_DIRS:
        allowed = ", ".join(sorted(supported_dataset_names()))
        raise MathDatasetError(f"Unsupported dataset_name='{dataset_name}'. Allowed: {allowed}")
    return normalized


def safe_dataset_cache_stem(dataset_name: str) -> str:
    """Return a filesystem-safe stem for dataset-specific artifacts."""
    return canonical_dataset_name(dataset_name).replace("/", "_").replace("-", "_")


def split_ratios_dict(ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS) -> dict[str, float]:
    """Normalize split ratios into named metadata."""
    if len(ratios) != 3:
        raise MathDatasetError("split ratios must contain train/validation/test values")
    total = float(sum(ratios))
    if total <= 0:
        raise MathDatasetError("split ratios must sum to a positive value")
    return {
        "train": float(ratios[0]) / total,
        "validation": float(ratios[1]) / total,
        "test": float(ratios[2]) / total,
    }


def prompt_format_hash(tokenizer: Optional[object] = None) -> str:
    """Hash prompt formatting inputs relevant to SENT compatibility."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        chat_template = ""
    payload = {
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "chat_template": chat_template,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _deterministic_counts(n: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    """Compute deterministic split sizes with non-empty validation/test for n >= 3."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    normalized = split_ratios_dict(ratios)
    n_val = max(1, int(round(n * normalized["validation"])))
    n_test = max(1, int(round(n * normalized["test"])))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def deterministic_split_indices(
    n: int,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, list[int]]:
    """Return stable train/validation/test index assignments."""
    indices = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(indices)

    n_train, n_val, n_test = _deterministic_counts(n, ratios)
    train_end = n_train
    val_end = train_end + n_val
    return {
        "train": sorted(indices[:train_end]),
        "validation": sorted(indices[train_end:val_end]),
        "test": sorted(indices[val_end : val_end + n_test]),
    }


def deterministic_train_validation_indices(
    n: int,
    seed: int = DEFAULT_SPLIT_SEED,
    validation_ratio: float = 0.05,
) -> dict[str, list[int]]:
    """Split an existing train split into train/validation only."""
    indices = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if n <= 1:
        n_val = 0
    else:
        n_val = max(1, int(round(n * float(validation_ratio))))
        n_val = min(n_val, n - 1)
    return {
        "train": sorted(indices[n_val:]),
        "validation": sorted(indices[:n_val]),
    }


def assert_split_separation(split_indices: Mapping[str, Sequence[int]]) -> None:
    """Fail if split assignments overlap."""
    seen: dict[int, str] = {}
    for split, indices in split_indices.items():
        for idx in indices:
            if idx in seen:
                raise MathDatasetError(
                    f"Dataset split leak: row index {idx} is in both {seen[idx]} and {split}"
                )
            seen[idx] = split


def _as_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    raise MathDatasetError(f"Expected dataset row mapping, got {type(row).__name__}")


def _nested_get(row: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_string(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = _nested_get(row, key) if "." in key else row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_user_from_chat(value: Any) -> Optional[str]:
    """Extract the latest user problem from list/chat style fields."""
    if not isinstance(value, list):
        return None
    user_texts: list[str] = []
    for message in value:
        if not isinstance(message, Mapping):
            if isinstance(message, str) and message.strip():
                user_texts.append(message.strip())
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = message.get("content") or message.get("value")
        if isinstance(content, str) and content.strip():
            if role in {"user", "human"}:
                user_texts.append(content.strip())
            elif not role:
                user_texts.append(content.strip())
    return user_texts[-1] if user_texts else None


def extract_question(row: Mapping[str, Any]) -> str:
    """Extract a user-facing math problem from common dataset schemas."""
    for key in ("messages", "conversations", "prompt"):
        chat_question = _extract_user_from_chat(row.get(key))
        if chat_question:
            return chat_question

    question = _first_string(
        row,
        (
            "question",
            "problem",
            "query",
            "instruction",
            "input",
            "prompt",
            "text",
        ),
    )
    if question:
        return question
    raise MathDatasetError("Could not extract question/prompt text from dataset row")


def _extract_after_gsm8k_marker(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    answer = text.rsplit("####", 1)[1].strip()
    return answer or None


def _extract_boxed(text: str) -> Optional[str]:
    marker = "\\boxed{"
    if marker not in text:
        return None
    start = text.rfind(marker) + len(marker)
    depth = 1
    chars: list[str] = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(ch)
    boxed = "".join(chars).strip()
    return boxed or None


def extract_answer(row: Mapping[str, Any]) -> str:
    """Extract final answer text from common math RL dataset fields."""
    direct = _first_string(
        row,
        (
            "answer",
            "gold_parsed",
            "reward_model.ground_truth",
            "extra_info.answer",
            "final_answer",
            "target",
            "label",
        ),
    )
    if direct:
        gsm8k_answer = _extract_after_gsm8k_marker(direct)
        return gsm8k_answer if gsm8k_answer else direct

    solution = _first_string(row, ("solution", "reference", "cot", "response"))
    if solution:
        marker_answer = _extract_after_gsm8k_marker(solution)
        if marker_answer:
            return marker_answer
        boxed = _extract_boxed(solution)
        if boxed:
            return boxed

    raise MathDatasetError(
        "Could not extract a final answer. Checked answer, gold_parsed, "
        "reward_model.ground_truth, extra_info.answer, and solution/reference fields."
    )


def extract_solution(row: Mapping[str, Any]) -> Optional[str]:
    """Extract optional reference reasoning text."""
    return _first_string(row, ("solution", "reference", "cot", "rationale", "answer"))


def normalize_math_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    split: str,
    raw_index: int,
) -> dict[str, Any]:
    """Normalize one raw row into trainer-ready metadata."""
    row = _as_mapping(row)
    question = extract_question(row)
    answer = extract_answer(row)
    solution = extract_solution(row)
    original_id = row.get("id") or row.get("uid") or row.get("uuid") or raw_index
    return {
        "question": question,
        "answer": answer,
        "solution": solution,
        "reference": solution,
        "original_id": original_id,
        "source": dataset_name,
        "raw_index": raw_index,
        "split": split,
        "dataset_name": dataset_name,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
    }


def _load_hf_dataset_dict(dataset_name: str) -> Any:
    spec = DATASET_SPECS[canonical_dataset_name(dataset_name)]
    if spec.hf_config:
        return load_dataset(spec.hf_path, spec.hf_config)
    return load_dataset(spec.hf_path)


def filtered_dataset_path(dataset_name: str, split: str) -> str:
    """Return the local JSONL path for a length-filtered dataset split."""
    dataset_name = canonical_dataset_name(dataset_name)
    if dataset_name not in FILTERED_DATASET_DIRS:
        raise MathDatasetError(f"Dataset '{dataset_name}' is not a filtered local dataset")
    if split not in SUPPORTED_SPLITS:
        raise MathDatasetError(f"Unsupported split='{split}'. Expected train/validation/test.")
    return os.path.join(FILTERED_DATASET_DIRS[dataset_name], f"{split}.jsonl")


def _load_filtered_split_rows(dataset_name: str, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an explicit filtered JSONL split without re-splitting it."""
    path = filtered_dataset_path(dataset_name, split)
    if not os.path.exists(path):
        raise MathDatasetError(
            f"Filtered dataset split not found: {path}. "
            "Run scripts/filter_dataset_by_response_length_vllm.py first."
        )

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MathDatasetError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            missing = [
                key
                for key in (
                    "question",
                    "answer",
                    "solution",
                    "original_id",
                    "raw_index",
                    "split",
                    "dataset_name",
                    "source",
                )
                if key not in row
            ]
            if missing:
                raise MathDatasetError(
                    f"Filtered row {path}:{line_number} is missing fields: {missing}"
                )
            row["split"] = split
            row["dataset_name"] = dataset_name
            rows.append(row)

    split_sizes: dict[str, int | None] = {}
    for split_name in SUPPORTED_SPLITS:
        split_path = filtered_dataset_path(dataset_name, split_name)
        if not os.path.exists(split_path):
            split_sizes[split_name] = None
            continue
        with open(split_path, "r", encoding="utf-8") as fh:
            split_sizes[split_name] = sum(1 for line in fh if line.strip())

    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "source_split": split,
        "split_seed": None,
        "split_ratios": None,
        "split_sizes": split_sizes,
        "num_rows": len(rows),
        "invalid_rows_dropped": 0,
        "filtered_local": True,
        "path": path,
    }
    return rows, metadata


def _sequence_from_split(dataset_obj: Any, split: str) -> Sequence[Any]:
    if isinstance(dataset_obj, Mapping):
        if split not in dataset_obj:
            raise MathDatasetError(f"Dataset object does not contain split '{split}'")
        return dataset_obj[split]
    return dataset_obj


def _slice_sequence(seq: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [seq[idx] for idx in indices]


def load_math_split_rows(
    dataset_name: str,
    split: str,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    strict_filter_invalid: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and normalize a dataset split with deterministic local splitting.

    If a dataset exposes only train, the full train split is split 90/5/5 by
    default. If it exposes train/test but no validation, validation is carved
    out of train while test remains the provided test split.
    """
    dataset_name = canonical_dataset_name(dataset_name)
    if split not in SUPPORTED_SPLITS:
        raise MathDatasetError(f"Unsupported split='{split}'. Expected train/validation/test.")
    if dataset_name in FILTERED_DATASET_DIRS:
        return _load_filtered_split_rows(dataset_name, split)

    raw = _load_hf_dataset_dict(dataset_name)
    split_sizes: dict[str, int] = {}

    if isinstance(raw, Mapping) and "train" in raw and "validation" in raw and "test" in raw:
        source_rows = _sequence_from_split(raw, split)
        raw_indices = list(range(len(source_rows)))
        split_assignments = {
            name: list(range(len(_sequence_from_split(raw, name))))
            for name in SUPPORTED_SPLITS
        }
        source_label = split
    elif isinstance(raw, Mapping) and "train" in raw and "test" in raw:
        train_rows = _sequence_from_split(raw, "train")
        train_assignments = deterministic_train_validation_indices(
            len(train_rows),
            seed=split_seed,
            validation_ratio=split_ratios[1],
        )
        split_assignments = {
            "train": train_assignments["train"],
            "validation": train_assignments["validation"],
            "test": list(range(len(_sequence_from_split(raw, "test")))),
        }
        assert_split_separation(
            {"train": split_assignments["train"], "validation": split_assignments["validation"]}
        )
        if split == "test":
            source_rows = _sequence_from_split(raw, "test")
            raw_indices = split_assignments["test"]
            source_label = "test"
        else:
            source_rows = train_rows
            raw_indices = split_assignments[split]
            source_label = "train"
    else:
        source_rows = _sequence_from_split(raw, "train") if isinstance(raw, Mapping) else raw
        split_assignments = deterministic_split_indices(
            len(source_rows), seed=split_seed, ratios=split_ratios
        )
        assert_split_separation(split_assignments)
        raw_indices = split_assignments[split]
        source_label = "train"

    for split_name in SUPPORTED_SPLITS:
        if split_name == "test" and isinstance(raw, Mapping) and "train" in raw and "test" in raw:
            split_sizes[split_name] = len(split_assignments[split_name])
        else:
            split_sizes[split_name] = len(split_assignments.get(split_name, []))

    normalized: list[dict[str, Any]] = []
    invalid_errors: list[str] = []
    for raw_index in raw_indices:
        try:
            normalized.append(
                normalize_math_row(
                    source_rows[raw_index],
                    dataset_name=dataset_name,
                    split=split,
                    raw_index=raw_index,
                )
            )
        except Exception as exc:
            message = (
                f"{dataset_name}/{split} row raw_index={raw_index} failed normalization: {exc}"
            )
            if strict_filter_invalid:
                invalid_errors.append(message)
                continue
            raise MathDatasetError(message) from exc

    if invalid_errors:
        logger.warning(
            "Dropped %d invalid rows from %s/%s. First error: %s",
            len(invalid_errors),
            dataset_name,
            split,
            invalid_errors[0],
        )

    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "source_split": source_label,
        "split_seed": int(split_seed),
        "split_ratios": split_ratios_dict(split_ratios),
        "split_sizes": split_sizes,
        "num_rows": len(normalized),
        "invalid_rows_dropped": len(invalid_errors),
    }
    return normalized, metadata


class GRPOMathDataset(Dataset):
    """Normalized math dataset for GRPO generation/reward training."""

    def __init__(
        self,
        tokenizer,
        *,
        dataset_name: str,
        split: str = "train",
        max_prompt_length: int = 512,
        split_seed: int = DEFAULT_SPLIT_SEED,
        split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
        strict_filter_invalid: bool = False,
        rows: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        from src.data.gsm8k_loader import format_grpo_prompt

        self.tokenizer = tokenizer
        self.dataset_name = canonical_dataset_name(dataset_name)
        self.split = split
        self.max_prompt_length = max_prompt_length
        self.requested_max_prompt_length = max_prompt_length
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(v) for v in split_ratios)
        self.use_sent = False
        self.num_stages = 1
        self._format_prompt = format_grpo_prompt

        if rows is None:
            rows, metadata = load_math_split_rows(
                self.dataset_name,
                split,
                split_seed=self.split_seed,
                split_ratios=self.split_ratios,
                strict_filter_invalid=strict_filter_invalid,
            )
        self.rows = rows
        self.metadata = metadata or {}
        logger.info(
            "Loaded %s split for %s: %d rows (sizes=%s)",
            split,
            self.dataset_name,
            len(self.rows),
            self.metadata.get("split_sizes"),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.rows[idx]
        prompt = self._format_prompt(self.tokenizer, item["question"])
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "question": item["question"],
            "answer": item["answer"],
            "solution": item.get("solution"),
            "original_id": item.get("original_id"),
            "raw_index": item.get("raw_index"),
            "split": item.get("split", self.split),
            "dataset_name": item.get("dataset_name", self.dataset_name),
            "source": item.get("source", self.dataset_name),
        }


def collate_grpo_math_batch(batch: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    """Collate normalized GRPO examples with tokenizer padding semantics."""
    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None or not isinstance(pad_token_id, int):
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None or not isinstance(pad_token_id, int):
        pad_token_id = 0
    padding_side = getattr(tokenizer, "padding_side", "right")

    input_ids_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        seq_len = len(input_ids)
        if padding_side == "left":
            input_ids_padded[i, -seq_len:] = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_padded[i, -seq_len:] = torch.tensor(
                attention_mask, dtype=torch.long
            )
        else:
            input_ids_padded[i, :seq_len] = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_padded[i, :seq_len] = torch.tensor(
                attention_mask, dtype=torch.long
            )

    collated = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "solutions": [item.get("solution") for item in batch],
        "dataset_names": [item.get("dataset_name") for item in batch],
        "splits": [item.get("split") for item in batch],
        "original_ids": [item.get("original_id") for item in batch],
        "raw_indices": [item.get("raw_index") for item in batch],
    }

    sent_rank_values = [item.get("sent_rank") for item in batch]
    sent_rank_fraction_values = [item.get("sent_rank_fraction") for item in batch]
    sent_rank_totals = [item.get("sent_rank_total") for item in batch]
    if (
        all(value is not None for value in sent_rank_values)
        and all(value is not None for value in sent_rank_fraction_values)
        and all(value is not None for value in sent_rank_totals)
    ):
        collated["sent_rank"] = torch.tensor(sent_rank_values, dtype=torch.long)
        collated["sent_rank_fraction"] = torch.tensor(
            sent_rank_fraction_values, dtype=torch.float32
        )
        collated["sent_rank_total"] = int(sent_rank_totals[0])

    return collated
