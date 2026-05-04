"""Result file writing and summary helpers."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


OUTPUT_FILES = (
    "raw_generations.jsonl",
    "parsed_predictions.jsonl",
    "scores_by_example.jsonl",
    "summary_by_dataset.csv",
    "summary_by_model.csv",
    "macro_family_summary.csv",
    "run_config.json",
    "README.md",
    "dataset_metadata.json",
)


def prepare_output_dir(path: str | Path, *, resume: bool, overwrite: bool) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not (resume or overwrite):
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --resume or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in OUTPUT_FILES:
            target = output_dir / name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
    return output_dir


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completed_example_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        model = row.get("model_key")
        dataset = row.get("dataset")
        protocol = row.get("protocol")
        example_id = row.get("example_id")
        if model and dataset and protocol and example_id is not None:
            keys.add((str(model), str(dataset), str(protocol), str(example_id)))
    return keys


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
