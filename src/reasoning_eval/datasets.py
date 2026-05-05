"""Dataset registry and normalization for reasoning evaluation."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import Dataset, load_dataset

from src.reasoning_eval.schema import DatasetSpec, EvalExample


def _candidates(*items: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for item in items:
        path = item[0]
        config = item[1] if len(item) > 1 else None
        split = item[2] if len(item) > 2 else None
        out.append({"path": path, "config": config, "split": split})
    return tuple(out)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec("gsm8k", "math", "math", _candidates(("gsm8k", "main", "test")), local_path_env="REASONING_EVAL_GSM8K_PATH", default_limit=1319),
    "gsm-plus": DatasetSpec("gsm-plus", "math", "math", _candidates(("qintongli/GSM-Plus", None, "test"), ("qintongli/GSM-Plus", None, "train")), local_path_env="REASONING_EVAL_GSM_PLUS_PATH"),
    "svamp": DatasetSpec("svamp", "math", "math", _candidates(("ChilleD/SVAMP", None, "test"), ("ChilleD/SVAMP", None, "train")), local_path_env="REASONING_EVAL_SVAMP_PATH"),
    "math500": DatasetSpec("math500", "math", "math", _candidates(("HuggingFaceH4/MATH-500", None, "test")), local_path_env="REASONING_EVAL_MATH500_PATH", default_limit=500),
    "aime24": DatasetSpec("aime24", "math", "math", _candidates(("HuggingFaceH4/aime_2024", None, "train"), ("Maxwell-Jia/AIME_2024", None, "train")), local_path_env="REASONING_EVAL_AIME24_PATH", default_limit=30),
    "aime25": DatasetSpec("aime25", "math", "math", _candidates(("math-ai/aime25", None, "train"), ("test-time-compute/aime_2025", None, "train")), local_path_env="REASONING_EVAL_AIME25_PATH", default_limit=30),
    "amc23": DatasetSpec("amc23", "math", "math", _candidates(("math-ai/amc23", None, "train"), ("AI-MO/aimo-validation-amc", None, "train")), local_path_env="REASONING_EVAL_AMC23_PATH", default_limit=40),
    "minerva": DatasetSpec("minerva", "math", "math", _candidates(("EleutherAI/hendrycks_math", "algebra", "test"), ("EleutherAI/hendrycks_math", "number_theory", "test")), local_path_env="REASONING_EVAL_MINERVA_PATH", default_limit=272),
    "olympiadbench": DatasetSpec("olympiadbench", "math", "math", _candidates(("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train"), ("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", "train")), local_path_env="REASONING_EVAL_OLYMPIADBENCH_PATH"),
    "gaokao": DatasetSpec("gaokao", "math", "math", _candidates(("FiveEye/GaokaoBench", None, "test"), ("FiveEye/GaokaoBench", None, "train")), local_path_env="REASONING_EVAL_GAOKAO_PATH"),
    "omni-math": DatasetSpec("omni-math", "math", "math", _candidates(("KbsdJames/Omni-MATH", None, "test"), ("KbsdJames/Omni-MATH", None, "train")), local_path_env="REASONING_EVAL_OMNI_MATH_PATH"),
    "open-rs": DatasetSpec("open-rs", "training_transfer", "math", _candidates(("knoveleng/open-rs", None, "test"), ("knoveleng/open-rs", None, "train")), local_path_env="REASONING_EVAL_OPEN_RS_PATH"),
    "dapo-math-17k": DatasetSpec("dapo-math-17k", "training_transfer", "math", _candidates(("OpenRLHF/dapo-math-17k", None, "train")), local_path_env="REASONING_EVAL_DAPO_MATH_17K_PATH"),
    "open-deepscaler": DatasetSpec("open-deepscaler", "training_transfer", "math", _candidates(("knoveleng/open-deepscaler", None, "test"), ("knoveleng/open-deepscaler", None, "train")), local_path_env="REASONING_EVAL_OPEN_DEEPSCALER_PATH"),
    "still": DatasetSpec("still", "training_transfer", "math", _candidates(("RUC-AIBOX/STILL-3-Preview-RL-Data", None, "train")), local_path_env="REASONING_EVAL_STILL_PATH"),
    "numinamath-cot": DatasetSpec("numinamath-cot", "training_transfer", "math", _candidates(("AI-MO/NuminaMath-CoT", None, "test"), ("AI-MO/NuminaMath-CoT", None, "train")), local_path_env="REASONING_EVAL_NUMINAMATH_COT_PATH"),
    "gpqa-diamond": DatasetSpec("gpqa-diamond", "science_general", "multiple_choice", _candidates(("fingertap/GPQA-Diamond", None, "test"), ("Idavidrein/gpqa", "gpqa_diamond", "train")), local_path_env="REASONING_EVAL_GPQA_DIAMOND_PATH", default_limit=198),
    "mmlu-pro": DatasetSpec("mmlu-pro", "science_general", "multiple_choice", _candidates(("TIGER-Lab/MMLU-Pro", None, "test")), local_path_env="REASONING_EVAL_MMLU_PRO_PATH"),
    "mmlu": DatasetSpec("mmlu", "science_general", "multiple_choice", _candidates(("cais/mmlu", "all", "test")), local_path_env="REASONING_EVAL_MMLU_PATH"),
    "bbh": DatasetSpec("bbh", "science_general", "math", _candidates(("lukaemon/bbh", "multistep_arithmetic_two", "test"), ("lukaemon/bbh", "logical_deduction_three_objects", "test")), local_path_env="REASONING_EVAL_BBH_PATH"),
    "arc-challenge": DatasetSpec("arc-challenge", "science_general", "multiple_choice", _candidates(("allenai/ai2_arc", "ARC-Challenge", "test")), local_path_env="REASONING_EVAL_ARC_CHALLENGE_PATH"),
    "drop": DatasetSpec("drop", "science_general", "drop", _candidates(("ucinlp/drop", None, "validation")), local_path_env="REASONING_EVAL_DROP_PATH"),
    "humaneval": DatasetSpec("humaneval", "coding", "code", _candidates(("openai/openai_humaneval", None, "test")), local_path_env="REASONING_EVAL_HUMANEVAL_PATH", default_limit=164),
    "mbpp": DatasetSpec("mbpp", "coding", "code", _candidates(("google-research-datasets/mbpp", "sanitized", "test"), ("google-research-datasets/mbpp", None, "test")), local_path_env="REASONING_EVAL_MBPP_PATH"),
    "livecodebench": DatasetSpec("livecodebench", "coding", "code", _candidates(("livecodebench/code_generation_lite", None, "test")), local_path_env="REASONING_EVAL_LIVECODEBENCH_PATH"),
}

MINIMAL_DATASETS = ("gsm8k", "gsm-plus", "math500", "aime24", "amc23", "humaneval")
STRONG_DATASETS = (
    "gsm8k",
    "gsm-plus",
    "math500",
    "aime24",
    "aime25",
    "amc23",
    "minerva",
    "gpqa-diamond",
    "humaneval",
    "mbpp",
    "bbh",
)
MAXIMAL_DATASETS = tuple(DATASET_REGISTRY)


def dataset_names_for_tier(tier: str) -> tuple[str, ...]:
    if tier == "minimal":
        return MINIMAL_DATASETS
    if tier == "strong":
        return STRONG_DATASETS
    if tier == "maximal":
        return MAXIMAL_DATASETS
    raise ValueError(f"Unknown tier: {tier}")


def resolve_dataset_names(selection: str, tier: str) -> list[str]:
    if selection.strip().lower() == "all":
        names = list(dataset_names_for_tier(tier))
    else:
        names = [item.strip().lower() for item in selection.split(",") if item.strip()]
    unknown = [name for name in names if name not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return names


def _read_local_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [dict(row) for row in obj]
        if isinstance(obj, dict):
            for key in ("data", "examples", "rows"):
                if isinstance(obj.get(key), list):
                    return [dict(row) for row in obj[key]]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    raise ValueError(f"Unsupported local dataset file: {path}")


def _load_hf_rows(spec: DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    for candidate in spec.hf_candidates:
        path = candidate["path"]
        config = candidate.get("config")
        split = candidate.get("split") or spec.split
        try:
            dataset = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
            return [dict(row) for row in dataset], {"source": "huggingface", "path": path, "config": config, "split": split}
        except Exception as exc:
            errors.append(f"{path}/{config or '-'}:{split}: {exc}")
    raise RuntimeError(f"Could not load dataset {spec.name}. Tried: {' | '.join(errors)}")


def _first(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        current: Any = row
        found = True
        for part in key.split("."):
            if not isinstance(current, Mapping):
                found = False
                break
            if part in current:
                current = current[part]
                continue
            lowered = {str(raw_key).lower(): raw_key for raw_key in current}
            raw_key = lowered.get(part.lower())
            if raw_key is None:
                found = False
                break
            current = current[raw_key]
        if found and current not in (None, ""):
            return current
    return None


def _extract_chat_question(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    texts: list[str] = []
    for message in value:
        if isinstance(message, Mapping):
            role = str(message.get("role") or message.get("from") or "").lower()
            content = message.get("content") or message.get("value")
            if isinstance(content, str) and content.strip() and role in {"user", "human", ""}:
                texts.append(content.strip())
        elif isinstance(message, str) and message.strip():
            texts.append(message.strip())
    return texts[-1] if texts else None


def _extract_question(row: Mapping[str, Any]) -> str | None:
    concat_question = _first(row, ("question_concat",))
    if isinstance(concat_question, str) and concat_question.strip():
        return concat_question.strip()

    body = _first(row, ("body",))
    direct_question = _first(row, ("question",))
    if isinstance(body, str) and body.strip() and isinstance(direct_question, str) and direct_question.strip():
        return f"{body.strip()} {direct_question.strip()}"

    for key in ("messages", "conversations", "prompt"):
        chat_question = _extract_chat_question(_first(row, (key,)))
        if chat_question:
            return chat_question

    question = _first(row, ("question", "problem", "prompt", "query", "instruction", "input", "text"))
    if isinstance(question, str) and question.strip():
        return question.strip()
    return None


def _extract_after_marker(text: str) -> str | None:
    if "####" in text:
        answer = text.rsplit("####", 1)[1].strip()
        if answer:
            return answer
    marker = "\\boxed{"
    if marker not in text:
        return None
    start = text.rfind(marker) + len(marker)
    depth = 1
    chars: list[str] = []
    for char in text[start:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
    answer = "".join(chars).strip()
    return answer or None


def _normalize_choices(row: Mapping[str, Any]) -> list[str]:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, Mapping):
        if "text" in choices and isinstance(choices["text"], list):
            return [str(item) for item in choices["text"]]
        return [str(choices[key]) for key in sorted(choices)]
    if isinstance(choices, list):
        if choices and isinstance(choices[0], Mapping):
            return [str(item.get("text") or item.get("content") or item) for item in choices]
        return [str(item) for item in choices]
    extracted: list[str] = []
    for letter in "ABCDEFGHIJ":
        value = row.get(letter) or row.get(f"option_{letter.lower()}") or row.get(f"option{letter}")
        if value:
            extracted.append(str(value))
    return extracted


def _normalize_answer(row: Mapping[str, Any], spec: DatasetSpec) -> Any:
    if spec.task_type == "drop":
        value = _first(row, ("answers_spans.spans", "answers", "answer", "validated_answers"))
        if isinstance(value, Mapping):
            for key in ("spans", "text", "answer"):
                if key in value:
                    return value[key]
        return value
    if spec.task_type == "code":
        return _first(row, ("canonical_solution", "solution", "answer", "target")) or ""
    value = _first(
        row,
        (
            "answer",
            "final_answer",
            "target",
            "label",
            "gold",
            "gold_parsed",
            "reward_model.ground_truth",
            "extra_info.answer",
            "correct_answer",
            "answerKey",
        ),
    )
    if isinstance(value, int) and spec.task_type == "multiple_choice":
        return chr(65 + value)
    if value not in (None, ""):
        return value

    solution = _first(row, ("solution", "reference", "rationale", "cot", "response"))
    if isinstance(solution, list):
        solution = "\n".join(str(item) for item in solution if item is not None)
    if isinstance(solution, str):
        marker_answer = _extract_after_marker(solution)
        if marker_answer:
            return marker_answer
    return value


def _build_code_tests(row: Mapping[str, Any], dataset_name: str) -> str:
    if dataset_name == "humaneval":
        tests = str(row.get("test") or "")
        entry_point = row.get("entry_point")
        return f"{tests}\ncheck({entry_point})" if tests and entry_point else tests
    if row.get("test"):
        tests = row["test"]
        if isinstance(tests, list):
            return "\n".join(str(item) for item in tests)
        return str(tests)
    if row.get("test_list"):
        return "\n".join(str(item) for item in row["test_list"])
    if row.get("input_output"):
        return "# LiveCodeBench input_output tests require a local harness; provide executable tests in local JSONL."
    return ""


def normalize_row(row: Mapping[str, Any], spec: DatasetSpec, index: int) -> EvalExample:
    question = _extract_question(row)
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Could not extract question for {spec.name} row {index}")

    choices = _normalize_choices(row) if spec.task_type == "multiple_choice" else []
    if spec.name == "arc-challenge" and not choices and isinstance(row.get("choices"), Mapping):
        choices = [str(item) for item in row["choices"].get("text", [])]

    answer = _normalize_answer(row, spec)
    metadata = {
        "raw_index": index,
        "source": spec.name,
    }
    if spec.task_type == "drop":
        metadata["passage"] = row.get("passage")
    if spec.task_type == "code":
        metadata["tests"] = _build_code_tests(row, spec.name)
        metadata["entry_point"] = row.get("entry_point")
        if row.get("prompt") and spec.name == "humaneval":
            question = str(row["prompt"])

    example_id = str(_first(row, ("id", "task_id", "uid", "problem_id", "ID")) or index)
    return EvalExample(
        dataset=spec.name,
        family=spec.family,
        task_type=spec.task_type,
        example_id=example_id,
        question=question.strip(),
        answer=answer,
        choices=choices,
        metadata=metadata,
    )


def load_examples(dataset_name: str, *, limit: int | None = None, seed: int = 42) -> tuple[list[EvalExample], dict[str, Any]]:
    spec = DATASET_REGISTRY[dataset_name]
    local_path = os.environ.get(spec.local_path_env or "")
    if local_path:
        rows = _read_local_rows(local_path)
        source = {"source": "local_file", "path": local_path, "env": spec.local_path_env}
    elif spec.requires_local_file:
        raise RuntimeError(f"{dataset_name} requires a local file via {spec.local_path_env}")
    else:
        rows, source = _load_hf_rows(spec)

    examples: list[EvalExample] = []
    for index, row in enumerate(rows):
        try:
            examples.append(normalize_row(row, spec, index))
        except Exception as exc:
            if len(examples) == 0:
                raise
            source.setdefault("normalization_errors", []).append({"index": index, "error": str(exc)})

    effective_limit = limit if limit is not None else spec.default_limit
    if effective_limit is not None:
        examples = examples[:effective_limit]
    rng = random.Random(seed)
    source["loaded_examples"] = len(examples)
    source["seed"] = seed
    source["order_checksum"] = rng.random()
    return examples, source
