"""Answer extraction and scoring for reasoning benchmarks."""

from __future__ import annotations

import contextlib
import io
import json
import math
import multiprocessing as mp
import re
import statistics
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from src.reasoning_eval.schema import EvalExample, Generation


BOXED_RE = re.compile(r"\\boxed\s*\{")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
FINAL_MARKER_RE = re.compile(
    r"(?:####|final answer|the answer is|answer)\s*:?\s*([^\n]+)",
    re.IGNORECASE,
)
OPTION_RE = re.compile(r"(?:\\boxed\s*\{)?\b([A-J])\b(?:\})?", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
)
LATEX_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


@dataclass
class ScoreResult:
    parsed: Any
    normalized_prediction: str | None
    normalized_gold: str | None
    correct: bool
    parsable: bool
    format_ok: bool
    details: dict[str, Any]


def _balanced_boxed(text: str) -> str | None:
    last_match = None
    for match in BOXED_RE.finditer(text):
        last_match = match
    if last_match is None:
        return None
    start = last_match.end()
    depth = 1
    chars: list[str] = []
    for char in text[start:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
    return None


def strip_latex_noise(text: str) -> str:
    text = str(text).strip()
    text = LATEX_FRAC_RE.sub(r"(\1)/(\2)", text)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\%", "%").replace("\\$", "$")
    text = text.replace("$", "").replace(",", "")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def extract_final_answer(text: str, *, task_type: str = "math") -> str | None:
    """Extract the final answer from model output."""
    text = text or ""
    tag_matches = ANSWER_TAG_RE.findall(text)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed = _balanced_boxed(text)
    if boxed:
        return boxed

    if task_type == "multiple_choice":
        tail = text[-300:]
        matches = OPTION_RE.findall(tail)
        if matches:
            return matches[-1].upper()

    marker_matches = FINAL_MARKER_RE.findall(text)
    if marker_matches:
        candidate = marker_matches[-1].strip()
        if task_type == "multiple_choice":
            option = OPTION_RE.findall(candidate)
            if option:
                return option[-1].upper()
        number = NUMBER_RE.findall(candidate)
        return number[-1] if number else candidate

    tail = text[-500:]
    if task_type == "multiple_choice":
        matches = OPTION_RE.findall(tail)
        return matches[-1].upper() if matches else None

    numbers = NUMBER_RE.findall(tail)
    if numbers:
        return numbers[-1].strip()
    return None


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = strip_latex_noise(str(value)).upper()
    match = OPTION_RE.search(text)
    return match.group(1).upper() if match else None


def normalize_numeric(value: Any) -> str | None:
    if value is None:
        return None
    text = strip_latex_noise(str(value))
    if not text:
        return None
    if text.endswith("%"):
        inner = normalize_numeric(text[:-1])
        if inner is None:
            return None
        return str(Decimal(inner) / Decimal(100)).rstrip("0").rstrip(".")

    frac_match = re.fullmatch(r"\(?\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*\)?", text)
    if frac_match:
        try:
            frac = Fraction(Decimal(frac_match.group(1))) / Fraction(Decimal(frac_match.group(2)))
            return str(frac)
        except (InvalidOperation, ZeroDivisionError):
            return None

    number_match = NUMBER_RE.findall(text)
    candidate = number_match[-1] if number_match else text
    candidate = candidate.replace(",", "").strip()
    try:
        dec = Decimal(candidate)
    except InvalidOperation:
        return text.lower()
    normalized = format(dec.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def numeric_equal(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    try:
        p = Fraction(Decimal(prediction)) if "/" not in prediction else Fraction(prediction)
        g = Fraction(Decimal(gold)) if "/" not in gold else Fraction(gold)
        return p == g or math.isclose(float(p), float(g), rel_tol=1e-9, abs_tol=1e-9)
    except Exception:
        return prediction.strip().lower() == gold.strip().lower()


def score_math_generation(example: EvalExample, generation: Generation) -> ScoreResult:
    parsed = extract_final_answer(generation.text, task_type="math")
    pred_norm = normalize_numeric(parsed)
    gold_norm = normalize_numeric(example.answer)
    correct = numeric_equal(pred_norm, gold_norm)
    return ScoreResult(
        parsed=parsed,
        normalized_prediction=pred_norm,
        normalized_gold=gold_norm,
        correct=correct,
        parsable=parsed is not None and pred_norm is not None,
        format_ok=bool(_balanced_boxed(generation.text) or ANSWER_TAG_RE.search(generation.text)),
        details={},
    )


def score_mc_generation(example: EvalExample, generation: Generation) -> ScoreResult:
    parsed = extract_final_answer(generation.text, task_type="multiple_choice")
    pred_norm = normalize_choice(parsed)
    gold_norm = normalize_choice(example.answer)
    return ScoreResult(
        parsed=parsed,
        normalized_prediction=pred_norm,
        normalized_gold=gold_norm,
        correct=pred_norm is not None and pred_norm == gold_norm,
        parsable=pred_norm is not None,
        format_ok=bool(_balanced_boxed(generation.text) or ANSWER_TAG_RE.search(generation.text)),
        details={},
    )


def normalize_text_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = strip_latex_noise(str(value)).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9.\-/ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_text_generation(example: EvalExample, generation: Generation) -> ScoreResult:
    parsed = extract_final_answer(generation.text, task_type="math")
    pred_norm = normalize_text_answer(parsed)
    gold_values = example.answer if isinstance(example.answer, list) else [example.answer]
    gold_norms = [normalize_text_answer(gold) for gold in gold_values]
    correct = pred_norm is not None and pred_norm in gold_norms
    return ScoreResult(
        parsed=parsed,
        normalized_prediction=pred_norm,
        normalized_gold="|".join(g for g in gold_norms if g is not None),
        correct=correct,
        parsable=pred_norm is not None,
        format_ok=bool(_balanced_boxed(generation.text) or ANSWER_TAG_RE.search(generation.text)),
        details={},
    )


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


def _code_worker(code: str, tests: str, queue: mp.Queue) -> None:
    started = time.time()
    stdout = io.StringIO()
    stderr = io.StringIO()
    namespace: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace)
            exec(tests, namespace)
        queue.put({"passed": True, "stderr": stderr.getvalue(), "stdout": stdout.getvalue(), "runtime_s": time.time() - started})
    except BaseException as exc:
        queue.put(
            {
                "passed": False,
                "stderr": f"{type(exc).__name__}: {exc}\n{stderr.getvalue()}",
                "stdout": stdout.getvalue(),
                "runtime_s": time.time() - started,
            }
        )


def run_code_tests(code: str, tests: str, timeout_s: float = 5.0) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_code_worker, args=(code, tests, queue))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return {"passed": False, "stderr": "timeout", "stdout": "", "runtime_s": timeout_s, "timeout": True}
    if queue.empty():
        return {"passed": False, "stderr": "no result from subprocess", "stdout": "", "runtime_s": None}
    result = queue.get()
    result["timeout"] = False
    return result


def score_code_generation(example: EvalExample, generation: Generation) -> ScoreResult:
    code = _extract_code(generation.text)
    tests = str(example.metadata.get("tests") or "")
    if not tests:
        return ScoreResult(code, None, None, False, False, bool(code), {"error": "missing_tests"})
    result = run_code_tests(code, tests, timeout_s=float(example.metadata.get("timeout_s", 5.0)))
    return ScoreResult(
        parsed=code,
        normalized_prediction=None,
        normalized_gold=None,
        correct=bool(result.get("passed")),
        parsable=bool(code),
        format_ok=bool(code),
        details=result,
    )


def score_generation(example: EvalExample, generation: Generation) -> ScoreResult:
    if example.task_type == "multiple_choice":
        return score_mc_generation(example, generation)
    if example.task_type == "drop":
        return score_text_generation(example, generation)
    if example.task_type == "code":
        return score_code_generation(example, generation)
    return score_math_generation(example, generation)


def pass_at_k(correct_values: list[bool], k: int) -> float:
    if not correct_values:
        return 0.0
    return float(any(correct_values[:k]))


def avg_at_k(correct_values: list[bool], k: int) -> float:
    if not correct_values:
        return 0.0
    subset = correct_values[:k]
    return sum(1 for item in subset if item) / max(1, len(subset))


def median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0

