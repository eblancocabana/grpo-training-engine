import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_reasoning_vllm import main, parse_args
from src.reasoning_eval.adapters import prepare_vllm_adapter
from src.reasoning_eval.datasets import DATASET_REGISTRY, load_examples, normalize_row, resolve_dataset_names
from src.reasoning_eval.io import completed_example_keys, prepare_output_dir
from src.reasoning_eval.models import ModelSpec, load_models_config
from src.reasoning_eval.prompts import build_prompt
from src.reasoning_eval.schema import EvalExample, Generation
from src.reasoning_eval.scoring import (
    extract_final_answer,
    normalize_numeric,
    score_code_generation,
    score_generation,
)
from src.reasoning_eval.vllm_engine import GenerationProtocol


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _models_yaml(path: Path) -> None:
    path.write_text(
        """
models:
  base:
    model_id: local/mock
    adapter: null
  sent_best:
    model_id: local/mock
    adapter: /tmp/sent-best
    selection_source: in_training_validation
""".strip(),
        encoding="utf-8",
    )


def test_cli_parsing_sets_tier_defaults(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--models-config",
            str(tmp_path / "models.yaml"),
            "--datasets",
            "all",
            "--tier",
            "strong",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.protocol == "both"
    assert args.max_new_tokens == 16384


def test_models_config_requires_in_training_selection_source(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  sent_best:\n    model_id: base\n    adapter: /tmp/a\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="in_training_validation"):
        load_models_config(path)


def test_manual_lora_pt_adapter_is_converted_for_vllm(tmp_path: Path) -> None:
    source = tmp_path / "manual_lora.pt"
    torch.save(
        {
            "lora_weights": {
                "model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
                "model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 2),
            }
        },
        source,
    )
    adapter_dir = Path(
        prepare_vllm_adapter(
            ModelSpec("adapter", "base-model", adapter=str(source)),
            tmp_path / "out",
        )
    )

    assert (adapter_dir / "adapter_model.safetensors").exists()
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["r"] == 2
    assert config["lora_alpha"] == 4
    assert config["target_modules"] == ["q_proj"]


def test_dataset_loading_from_local_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "gsm8k.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"id": "a", "question": "What is 40+2?", "answer": "#### 42"},
            {"id": "b", "question": "What is 1+1?", "answer": "2"},
        ],
    )
    monkeypatch.setenv("REASONING_EVAL_GSM8K_PATH", str(dataset_path))

    examples, metadata = load_examples("gsm8k", limit=1, seed=7)

    assert metadata["source"] == "local_file"
    assert examples[0].example_id == "a"
    assert examples[0].answer == "#### 42"


def test_normalize_svamp_capitalized_schema() -> None:
    row = {
        "ID": "chal-736",
        "Body": "There are 35 birds in Asia and 62 in Africa.",
        "Question": "How many more are in Africa?",
        "Answer": "27",
        "question_concat": "There are 35 birds in Asia and 62 in Africa. How many more are in Africa?",
    }

    example = normalize_row(row, DATASET_REGISTRY["svamp"], 0)

    assert example.example_id == "chal-736"
    assert example.question.startswith("There are 35 birds")
    assert example.answer == "27"


def test_normalize_dapo_chat_prompt_schema() -> None:
    row = {
        "prompt": [{"role": "user", "content": "Solve 40+2."}],
        "label": "42",
    }

    example = normalize_row(row, DATASET_REGISTRY["dapo-math-17k"], 0)

    assert example.question == "Solve 40+2."
    assert example.answer == "42"


def test_prompt_construction_for_math_and_multiple_choice() -> None:
    math_example = EvalExample("gsm8k", "math", "math", "1", "What is 6*7?", "42")
    mc_example = EvalExample(
        "gpqa-diamond",
        "science_general",
        "multiple_choice",
        "2",
        "Pick one.",
        "B",
        choices=["alpha", "beta"],
    )

    assert "\\boxed{}" in build_prompt(math_example)
    mc_prompt = build_prompt(mc_example)
    assert "A. alpha" in mc_prompt
    assert "B. beta" in mc_prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ("work #### 42", "42"),
        ("therefore \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("Final answer: 12.5%", "12.5%"),
        ("The answer is (C)", "C"),
    ],
)
def test_answer_extraction_patterns(text: str, expected: str) -> None:
    task_type = "multiple_choice" if expected == "C" else "math"
    assert extract_final_answer(text, task_type=task_type) == expected


def test_math_scoring_normalizes_fraction_decimal_and_percent() -> None:
    example = EvalExample("math500", "math", "math", "1", "x?", "1/2")
    generation = Generation("base", "math500", "deterministic", "1", 0, "p", "\\boxed{0.5}", 4, "stop")
    assert score_generation(example, generation).correct is True
    assert normalize_numeric("50%") == "0.5"


def test_multiple_choice_scoring() -> None:
    example = EvalExample("arc-challenge", "science_general", "multiple_choice", "1", "x?", "B", ["a", "b"])
    generation = Generation("base", "arc-challenge", "deterministic", "1", 0, "p", "Final answer: B", 3, "stop")
    assert score_generation(example, generation).correct is True


def test_code_scoring_runs_subprocess_tests() -> None:
    example = EvalExample(
        "humaneval",
        "coding",
        "code",
        "task",
        "write add",
        "",
        metadata={"tests": "assert add(2, 3) == 5", "timeout_s": 2.0},
    )
    generation = Generation("base", "humaneval", "deterministic", "task", 0, "p", "def add(a, b):\n    return a + b\n", 6, "stop")

    result = score_code_generation(example, generation)

    assert result.correct is True
    assert result.details["timeout"] is False


def test_output_dir_protects_previous_results(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "raw_generations.jsonl").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_output_dir(out, resume=False, overwrite=False)


def test_completed_example_keys_for_resume(tmp_path: Path) -> None:
    path = tmp_path / "scores_by_example.jsonl"
    _write_jsonl(
        path,
        [{"model_key": "base", "dataset": "gsm8k", "protocol": "deterministic", "example_id": "a"}],
    )

    assert completed_example_keys(path) == {("base", "gsm8k", "deterministic", "a")}


def test_protocol_dataset_resolution() -> None:
    assert resolve_dataset_names("gsm8k,math500", "minimal") == ["gsm8k", "math500"]
    assert "humaneval" in resolve_dataset_names("all", "minimal")


def test_end_to_end_mock_evaluation_and_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "gsm8k.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"id": "a", "question": "What is 40+2?", "answer": "42"},
            {"id": "b", "question": "What is 41+1?", "answer": "42"},
        ],
    )
    monkeypatch.setenv("REASONING_EVAL_GSM8K_PATH", str(dataset_path))
    models_path = tmp_path / "models.yaml"
    _models_yaml(models_path)
    out = tmp_path / "eval"

    argv = [
        "--models-config",
        str(models_path),
        "--models",
        "base",
        "--datasets",
        "gsm8k",
        "--tier",
        "minimal",
        "--output-dir",
        str(out),
        "--limit-per-dataset",
        "2",
        "--batch-size",
        "1",
        "--mock-generator",
    ]

    assert main(argv) == 0
    first_raw = (out / "raw_generations.jsonl").read_text(encoding="utf-8")
    assert (out / "summary_by_dataset.csv").exists()
    assert (out / "summary_by_model.csv").exists()
    assert (out / "macro_family_summary.csv").exists()

    assert main([*argv, "--resume"]) == 0
    second_raw = (out / "raw_generations.jsonl").read_text(encoding="utf-8")
    assert second_raw == first_raw
