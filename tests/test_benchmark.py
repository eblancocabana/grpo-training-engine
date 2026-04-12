import importlib
import sys
from unittest.mock import Mock, patch

import pytest


def test_benchmark_import_is_safe_without_wandb():
    with patch.dict(sys.modules, {"wandb": None}):
        sys.modules.pop("src.grpo.benchmark", None)
        benchmark = importlib.import_module("src.grpo.benchmark")

    assert benchmark.WANDB_AVAILABLE is False
    sys.modules["src.grpo.benchmark"] = benchmark


def test_benchmark_requires_single_response_generate_fn():
    from src.grpo.benchmark import GSM8KBenchmark

    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": [1]}
    memory_manager = Mock()
    memory_manager.optimize_for_inference = Mock()
    memory_manager.clear_cache = Mock()

    with patch("src.grpo.benchmark.GRPOGSM8KDataset") as dataset_cls:
        dataset_cls.return_value = [
            {
                "input_ids": [1],
                "attention_mask": [1],
                "answer": "42",
                "question": "What is 40+2?",
            }
        ]
        benchmark = GSM8KBenchmark(
            model=Mock(),
            tokenizer=tokenizer,
            memory_manager=memory_manager,
            generate_fn=lambda input_ids, attention_mask: ["wrong", "42", "42", "42"],
        )

    with pytest.raises(ValueError, match="exactly one response"):
        benchmark.run(step=0)


def test_benchmark_scores_single_generated_response():
    from src.grpo.benchmark import GSM8KBenchmark

    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": [1, 2, 3]}
    memory_manager = Mock()
    memory_manager.optimize_for_inference = Mock()
    memory_manager.clear_cache = Mock()

    with patch("src.grpo.benchmark.GRPOGSM8KDataset") as dataset_cls:
        dataset_cls.return_value = [
            {
                "input_ids": [1],
                "attention_mask": [1],
                "answer": "42",
                "question": "What is 40+2?",
            }
        ]
        benchmark = GSM8KBenchmark(
            model=Mock(),
            tokenizer=tokenizer,
            memory_manager=memory_manager,
            generate_fn=lambda input_ids, attention_mask: ["The answer is \\boxed{42} </think>"],
        )

    metrics = benchmark.run(step=0)
    assert metrics["val/acc"] == 1.0
    assert metrics["val/avg_len"] == 3.0
