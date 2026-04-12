import importlib
import sys
from unittest.mock import patch


def test_benchmark_import_is_safe_without_wandb():
    with patch.dict(sys.modules, {"wandb": None}):
        sys.modules.pop("src.grpo.benchmark", None)
        benchmark = importlib.import_module("src.grpo.benchmark")

    assert benchmark.WANDB_AVAILABLE is False
    sys.modules.pop("src.grpo.benchmark", None)
