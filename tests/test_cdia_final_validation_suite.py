import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "cdia_final_validation_suite.py"
)


def _load_suite_module():
    spec = importlib.util.spec_from_file_location(
        "cdia_final_validation_suite", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_suite_matches_expected_final_validation_configs():
    module = _load_suite_module()

    configs = module.build_suite()

    assert [cfg.name for cfg in configs] == [
        "length_penalty_0_seed1",
        "length_penalty_0_seed2",
        "length_penalty_0005_mask_truncated_off_seed1",
        "length_penalty_0005_mask_truncated_off_seed2",
        "length_penalty_0_difficulty_weighting_seed1",
    ]

    resolved = {cfg.name: cfg.resolved() for cfg in configs}
    assert resolved["length_penalty_0_seed1"]["seed"] == 1
    assert resolved["length_penalty_0_seed1"]["sent_enabled"] is False
    assert resolved["length_penalty_0_seed1"]["length_penalty_coef"] == pytest.approx(0.0)
    assert resolved["length_penalty_0_seed2"]["seed"] == 2
    assert (
        resolved["length_penalty_0005_mask_truncated_off_seed1"][
            "mask_truncated_completions"
        ]
        is False
    )
    assert (
        resolved["length_penalty_0005_mask_truncated_off_seed2"][
            "length_penalty_coef"
        ]
        == pytest.approx(0.0005)
    )
    assert resolved["length_penalty_0_difficulty_weighting_seed1"]["sent_enabled"] is True
    assert (
        resolved["length_penalty_0_difficulty_weighting_seed1"][
            "difficulty_weighting_mode"
        ]
        == "sent_rank_linear"
    )
    assert (
        resolved["length_penalty_0_difficulty_weighting_seed1"][
            "difficulty_weighting_min_weight"
        ]
        == pytest.approx(1.0)
    )
    assert (
        resolved["length_penalty_0_difficulty_weighting_seed1"][
            "difficulty_weighting_max_weight"
        ]
        == pytest.approx(1.2)
    )


def test_print_summary_lists_strict_and_exploratory_runs(capsys: pytest.CaptureFixture[str]):
    module = _load_suite_module()

    module.print_summary(module.build_suite(), trainer_steps=120)
    output = capsys.readouterr().out

    assert "Strict Finalist Comparison" in output
    assert "Exploratory Ablation" in output
    assert "length_penalty_0_seed1" in output
    assert "length_penalty_0_seed2" in output
    assert "length_penalty_0005_mask_truncated_off_seed1" in output
    assert "length_penalty_0005_mask_truncated_off_seed2" in output
    assert "length_penalty_0_difficulty_weighting_seed1" in output


def test_validation_suite_report_keeps_strict_and_exploratory_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_suite_module()
    output_dir = tmp_path / "output"
    plan_path = tmp_path / "cdia_final_validation_suite_plan.json"
    suite = module.CdiaBenchmarkSuite(
        output_dir=output_dir,
        trainer_steps=120,
        use_wandb=False,
        run_prefix="cdia",
        plan_path=plan_path,
    )
    strict_config = next(
        cfg for cfg in module.build_suite() if cfg.name == "length_penalty_0_seed1"
    )
    exploratory_config = next(
        cfg
        for cfg in module.build_suite()
        if cfg.name == "length_penalty_0_difficulty_weighting_seed1"
    )

    metrics_entries = [
        {
            "event": "train_metrics",
            "step": 1,
            "train/avg_reward": 0.2,
            "train/reward_std": 0.6,
            "train/reward_min": -1.0,
            "train/reward_max": 1.0,
            "train/avg_response_length": 500.0,
            "train/entropy_masked_ratio": 0.45,
            "train/actual_truncated_completions_ratio": 0.4,
            "train/truncated_completions_masked_out_of_loss_ratio": 0.0,
            "train/truncation_masking_active": 0.0,
            "train/positive_advantages_ratio": 0.5,
            "train/oom_backoff_count": 1,
            "train/optimizer_step": 1,
        },
        {
            "event": "train_metrics",
            "step": 2,
            "train/avg_reward": 0.5,
            "train/reward_std": 0.2,
            "train/reward_min": 0.0,
            "train/reward_max": 0.9,
            "train/avg_response_length": 430.0,
            "train/entropy_masked_ratio": 0.42,
            "train/actual_truncated_completions_ratio": 0.3,
            "train/truncated_completions_masked_out_of_loss_ratio": 0.0,
            "train/truncation_masking_active": 0.0,
            "train/positive_advantages_ratio": 0.6,
            "train/oom_backoff_count": 2,
            "train/optimizer_step": 2,
        },
    ]

    class DummyProcess:
        returncode = 0

        def wait(self):
            return 0

    def fake_run_with_conda(cmd, **kwargs):
        if "--difficulty-weighting-mode" in cmd:
            assert "--difficulty-weighting-mode" in cmd
            assert cmd[cmd.index("--difficulty-weighting-mode") + 1] == "sent_rank_linear"
            assert "--no-sent" not in cmd
        else:
            assert "--no-sent" in cmd

        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        metrics_path = out_dir / "metrics.jsonl"
        metrics_path.write_text(
            "\n".join(json.dumps(entry) for entry in metrics_entries),
            encoding="utf-8",
        )

        stdout = kwargs["stdout"]
        stdout.write("step=1 30.0s/it tokens_per_sec=50.0 loss=0.300 reward=0.200 VRAM: 5.5 GB\n")
        stdout.write("step=2 28.0s/it tokens_per_sec=52.0 loss=0.220 reward=0.500 VRAM: 5.6 GB\n")
        stdout.flush()
        return DummyProcess()

    monkeypatch.setattr(module, "run_with_conda", fake_run_with_conda)

    strict_result = suite.run_benchmark(strict_config)
    exploratory_result = suite.run_benchmark(exploratory_config)

    assert strict_result.success is True
    assert exploratory_result.success is True

    suite.results = [strict_result, exploratory_result]
    suite._generate_report()
    report = (output_dir / "report.md").read_text(encoding="utf-8")

    assert "## Strict Finalist Comparison" in report
    assert "## Exploratory Difficulty-Weighting Ablation" in report
    assert "length_penalty_0_seed1" in report
    assert "length_penalty_0_difficulty_weighting_seed1" in report
    assert "- SENT enabled: False" in report
    assert "- SENT enabled: True" in report
    assert "- Difficulty weighting mode: sent_rank_linear" in report
