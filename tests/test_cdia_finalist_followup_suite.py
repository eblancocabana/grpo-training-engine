import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "cdia_finalist_followup_suite.py"
)


def _load_suite_module():
    spec = importlib.util.spec_from_file_location(
        "cdia_finalist_followup_suite", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_suite_matches_expected_finalist_configs():
    module = _load_suite_module()

    configs = module.build_suite()

    assert [cfg.name for cfg in configs] == [
        "baseline",
        "length_penalty_0",
        "length_penalty_00025",
        "length_penalty_0005",
        "length_penalty_0005_batch_eff_32",
        "length_penalty_0005_eps_high_02",
        "length_penalty_0_batch_eff_32",
        "length_penalty_0_eps_high_02",
        "length_penalty_0005_mask_truncated_off",
    ]

    resolved = {cfg.name: cfg.resolved() for cfg in configs}
    assert resolved["baseline"]["length_penalty_coef"] == pytest.approx(0.001)
    assert resolved["length_penalty_0"]["length_penalty_coef"] == pytest.approx(0.0)
    assert resolved["length_penalty_00025"]["length_penalty_coef"] == pytest.approx(
        0.00025
    )
    assert resolved["length_penalty_0005"]["length_penalty_coef"] == pytest.approx(
        0.0005
    )
    assert (
        resolved["length_penalty_0005_batch_eff_32"]["gradient_accumulation_steps"]
        == 8
    )
    assert resolved["length_penalty_0005_eps_high_02"]["epsilon_high"] == pytest.approx(
        0.2
    )
    assert (
        resolved["length_penalty_0_batch_eff_32"]["gradient_accumulation_steps"] == 8
    )
    assert resolved["length_penalty_0_eps_high_02"]["epsilon_high"] == pytest.approx(
        0.2
    )
    assert (
        resolved["length_penalty_0005_mask_truncated_off"][
            "mask_truncated_completions"
        ]
        is False
    )


def test_followup_suite_surfaces_peak_vs_final_and_truncation_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_suite_module()
    output_dir = tmp_path / "output"
    plan_path = tmp_path / "cdia_finalist_followup_suite_plan.json"
    suite = module.CdiaBenchmarkSuite(
        output_dir=output_dir,
        trainer_steps=120,
        use_wandb=False,
        run_prefix="cdia",
        plan_path=plan_path,
    )
    config = next(
        cfg
        for cfg in module.build_suite()
        if cfg.name == "length_penalty_0005_mask_truncated_off"
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
            "train/avg_reward": 0.8,
            "train/reward_std": 0.3,
            "train/reward_min": -0.2,
            "train/reward_max": 1.0,
            "train/avg_response_length": 640.0,
            "train/entropy_masked_ratio": 0.48,
            "train/actual_truncated_completions_ratio": 0.7,
            "train/truncated_completions_masked_out_of_loss_ratio": 0.0,
            "train/truncation_masking_active": 0.0,
            "train/positive_advantages_ratio": 0.75,
            "train/oom_backoff_count": 1,
            "train/optimizer_step": 1,
        },
        {
            "event": "train_metrics",
            "step": 3,
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
        assert "--no-mask-truncated" in cmd
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        metrics_path = out_dir / "metrics.jsonl"
        metrics_path.write_text(
            "\n".join(json.dumps(entry) for entry in metrics_entries),
            encoding="utf-8",
        )

        stdout = kwargs["stdout"]
        stdout.write("step=1 30.0s/it tokens_per_sec=50.0 loss=0.300 reward=0.200 VRAM: 5.5 GB\n")
        stdout.write("step=2 32.0s/it tokens_per_sec=48.0 loss=0.250 reward=0.800 VRAM: 5.7 GB\n")
        stdout.write("step=3 28.0s/it tokens_per_sec=52.0 loss=0.220 reward=0.500 VRAM: 5.6 GB\n")
        stdout.flush()
        return DummyProcess()

    monkeypatch.setattr(module, "run_with_conda", fake_run_with_conda)

    result = suite.run_benchmark(config)

    assert result.success is True
    assert result.steps_completed == 3
    assert result.reward_avg == pytest.approx(0.5)
    assert result.reward_final == pytest.approx(0.5)
    assert result.reward_peak == pytest.approx(0.8)
    assert result.reward_peak_step == 2
    assert result.reward_drop_from_peak_to_final == pytest.approx(0.3)
    assert result.avg_response_length_final == pytest.approx(430.0)
    assert result.response_length_peak == pytest.approx(640.0)
    assert result.response_length_peak_step == 2
    assert result.response_length_at_reward_peak == pytest.approx(640.0)
    assert result.response_length_drop_from_peak_to_final == pytest.approx(210.0)
    assert result.actual_truncated_completions_ratio_avg == pytest.approx(
        (0.4 + 0.7 + 0.3) / 3.0
    )
    assert result.actual_truncated_completions_ratio_final == pytest.approx(0.3)
    assert result.actual_truncated_completions_ratio_peak == pytest.approx(0.7)
    assert result.actual_truncated_completions_ratio_peak_step == 2
    assert result.truncated_completions_masked_out_of_loss_ratio_final == pytest.approx(
        0.0
    )
    assert result.truncation_masking_active_final == pytest.approx(0.0)
    assert result.optimizer_step_final == 2
    assert result.optimizer_step_fraction_of_trainer_steps == pytest.approx(2.0 / 3.0)
    assert result.trainer_steps_per_optimizer_step == pytest.approx(1.5)

    suite.results = [result]
    suite._generate_report()
    report = (output_dir / "report.md").read_text(encoding="utf-8")

    assert "Reward Peak@Step" in report
    assert "Peak reward: 0.8000 at step 2" in report
    assert "Reward drop from peak to final: 0.3000" in report
    assert "Peak avg response length: 640.0 at step 2" in report
    assert "Avg response length at reward peak step 2: 640.0" in report
    assert "Peak actual truncated completions ratio: 0.7000 at step 2" in report
    assert "Final truncated completions masked out of loss ratio: 0.0000" in report
    assert "Truncation masking active in loss: False" in report
