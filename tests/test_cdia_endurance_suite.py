import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "cdia_endurance_suite.py"
)


def _load_suite_module():
    spec = importlib.util.spec_from_file_location("cdia_endurance_suite", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_suite_matches_expected_endurance_configs():
    module = _load_suite_module()

    configs = module.build_suite()

    assert [cfg.name for cfg in configs] == [
        "length_penalty_0_seed1",
        "length_penalty_0_seed2",
        "length_penalty_0_seed3",
        "length_penalty_0005_mask_truncated_off_seed1",
        "length_penalty_0005_mask_truncated_off_seed2",
        "length_penalty_0005_mask_truncated_off_seed3",
    ]

    resolved = {cfg.name: cfg.resolved() for cfg in configs}
    assert resolved["length_penalty_0_seed1"]["seed"] == 1
    assert resolved["length_penalty_0_seed2"]["seed"] == 2
    assert resolved["length_penalty_0_seed3"]["seed"] == 3
    assert resolved["length_penalty_0_seed1"]["sent_enabled"] is False
    assert resolved["length_penalty_0_seed1"]["length_penalty_coef"] == pytest.approx(0.0)
    assert (
        resolved["length_penalty_0005_mask_truncated_off_seed1"][
            "mask_truncated_completions"
        ]
        is False
    )
    assert (
        resolved["length_penalty_0005_mask_truncated_off_seed3"][
            "length_penalty_coef"
        ]
        == pytest.approx(0.0005)
    )
    assert module.DEFAULT_TRAINER_STEPS == 240


def test_write_plan_keeps_default_finalist_only_run_set(tmp_path: Path):
    module = _load_suite_module()
    plan_path = tmp_path / "cdia_endurance_suite_plan.json"

    module.write_plan(module.build_suite(), plan_path, trainer_steps=240)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert payload["trainer_steps"] == 240
    assert payload["benchmark_cadence_trainer_steps"] == 100
    assert payload["final_checkpoint_benchmark"] is True
    assert payload["comparison_groups"] == [
        "length_penalty_0",
        "length_penalty_0005_mask_truncated_off",
    ]
    assert payload["default_run_names"] == [
        "length_penalty_0_seed1",
        "length_penalty_0_seed2",
        "length_penalty_0_seed3",
        "length_penalty_0005_mask_truncated_off_seed1",
        "length_penalty_0005_mask_truncated_off_seed2",
        "length_penalty_0005_mask_truncated_off_seed3",
    ]


def test_endurance_suite_surfaces_benchmark_accuracy_and_selection_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_suite_module()
    output_dir = tmp_path / "output"
    plan_path = tmp_path / "cdia_endurance_suite_plan.json"
    suite = module.CdiaEnduranceSuite(
        output_dir=output_dir,
        trainer_steps=240,
        use_wandb=False,
        run_prefix="cdia",
        plan_path=plan_path,
    )
    length_zero = next(
        cfg for cfg in module.build_suite() if cfg.name == "length_penalty_0_seed1"
    )
    penalized = next(
        cfg
        for cfg in module.build_suite()
        if cfg.name == "length_penalty_0005_mask_truncated_off_seed1"
    )

    metrics_by_run = {
        "length_penalty_0_seed1": [
            {
                "event": "train_metrics",
                "step": 100,
                "train/avg_reward": 0.40,
                "train/reward_std": 0.30,
                "train/reward_min": -0.10,
                "train/reward_max": 0.90,
                "train/avg_response_length": 640.0,
                "train/entropy_masked_ratio": 0.45,
                "train/actual_truncated_completions_ratio": 0.10,
                "train/truncated_completions_masked_out_of_loss_ratio": 0.10,
                "train/truncation_masking_active": 1.0,
                "train/positive_advantages_ratio": 0.55,
                "train/oom_backoff_count": 0,
                "train/optimizer_step": 25,
            },
            {
                "event": "benchmark_metrics",
                "step": 100,
                "benchmark_phase": "periodic",
                "val/acc": 0.42,
                "val/format_compliance": 0.92,
                "val/avg_len": 610.0,
            },
            {
                "event": "train_metrics",
                "step": 200,
                "train/avg_reward": 0.52,
                "train/reward_std": 0.20,
                "train/reward_min": 0.00,
                "train/reward_max": 0.95,
                "train/avg_response_length": 700.0,
                "train/entropy_masked_ratio": 0.44,
                "train/actual_truncated_completions_ratio": 0.12,
                "train/truncated_completions_masked_out_of_loss_ratio": 0.12,
                "train/truncation_masking_active": 1.0,
                "train/positive_advantages_ratio": 0.60,
                "train/oom_backoff_count": 0,
                "train/optimizer_step": 50,
            },
            {
                "event": "benchmark_metrics",
                "step": 200,
                "benchmark_phase": "periodic",
                "val/acc": 0.48,
                "val/format_compliance": 0.94,
                "val/avg_len": 640.0,
            },
            {
                "event": "train_metrics",
                "step": 240,
                "train/avg_reward": 0.49,
                "train/reward_std": 0.18,
                "train/reward_min": 0.05,
                "train/reward_max": 0.90,
                "train/avg_response_length": 680.0,
                "train/entropy_masked_ratio": 0.43,
                "train/actual_truncated_completions_ratio": 0.08,
                "train/truncated_completions_masked_out_of_loss_ratio": 0.08,
                "train/truncation_masking_active": 1.0,
                "train/positive_advantages_ratio": 0.58,
                "train/oom_backoff_count": 0,
                "train/optimizer_step": 60,
            },
            {
                "event": "benchmark_metrics",
                "step": 240,
                "benchmark_phase": "final_checkpoint",
                "val/acc": 0.50,
                "val/format_compliance": 0.96,
                "val/avg_len": 650.0,
            },
        ],
        "length_penalty_0005_mask_truncated_off_seed1": [
            {
                "event": "train_metrics",
                "step": 100,
                "train/avg_reward": 0.28,
                "train/reward_std": 0.35,
                "train/reward_min": -0.20,
                "train/reward_max": 0.80,
                "train/avg_response_length": 860.0,
                "train/entropy_masked_ratio": 0.46,
                "train/actual_truncated_completions_ratio": 0.22,
                "train/truncated_completions_masked_out_of_loss_ratio": 0.00,
                "train/truncation_masking_active": 0.0,
                "train/positive_advantages_ratio": 0.52,
                "train/oom_backoff_count": 1,
                "train/optimizer_step": 25,
            },
            {
                "event": "benchmark_metrics",
                "step": 100,
                "benchmark_phase": "periodic",
                "val/acc": 0.40,
                "val/format_compliance": 0.90,
                "val/avg_len": 820.0,
            },
            {
                "event": "train_metrics",
                "step": 200,
                "train/avg_reward": 0.32,
                "train/reward_std": 0.28,
                "train/reward_min": -0.10,
                "train/reward_max": 0.78,
                "train/avg_response_length": 1018.0,
                "train/entropy_masked_ratio": 0.47,
                "train/actual_truncated_completions_ratio": 0.58,
                "train/truncated_completions_masked_out_of_loss_ratio": 0.00,
                "train/truncation_masking_active": 0.0,
                "train/positive_advantages_ratio": 0.54,
                "train/oom_backoff_count": 1,
                "train/optimizer_step": 50,
            },
            {
                "event": "benchmark_metrics",
                "step": 200,
                "benchmark_phase": "periodic",
                "val/acc": 0.46,
                "val/format_compliance": 0.91,
                "val/avg_len": 980.0,
            },
            {
                "event": "train_metrics",
                "step": 240,
                "train/avg_reward": 0.18,
                "train/reward_std": 0.40,
                "train/reward_min": -0.30,
                "train/reward_max": 0.70,
                "train/avg_response_length": 1020.0,
                "train/entropy_masked_ratio": 0.48,
                "train/actual_truncated_completions_ratio": 0.64,
                "train/truncated_completions_masked_out_of_loss_ratio": 0.00,
                "train/truncation_masking_active": 0.0,
                "train/positive_advantages_ratio": 0.49,
                "train/oom_backoff_count": 2,
                "train/optimizer_step": 60,
            },
            {
                "event": "benchmark_metrics",
                "step": 240,
                "benchmark_phase": "final_checkpoint",
                "val/acc": 0.38,
                "val/format_compliance": 0.88,
                "val/avg_len": 1005.0,
            },
        ],
    }

    log_lines_by_run = {
        "length_penalty_0_seed1": [
            "step=100 30.0s/it tokens_per_sec=48.0 loss=0.300 reward=0.400 VRAM: 5.5 GB\n",
            "step=200 31.0s/it tokens_per_sec=47.0 loss=0.250 reward=0.520 VRAM: 5.6 GB\n",
            "step=240 29.0s/it tokens_per_sec=49.0 loss=0.240 reward=0.490 VRAM: 5.6 GB\n",
        ],
        "length_penalty_0005_mask_truncated_off_seed1": [
            "step=100 31.0s/it tokens_per_sec=45.0 loss=0.310 reward=0.280 VRAM: 5.6 GB\n",
            "step=200 33.0s/it tokens_per_sec=43.0 loss=0.290 reward=0.320 VRAM: 5.8 GB\n",
            "step=240 35.0s/it tokens_per_sec=41.0 loss=0.340 reward=0.180 VRAM: 5.9 GB\n",
        ],
    }

    class DummyProcess:
        returncode = 0

        def wait(self):
            return 0

    def fake_run_with_conda(cmd, **kwargs):
        assert "--no-sent" in cmd
        run_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_name = run_dir.name.removeprefix("cdia_")
        metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text(
            "\n".join(json.dumps(entry) for entry in metrics_by_run[run_name]),
            encoding="utf-8",
        )
        stdout = kwargs["stdout"]
        for line in log_lines_by_run[run_name]:
            stdout.write(line)
        stdout.flush()
        if run_name == "length_penalty_0005_mask_truncated_off_seed1":
            assert "--no-mask-truncated" in cmd
        else:
            assert "--no-mask-truncated" not in cmd
        return DummyProcess()

    monkeypatch.setattr(module, "run_with_conda", fake_run_with_conda)

    zero_result = suite.run_benchmark(length_zero)
    penalized_result = suite.run_benchmark(penalized)

    assert zero_result.comparison_group == "length_penalty_0"
    assert zero_result.seed == 1
    assert zero_result.in_training_heldout_benchmark_accuracy_values == [0.42, 0.48]
    assert zero_result.in_training_heldout_benchmark_accuracy_steps == [100, 200]
    assert zero_result.in_training_heldout_benchmark_accuracy_final == pytest.approx(0.48)
    assert zero_result.final_checkpoint_heldout_benchmark_accuracy == pytest.approx(0.50)

    assert penalized_result.comparison_group == "length_penalty_0005_mask_truncated_off"
    assert penalized_result.seed == 1
    assert penalized_result.in_training_heldout_benchmark_accuracy_peak == pytest.approx(0.46)
    assert penalized_result.in_training_heldout_benchmark_accuracy_peak_step == 200
    assert penalized_result.final_checkpoint_heldout_benchmark_accuracy == pytest.approx(0.38)
    assert penalized_result.avg_response_length_final == pytest.approx(1020.0)
    assert penalized_result.response_budget_usage_final == pytest.approx(1020.0 / 1024.0)

    suite.results = [zero_result, penalized_result]
    suite._save_results()
    suite._generate_report()

    results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert results_payload["trainer_steps"] == 240
    assert results_payload["total_runs"] == 2
    assert results_payload["results"][0]["comparison_group"] == "length_penalty_0"

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Selection Guidance" in report
    assert "Raw reward is not directly comparable" in report
    assert "explicitly treats drift toward the 1024 token cap as a failure signal" in report
    assert "In-training held-out benchmark accuracy values: 0.4200@100, 0.4800@200" in report
    assert "Final checkpoint held-out benchmark accuracy: 0.5000" in report
    assert "Final checkpoint held-out benchmark accuracy: 0.3800" in report
    assert "Recommended single winner for the later heavy run: `length_penalty_0`." in report
