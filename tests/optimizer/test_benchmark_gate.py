import json
from pathlib import Path
from typing import cast

from optimizer.benchmark_log_parser import parse_benchmark_log
from optimizer.evaluation.benchmark_gate import classify_run
from optimizer.records import BenchmarkRunRecord, JsonValue


def _run(**overrides: JsonValue) -> BenchmarkRunRecord:
    payload: dict[str, JsonValue] = {
        "input": "candidate",
        "label": "candidate",
        "commit": "abc123",
        "status": "ok",
        "valid": True,
        "steps_requested": 10,
        "steps_observed": 10,
        "tokens_per_sec": 10.0,
        "reward_avg": 0.4,
        "loss_avg": 0.9,
        "time_avg_s": 1.0,
        "effective_batch": 16,
        "oom_events": 0,
        "triton_mode": "on",
        "triton_arg": "--use-triton",
    }
    payload.update(overrides)
    return BenchmarkRunRecord.from_dict(payload)


def test_classifies_ok_run_as_clean_comparable() -> None:
    classification = classify_run(_run(status="ok", valid=True))

    assert classification.comparability == "clean_comparable"
    assert classification.promotable is True
    assert classification.complete is True


def test_classifies_completed_oom_recovery_as_recovered_comparable() -> None:
    classification = classify_run(
        _run(status="oom_recovered", valid=True, oom_events=2)
    )

    assert classification.comparability == "recovered_comparable"
    assert classification.promotable is True
    assert "oom_backoff_recovered" in classification.reasons


def test_rejects_incomplete_or_missing_metrics() -> None:
    incomplete = classify_run(_run(steps_observed=3))
    missing_batch = classify_run(_run(effective_batch=None))
    invalid = classify_run(_run(valid=False))
    incomplete_recovered = classify_run(
        _run(status="oom_recovered", valid=True, oom_events=1, steps_observed=3)
    )

    assert incomplete.comparability == "not_comparable"
    assert "benchmark_incomplete" in incomplete.reasons
    assert missing_batch.comparability == "not_comparable"
    assert "effective_batch_missing" in missing_batch.reasons
    assert invalid.comparability == "not_comparable"
    assert "benchmark_invalid" in invalid.reasons
    assert incomplete_recovered.comparability == "not_comparable"
    assert "oom_backoff_recovered" in incomplete_recovered.reasons
    assert "benchmark_incomplete" in incomplete_recovered.reasons


def test_rejects_runs_without_forced_triton_or_step_time() -> None:
    missing_triton = classify_run(_run(triton_mode="off", triton_arg="--no-triton"))
    missing_flag = classify_run(_run(triton_mode="on", triton_arg=None))
    missing_step_time = classify_run(_run(time_avg_s=None))

    assert missing_triton.comparability == "not_comparable"
    assert "triton_not_forced_on" in missing_triton.reasons
    assert missing_flag.comparability == "not_comparable"
    assert "triton_flag_missing" in missing_flag.reasons
    assert missing_step_time.comparability == "not_comparable"
    assert "step_time_missing" in missing_step_time.reasons


def test_compare_bench_script_marks_complete_recovered_oom_runs_valid() -> None:
    script = Path(__file__).resolve().parents[2] / "tools" / "compare_bench.sh"
    content = script.read_text(encoding="utf-8")

    assert 'return "oom_recovered", True, None, parsed["terminal_error"]' in content
    assert '--log-metrics-jsonl' in content
    assert 'metrics_path' in content


def test_parse_benchmark_log_handles_tqdm_carriage_return_updates(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.log"
    _ = log_path.write_text(
        (
            "Effective batch size: 16\n"
            "Epoch 1:  20%|██        | 1/5 [00:01<00:04, 1.00s/it, loss=0.9000, reward=0.400, tokens_per_sec=80.000, vram=6.1GB, step=1]\r"
            "Epoch 1:  40%|████      | 2/5 [00:02<00:03, 1.00s/it, loss=0.8500, reward=0.450, tokens_per_sec=100.000, vram=6.2GB, step=2]\r"
            "Epoch 1:  60%|██████    | 3/5 [00:03<00:02, 1.00s/it, loss=0.8000, reward=0.500, tokens_per_sec=120.000, vram=6.3GB, step=3]\n"
        ),
        encoding="utf-8",
    )

    parsed = parse_benchmark_log(log_path)

    assert parsed["last_step"] == 3
    assert parsed["effective_batch"] == 16
    step_tokens_obj = cast(dict[object, object], parsed["step_tokens_per_sec"])
    step_tokens: dict[int, float] = {}
    for raw_step, raw_value in step_tokens_obj.items():
        assert isinstance(raw_step, int)
        assert isinstance(raw_value, float)
        step_tokens[raw_step] = raw_value
    assert step_tokens[1] == 80.0
    assert step_tokens[2] == 100.0
    assert step_tokens[3] == 120.0
    samples_after_warmup: list[float] = [
        value for step, value in step_tokens.items() if step > 1
    ]
    assert sum(samples_after_warmup) / len(samples_after_warmup) == 110.0


def test_parse_benchmark_log_extracts_failed_response_samples(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    _ = log_path.write_text(
        (
            "[DEBUG][grpo.grpo.trainer] [FAILED] Response 1 (Reward: 0.00):\n"
            "[DEBUG][grpo.grpo.trainer]   -> Text: <think>2+2=5</think>\n"
            "[DEBUG][grpo.grpo.trainer]   -> Extracted: 5\n"
            "[DEBUG][grpo.grpo.trainer]   -> GT: 4\n"
            "[DEBUG][grpo.grpo.trainer]   -> Match: False\n"
            "[DEBUG][grpo.grpo.trainer] ----------\n"
        ),
        encoding="utf-8",
    )

    parsed = parse_benchmark_log(log_path)

    assert parsed["failed_response_count"] == 1
    examples = cast(list[object], parsed["failed_response_examples"])
    assert len(examples) == 1
    assert "text=<think>2+2=5</think>" in str(examples[0])


def test_parse_benchmark_log_prefers_structured_metrics_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    metrics_path = tmp_path / "metrics.jsonl"
    _ = log_path.write_text(
        "Effective batch size: 16\n"
        "Epoch 1:  20%|██        | 1/5 [00:01<00:04, 1.00s/it, loss=9.9000, reward=9.400, tokens_per_sec=8.000, vram=6.1GB, step=1]\n",
        encoding="utf-8",
    )
    entries = [
        {"event": "run_info", "effective_batch": 16, "step": 0, "epoch": 0},
        {
            "event": "train_metrics",
            "step": 1,
            "epoch": 0,
            "train/loss": 0.9,
            "train/avg_reward": 0.4,
            "train/tokens_per_sec": 80.0,
            "perf/step_time_s": 1.0,
            "memory/vram_used_gb": 6.1,
        },
        {
            "event": "train_metrics",
            "step": 2,
            "epoch": 0,
            "train/loss": 0.8,
            "train/avg_reward": 0.5,
            "train/tokens_per_sec": 100.0,
            "perf/step_time_s": 1.2,
            "memory/vram_used_gb": 6.2,
        },
    ]
    _ = metrics_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    parsed = parse_benchmark_log(log_path, metrics_path=metrics_path)

    assert parsed["last_step"] == 2
    assert parsed["effective_batch"] == 16
    step_tokens = cast(dict[int, float], parsed["step_tokens_per_sec"])
    step_loss = cast(dict[int, float], parsed["step_loss"])
    step_reward = cast(dict[int, float], parsed["step_reward"])
    step_times = cast(dict[int, float], parsed["step_times"])
    assert step_tokens[1] == 80.0
    assert step_tokens[2] == 100.0
    assert step_loss[1] == 0.9
    assert step_reward[2] == 0.5
    assert step_times[2] == 1.2
