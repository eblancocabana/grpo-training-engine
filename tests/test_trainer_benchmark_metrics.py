import json
from pathlib import Path

import pytest

from src.grpo.trainer import GRPOTrainerLoop


def test_record_benchmark_metrics_writes_jsonl(tmp_path: Path):
    loop = GRPOTrainerLoop.__new__(GRPOTrainerLoop)
    loop._metrics_jsonl_path = str(tmp_path / "metrics.jsonl")
    loop.global_step = 240
    loop.current_epoch = 3

    GRPOTrainerLoop._record_benchmark_metrics(
        loop,
        {
            "val/acc": 0.54,
            "val/format_compliance": 0.92,
            "val/avg_len": 612.0,
        },
        phase="final_checkpoint",
    )

    entry = json.loads(Path(loop._metrics_jsonl_path).read_text(encoding="utf-8"))
    assert entry["event"] == "benchmark_metrics"
    assert entry["benchmark_phase"] == "final_checkpoint"
    assert entry["step"] == 240
    assert entry["epoch"] == 3
    assert entry["val/acc"] == pytest.approx(0.54)
    assert entry["val/format_compliance"] == pytest.approx(0.92)
    assert entry["val/avg_len"] == pytest.approx(612.0)
