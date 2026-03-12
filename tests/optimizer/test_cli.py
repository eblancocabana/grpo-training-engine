from __future__ import annotations

import json
from pathlib import Path

from optimizer.cli import main


def test_cli_dry_run_prints_sequential_summary(capsys, tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "baseline": "main",
                "candidates": ["feat/example"],
                "iteration_budget": "2",
                "open_ended": "false",
                "trace_summary_path": "summary.json",
                "benchmark": {"steps": 5, "triton": "auto"},
                "policy": {"allow_recovered_oom_promotion": True},
                "output": {"artifacts_dir": "optimizer/artifacts/records"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--template", str(template), "--dry-run"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "mode=dry-run" in captured
    assert "sequential=true" in captured
    assert "decisive_tool=compare_bench.sh" in captured
    assert "diagnostic_tools=compare_triton.py,profiler" in captured
    assert "handoff_mode=external_ai_agent" in captured
    assert f"trace_summary_path={(tmp_path / 'summary.json').resolve()}" in captured
    assert "allow_recovered_oom_promotion=True" in captured
    assert "iteration_budget=2" in captured
    assert "open_ended=False" in captured
    assert "iteration_targets=['feat/example']" in captured
