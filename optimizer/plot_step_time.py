from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer.records import ExperimentLedgerRecord


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot step time (s/it) history for optimizer attempts"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("optimizer/artifacts/records/experiments/experiment_ledger.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimizer/artifacts/reports"),
    )
    args = parser.parse_args(argv)

    rows = load_ledger(args.ledger)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "step_time.svg"
    md_path = args.output_dir / "step_time.md"

    write_svg_plot(rows, svg_path)
    write_report(rows, md_path, svg_path)

    print(f"wrote_plot={svg_path}")
    print(f"wrote_report={md_path}")
    return 0


def load_ledger(path: Path) -> list[ExperimentLedgerRecord]:
    if not path.exists():
        return []
    rows: list[ExperimentLedgerRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(ExperimentLedgerRecord.from_dict(payload))
    return rows


def write_svg_plot(rows: list[ExperimentLedgerRecord], svg_path: Path) -> None:
    width = 800
    height = 420
    margin = 40
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    attempt_numbers = [row.attempt_number or row.experiment_number for row in rows]
    # Use step time (time_avg_s) - lower is better
    frontier_values = [
        row.incumbent_step_time_after_decision or 0.0 for row in rows
    ]
    candidate_values = [row.candidate_snapshot.time_avg_s or 0.0 for row in rows]
    all_values = frontier_values + candidate_values
    max_value = max(all_values) if all_values else 1.0
    min_value = min(all_values) if all_values else 0.0
    value_range = max(max_value - min_value, 1.0)

    def x_coord(index: int) -> float:
        if len(attempt_numbers) <= 1:
            return margin + plot_width / 2
        return margin + (plot_width * index / (len(attempt_numbers) - 1))

    def y_coord(value: float) -> float:
        return (
            margin + plot_height - (((value - min_value) / value_range) * plot_height)
        )

    def polyline(values: list[float], color: str) -> str:
        if not values:
            return ""
        points = " ".join(
            f"{x_coord(index):.2f},{y_coord(value):.2f}"
            for index, value in enumerate(values)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />'

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
<rect width=\"100%\" height=\"100%\" fill=\"white\" />
<text x=\"{margin}\" y=\"24\" font-family=\"sans-serif\" font-size=\"18\">Step time (s/it) over attempts - lower is better</text>
<line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#444\" />
<line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{height - margin}\" stroke=\"#444\" />
{polyline(candidate_values, "#1f77b4")}
{polyline(frontier_values, "#d62728")}
<text x=\"{width - 220}\" y=\"{margin}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#1f77b4\">Candidate step time</text>
<text x=\"{width - 220}\" y=\"{margin + 18}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#d62728\">Frontier after decision</text>
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def write_report(
    rows: list[ExperimentLedgerRecord], md_path: Path, plot_path: Path
) -> None:
    latest = rows[-1] if rows else None
    total_attempts = len(rows)
    kept = sum(1 for row in rows if row.outcome == "kept")
    discarded = sum(1 for row in rows if row.outcome == "discarded")
    latest_frontier = latest.incumbent_step_time_after_decision if latest else None
    md_path.write_text(
        "# Step Time (s/it) History\n\n"
        f"- Total attempts: {total_attempts}\n"
        f"- Kept: {kept}\n"
        f"- Discarded: {discarded}\n"
        f"- Latest frontier step time (s/it): {latest_frontier}\n"
        f"- Plot: `{plot_path}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
