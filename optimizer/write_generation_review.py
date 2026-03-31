from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer.records import GenerationReviewRecord, utc_timestamp, write_json_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an explicit failed-generation review artifact"
    )
    parser.add_argument("--candidate-target", required=True)
    parser.add_argument("--verdict", choices=["gibberish", "not_gibberish"], required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--reviewer", default="agent")
    parser.add_argument("--benchmark-log-path")
    parser.add_argument("--example", action="append", dest="examples_reviewed", default=[])
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("optimizer/artifacts/records"),
    )
    args = parser.parse_args(argv)

    timestamp = utc_timestamp()
    record = GenerationReviewRecord(
        schema_version=1,
        review_id=timestamp,
        generated_at=timestamp,
        reviewer=args.reviewer,
        candidate_target=args.candidate_target,
        benchmark_log_path=args.benchmark_log_path,
        verdict=args.verdict,
        reason=args.reason,
        examples_reviewed=list(args.examples_reviewed),
    )
    path = write_json_record(
        args.artifacts_dir.resolve() / "reviews" / f"{timestamp}_generation_review.json",
        record,
    )
    print(f"generation_review_path={path}")
    print(f"verdict={record.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
