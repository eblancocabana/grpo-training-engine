from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer.records import TestRunRecord, utc_timestamp, write_json_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the optimizer acceptance hard-gate test suite"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("optimizer/artifacts/records"),
    )
    parser.add_argument(
        "--command",
        type=str,
        default=_default_test_command(),
    )
    args = parser.parse_args(argv)

    artifacts_dir = args.artifacts_dir.resolve()
    timestamp = utc_timestamp()
    log_path = artifacts_dir / "tests" / f"{timestamp}_pytest.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        ["zsh", "-lc", args.command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd(),
    )
    log_path.write_text(completed.stdout, encoding="utf-8")

    counts = _parse_pytest_counts(completed.stdout)
    record = TestRunRecord(
        schema_version=1,
        generated_at=timestamp,
        command=args.command,
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        failed_count=counts["failed"],
        passed_count=counts["passed"],
        skipped_count=counts["skipped"],
        xfailed_count=counts["xfailed"],
        log_path=str(log_path),
    )
    record_path = write_json_record(
        artifacts_dir / "tests" / f"{timestamp}_pytest.json",
        record,
    )
    print(f"test_record_path={record_path}")
    print(f"passed={record.passed}")
    print(f"exit_code={record.exit_code}")
    return 0 if record.passed else 1


def _parse_pytest_counts(output: str) -> dict[str, int]:
    counts = {
        "failed": 0,
        "passed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    summary_line = ""
    for line in output.splitlines():
        if " in " in line and any(
            token in line for token in (" passed", " failed", " skipped", " xfailed")
        ):
            summary_line = line.strip()
    if not summary_line:
        return counts

    for key in counts:
        match = re.search(rf"(\d+)\s+{re.escape(key)}", summary_line)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _default_test_command() -> str:
    base_pytest = "pytest -q --ignore=optimizer/backups -m 'not performance'"
    if os.environ.get("CONDA_DEFAULT_ENV") == "grpo-3060ti":
        return base_pytest
    return (
        "source ~/.zshrc >/dev/null 2>&1; "
        "conda run --no-capture-output -n grpo-3060ti "
        f"{base_pytest}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
