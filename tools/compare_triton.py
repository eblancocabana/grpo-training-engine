#!/usr/bin/env python3
"""Compare Triton vs PyTorch kernels in training.

Runs two 5-minute training windows (Triton on/off), samples GPU memory
usage with nvidia-smi, and estimates throughput from log output.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STEP_PATTERN = re.compile(r"step=([0-9]+)")
ALT_STEP_PATTERN = re.compile(r"\bStep\s+([0-9]+)\b", re.IGNORECASE)
IT_S_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)it/s")
S_PER_IT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)s/it")
PBar_IT_S_PATTERN = re.compile(r"\[.*?(?:\s|^)([0-9]+(?:\.[0-9]+)?)it/s\]")
VRAM_PROGRESS_PATTERN = re.compile(r"vram=([0-9]+(?:\.[0-9]+)?)GB")
MAX_ALLOC_PATTERN = re.compile(r"Max Allocated:\s*([0-9]+(?:\.[0-9]+)?)\s*GB")
RESERVED_PATTERN = re.compile(r"Reserved:\s*([0-9]+(?:\.[0-9]+)?)\s*GB")


@dataclass
class RunStats:
    label: str
    vram_samples_gb: list[float]
    it_s_samples: list[float]
    vram_progress_samples_gb: list[float]
    max_allocated_gb: float | None
    max_reserved_gb: float | None
    first_step: int | None
    last_step: int | None
    first_step_time: float | None
    last_step_time: float | None
    log_path: Path

    def vram_avg(self) -> float | None:
        if not self.vram_samples_gb:
            return None
        return sum(self.vram_samples_gb) / len(self.vram_samples_gb)

    def vram_peak(self) -> float | None:
        if not self.vram_samples_gb:
            return None
        return max(self.vram_samples_gb)

    def vram_progress_peak(self) -> float | None:
        if not self.vram_progress_samples_gb:
            return None
        return max(self.vram_progress_samples_gb)

    def it_s_avg(self) -> float | None:
        if self.it_s_samples:
            return sum(self.it_s_samples) / len(self.it_s_samples)
        if (
            self.first_step is None
            or self.last_step is None
            or self.first_step_time is None
            or self.last_step_time is None
            or self.last_step_time <= self.first_step_time
        ):
            return None
        step_delta = self.last_step - self.first_step
        if step_delta <= 0:
            return None
        return step_delta / (self.last_step_time - self.first_step_time)


def _sample_vram_gb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if not output:
        return None
    try:
        mb = float(output.splitlines()[0].strip())
    except ValueError:
        return None
    return mb / 1024.0


def _reader_thread(
    process: subprocess.Popen[str],
    stats: RunStats,
    log_file,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.05)
            continue
        log_file.write(line)
        log_file.flush()

        step_matches = STEP_PATTERN.findall(line) + ALT_STEP_PATTERN.findall(line)
        if step_matches:
            step = int(step_matches[-1])
            now = time.monotonic()
            if stats.first_step is None:
                stats.first_step = step
                stats.first_step_time = now
            if stats.last_step is None or step > stats.last_step:
                stats.last_step = step
                stats.last_step_time = now

        for match in IT_S_PATTERN.finditer(line):
            stats.it_s_samples.append(float(match.group(1)))
        for match in S_PER_IT_PATTERN.finditer(line):
            s_per_it = float(match.group(1))
            if s_per_it > 0:
                stats.it_s_samples.append(1.0 / s_per_it)
        pbar_match = IT_S_PATTERN.search(line)
        if pbar_match:
            value = pbar_match.group(1)
            if value != "?":
                stats.it_s_samples.append(float(value))
        for match in VRAM_PROGRESS_PATTERN.finditer(line):
            stats.vram_progress_samples_gb.append(float(match.group(1)))
        max_alloc_match = MAX_ALLOC_PATTERN.search(line)
        if max_alloc_match:
            max_alloc = float(max_alloc_match.group(1))
            if stats.max_allocated_gb is None or max_alloc > stats.max_allocated_gb:
                stats.max_allocated_gb = max_alloc
        reserved_match = RESERVED_PATTERN.search(line)
        if reserved_match:
            reserved = float(reserved_match.group(1))
            if stats.max_reserved_gb is None or reserved > stats.max_reserved_gb:
                stats.max_reserved_gb = reserved


def _run_training(
    label: str,
    command: list[str],
    duration_s: int,
    log_path: Path,
    sample_interval_s: float,
) -> RunStats:
    stats = RunStats(
        label=label,
        vram_samples_gb=[],
        it_s_samples=[],
        vram_progress_samples_gb=[],
        max_allocated_gb=None,
        max_reserved_gb=None,
        first_step=None,
        last_step=None,
        first_step_time=None,
        last_step_time=None,
        log_path=log_path,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stop_event = threading.Event()
        reader = threading.Thread(
            target=_reader_thread,
            args=(process, stats, log_file, stop_event),
            daemon=True,
        )
        reader.start()

        start = time.monotonic()
        while time.monotonic() - start < duration_s:
            vram_gb = _sample_vram_gb()
            if vram_gb is not None:
                stats.vram_samples_gb.append(vram_gb)
            time.sleep(sample_interval_s)

        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        stop_event.set()
        reader.join(timeout=5)

    return stats


def _print_summary(stats: RunStats) -> None:
    print(f"\n[{stats.label}] Summary")
    print(f"  Log: {stats.log_path}")
    avg_vram = stats.vram_avg()
    peak_vram = stats.vram_peak()
    it_s = stats.it_s_avg()
    progress_peak = stats.vram_progress_peak()
    if avg_vram is not None:
        print(f"  VRAM avg:  {avg_vram:.2f} GB")
        print(f"  VRAM peak: {peak_vram:.2f} GB")
    else:
        print("  VRAM avg/peak: unavailable")
    if progress_peak is not None:
        print(f"  VRAM peak (log): {progress_peak:.2f} GB")
    if stats.max_allocated_gb is not None:
        print(f"  Max allocated (log): {stats.max_allocated_gb:.2f} GB")
    if stats.max_reserved_gb is not None:
        print(f"  Max reserved (log): {stats.max_reserved_gb:.2f} GB")
    if it_s is not None:
        print(f"  it/s avg:  {it_s:.4f}")
    else:
        print("  it/s avg: unavailable")


def _load_summary(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Triton on/off metrics")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--triton-lora-prefer-base", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument(
        "--mode",
        choices=("on", "off", "both"),
        default="both",
        help="Run Triton on, off, or both (default: both)",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("logs/compare_triton_summary.json"),
        help="Path to write summary JSON",
    )
    parser.add_argument("--verbose", action="count", default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_cmd = [
        "python",
        "-u",
        "train.py",
        "--no-wandb",
        "--group-size",
        str(args.group_size),
        "--epochs",
        str(args.epochs),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-response-length",
        str(args.max_response_length),
    ]
    if args.gradient_accumulation_steps is not None:
        base_cmd += [
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
        ]
    if args.max_steps is not None:
        base_cmd += [
            "--max-steps",
            str(args.max_steps),
        ]
    if args.triton_lora_prefer_base:
        base_cmd += [
            "--triton-lora-prefer-base",
        ]
    if args.verbose:
        base_cmd.append("-" + "v" * min(args.verbose, 3))

    triton_on_cmd = base_cmd[:]
    triton_off_cmd = base_cmd[:] + ["--no-triton"]

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    triton_on_log = logs_dir / "compare_triton_on.log"
    triton_off_log = logs_dir / "compare_triton_off.log"

    summary = _load_summary(args.summary_path)
    if args.mode in {"on", "both"}:
        print("Running Triton ON...")
        triton_on = _run_training(
            "Triton ON",
            triton_on_cmd,
            args.duration,
            triton_on_log,
            args.sample_interval,
        )
        summary["triton_on"] = {
            "vram_avg_gb": triton_on.vram_avg(),
            "vram_peak_gb": triton_on.vram_peak(),
            "vram_peak_log_gb": triton_on.vram_progress_peak(),
            "max_allocated_log_gb": triton_on.max_allocated_gb,
            "max_reserved_log_gb": triton_on.max_reserved_gb,
            "it_s_avg": triton_on.it_s_avg(),
            "log": str(triton_on.log_path),
        }
        _print_summary(triton_on)
    if args.mode in {"off", "both"}:
        print("Running Triton OFF...")
        triton_off = _run_training(
            "Triton OFF",
            triton_off_cmd,
            args.duration,
            triton_off_log,
            args.sample_interval,
        )
        summary["triton_off"] = {
            "vram_avg_gb": triton_off.vram_avg(),
            "vram_peak_gb": triton_off.vram_peak(),
            "vram_peak_log_gb": triton_off.vram_progress_peak(),
            "max_allocated_log_gb": triton_off.max_allocated_gb,
            "max_reserved_log_gb": triton_off.max_reserved_gb,
            "it_s_avg": triton_off.it_s_avg(),
            "log": str(triton_off.log_path),
        }
        _print_summary(triton_off)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nSummary JSON written to {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
