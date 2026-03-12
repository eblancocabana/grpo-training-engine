from __future__ import annotations

import re
from pathlib import Path


step_pattern = re.compile(r"\bstep[:=]\s*([0-9]+)\b", re.IGNORECASE)
alt_step_pattern = re.compile(r"\bStep\s+([0-9]+)\b", re.IGNORECASE)
number_pattern = r"-?[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?"
loss_pattern = re.compile(rf"\bloss=({number_pattern})", re.IGNORECASE)
reward_pattern = re.compile(rf"\breward=({number_pattern})", re.IGNORECASE)
tokens_per_sec_pattern = re.compile(
    rf"\btokens_per_sec=({number_pattern})", re.IGNORECASE
)
it_s_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)it/s")
s_it_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)s/it")
vram_postfix_pattern = re.compile(r"\bvram=([0-9]+(?:\.[0-9]+)?)GB")
vram_log_pattern = re.compile(r"VRAM:\s*([0-9]+(?:\.[0-9]+)?)GB")
effective_batch_pattern = re.compile(r"Effective batch size:\s*([0-9]+)")
oom_pattern = re.compile(r"\[OOM\]|out of memory", re.IGNORECASE)
traceback_pattern = re.compile(r"Traceback \(most recent call last\):")
exception_pattern = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(?P<message>.+)$"
)


def parse_benchmark_log(path: Path) -> dict[str, object]:
    step_times: dict[int, float] = {}
    step_loss: dict[int, float] = {}
    step_reward: dict[int, float] = {}
    step_tokens_per_sec: dict[int, float] = {}
    vram_samples: list[float] = []
    effective_batch: int | None = None
    oom_events = 0
    last_step: int | None = None
    saw_traceback = False
    terminal_error: dict[str, str] | None = None

    content = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    for line in content.splitlines():
        if oom_pattern.search(line):
            oom_events += 1
        if traceback_pattern.search(line):
            saw_traceback = True

        exception_match = exception_pattern.search(line.strip())
        if exception_match:
            terminal_error = {
                "type": exception_match.group("name"),
                "message": exception_match.group("message"),
            }

        if effective_batch is None:
            match = effective_batch_pattern.search(line)
            if match:
                effective_batch = int(match.group(1))

        step_match = step_pattern.search(line) or alt_step_pattern.search(line)
        step = int(step_match.group(1)) if step_match else None
        if step is not None:
            last_step = step

        it_s_match = it_s_pattern.search(line)
        s_it_match = s_it_pattern.search(line)
        step_time_s = None
        if s_it_match:
            step_time_s = float(s_it_match.group(1))
        elif it_s_match:
            it_s = float(it_s_match.group(1))
            if it_s > 0:
                step_time_s = 1.0 / it_s
        if step is not None and step_time_s is not None:
            step_times[step] = step_time_s

        loss_match = loss_pattern.search(line)
        if step is not None and loss_match:
            step_loss[step] = float(loss_match.group(1))

        reward_match = reward_pattern.search(line)
        if step is not None and reward_match:
            step_reward[step] = float(reward_match.group(1))

        tokens_per_sec_match = tokens_per_sec_pattern.search(line)
        if step is not None and tokens_per_sec_match:
            step_tokens_per_sec[step] = float(tokens_per_sec_match.group(1))

        vram_match = vram_postfix_pattern.search(line)
        if vram_match:
            vram_samples.append(float(vram_match.group(1)))

        vram_log_match = vram_log_pattern.search(line)
        if vram_log_match:
            vram_samples.append(float(vram_log_match.group(1)))

    return {
        "step_times": step_times,
        "step_loss": step_loss,
        "step_reward": step_reward,
        "step_tokens_per_sec": step_tokens_per_sec,
        "vram_samples": vram_samples,
        "effective_batch": effective_batch,
        "oom_events": oom_events,
        "last_step": last_step,
        "saw_traceback": saw_traceback,
        "terminal_error": terminal_error,
    }
