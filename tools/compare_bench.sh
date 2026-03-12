#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'USAGE'
Usage: tools/compare_bench.sh [--steps N] <rev_or_path> [<rev_or_path> ...]

Arguments can be:
  - Git refs (branches like main, feat/xyz)
  - Commit hashes (abc123)
  - Worktree paths (/path/to/worktree)

Options:
  --steps N   Number of training steps (default: 5)
  --triton MODE  Triton mode: auto, on, off (default: auto)
  -h, --help  Show this help
USAGE
}

steps=5
triton_mode="auto"
targets=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      steps="$2"
      shift 2
      ;;
    --triton)
      triton_mode="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      targets+=("$1")
      shift
      ;;
  esac
done

if [[ ${#targets[@]} -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$triton_mode" != "auto" && "$triton_mode" != "on" && "$triton_mode" != "off" ]]; then
  echo "[ERROR] --triton must be one of: auto, on, off" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"

if [[ -f "$repo_root/.opencode/env" ]]; then
  set +u
  source "$repo_root/.opencode/env"
  set -u
fi

run_python() {
  "${python_runner[@]}" "$@"
}

output_root="$repo_root/benchmarks/output"
logs_dir="$output_root/logs"
worktrees_dir="$output_root/worktrees"
run_meta_path="$output_root/run_metadata.jsonl"

mkdir -p "$logs_dir" "$worktrees_dir"
rm -f "$run_meta_path"

timestamp="$(date +%Y%m%d-%H%M%S)"
gpu_id="${GPU_ID:-0}"
vram_sample_ms="${VRAM_SAMPLE_MS:-1}"
conda_env_name="${BENCHMARK_CONDA_ENV:-grpo-3060ti}"
python_runner=()
sample_interval_s=""

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_runner=("$PYTHON_BIN")
elif command -v conda >/dev/null 2>&1; then
  if ! conda env list | grep -qE "^[[:space:]]*${conda_env_name}[[:space:]]"; then
    echo "[ERROR] Conda env ${conda_env_name} not found." >&2
    exit 1
  fi
  python_runner=(conda run --no-capture-output -n "$conda_env_name" python)
else
  conda_python="${HOME}/.conda/envs/${conda_env_name}/bin/python"
  if [[ -x "$conda_python" ]]; then
    python_runner=("$conda_python")
  else
    echo "[ERROR] Could not resolve python for conda env ${conda_env_name}. Set PYTHON_BIN explicitly or ensure conda is available." >&2
    exit 1
  fi
fi

if [[ ! "$vram_sample_ms" =~ ^[0-9]+$ ]]; then
  echo "[WARN] VRAM_SAMPLE_MS is not numeric; defaulting to 1ms" >&2
  vram_sample_ms=1
fi
if (( vram_sample_ms < 1 )); then
  echo "[WARN] VRAM_SAMPLE_MS < 1ms; clamping to 1ms" >&2
  vram_sample_ms=1
elif (( vram_sample_ms < 10 )); then
  echo "[WARN] VRAM_SAMPLE_MS < 10ms may be unreliable with nvidia-smi" >&2
fi

sample_interval_s="$(python - <<PY
ms = int(${vram_sample_ms})
print(f"{ms / 1000:.3f}")
PY
)"

current_vram_sampler_pid=""
current_exit_code_path=""

cleanup_current_run() {
  if [[ -n "$current_vram_sampler_pid" ]]; then
    kill "$current_vram_sampler_pid" >/dev/null 2>&1 || true
    wait "$current_vram_sampler_pid" >/dev/null 2>&1 || true
    current_vram_sampler_pid=""
  fi
  if [[ -n "$current_exit_code_path" ]]; then
    rm -f "$current_exit_code_path"
    current_exit_code_path=""
  fi
}

trap cleanup_current_run EXIT INT TERM

sanitize_label() {
  local raw="$1"
  raw="${raw//\//-}"
  raw="${raw// /-}"
  raw="${raw//:/-}"
  raw="${raw//\\/-}"
  echo "$raw"
}

resolve_workdir() {
  local target="$1"
  local workdir=""
  local cleanup="false"
  local commit=""
  local label=""

  if [[ -d "$target" && -f "$target/train.py" ]]; then
    workdir="$(cd "$target" && pwd)"
    label="$(sanitize_label "$target")"
    if git -C "$workdir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      commit="$(git -C "$workdir" rev-parse HEAD)"
    else
      commit="unknown"
    fi
  else
    commit="$(git -C "$repo_root" rev-parse "$target^{commit}")"
    local short_commit
    short_commit="$(git -C "$repo_root" rev-parse --short "$commit")"
    label="$(sanitize_label "$target")-$short_commit"
    workdir="$worktrees_dir/$label"
    if [[ -d "$workdir" ]]; then
      git -C "$repo_root" worktree remove --force "$workdir" >/dev/null 2>&1 || true
      rm -rf "$workdir"
    fi
    git -C "$repo_root" worktree add --detach "$workdir" "$commit" >/dev/null
    cleanup="true"
  fi

  local source_cache_dir="$repo_root/data/cache"
  local dest_cache_dir="$workdir/data/cache"
  if [[ -d "$source_cache_dir" && "$source_cache_dir" != "$dest_cache_dir" ]]; then
    mkdir -p "$dest_cache_dir"
    cp -R "$source_cache_dir/." "$dest_cache_dir/"
  fi

  echo "$workdir|$cleanup|$commit|$label"
}

run_training() {
  local target="$1"
  local resolved
  resolved="$(resolve_workdir "$target")"
  local workdir cleanup commit label
  IFS='|' read -r workdir cleanup commit label <<<"$resolved"

  local no_checkpoint_arg=""
  local no_initial_benchmark_arg=""
  local triton_arg=""
  local help_output
  help_output="$(run_python "$workdir/train.py" --help 2>&1 || true)"
  if [[ "$help_output" == *"--no-checkpoints"* ]]; then
    no_checkpoint_arg="--no-checkpoints"
  fi
  if [[ "$help_output" == *"--no-initial-benchmark"* ]]; then
    no_initial_benchmark_arg="--no-initial-benchmark"
  fi
  if [[ "$help_output" == *"--use-triton"* && "$help_output" == *"--no-triton"* ]]; then
    case "$triton_mode" in
      on)
        triton_arg="--use-triton"
        ;;
      off)
        triton_arg="--no-triton"
        ;;
    esac
  fi

  local run_id="${label}-${timestamp}"
  local log_path="$logs_dir/${run_id}.log"
  local run_output_dir="$output_root/run_${run_id}"
  local vram_samples_path="$output_root/vram_${run_id}.txt"
  local exit_code_path="$output_root/exit_${run_id}.txt"

  local start_ts
  start_ts="$(date +%s)"

  rm -f "$vram_samples_path" "$exit_code_path"
  current_exit_code_path="$exit_code_path"

  # Create baseline benchmark marker to skip initial benchmark
  mkdir -p "$run_output_dir"
  echo '{"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "skipped": true}' > "$run_output_dir/baseline_benchmark_done.json"

  (
    cd "$workdir"
    WANDB_DISABLED=true \
      run_python "$workdir/train.py" \
      --no-wandb \
      $no_checkpoint_arg \
      $no_initial_benchmark_arg \
      $triton_arg \
      --epochs 1 \
      --max-steps "$steps" \
      --output-dir "$run_output_dir" \
      > "$log_path" 2>&1
    echo $? > "$exit_code_path"
  ) &
  local train_pid=$!

  local vram_sampler_pid=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while kill -0 "$train_pid" >/dev/null 2>&1; do
        nvidia-smi \
          --id="$gpu_id" \
          --query-gpu=timestamp,memory.used \
          --format=csv,noheader,nounits >> "$vram_samples_path" 2>/dev/null || true
        sleep "$sample_interval_s"
      done
    ) &
    vram_sampler_pid=$!
    current_vram_sampler_pid="$vram_sampler_pid"
  fi

  set +e
  wait "$train_pid"
  local train_status=$?
  set -e

  local exit_code
  if [[ -f "$exit_code_path" ]]; then
    exit_code="$(cat "$exit_code_path")"
  else
    exit_code="$train_status"
  fi

  if [[ -n "$vram_sampler_pid" ]]; then
    kill "$vram_sampler_pid" >/dev/null 2>&1 || true
    wait "$vram_sampler_pid" >/dev/null 2>&1 || true
    current_vram_sampler_pid=""
  fi

  local end_ts
  end_ts="$(date +%s)"

  rm -rf "$run_output_dir" "$exit_code_path"
  current_exit_code_path=""

  if [[ "$cleanup" == "true" ]]; then
    git -C "$repo_root" worktree remove --force "$workdir" >/dev/null
  fi

  cat <<EOF >>"$run_meta_path"
{"input":"$target","label":"$label","commit":"$commit","log_path":"$log_path","exit_code":$exit_code,"start_ts":$start_ts,"end_ts":$end_ts,"steps_requested":$steps,"timestamp":"$timestamp","vram_samples_path":"$vram_samples_path","run_origin":"isolated_worktree","triton_mode":"$triton_mode","triton_arg":"$triton_arg"}
EOF
}

for target in "${targets[@]}"; do
  run_training "$target"
done

run_python <<PY
import json
import re
from pathlib import Path

run_meta = Path("$run_meta_path")
output_root = Path("$output_root")
timestamp = ""

if not run_meta.exists():
    raise SystemExit("No run metadata found.")

step_pattern = re.compile(r"\bstep[:=]\s*([0-9]+)\b", re.IGNORECASE)
alt_step_pattern = re.compile(r"\bStep\s+([0-9]+)\b", re.IGNORECASE)
number_pattern = r"-?[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?"
loss_pattern = re.compile(rf"\bloss=({number_pattern})", re.IGNORECASE)
reward_pattern = re.compile(rf"\breward=({number_pattern})", re.IGNORECASE)
it_s_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)it/s")
s_it_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)s/it")
vram_postfix_pattern = re.compile(r"\bvram=([0-9]+(?:\.[0-9]+)?)GB")
vram_log_pattern = re.compile(r"VRAM:\s*([0-9]+(?:\.[0-9]+)?)GB")
effective_batch_pattern = re.compile(r"Effective batch size:\s*([0-9]+)")
oom_pattern = re.compile(r"\[OOM\]|out of memory", re.IGNORECASE)
traceback_pattern = re.compile(r"Traceback \(most recent call last\):")
exception_pattern = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(?P<message>.+)$")

def parse_log(path: Path):
    step_times = {}
    step_loss = {}
    step_reward = {}
    vram_samples = []
    effective_batch = None
    oom_events = 0
    last_step = None
    saw_traceback = False
    terminal_error = None

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
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
        "vram_samples": vram_samples,
        "effective_batch": effective_batch,
        "oom_events": oom_events,
        "last_step": last_step,
        "saw_traceback": saw_traceback,
        "terminal_error": terminal_error,
    }

def classify_run(exit_code, parsed, meta):
    if exit_code == 0:
        if parsed["oom_events"] > 0:
            return "oom_recovered", True, None, parsed["terminal_error"]
        return "ok", True, None, parsed["terminal_error"]

    failure_phase = "startup" if parsed["last_step"] is None else "training"
    error = parsed["terminal_error"]

    if meta.get("run_origin") != "isolated_worktree":
        return "contaminated_run", False, failure_phase, error

    if parsed["saw_traceback"] and parsed["last_step"] is None:
        if error and error["type"] in {"CalledProcessError", "FileNotFoundError"}:
            return "tool_error", False, failure_phase, error
        return "invalid_revision", False, failure_phase, error

    return "failed", False, failure_phase, error

def safe_avg(values):
    return sum(values) / len(values) if values else None

def safe_min(values):
    return min(values) if values else None

def safe_max(values):
    return max(values) if values else None

runs = []

for line in run_meta.read_text(encoding="utf-8").splitlines():
    meta = json.loads(line)
    if not timestamp:
        timestamp = meta.get("timestamp", "")
    log_path = Path(meta["log_path"])
    parsed = parse_log(log_path)

    smi_samples = []
    smi_path = Path(meta.get("vram_samples_path", ""))
    if smi_path.is_file():
        for raw in smi_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            candidate = raw.split(",")[-1].strip()
            try:
                value = float(candidate)
            except ValueError:
                continue
            smi_samples.append(value / 1024.0)

    time_samples = [v for k, v in parsed["step_times"].items() if k > 1]
    loss_samples = [v for _, v in parsed["step_loss"].items()]
    reward_samples = [v for _, v in parsed["step_reward"].items()]
    vram_samples = smi_samples or parsed["vram_samples"]

    status, valid, failure_phase, terminal_error = classify_run(
        meta["exit_code"], parsed, meta
    )

    run = {
        "input": meta["input"],
        "label": meta["label"],
        "commit": meta["commit"],
        "triton_mode": meta.get("triton_mode", "auto"),
        "triton_arg": meta.get("triton_arg") or None,
        "status": status,
        "valid": valid,
        "steps_requested": meta["steps_requested"],
        "steps_observed": parsed["last_step"],
        "time_avg_s": safe_avg(time_samples),
        "time_min_s": safe_min(time_samples),
        "time_max_s": safe_max(time_samples),
        "vram_avg_gb": safe_avg(vram_samples),
        "vram_peak_gb": safe_max(vram_samples),
        "loss_avg": safe_avg(loss_samples),
        "reward_avg": safe_avg(reward_samples),
        "effective_batch": parsed["effective_batch"],
        "oom_events": parsed["oom_events"],
        "failure_phase": failure_phase,
        "error_type": terminal_error["type"] if terminal_error else None,
        "error_message": terminal_error["message"] if terminal_error else None,
        "log_path": str(log_path),
    }
    runs.append(run)

summary = {
    "schema_version": 1,
    "generated_at": timestamp or "manual",
    "runs": runs,
}

json_path = output_root / f"compare_bench_{summary['generated_at']}.json"
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

columns = [
    "label",
    "input",
    "commit",
    "triton_mode",
    "triton_arg",
    "status",
    "valid",
    "failure_phase",
    "error_type",
    "error_message",
    "steps_requested",
    "steps_observed",
    "time_avg_s",
    "time_min_s",
    "time_max_s",
    "vram_avg_gb",
    "vram_peak_gb",
    "loss_avg",
    "reward_avg",
    "effective_batch",
    "oom_events",
    "log_path",
]

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

csv_path = output_root / f"compare_bench_{summary['generated_at']}.csv"
lines = [",".join(columns)]
for run in runs:
    lines.append(",".join(fmt(run.get(col)) for col in columns))
csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {json_path}")
print(f"Wrote {csv_path}")
PY
