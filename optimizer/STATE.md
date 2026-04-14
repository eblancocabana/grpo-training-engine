# State

## Locked Configuration

The following configuration values are locked and must not be modified:

- **group_size = 4** - Locked. Do not attempt to change.
- **entropy_mask = True** - Locked. Do not attempt to disable.

Any attempt markdown proposing changes to these parameters will be rejected.

## Active state files

### Frontier

`optimizer/artifacts/records/frontier/frontier_state.json`

This file now tracks frontier state separately for two Triton modes:

- `modes.triton_on.current`
- `modes.triton_on.running_best`
- `modes.triton_off.current`
- `modes.triton_off.running_best`

The active mode is selected from `optimizer/artifacts/records/policy/runtime_policy.json`.

Read:

- `active_mode`
- `modes.<active_mode>.current.candidate_id`
- `modes.<active_mode>.current.target`
- `modes.<active_mode>.current.comparability`
- `modes.<active_mode>.current.benchmark`
- `modes.<active_mode>.running_best.candidate_id`
- `modes.<active_mode>.running_best.target`
- `modes.<active_mode>.running_best.comparability`
- `modes.<active_mode>.running_best.benchmark`

Rule:

- `current` may drift because of noisy reruns, machine-state changes, or temporary regressions.
- `running_best` is the protected historical anchor for the active mode.
- If active-mode `current` becomes worse than active-mode `running_best`, restore `current = running_best` inside that mode before the next iteration.
- Never let Triton-on and Triton-off overwrite each other’s frontier lane.

### Experiment ledger

`optimizer/artifacts/records/experiments/experiment_ledger.jsonl`

This is the append-only history of attempts.

Read:

- `experiment_number`
- `change_summary`
- `outcome`
- `reason`
- `selected_target`
- `hypothesis`
- `baseline_snapshot`
- `candidate_snapshot`
- `running_best_step_time`
- `running_best_tokens_per_sec`

Rule:

- `running_best_step_time` is the authoritative historical floor for the active mode.
- Do not accept a candidate that improves active-mode `current` but still fails to beat active-mode `running_best_step_time`.

### Raw benchmark artifacts

`benchmarks/output/compare_bench_*.json`

These are the benchmark authority inputs consumed by the reducer.

### Reports

- `optimizer/artifacts/reports/tokens_per_sec.svg`
- `optimizer/artifacts/reports/tokens_per_sec.md`

## Attempt markdown contract

Each active attempt markdown should define at least:

- Frontier target
- Candidate id
- Candidate worktree
- Selected target
- Hypothesis
- Change summary

The reducer reads this markdown plus the benchmark JSON.

## Legacy history

Older session, iteration, observation, mutation-plan, worktree, and generated attempt artifacts are preserved on disk as archive history. They are not active inputs to the new loop.

## Runtime policy

`optimizer/artifacts/records/policy/runtime_policy.json`

This file selects the active Triton mode.

- `triton_mode = on` → run compare bench with `--triton on`, and Triton kernel/structural modifications are allowed.
- `triton_mode = off` → run compare bench with `--triton off`, and Triton kernel/structural modifications are forbidden.

The optional `note` field is operator guidance only.
