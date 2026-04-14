# Artifact Schema

## Active artifacts

### Frontier state

`optimizer/artifacts/records/frontier/frontier_state.json`

Overwrite-in-place incumbent state.

### Experiment ledger

`optimizer/artifacts/records/experiments/experiment_ledger.jsonl`

Append-only experiment history. Old rows are preserved and still readable.

### Benchmark authority artifacts

`benchmarks/output/compare_bench_*.json`

These are produced by `tools/compare_bench.sh` and consumed by the reducer.

### Attempt markdowns

`optimizer/artifacts/attempts/*.md`

These are active agent-facing inputs.

### Reports

- `optimizer/artifacts/reports/tokens_per_sec.svg`
- `optimizer/artifacts/reports/tokens_per_sec.md`

## Legacy artifacts

These remain preserved on disk but are no longer active loop inputs:

- `optimizer/artifacts/records/sessions/`
- `optimizer/artifacts/records/iterations/`
- `optimizer/artifacts/records/observations/`
- `optimizer/artifacts/records/mutation_plans/`
- `optimizer/artifacts/records/worktrees/`

## Reducer outputs

The reducer writes:

- frontier state
- decision record
- normalized experiment ledger row
- refreshed tokens/sec report outputs
