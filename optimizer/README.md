# Optimizer Loop

This directory contains the benchmark-driven optimization loop used to evolve generation performance without treating ad hoc local timings as authority.

## What Is Here

- `LOOP.md`: operator procedure for each iteration
- `CONSTRAINTS.md`: locked parameters and non-negotiables
- `ACCEPTANCE.md`: promotion criteria
- `STATE.md`: current operating notes
- `frontier.py`: frontier state helpers
- `records.py`: persistent record models
- `reducer.py`: attempt + benchmark reduction into acceptance decisions
- `evaluation/`: benchmark classification and acceptance logic
- `run_one_cron_iteration.py`: one automated loop pass
- `run_one_cron_iteration.sh`: shell entrypoint for cron-style execution
- `run_acceptance_tests.py`: focused acceptance test runner
- `plot_step_time.py` and `plot_tokens_per_sec.py`: report generation
- `write_generation_review.py`: review artifact writer
- `artifacts/`: persisted attempts, decisions, reports, tests, frontiers, and ledgers

## Source of Truth

The active machine state lives under:

- `optimizer/artifacts/frontier/frontier_state.json`
- `optimizer/artifacts/experiments/experiment_ledger.jsonl`
- `optimizer/artifacts/reports/`
- `benchmarks/output/compare_bench_*.json`

The loop is benchmark-authoritative. Candidate promotions depend on benchmark evidence, not on isolated local intuition.

## Active Operating Model

1. Read `LOOP.md`, `ACCEPTANCE.md`, `STATE.md`, and `CONSTRAINTS.md`.
2. Read runtime policy and choose the active Triton mode.
3. Load the active frontier, recent ledger entries, and benchmark outputs.
4. If `current` drifted above `running_best`, restore `current = running_best`.
5. Form one bounded hypothesis.
6. Create or refresh one candidate worktree.
7. Make one generation-focused change.
8. Run `tools/compare_bench.sh` against frontier and candidate.
9. Reduce the benchmark output with `python -m optimizer.reducer`.
10. Promote only if the acceptance gate passes.

## Current Constraints

At the time of this README update, the loop documents two locked runtime constraints explicitly:

- `group_size = 4`
- `entropy_mask = True`

Check `CONSTRAINTS.md` before changing optimizer-targeted runs.

## Tests

Optimizer coverage lives in `tests/optimizer/`:

```bash
pytest tests/optimizer/
```

Key test areas include:

- acceptance decisions
- frontier handling
- benchmark gate behavior
- record serialization
- reducer behavior

## Typical Commands

One compare run:

```bash
tools/compare_bench.sh --steps 10 --triton on <frontier-worktree> <candidate-worktree>
```

Reduce an attempt:

```bash
python -m optimizer.reducer \
  --attempt-markdown optimizer/artifacts/attempts/iter001_attempt.md \
  --benchmark-report benchmarks/output/compare_bench_example.json \
  --allow-recovered-oom-promotion
```

Run optimizer tests:

```bash
pytest tests/optimizer/
```

## Notes

- Historical planning and attempt artifacts are intentionally preserved on disk.
- The optimizer is tightly coupled to the repository’s Triton-on/Triton-off benchmark workflow.
- This directory documents the current loop mechanics; benchmark reports remain the authority for promotions.
