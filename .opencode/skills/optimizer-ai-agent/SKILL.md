---
name: optimizer-ai-agent
description: "Own the sequential autoresearch-style optimization loop for this repository using benchmark authority, frontier state, experiment history, and attempt markdowns."
compatibility: opencode
metadata:
  audience: optimization
---

# What I do

This skill makes the agent the owner of the optimization loop.

The repository does not plan the loop for the agent anymore. The repository only provides benchmark authority, state persistence, and reporting.

# Active inputs

Read these before acting:

- `optimizer/LOOP.md`
- `optimizer/ACCEPTANCE.md`
- `optimizer/STATE.md`
- `optimizer/artifacts/records/frontier/frontier_state.json`
- `optimizer/artifacts/records/experiments/experiment_ledger.jsonl`
- recent `benchmarks/output/compare_bench_*.json`
- the active attempt markdown under `optimizer/artifacts/attempts/`

# Core loop

1. Read the current frontier and recent experiment history.
2. Form one hypothesis based on that history.
3. Create or refresh one candidate worktree.
4. Write or update one attempt markdown.
5. Make one bounded generation-only change in that worktree.
6. Run `tools/compare_bench.sh --steps 10` against the frontier and candidate worktree.
7. Run `python -m optimizer.reducer --attempt-markdown <attempt.md> --benchmark-report <compare_bench.json> --allow-recovered-oom-promotion`.
8. Read the updated frontier and ledger.
9. Repeat sequentially.

# Hard requirements

- Stay strictly sequential.
- Work on one candidate only.
- Keep scope generation-only.
- Prefer the smallest causal change.
- Do not treat profiler evidence as authority.
- Do not rewrite experiment history.
- Do not replace benchmark authority.
- **CRITICAL: group_size must remain at 8. Do NOT modify group_size under any circumstances.**
- **CRITICAL: entropy_mask must remain at True (use_entropy_mask = True). Do NOT disable entropy_mask.**

# Attempt markdown contract

The attempt markdown must define at least:

- Frontier target
- Candidate id
- Candidate worktree
- Selected target
- Hypothesis
- Change summary

Everything else may remain prose.

# What not to read as active control

These directories may still exist for historical reasons, but they are no longer active loop inputs:

- `optimizer/artifacts/records/sessions/`
- `optimizer/artifacts/records/iterations/`
- `optimizer/artifacts/records/observations/`
- `optimizer/artifacts/records/mutation_plans/`
- `optimizer/artifacts/records/worktrees/`

# Success condition

The loop is successful only when the benchmark promotes the candidate and the reducer updates frontier, ledger, and tokens/sec reporting accordingly.
