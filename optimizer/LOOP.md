# Loop

This repository uses a strict sequential loop.

The agent is the optimizer.

## Sequence

1. Read `optimizer/artifacts/records/policy/runtime_policy.json` and determine the active Triton mode (`on` or `off`).
2. Read the active-mode frontier, the active-mode running-best frontier, and the recent experiment ledger.
3. If active-mode `current` is worse than active-mode `running_best`, restore `current = running_best` inside that mode before continuing.
4. Read recent benchmark artifacts for context.
5. Form one bounded hypothesis.
5. Prepare or refresh one candidate worktree.
6. Write or update one attempt markdown file.
7. Edit only inside that worktree.
9. Run `tools/compare_bench.sh --steps 10 --triton <mode>` against the active-mode current frontier and the candidate worktree.
10. If the active-mode current frontier benchmark is more than 10% slower than active-mode `running_best`, rerun once with reversed order (`candidate` first, then `frontier`) before deciding promotion.
11. Run the reducer to persist the outcome.
12. Read the updated active-mode frontier and continue from there.

## Hard rules

- One current benchmark frontier plus one protected running-best frontier per Triton mode.
- The active Triton mode comes from `optimizer/artifacts/records/policy/runtime_policy.json`.
- `triton_mode=on` means Triton kernel / structural changes are allowed.
- `triton_mode=off` means Triton kernel / structural changes are forbidden.
- One active candidate only.
- One benchmark decision path only.
- No parallel benchmark runs.
- Benchmark output is authoritative.
- Scope is generation-only.
- **CRITICAL: group_size must remain at 4. Do NOT modify group_size.**
- **CRITICAL: entropy_mask must remain at True. Do NOT disable entropy_mask.**
