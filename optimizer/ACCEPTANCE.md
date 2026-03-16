# Acceptance

The candidate replaces the frontier only when benchmark comparison says it is better.

## Blocked Parameters

The following parameters are **BLOCKED** from modification:

- **group_size** - Must remain at 8. Any candidate modifying this will be rejected.
- **entropy_mask** - Must remain at True. Any candidate disabling this will be rejected.

## Authority

Use only `tools/compare_bench.sh --steps 10` output.

Profiler traces, intuition, and code inspection are diagnostic inputs only.

## Comparability

The candidate must be benchmark-comparable.

- `status=ok` and `valid=true` are clean comparable runs.
- `status=oom_recovered` can still be promotable when recovered OOM promotion is allowed and the run is otherwise complete and valid.
- Incomplete, invalid, or non-comparable runs must not become the frontier.

## Promotion rule

1. Reject if the frontier is not comparable.
2. Reject if the candidate is not comparable.
3. Reject if effective batch differs.
4. Reject if blocked parameters were modified.
5. Accept if tokens/sec improves.
6. Otherwise discard the candidate.
