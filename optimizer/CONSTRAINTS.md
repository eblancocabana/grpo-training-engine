# Optimization Constraints

## Hard Constraints

These parameters are **LOCKED** and must not be modified:

### group_size
- **Value:** 4
- **Status:** LOCKED
- **Reason:** Optimal value found through extensive testing. Any modification will be rejected.

### entropy_mask
- **Value:** True (use_entropy_mask = True)
- **Status:** LOCKED
- **Reason:** Required for training stability. Must remain enabled despite performance impact.
- **Note:** While testing showed +34.7% improvement with False, stability requirements override speed.

## Triton Mode

Runtime policy file: `optimizer/artifacts/records/policy/runtime_policy.json`

- `triton_mode = on` → Triton kernel / structural modifications are allowed.
- `triton_mode = off` → Triton kernel / structural modifications are forbidden.

This flag controls only whether Triton modifications are allowed. It does not change the benchmark authority, promotion rules, or locked parameters.

## Mutable Parameters

These parameters can still be optimized:
- clear_cache frequency
- CPU offload settings
- TF32 precision
- cudnn.benchmark
- Other generation-specific parameters

## Procedure

When forming hypotheses:
1. Check this file first
2. Do not propose changes to locked parameters
3. Focus optimization efforts on mutable parameters only
4. If unsure, prefer smaller changes over larger ones

## Important Notes

- **entropy_mask=False** showed +34.7% speed improvement in testing
- **However**, stability requirements mandate keeping it at True
- Document any performance trade-offs in experiment notes
