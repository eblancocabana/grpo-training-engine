# Combined Benchmark Analysis

Generated: 2026-04-14

## Scope

This note consolidates the benchmark output families that matter for the current
default-training decision:

- `benchmarks/output/2026-04-13/triton_matrix`
- `benchmarks/output/2026-04-13/ing_inf`
- `benchmarks/output/deepseek_generation_suite`
- `benchmarks/output/deepseek_generation_default_resp384`
- `benchmarks/output/deepseek_g8_probe_iter2`
- `benchmarks/output/triton_train_focus_20260412`
- `benchmarks/output/triton_train_smoke_20260412`
- `benchmarks/output/response_len_512`
- `benchmarks/output/diagnostic_512_40step_final_2026-04-14`
- `benchmarks/output/diagnostic_768_40step_2026-04-14`

I did not treat temporary or log-only directories such as `run_*`,
`default_run_logs`, or older backups as primary evidence unless they were needed
to confirm a pattern already visible in the structured suites above.

## Executive Conclusion

Three claims are simultaneously true:

1. Triton is useful in this project, but mainly through the non-generation
   kernels.
2. Triton generation is not the strongest default based on the benchmark outputs
   on disk.
3. The best fast-versus-quality balance right now is:

- Triton on
- Triton generation off
- Triton GRPO loss on
- Triton entropy mask on
- Triton LoRA on
- `group_size=4`
- `gradient_accumulation_steps=16`
- `max_response_length=768`
- `lora_rank=16`
- `lora_adapter_quant=none`

That is now the code default.

## Why Triton Generation Is Not the Default

The most important long-response matrix is
`benchmarks/output/2026-04-13/triton_matrix/results.json`:

| Config | Step Time (s) | Tok/s |
| --- | ---: | ---: |
| `torch_baseline` | `136.41` | `16.80` |
| `triton_generation_only` | `133.20` | `17.21` |
| `triton_all` | `139.68` | `16.02` |
| `triton_all_except_generation` | `59.71` | `31.07` |

This is the main source of confusion in the earlier discussion.

- `triton_generation_only` being slightly better than `torch_baseline` means
  Triton generation was a small isolated win over full torch.
- It does **not** mean Triton generation was the best overall stack.
- The best overall long-response result was clearly
  `triton_all_except_generation`.

The logs explain why:

- `triton_all` hit more OOM backoffs and collapsed generation micro-batch size.
- `triton_all_except_generation` stayed much healthier.

So the benefit came from better end-to-end stability under memory pressure, not
from Triton generation being globally faster.

## What the DeepSeek Generation Suites Showed

The shorter structured generation suites mostly favored torch generation.

### `deepseek_generation_suite` at `max_response_length=128`

From `benchmarks/output/deepseek_generation_suite/training_matrix.json`:

- `g1`: torch `7.21s` vs Triton `7.96s`
- `g2`: torch `8.28s` vs Triton `8.96s`
- `g4`: torch `8.45s` vs Triton `9.30s`
- `g8`: torch `15.64s` vs Triton `16.84s`
- `g16`: torch `29.14s` vs Triton `31.81s`

Torch also led on tokens/sec at every group size in that suite.

### `deepseek_generation_default_resp384`

The `384` response-length matrix continued the same pattern for the completed
pairs on disk: torch generation beat Triton generation for the tested `g1`,
`g2`, and `g4` runs.

### `deepseek_g8_probe_iter2`

There was one narrow counterexample. In that probe, Triton generation had a
small tokens/sec edge at `g8`, but step time was still worse than torch. That
is weak evidence for a training default because the training loop still finished
slower.

### Lower-weight consistency checks

The smaller `triton_train_focus_20260412` and `triton_train_smoke_20260412`
directories did not overturn the conclusion above. They were useful as
consistency checks, but the structured suites carried more weight.

## What the April 13 `ing_inf` Suite Actually Means

Yes, the `ing_inf` suite default response length is `1024`.

That is defined in `benchmarks/ing_inf_suite.py`, so many runs inside
`benchmarks/output/2026-04-13/ing_inf` were indeed `1024` runs.

Important distinction:

- some tuned `1024` configurations were reasonably fast
- the matched response-length sweep showed that pushing the cap from `768` to
  `1024` was expensive under one fixed Triton-on configuration

### Matched response-length sweep

From `benchmarks/output/2026-04-13/ing_inf/results.json`:

| Config | Step Time (s) | Tok/s |
| --- | ---: | ---: |
| `triton_on_response_len_256` | `34.79` | `30.70` |
| `triton_on_response_len_512` | `47.20` | `36.34` |
| `triton_on_response_len_768` | `51.70` | `36.13` |
| `triton_on_response_len_1024` | `156.41` | `15.33` |

That `1024` result showed the cliff clearly.

The `1024` train log also recorded two explicit OOM backoffs, even though the
JSON parser only summarizes OOM presence as `1`:

- Backoff #1: `gen_batch=2`, `train_batch=2`
- Backoff #2: `gen_batch=1`, `train_batch=1`

### Tuned `1024` runs in the same suite

`1024` is not always bad. The same April 13 suite includes faster tuned
variants:

- `triton_on_current`: `94.60s/step`
- `triton_on_grad_accum_8`: `86.33s/step`
- `triton_on_lora_rank_16`: `71.23s/step`
- `triton_on_lora_quant_none`: `75.85s/step`

So the right interpretation is:

- `1024` was the suite default
- some tuned `1024` configs were fine
- but the fixed-config response sweep shows `1024` is more fragile than `768`

## What the `response_len_512` Workflow Showed

The automatic workflow did run successfully.

With `--include-rollout-diversity`, the suite is supposed to run:

1. four core configs
2. the winner with `grad_accum=8`
3. the winner with `grad_accum=32`
4. the winner with `group_size=8`

The report shows `7/7` successful runs. It does **not** mean all 16 configs
displayed by `--list` were executed.

### Step times from the completed 512 workflow

From `benchmarks/output/response_len_512/report.md`:

| Config | Step Time (s) | Tok/s |
| --- | ---: | ---: |
| `triton_gen_off_g4_ga16` | `25.44` | `66.45` |
| `triton_gen_auto_g4_ga16` | `24.71` | `67.56` |
| `triton_gen_on_g4_ga16` | `25.50` | `65.09` |
| `torch_baseline_g4_ga16` | `23.03` | `75.22` |
| `torch_baseline_g4_ga8` | `22.95` | `73.97` |
| `torch_baseline_g4_ga32` | `23.33` | `72.98` |
| `torch_baseline_g8_ga16` | `45.33` | `74.57` |

At `512`, torch won cleanly:

- `torch_baseline_g4_ga16` beat `triton_gen_off_g4_ga16` by `10.47%` on step
  time
- it beat `triton_gen_on_g4_ga16` by `10.71%`
- `group_size=8` nearly doubled step time without improving throughput

### Important caveat about the `auto` run

The saved config for the supposed auto-policy run ended up with:

- `triton_generation_mode: "on"`

So the suite did **not** really compare:

- off
- auto
- on
- torch

It effectively compared:

- off
- on
- on
- torch

That means the auto-vs-on question remains unresolved in this particular
workflow.

## Why `512` Is Not the Best Long-Run Default

The 7-run benchmark above only covered short benchmark windows. To test whether
`512` was actually enough headroom for real rollouts, I ran a 40-step training
diagnostic.

Path:

- `benchmarks/output/diagnostic_512_40step_final_2026-04-14`

Configuration:

- full torch
- `group_size=4`
- `grad_accum=16`
- `max_response_length=512`
- `lora_adapter_quant=none`
- `--no-sent`
- `--no-wandb --no-initial-benchmark --no-checkpoints -vvv`

Results:

- `40/40` steps completed
- average step time: `33.41s`
- median step time: `26.26s`
- average response length: `421.67`
- steps with any truncation: `19/40`
- steps with majority truncation: `17/40`
- steps with full truncation: `7/40`
- OOM backoff events: `1`
- backoff step: `22`
- first 20 steps: `22.39s/step`
- last 20 steps: `44.43s/step`

This is why `512` stopped looking like the right default. It was fast early,
but too many rollouts were being clipped or destabilized later in the run.

## Why `768` Is the Current Default

I then ran the same kind of 40-step diagnostic for the best-balance candidate:

Path:

- `benchmarks/output/diagnostic_768_40step_2026-04-14`

Configuration:

- Triton on
- Triton generation off
- Triton GRPO loss on
- Triton entropy mask on
- Triton LoRA on
- `group_size=4`
- `grad_accum=16`
- `max_response_length=768`
- `lora_rank=16`
- `lora_adapter_quant=none`
- `--no-wandb --no-initial-benchmark --no-checkpoints -vvv`

Important caveat:

- this `768` run used SENT-on defaults
- the earlier `512` diagnostic used `--no-sent`

So the two diagnostics are not perfectly apples-to-apples. Even with that
caveat, the `768` run was clearly healthier.

### 768 diagnostic result

- `40/40` steps completed
- average step time: `27.59s`
- median step time: `25.45s`
- average tokens/sec: `63.78`
- average response length: `440.49`
- steps with any truncation: `9/40`
- steps with majority truncation: `7/40`
- steps with full truncation: `3/40`
- OOM backoff events: `0`
- generation micro-batch values: always `4`
- training micro-batch values: always `4`

That is the best balance currently demonstrated on disk:

- much healthier than `512`
- much cheaper and less fragile than the `1024` response-length cliff
- aligned with the earlier evidence that the best Triton stack excludes Triton
  generation

## Final Recommendation

The current default should be:

```bash
python train.py \
  --use-triton \
  --no-triton-generation \
  --triton-grpo-loss \
  --triton-entropy-mask \
  --triton-lora \
  --group-size 4 \
  --gradient-accumulation-steps 16 \
  --max-response-length 768 \
  --lora-rank 16 \
  --lora-adapter-quant none
```

Rationale:

- `512` is faster in short benchmarks, but too truncation-heavy in a longer run
- `1024` can work, but is more fragile and more expensive
- `768` gives enough room for the model to think without returning to the clear
  `1024` instability seen in the matched sweep
- Triton still matters, but mainly outside the generation path

## Code Status

The codebase has been updated to make this the default preset:

- `src/utils/config.py` now defaults to the `768` Triton-with-generation-off
  profile
- `train.py` now preserves the preset instead of overriding Triton defaults back
  to `auto`

That makes the benchmark-backed configuration the real default instead of a
documentation-only recommendation.
