# CDIA Endurance Report

Generated: 2026-04-19 14:33:31
Plan: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_endurance_suite_plan.json
Trainer steps: 240
Total runs: 6
Successful: 6
Derived from benchmarks/cdia_final_validation_suite.py with the same output flow and per-run layout, but narrowed to a strict 6-run endurance selector across the two real finalist families.

## Methodology

- Strict finalist-only scope: `length_penalty_0` versus `length_penalty_0005_mask_truncated_off`, three seeds each, no other variants.
- Apples-to-apples budget: every run gets 240 trainer steps, the same held-out benchmark path, the same benchmark cadence, and SENT disabled.
- Held-out benchmark cadence: periodic every 100 trainer steps plus one final benchmark immediately after the final checkpoint is saved.
- Hardware/output assumptions match the current CDIA suites: same train.py path, same per-run folder structure, same checkpoint markers, same metrics JSONL logging.
- Critical interpretation rule: raw reward is not directly comparable between the penalized and unpenalized finalist families. A penalized run can show lower raw reward simply because the objective subtracts length cost.

## Summary

| Config | Group | Seed | Status | Steps | Opt Steps | Tok/s | In-Train Bench | Final Ckpt Bench | Reward Final | Reward Peak@Step | Drop | Resp Final | Resp Peak@Step | Trunc Avg | Masked-Out Final |
|--------|-------|------|--------|-------|-----------|-------|----------------|------------------|--------------|------------------|------|------------|----------------|-----------|------------------|
| length_penalty_0_seed1 | length_penalty_0 | 1 | OK | 240 | 60 | 64.97 | 0.8000 | 0.7400 | 1.0000 | 1.0000@2 | 0.0000 | 434.2 | 1024.0@28 | 0.1167 | 0.0000 |
| length_penalty_0_seed2 | length_penalty_0 | 2 | OK | 240 | 60 | 66.08 | 0.7200 | 0.7600 | 0.5000 | 1.0000@1 | 0.5000 | 765.8 | 1024.0@30 | 0.0917 | 0.5000 |
| length_penalty_0_seed3 | length_penalty_0 | 3 | OK | 240 | 60 | 65.65 | 0.6600 | 0.6800 | 0.7500 | 1.0000@1 | 0.2500 | 880.8 | 1024.0@103 | 0.1083 | 0.5000 |
| length_penalty_0005_mask_truncated_off_seed1 | length_penalty_0005_mask_truncated_off | 1 | OK | 240 | 60 | 65.47 | 0.7200 | 0.7600 | 0.8148 | 0.8917@117 | 0.0770 | 370.5 | 1024.0@28 | 0.1167 | 0.0000 |
| length_penalty_0005_mask_truncated_off_seed2 | length_penalty_0005_mask_truncated_off | 2 | OK | 240 | 60 | 65.75 | 0.7800 | 0.8000 | 0.3720 | 0.8874@10 | 0.5154 | 756.0 | 1024.0@30 | 0.0990 | 0.0000 |
| length_penalty_0005_mask_truncated_off_seed3 | length_penalty_0005_mask_truncated_off | 3 | OK | 240 | 60 | 66.04 | 0.7000 | 0.7000 | 0.0444 | 0.8911@134 | 0.8468 | 911.2 | 1024.0@103 | 0.1021 | 0.0000 |

## Detailed Results

### length_penalty_0_seed1

- Comparison group: length_penalty_0
- Seed: 1
- Success: True
- Duration: 10486.8s
- Steps completed: 240
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0_seed1/train.log
- Metrics: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0_seed1/metrics.jsonl
- Step time: 39.77s (min: 21.06, max: 318.40)
- Throughput: 64.97 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 60
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7792
- Final reward: 1.0000
- Peak reward: 1.0000 at step 2
- Reward drop from peak to final: 0.0000
- Final reward std: 0.0000
- Whole-run reward std avg: 0.1355
- Reward range avg: [0.6167, 0.9167]
- Whole-run avg response length: 515.9
- Final avg response length: 434.2
- Peak avg response length: 1024.0 at step 28
- Avg response length at reward peak step 2: 388.0
- Response length drop from peak to final: 589.8
- Response budget usage avg: 0.5038
- Final response budget usage: 0.4241
- Cap-drift failure signal: True (drift toward the 1024-token cap)
- Entropy-masked ratio avg: 0.3362
- Legacy truncated-completions ratio avg (only when masking active): 0.1167
- Actual truncated completions ratio avg: 0.1167
- Peak actual truncated completions ratio: 1.0000 at step 28
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.1167
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.1000
- OOM backoff count final: 1
- In-training held-out benchmark accuracy values: 0.7000@100, 0.8000@200
- Final in-training held-out benchmark accuracy: 0.8000
- Peak in-training held-out benchmark accuracy: 0.8000 at step 200
- Final checkpoint held-out benchmark accuracy: 0.7400
- Final checkpoint held-out benchmark step: 240
- Final checkpoint benchmark format compliance: 0.9800
- Final checkpoint benchmark avg length: 451.3
- Late-collapse flag: True

### length_penalty_0_seed2

- Comparison group: length_penalty_0
- Seed: 2
- Success: True
- Duration: 9789.0s
- Steps completed: 240
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0_seed2/train.log
- Metrics: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0_seed2/metrics.jsonl
- Step time: 36.90s (min: 20.06, max: 305.02)
- Throughput: 66.08 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 60
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7958
- Final reward: 0.5000
- Peak reward: 1.0000 at step 1
- Reward drop from peak to final: 0.5000
- Final reward std: 0.5000
- Whole-run reward std avg: 0.1169
- Reward range avg: [0.6542, 0.9125]
- Whole-run avg response length: 481.0
- Final avg response length: 765.8
- Peak avg response length: 1024.0 at step 30
- Avg response length at reward peak step 1: 278.8
- Response length drop from peak to final: 258.2
- Response budget usage avg: 0.4697
- Final response budget usage: 0.7478
- Cap-drift failure signal: True (drift toward the 1024-token cap)
- Entropy-masked ratio avg: 0.3554
- Legacy truncated-completions ratio avg (only when masking active): 0.0917
- Actual truncated completions ratio avg: 0.0917
- Peak actual truncated completions ratio: 1.0000 at step 30
- Final actual truncated completions ratio: 0.5000
- Truncated completions masked out of loss ratio avg: 0.0917
- Final truncated completions masked out of loss ratio: 0.5000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.1021
- OOM backoff count final: 1
- In-training held-out benchmark accuracy values: 0.7000@100, 0.7200@200
- Final in-training held-out benchmark accuracy: 0.7200
- Peak in-training held-out benchmark accuracy: 0.7200 at step 200
- Final checkpoint held-out benchmark accuracy: 0.7600
- Final checkpoint held-out benchmark step: 240
- Final checkpoint benchmark format compliance: 0.9800
- Final checkpoint benchmark avg length: 456.7
- Late-collapse flag: True

### length_penalty_0_seed3

- Comparison group: length_penalty_0
- Seed: 3
- Success: True
- Duration: 10249.5s
- Steps completed: 240
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0_seed3/train.log
- Metrics: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0_seed3/metrics.jsonl
- Step time: 38.93s (min: 20.42, max: 313.05)
- Throughput: 65.65 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 60
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7719
- Final reward: 0.7500
- Peak reward: 1.0000 at step 1
- Reward drop from peak to final: 0.2500
- Final reward std: 0.4330
- Whole-run reward std avg: 0.1047
- Reward range avg: [0.6417, 0.8750]
- Whole-run avg response length: 508.9
- Final avg response length: 880.8
- Peak avg response length: 1024.0 at step 103
- Avg response length at reward peak step 1: 419.8
- Response length drop from peak to final: 143.2
- Response budget usage avg: 0.4969
- Final response budget usage: 0.8601
- Cap-drift failure signal: True (drift toward the 1024-token cap)
- Entropy-masked ratio avg: 0.3441
- Legacy truncated-completions ratio avg (only when masking active): 0.1083
- Actual truncated completions ratio avg: 0.1083
- Peak actual truncated completions ratio: 1.0000 at step 103
- Final actual truncated completions ratio: 0.5000
- Truncated completions masked out of loss ratio avg: 0.1083
- Final truncated completions masked out of loss ratio: 0.5000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.0708
- OOM backoff count final: 1
- In-training held-out benchmark accuracy values: 0.7000@100, 0.6600@200
- Final in-training held-out benchmark accuracy: 0.6600
- Peak in-training held-out benchmark accuracy: 0.7000 at step 100
- Final checkpoint held-out benchmark accuracy: 0.6800
- Final checkpoint held-out benchmark step: 240
- Final checkpoint benchmark format compliance: 0.9200
- Final checkpoint benchmark avg length: 445.7
- Late-collapse flag: True

### length_penalty_0005_mask_truncated_off_seed1

- Comparison group: length_penalty_0005_mask_truncated_off
- Seed: 1
- Success: True
- Duration: 10393.4s
- Steps completed: 240
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0005_mask_truncated_off_seed1/train.log
- Metrics: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0005_mask_truncated_off_seed1/metrics.jsonl
- Step time: 39.47s (min: 21.82, max: 323.57)
- Throughput: 65.47 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 60
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0104
- Whole-run reward avg: 0.5291
- Final reward: 0.8148
- Peak reward: 0.8917 at step 117
- Reward drop from peak to final: 0.0770
- Final reward std: 0.0046
- Whole-run reward std avg: 0.1508
- Reward range avg: [0.3419, 0.6834]
- Whole-run avg response length: 512.7
- Final avg response length: 370.5
- Peak avg response length: 1024.0 at step 28
- Avg response length at reward peak step 117: 216.5
- Response length drop from peak to final: 653.5
- Response budget usage avg: 0.5007
- Final response budget usage: 0.3618
- Cap-drift failure signal: True (drift toward the 1024-token cap)
- Entropy-masked ratio avg: 0.3936
- Actual truncated completions ratio avg: 0.1167
- Peak actual truncated completions ratio: 1.0000 at step 28
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.0000
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: False
- Positive advantages ratio avg: 0.5250
- OOM backoff count final: 1
- In-training held-out benchmark accuracy values: 0.6800@100, 0.7200@200
- Final in-training held-out benchmark accuracy: 0.7200
- Peak in-training held-out benchmark accuracy: 0.7200 at step 200
- Final checkpoint held-out benchmark accuracy: 0.7600
- Final checkpoint held-out benchmark step: 240
- Final checkpoint benchmark format compliance: 0.9800
- Final checkpoint benchmark avg length: 440.9

### length_penalty_0005_mask_truncated_off_seed2

- Comparison group: length_penalty_0005_mask_truncated_off
- Seed: 2
- Success: True
- Duration: 9758.2s
- Steps completed: 240
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0005_mask_truncated_off_seed2/train.log
- Metrics: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0005_mask_truncated_off_seed2/metrics.jsonl
- Step time: 36.78s (min: 20.04, max: 313.39)
- Throughput: 65.75 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 60
- Trainer steps per optimizer step: 4.00
- Final loss: 23.8741
- Whole-run reward avg: 0.5353
- Final reward: 0.3720
- Peak reward: 0.8874 at step 10
- Reward drop from peak to final: 0.5154
- Final reward std: 0.5182
- Whole-run reward std avg: 0.1319
- Reward range avg: [0.3807, 0.6879]
- Whole-run avg response length: 477.4
- Final avg response length: 756.0
- Peak avg response length: 1024.0 at step 30
- Avg response length at reward peak step 10: 225.2
- Response length drop from peak to final: 268.0
- Response budget usage avg: 0.4662
- Final response budget usage: 0.7383
- Cap-drift failure signal: True (drift toward the 1024-token cap)
- Entropy-masked ratio avg: 0.3962
- Actual truncated completions ratio avg: 0.0990
- Peak actual truncated completions ratio: 1.0000 at step 30
- Final actual truncated completions ratio: 0.2500
- Truncated completions masked out of loss ratio avg: 0.0000
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: False
- Positive advantages ratio avg: 0.4938
- OOM backoff count final: 1
- In-training held-out benchmark accuracy values: 0.6800@100, 0.7800@200
- Final in-training held-out benchmark accuracy: 0.7800
- Peak in-training held-out benchmark accuracy: 0.7800 at step 200
- Final checkpoint held-out benchmark accuracy: 0.8000
- Final checkpoint held-out benchmark step: 240
- Final checkpoint benchmark format compliance: 0.9800
- Final checkpoint benchmark avg length: 454.4
- Late-collapse flag: True

### length_penalty_0005_mask_truncated_off_seed3

- Comparison group: length_penalty_0005_mask_truncated_off
- Seed: 3
- Success: True
- Duration: 10116.3s
- Steps completed: 240
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0005_mask_truncated_off_seed3/train.log
- Metrics: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_endurance/cdia_length_penalty_0005_mask_truncated_off_seed3/metrics.jsonl
- Step time: 38.26s (min: 20.23, max: 304.79)
- Throughput: 66.04 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 60
- Trainer steps per optimizer step: 4.00
- Final loss: 15.9288
- Whole-run reward avg: 0.5199
- Final reward: 0.0444
- Peak reward: 0.8911 at step 134
- Reward drop from peak to final: 0.8468
- Final reward std: 0.5568
- Whole-run reward std avg: 0.1525
- Reward range avg: [0.3173, 0.6741]
- Whole-run avg response length: 504.0
- Final avg response length: 911.2
- Peak avg response length: 1024.0 at step 103
- Avg response length at reward peak step 134: 217.8
- Response length drop from peak to final: 112.8
- Response budget usage avg: 0.4922
- Final response budget usage: 0.8899
- Cap-drift failure signal: True (drift toward the 1024-token cap)
- Entropy-masked ratio avg: 0.3948
- Actual truncated completions ratio avg: 0.1021
- Peak actual truncated completions ratio: 1.0000 at step 103
- Final actual truncated completions ratio: 0.5000
- Truncated completions masked out of loss ratio avg: 0.0000
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: False
- Positive advantages ratio avg: 0.5177
- OOM backoff count final: 1
- In-training held-out benchmark accuracy values: 0.7000@100, 0.7000@200
- Final in-training held-out benchmark accuracy: 0.7000
- Peak in-training held-out benchmark accuracy: 0.7000 at step 100
- Final checkpoint held-out benchmark accuracy: 0.7000
- Final checkpoint held-out benchmark step: 240
- Final checkpoint benchmark format compliance: 0.9800
- Final checkpoint benchmark avg length: 447.2
- Late-collapse flag: True

## Selection Guidance

Compare only the two finalist families below.

- Raw reward is not directly comparable between the penalized and unpenalized families because the penalized objective subtracts length cost.
- Final winner selection should weigh held-out benchmark accuracy, repeatability across seeds, late collapse or survival, response-length and truncation behavior, throughput/runtime cost, and reward only in context.
- This report explicitly treats drift toward the 1024 token cap as a failure signal. A run is flagged when final or peak average response length reaches at least 95% of the cap.

| Finalist Family | Success | Final Ckpt Acc Mean+-Spread | In-Train Final Acc Mean+-Spread | In-Train Peak Acc Mean+-Spread | Late Collapse Flags | Cap-Drift Flags | Final Resp Len Mean+-Spread | Final Trunc Mean+-Spread | Tok/s Mean+-Spread | Reward Mean+-Spread (Context Only) |
|-----------------|---------|-----------------------------|----------------------------------|---------------------------------|--------------------|-----------------|-----------------------------|--------------------------|--------------------|------------------------------------|
| length_penalty_0 | 3/3 | 0.7267 +- 0.0340 | 0.7267 +- 0.0573 | 0.7400 +- 0.0432 | 3 | 3 | 693.6 +- 189.3 | 0.3333 +- 0.2357 | 65.56 +- 0.46 | 0.7500 +- 0.2041 |
| length_penalty_0005_mask_truncated_off | 3/3 | 0.7533 +- 0.0411 | 0.7333 +- 0.0340 | 0.7333 +- 0.0340 | 2 | 3 | 679.2 +- 227.3 | 0.2500 +- 0.2041 | 65.75 +- 0.24 | 0.4104 +- 0.3157 |

Recommended single winner for the later heavy run: `length_penalty_0005_mask_truncated_off`.

- Benchmark accuracy: `length_penalty_0005_mask_truncated_off` leads on the primary selector signal when compared against `length_penalty_0` using final-checkpoint held-out accuracy (0.7533 vs 0.7267).
- Repeatability and survival: `length_penalty_0005_mask_truncated_off` has 2 late-collapse flag(s) across 3 seed(s) versus 3 for `length_penalty_0`.
- Response-length drift: `length_penalty_0005_mask_truncated_off` has 3 cap-drift failure signal(s) versus 3 for `length_penalty_0`. Any drift toward the 1024-token cap is treated here as negative evidence, not as progress.
- Throughput/runtime context: `length_penalty_0005_mask_truncated_off` averages 65.75 tok/s versus 65.56 tok/s for `length_penalty_0`.
- Reward context only: keep reward in the report for within-family drift and collapse reading, but do not use raw reward alone to pick between the penalized and unpenalized finalists.
