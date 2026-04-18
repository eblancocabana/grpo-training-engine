# CDIA Final Validation Report

Generated: 2026-04-18 01:31:34
Plan: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_final_validation/cdia_final_validation_suite_plan.json
Trainer steps: 120
Total runs: 5
Successful: 5
Derived from benchmarks/cdia_finalist_followup_suite.py with the same output flow and results.json layout, but reduced to a 4-run strict finalist comparison plus 1 exploratory difficulty-weighting ablation.

## Methodology

- Strict finalist comparison: 4 runs, 2 finalist configs x 2 seeds, SENT disabled throughout to preserve apples-to-apples comparability with the existing finalist-followup evidence.
- Exploratory difficulty-weighting ablation: 1 run, separate from the strict comparison, with weighting enabled explicitly and SENT allowed only because the weighting signal is a SENT-rank difficulty proxy.

## Strict Finalist Comparison

| Config | Seed | SENT | Weighting | Status | Steps | Opt Steps | Tok/s | Reward Final | Reward Peak@Step | Drop | Resp Final | Resp Peak@Step | Trunc Avg | Masked-Out Avg |
|--------|------|------|-----------|--------|-------|-----------|-------|--------------|------------------|------|------------|----------------|-----------|----------------|
| length_penalty_0_seed1 | 1 | False | off | OK | 120 | 30 | 63.90 | 0.5000 | 1.0000@2 | 0.5000 | 999.2 | 1024.0@64 | 0.0979 | 0.0979 |
| length_penalty_0_seed2 | 2 | False | off | OK | 120 | 30 | 63.63 | 1.0000 | 1.0000@1 | 0.0000 | 493.0 | 1024.0@30 | 0.0979 | 0.0979 |
| length_penalty_0005_mask_truncated_off_seed1 | 1 | False | off | OK | 120 | 30 | 64.06 | 0.0277 | 0.8913@117 | 0.8635 | 944.5 | 1024.0@28 | 0.1021 | 0.0000 |
| length_penalty_0005_mask_truncated_off_seed2 | 2 | False | off | OK | 120 | 30 | 63.25 | 0.7592 | 0.8813@85 | 0.1220 | 481.5 | 1024.0@30 | 0.0896 | 0.0000 |

## Exploratory Difficulty-Weighting Ablation

| Config | Seed | SENT | Weighting | Status | Steps | Opt Steps | Tok/s | Reward Final | Reward Peak@Step | Drop | Resp Final | Resp Peak@Step | Trunc Avg | Masked-Out Avg |
|--------|------|------|-----------|--------|-------|-----------|-------|--------------|------------------|------|------------|----------------|-----------|----------------|
| length_penalty_0_difficulty_weighting_seed1 | 1 | True | sent_rank_linear | OK | 120 | 30 | 64.33 | 1.0000 | 1.0000@2 | 0.0000 | 371.8 | 1024.0@3 | 0.1250 | 0.1250 |

## Detailed Results

### length_penalty_0_seed1

- Success: True
- Duration: 4860.2s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_final_validation/cdia_length_penalty_0_seed1/train.log
- Comparison group: strict_finalist_comparison
- Seed: 1
- SENT enabled: False
- Difficulty weighting mode: off
- Step time: 40.08s (min: 21.32, max: 314.06)
- Throughput: 63.90 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7646
- Final reward: 0.5000
- Peak reward: 1.0000 at step 2
- Reward drop from peak to final: 0.5000
- Final reward std: 0.5000
- Whole-run reward std avg: 0.1433
- Reward range avg: [0.5833, 0.9000]
- Final avg response length: 999.2
- Peak avg response length: 1024.0 at step 64
- Avg response length at reward peak step 2: 388.0
- Response length drop from peak to final: 24.8
- Final response budget usage: 0.9758
- Entropy-masked ratio avg: 0.3442
- Legacy truncated-completions ratio avg (only when masking active): 0.0979
- Actual truncated completions ratio avg: 0.0979
- Peak actual truncated completions ratio: 1.0000 at step 64
- Final actual truncated completions ratio: 0.7500
- Truncated completions masked out of loss ratio avg: 0.0979
- Final truncated completions masked out of loss ratio: 0.7500
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.1104
- OOM backoff count final: 1

### length_penalty_0_seed2

- Success: True
- Duration: 4624.2s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_final_validation/cdia_length_penalty_0_seed2/train.log
- Comparison group: strict_finalist_comparison
- Seed: 2
- SENT enabled: False
- Difficulty weighting mode: off
- Step time: 38.34s (min: 20.13, max: 317.46)
- Throughput: 63.63 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7438
- Final reward: 1.0000
- Peak reward: 1.0000 at step 1
- Reward drop from peak to final: 0.0000
- Final reward std: 0.0000
- Whole-run reward std avg: 0.1216
- Reward range avg: [0.6000, 0.8667]
- Final avg response length: 493.0
- Peak avg response length: 1024.0 at step 30
- Avg response length at reward peak step 1: 278.8
- Response length drop from peak to final: 531.0
- Final response budget usage: 0.4814
- Entropy-masked ratio avg: 0.3471
- Legacy truncated-completions ratio avg (only when masking active): 0.0979
- Actual truncated completions ratio avg: 0.0979
- Peak actual truncated completions ratio: 1.0000 at step 30
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.0979
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.1062
- OOM backoff count final: 1

### length_penalty_0005_mask_truncated_off_seed1

- Success: True
- Duration: 4868.3s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_final_validation/cdia_length_penalty_0005_mask_truncated_off_seed1/train.log
- Comparison group: strict_finalist_comparison
- Seed: 1
- SENT enabled: False
- Difficulty weighting mode: off
- Step time: 40.17s (min: 20.97, max: 315.88)
- Throughput: 64.06 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 12.3798
- Whole-run reward avg: 0.5130
- Final reward: 0.0277
- Peak reward: 0.8913 at step 117
- Reward drop from peak to final: 0.8635
- Final reward std: 0.5427
- Whole-run reward std avg: 0.1864
- Reward range avg: [0.2688, 0.7033]
- Final avg response length: 944.5
- Peak avg response length: 1024.0 at step 28
- Avg response length at reward peak step 117: 217.5
- Response length drop from peak to final: 79.5
- Final response budget usage: 0.9224
- Entropy-masked ratio avg: 0.3906
- Actual truncated completions ratio avg: 0.1021
- Peak actual truncated completions ratio: 1.0000 at step 28
- Final actual truncated completions ratio: 0.7500
- Truncated completions masked out of loss ratio avg: 0.0000
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: False
- Positive advantages ratio avg: 0.5375
- OOM backoff count final: 1

### length_penalty_0005_mask_truncated_off_seed2

- Success: True
- Duration: 4652.5s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_final_validation/cdia_length_penalty_0005_mask_truncated_off_seed2/train.log
- Comparison group: strict_finalist_comparison
- Seed: 2
- SENT enabled: False
- Difficulty weighting mode: off
- Step time: 38.60s (min: 20.10, max: 319.93)
- Throughput: 63.25 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.2678
- Whole-run reward avg: 0.5048
- Final reward: 0.7592
- Peak reward: 0.8813 at step 85
- Reward drop from peak to final: 0.1220
- Final reward std: 0.0231
- Whole-run reward std avg: 0.1580
- Reward range avg: [0.3176, 0.6796]
- Final avg response length: 481.5
- Peak avg response length: 1024.0 at step 30
- Avg response length at reward peak step 85: 237.5
- Response length drop from peak to final: 542.5
- Final response budget usage: 0.4702
- Entropy-masked ratio avg: 0.3902
- Actual truncated completions ratio avg: 0.0896
- Peak actual truncated completions ratio: 1.0000 at step 30
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.0000
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: False
- Positive advantages ratio avg: 0.4875
- OOM backoff count final: 1

### length_penalty_0_difficulty_weighting_seed1

- Success: True
- Duration: 4579.7s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_final_validation/cdia_length_penalty_0_difficulty_weighting_seed1/train.log
- Comparison group: exploratory_difficulty_weighting
- Seed: 1
- SENT enabled: True
- Difficulty weighting mode: sent_rank_linear
- Difficulty weighting range: [1.000, 1.200]
- Step time: 38.01s (min: 20.25, max: 320.00)
- Throughput: 64.33 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7417
- Final reward: 1.0000
- Peak reward: 1.0000 at step 2
- Reward drop from peak to final: 0.0000
- Final reward std: 0.0000
- Whole-run reward std avg: 0.1241
- Reward range avg: [0.6000, 0.8750]
- Final avg response length: 371.8
- Peak avg response length: 1024.0 at step 3
- Avg response length at reward peak step 2: 275.0
- Response length drop from peak to final: 652.2
- Final response budget usage: 0.3630
- Entropy-masked ratio avg: 0.3351
- Legacy truncated-completions ratio avg (only when masking active): 0.1250
- Actual truncated completions ratio avg: 0.1250
- Peak actual truncated completions ratio: 1.0000 at step 3
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.1250
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.0958
- OOM backoff count final: 1
