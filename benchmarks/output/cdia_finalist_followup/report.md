# CDIA Finalist Follow-Up Report

Generated: 2026-04-17 05:18:21
Plan: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_finalist_followup_suite_plan.json
Trainer steps: 120
Total runs: 9
Successful: 9
Derived from benchmarks/cdia_suite.py with the same output flow but finalist-specific config coverage and stronger observability.

## Summary

| Config | Status | Steps | Opt Steps | Tok/s | Reward Final | Reward Peak@Step | Drop | Resp Final | Resp Peak@Step | Trunc Avg | Masked-Out Avg |
|--------|--------|-------|-----------|-------|--------------|------------------|------|------------|----------------|-----------|----------------|
| baseline | OK | 120 | 30 | 64.64 | 0.0505 | 0.7568@108 | 0.7063 | 449.5 | 1024.0@25 | 0.1146 | 0.1146 |
| length_penalty_0 | OK | 120 | 30 | 65.03 | 1.0000 | 1.0000@1 | 0.0000 | 402.8 | 1024.0@6 | 0.1292 | 0.1292 |
| length_penalty_00025 | OK | 120 | 30 | 64.02 | -0.0060 | 0.9562@35 | 0.9622 | 1024.0 | 1024.0@9 | 0.1021 | 0.1021 |
| length_penalty_0005 | OK | 120 | 30 | 64.86 | 0.4945 | 0.8825@87 | 0.3880 | 511.0 | 1024.0@16 | 0.1250 | 0.1250 |
| length_penalty_0005_batch_eff_32 | OK | 120 | 15 | 64.46 | -0.5120 | 0.8727@107 | 1.3847 | 1024.0 | 1024.0@25 | 0.0854 | 0.0854 |
| length_penalty_0005_eps_high_02 | OK | 120 | 30 | 63.72 | 0.7404 | 0.8855@56 | 0.1451 | 519.2 | 1024.0@6 | 0.0729 | 0.0729 |
| length_penalty_0_batch_eff_32 | OK | 120 | 15 | 63.80 | 1.0000 | 1.0000@1 | 0.0000 | 482.2 | 1024.0@13 | 0.0958 | 0.0958 |
| length_penalty_0_eps_high_02 | OK | 120 | 30 | 65.34 | 0.5000 | 1.0000@1 | 0.5000 | 928.0 | 1024.0@18 | 0.1167 | 0.1167 |
| length_penalty_0005_mask_truncated_off | OK | 120 | 30 | 64.62 | 0.8109 | 0.9095@111 | 0.0986 | 378.2 | 1024.0@70 | 0.0500 | 0.0000 |

## Detailed Results

### baseline

- Success: True
- Duration: 4785.4s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_baseline/train.log
- Step time: 39.94s (min: 22.51, max: 311.25)
- Throughput: 64.64 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 2.1628
- Whole-run reward avg: 0.2793
- Final reward: 0.0505
- Peak reward: 0.7568 at step 108
- Reward drop from peak to final: 0.7063
- Final reward std: 0.5129
- Whole-run reward std avg: 0.1688
- Reward range avg: [0.0644, 0.4655]
- Final avg response length: 449.5
- Peak avg response length: 1024.0 at step 25
- Avg response length at reward peak step 108: 243.2
- Response length drop from peak to final: 574.5
- Final response budget usage: 0.4390
- Entropy-masked ratio avg: 0.3422
- Legacy truncated-completions ratio avg (only when masking active): 0.1146
- Actual truncated completions ratio avg: 0.1146
- Peak actual truncated completions ratio: 1.0000 at step 25
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.1146
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.4437
- OOM backoff count final: 1

### length_penalty_0

- Success: True
- Duration: 4916.0s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0/train.log
- Step time: 40.07s (min: 21.57, max: 323.50)
- Throughput: 65.03 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7354
- Final reward: 1.0000
- Peak reward: 1.0000 at step 1
- Reward drop from peak to final: 0.0000
- Final reward std: 0.0000
- Whole-run reward std avg: 0.1391
- Reward range avg: [0.5667, 0.8750]
- Final avg response length: 402.8
- Peak avg response length: 1024.0 at step 6
- Avg response length at reward peak step 1: 454.5
- Response length drop from peak to final: 621.2
- Final response budget usage: 0.3933
- Entropy-masked ratio avg: 0.3392
- Legacy truncated-completions ratio avg (only when masking active): 0.1292
- Actual truncated completions ratio avg: 0.1292
- Peak actual truncated completions ratio: 1.0000 at step 6
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.1292
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.1167
- OOM backoff count final: 1

### length_penalty_00025

- Success: True
- Duration: 4644.6s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_00025/train.log
- Step time: 38.64s (min: 22.07, max: 301.04)
- Throughput: 64.02 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.6246
- Final reward: -0.0060
- Peak reward: 0.9562 at step 35
- Reward drop from peak to final: 0.9622
- Final reward std: 0.4330
- Whole-run reward std avg: 0.1353
- Reward range avg: [0.4632, 0.7681]
- Final avg response length: 1024.0
- Peak avg response length: 1024.0 at step 9
- Avg response length at reward peak step 35: 175.0
- Response length drop from peak to final: 0.0
- Final response budget usage: 1.0000
- Entropy-masked ratio avg: 0.3492
- Legacy truncated-completions ratio avg (only when masking active): 0.1021
- Actual truncated completions ratio avg: 0.1021
- Peak actual truncated completions ratio: 1.0000 at step 9
- Final actual truncated completions ratio: 1.0000
- Truncated completions masked out of loss ratio avg: 0.1021
- Final truncated completions masked out of loss ratio: 1.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.4437
- OOM backoff count final: 1

### length_penalty_0005

- Success: True
- Duration: 4527.4s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0005/train.log
- Step time: 37.56s (min: 19.31, max: 304.54)
- Throughput: 64.86 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 3.8006
- Whole-run reward avg: 0.5732
- Final reward: 0.4945
- Peak reward: 0.8825 at step 87
- Reward drop from peak to final: 0.3880
- Final reward std: 0.4496
- Whole-run reward std avg: 0.1137
- Reward range avg: [0.4235, 0.6907]
- Final avg response length: 511.0
- Peak avg response length: 1024.0 at step 16
- Avg response length at reward peak step 87: 235.0
- Response length drop from peak to final: 513.0
- Final response budget usage: 0.4990
- Entropy-masked ratio avg: 0.3416
- Legacy truncated-completions ratio avg (only when masking active): 0.1250
- Actual truncated completions ratio avg: 0.1250
- Peak actual truncated completions ratio: 1.0000 at step 16
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.1250
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.4583
- OOM backoff count final: 1

### length_penalty_0005_batch_eff_32

- Success: True
- Duration: 4663.2s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0005_batch_eff_32/train.log
- Step time: 38.48s (min: 21.38, max: 312.55)
- Throughput: 64.46 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 15
- Trainer steps per optimizer step: 8.00
- Final loss: 0.0000
- Whole-run reward avg: 0.5133
- Final reward: -0.5120
- Peak reward: 0.8727 at step 107
- Reward drop from peak to final: 1.3847
- Final reward std: 0.0000
- Whole-run reward std avg: 0.1367
- Reward range avg: [0.3342, 0.6572]
- Final avg response length: 1024.0
- Peak avg response length: 1024.0 at step 25
- Avg response length at reward peak step 107: 254.5
- Response length drop from peak to final: 0.0
- Final response budget usage: 1.0000
- Entropy-masked ratio avg: 0.3544
- Legacy truncated-completions ratio avg (only when masking active): 0.0854
- Actual truncated completions ratio avg: 0.0854
- Peak actual truncated completions ratio: 1.0000 at step 25
- Final actual truncated completions ratio: 1.0000
- Truncated completions masked out of loss ratio avg: 0.0854
- Final truncated completions masked out of loss ratio: 1.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.4771
- OOM backoff count final: 1

### length_penalty_0005_eps_high_02

- Success: True
- Duration: 4524.7s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0005_eps_high_02/train.log
- Step time: 37.38s (min: 17.51, max: 300.82)
- Throughput: 63.72 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.9157
- Whole-run reward avg: 0.5334
- Final reward: 0.7404
- Peak reward: 0.8855 at step 56
- Reward drop from peak to final: 0.1451
- Final reward std: 0.0428
- Whole-run reward std avg: 0.1494
- Reward range avg: [0.3503, 0.6935]
- Final avg response length: 519.2
- Peak avg response length: 1024.0 at step 6
- Avg response length at reward peak step 56: 229.0
- Response length drop from peak to final: 504.8
- Final response budget usage: 0.5071
- Entropy-masked ratio avg: 0.3597
- Legacy truncated-completions ratio avg (only when masking active): 0.0729
- Actual truncated completions ratio avg: 0.0729
- Peak actual truncated completions ratio: 1.0000 at step 6
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.0729
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.4875
- OOM backoff count final: 1

### length_penalty_0_batch_eff_32

- Success: True
- Duration: 4719.7s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0_batch_eff_32/train.log
- Step time: 39.46s (min: 18.76, max: 317.52)
- Throughput: 63.80 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 15
- Trainer steps per optimizer step: 8.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7542
- Final reward: 1.0000
- Peak reward: 1.0000 at step 1
- Reward drop from peak to final: 0.0000
- Final reward std: 0.0000
- Whole-run reward std avg: 0.1397
- Reward range avg: [0.5750, 0.8833]
- Final avg response length: 482.2
- Peak avg response length: 1024.0 at step 13
- Avg response length at reward peak step 1: 600.2
- Response length drop from peak to final: 541.8
- Final response budget usage: 0.4709
- Entropy-masked ratio avg: 0.3491
- Legacy truncated-completions ratio avg (only when masking active): 0.0958
- Actual truncated completions ratio avg: 0.0958
- Peak actual truncated completions ratio: 1.0000 at step 13
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.0958
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.1083
- OOM backoff count final: 1

### length_penalty_0_eps_high_02

- Success: True
- Duration: 4649.5s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0_eps_high_02/train.log
- Step time: 38.35s (min: 22.78, max: 317.95)
- Throughput: 65.34 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0000
- Whole-run reward avg: 0.7354
- Final reward: 0.5000
- Peak reward: 1.0000 at step 1
- Reward drop from peak to final: 0.5000
- Final reward std: 0.5000
- Whole-run reward std avg: 0.1235
- Reward range avg: [0.6083, 0.8833]
- Final avg response length: 928.0
- Peak avg response length: 1024.0 at step 18
- Avg response length at reward peak step 1: 428.2
- Response length drop from peak to final: 96.0
- Final response budget usage: 0.9062
- Entropy-masked ratio avg: 0.3414
- Legacy truncated-completions ratio avg (only when masking active): 0.1167
- Actual truncated completions ratio avg: 0.1167
- Peak actual truncated completions ratio: 1.0000 at step 18
- Final actual truncated completions ratio: 0.5000
- Truncated completions masked out of loss ratio avg: 0.1167
- Final truncated completions masked out of loss ratio: 0.5000
- Truncation masking active in loss: True
- Positive advantages ratio avg: 0.0875
- OOM backoff count final: 1

### length_penalty_0005_mask_truncated_off

- Success: True
- Duration: 4468.4s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia_finalist_followup/cdia_length_penalty_0005_mask_truncated_off/train.log
- Step time: 36.99s (min: 20.61, max: 315.53)
- Throughput: 64.62 tok/s
- VRAM peak: 7.60 GB
- Optimizer steps completed: 30
- Trainer steps per optimizer step: 4.00
- Final loss: 0.0132
- Whole-run reward avg: 0.5823
- Final reward: 0.8109
- Peak reward: 0.9095 at step 111
- Reward drop from peak to final: 0.0986
- Final reward std: 0.0051
- Whole-run reward std avg: 0.1365
- Reward range avg: [0.4081, 0.7177]
- Final avg response length: 378.2
- Peak avg response length: 1024.0 at step 70
- Avg response length at reward peak step 111: 181.0
- Response length drop from peak to final: 645.8
- Final response budget usage: 0.3694
- Entropy-masked ratio avg: 0.3964
- Actual truncated completions ratio avg: 0.0500
- Peak actual truncated completions ratio: 1.0000 at step 70
- Final actual truncated completions ratio: 0.0000
- Truncated completions masked out of loss ratio avg: 0.0000
- Final truncated completions masked out of loss ratio: 0.0000
- Truncation masking active in loss: False
- Positive advantages ratio avg: 0.5229
- OOM backoff count final: 1
