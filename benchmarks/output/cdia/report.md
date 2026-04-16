# CDIA Quick-Screen Report

Generated: 2026-04-16 12:21:24
Plan: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_suite_plan.json
Trainer steps: 120
Total runs: 10
Successful: 10

## Summary

| Config | Status | Steps | Step Time (s) | Tok/s | VRAM (GB) | Reward | Reward Std | Avg Resp Len | Trunc Ratio |
|--------|--------|-------|---------------|-------|-----------|--------|------------|--------------|-------------|
| baseline | OK | 120 | 36.62 | 65.75 | 7.60 | 0.5752 | 0.0518 | 424.8 | 0.0813 |
| lr_5e-5 | OK | 120 | 39.08 | 63.75 | 7.60 | 0.3640 | 0.4624 | 386.0 | 0.1250 |
| batch_eff_32 | OK | 120 | 38.55 | 64.50 | 7.60 | 0.6890 | 0.0017 | 311.0 | 0.0708 |
| entropy_mask_off | OK | 120 | 38.63 | 63.65 | 7.60 | 0.1500 | 0.4070 | 600.0 | 0.0854 |
| entropy_pct_03 | OK | 120 | 41.47 | 65.69 | 7.60 | 0.3762 | 0.0114 | 623.8 | 0.1542 |
| entropy_pct_07 | OK | 120 | 39.15 | 64.56 | 7.60 | 0.3320 | 0.4126 | 418.0 | 0.1146 |
| eps_high_02 | OK | 120 | 38.51 | 66.09 | 7.60 | 0.5340 | 0.0046 | 466.0 | 0.0979 |
| length_penalty_0005 | OK | 120 | 38.56 | 64.64 | 7.60 | 0.7694 | 0.0186 | 461.2 | 0.0854 |
| length_penalty_002 | OK | 120 | 39.79 | 64.07 | 7.60 | -1.4510 | 0.1705 | 725.5 | 0.0979 |
| mask_truncated_off | OK | 120 | 39.27 | 66.47 | 7.60 | 0.6920 | 0.0225 | 308.0 | N/A |

## Detailed Results

### baseline

- Success: True
- Duration: 4419.9s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_baseline/train.log
- Step time: 36.62s (min: 20.42, max: 304.53)
- Throughput: 65.75 tok/s
- VRAM peak: 7.60 GB
- Final loss: 1.3453
- Final reward: 0.5752
- Final reward std: 0.0518
- Reward range avg: [0.1278, 0.4939]
- Final avg response length: 424.8
- Final response budget usage: 0.4148
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.0813
- Positive advantages ratio avg: 0.4708
- OOM backoff count final: 1

### lr_5e-5

- Success: True
- Duration: 4701.2s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_lr_5e-5/train.log
- Step time: 39.08s (min: 18.94, max: 312.69)
- Throughput: 63.75 tok/s
- VRAM peak: 7.60 GB
- Final loss: 3.5075
- Final reward: 0.3640
- Final reward std: 0.4624
- Reward range avg: [-0.0391, 0.4636]
- Final avg response length: 386.0
- Final response budget usage: 0.3770
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.1250
- Positive advantages ratio avg: 0.4562
- OOM backoff count final: 1

### batch_eff_32

- Success: True
- Duration: 4670.4s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_batch_eff_32/train.log
- Step time: 38.55s (min: 21.83, max: 314.23)
- Throughput: 64.50 tok/s
- VRAM peak: 7.60 GB
- Final loss: 0.0017
- Final reward: 0.6890
- Final reward std: 0.0017
- Reward range avg: [0.0407, 0.4875]
- Final avg response length: 311.0
- Final response budget usage: 0.3037
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.0708
- Positive advantages ratio avg: 0.4792
- OOM backoff count final: 1

### entropy_mask_off

- Success: True
- Duration: 4675.8s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_entropy_mask_off/train.log
- Step time: 38.63s (min: 21.49, max: 312.11)
- Throughput: 63.65 tok/s
- VRAM peak: 7.60 GB
- Final loss: -1.1893
- Final reward: 0.1500
- Final reward std: 0.4070
- Reward range avg: [0.0584, 0.4574]
- Final avg response length: 600.0
- Final response budget usage: 0.5859
- Truncated completions ratio avg: 0.0854
- Positive advantages ratio avg: 0.4646
- OOM backoff count final: 1

### entropy_pct_03

- Success: True
- Duration: 5006.8s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_entropy_pct_03/train.log
- Step time: 41.47s (min: 23.95, max: 329.17)
- Throughput: 65.69 tok/s
- VRAM peak: 7.60 GB
- Final loss: 0.0314
- Final reward: 0.3762
- Final reward std: 0.0114
- Reward range avg: [-0.0050, 0.4149]
- Final avg response length: 623.8
- Final response budget usage: 0.6091
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.1542
- Positive advantages ratio avg: 0.4396
- OOM backoff count final: 1

### entropy_pct_07

- Success: True
- Duration: 4702.1s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_entropy_pct_07/train.log
- Step time: 39.15s (min: 21.03, max: 318.96)
- Throughput: 64.56 tok/s
- VRAM peak: 7.60 GB
- Final loss: -2.0528
- Final reward: 0.3320
- Final reward std: 0.4126
- Reward range avg: [0.0442, 0.4348]
- Final avg response length: 418.0
- Final response budget usage: 0.4082
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.1146
- Positive advantages ratio avg: 0.4625
- OOM backoff count final: 1

### eps_high_02

- Success: True
- Duration: 4611.7s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_eps_high_02/train.log
- Step time: 38.51s (min: 21.82, max: 307.01)
- Throughput: 66.09 tok/s
- VRAM peak: 7.60 GB
- Final loss: 0.0102
- Final reward: 0.5340
- Final reward std: 0.0046
- Reward range avg: [0.0790, 0.4514]
- Final avg response length: 466.0
- Final response budget usage: 0.4551
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.0979
- Positive advantages ratio avg: 0.4625
- OOM backoff count final: 1

### length_penalty_0005

- Success: True
- Duration: 4645.4s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_length_penalty_0005/train.log
- Step time: 38.56s (min: 21.93, max: 322.18)
- Throughput: 64.64 tok/s
- VRAM peak: 7.60 GB
- Final loss: 0.3469
- Final reward: 0.7694
- Final reward std: 0.0186
- Reward range avg: [0.3317, 0.6613]
- Final avg response length: 461.2
- Final response budget usage: 0.4504
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.0854
- Positive advantages ratio avg: 0.4688
- OOM backoff count final: 1

### length_penalty_002

- Success: True
- Duration: 4821.2s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_length_penalty_002/train.log
- Step time: 39.79s (min: 22.11, max: 316.41)
- Throughput: 64.07 tok/s
- VRAM peak: 7.60 GB
- Final loss: 3.6579
- Final reward: -1.4510
- Final reward std: 0.1705
- Reward range avg: [-0.5236, 0.0071]
- Final avg response length: 725.5
- Final response budget usage: 0.7085
- Entropy-masked ratio avg: 0.0000
- Truncated completions ratio avg: 0.0979
- Positive advantages ratio avg: 0.4500
- OOM backoff count final: 1

### mask_truncated_off

- Success: True
- Duration: 4688.7s
- Steps completed: 120
- Log: /home/ndk/proyectos/clase/pfg/benchmarks/output/cdia/cdia_mask_truncated_off/train.log
- Step time: 39.27s (min: 20.45, max: 326.41)
- Throughput: 66.47 tok/s
- VRAM peak: 7.60 GB
- Final loss: 0.1258
- Final reward: 0.6920
- Final reward std: 0.0225
- Reward range avg: [0.0479, 0.4066]
- Final avg response length: 308.0
- Final response budget usage: 0.3008
- Entropy-masked ratio avg: 0.0000
- Positive advantages ratio avg: 0.4667
- OOM backoff count final: 1
