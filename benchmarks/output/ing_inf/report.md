# ING INF Benchmark Report

Generated: 2026-04-09 10:57:47
Total runs: 37
Successful: 34

## Summary

| Config | Status | Step Time (s) | Tok/s | VRAM (GB) |
|--------|--------|---------------|-------|-----------|
| triton_on_baseline | ✅ | 74.99 | 17.89 | 7.70 |
| triton_on_current | ✅ | 78.29 | 17.90 | 7.70 |
| triton_off_baseline | ✅ | 153.23 | 11.00 | 7.70 |
| triton_off_current | ✅ | 147.02 | 11.29 | 7.70 |
| triton_on_grad_accum_1 | ✅ | 80.27 | 17.84 | 7.70 |
| triton_off_grad_accum_1 | ✅ | 158.00 | 10.99 | 7.70 |
| triton_on_grad_accum_2 | ✅ | 79.24 | 18.37 | 7.70 |
| triton_off_grad_accum_2 | ❌ | 161.63 | 11.32 | 7.70 |
| triton_on_grad_accum_4 | ✅ | 77.57 | 17.97 | 7.70 |
| triton_off_grad_accum_4 | ✅ | 155.32 | 11.04 | 7.70 |
| triton_on_grad_accum_8 | ✅ | 82.42 | 18.42 | 7.70 |
| triton_off_grad_accum_8 | ✅ | 157.07 | 11.24 | 7.70 |
| triton_on_grad_accum_16 | ✅ | 79.27 | 17.98 | 7.70 |
| triton_off_grad_accum_16 | ✅ | 161.30 | 10.99 | 7.70 |
| triton_on_grad_accum_32 | ✅ | 81.57 | 18.08 | 7.70 |
| triton_off_grad_accum_32 | ✅ | 158.76 | 10.99 | 7.70 |
| triton_on_group_size_2 | ✅ | 26.30 | 24.87 | 7.70 |
| triton_on_group_size_4 | ✅ | 77.64 | 18.25 | 7.70 |
| triton_on_group_size_8 | ✅ | 152.73 | 18.47 | 7.70 |
| triton_on_group_size_16 | ❌ | 261.47 | 18.41 | 7.70 |
| triton_on_lora_rank_4 | ✅ | 110.00 | 13.01 | 7.70 |
| triton_on_lora_rank_8 | ✅ | 105.33 | 12.99 | 7.70 |
| triton_on_lora_rank_16 | ✅ | 74.83 | 17.59 | 7.70 |
| triton_on_lora_rank_32 | ✅ | 79.31 | 17.63 | 7.70 |
| triton_on_lora_rank_64 | ✅ | 54.19 | 27.14 | 7.70 |
| triton_on_response_len_256 | ✅ | 16.44 | 46.88 | 7.70 |
| triton_on_response_len_512 | ✅ | 40.34 | 29.20 | 7.70 |
| triton_on_response_len_768 | ✅ | 74.47 | 17.55 | 7.70 |
| triton_on_response_len_1024 | ✅ | 76.50 | 17.72 | 7.70 |
| triton_on_entropy_mask_on | ✅ | 77.71 | 17.92 | 7.70 |
| triton_on_entropy_mask_off | ✅ | 52.19 | 27.43 | 7.70 |
| triton_on_checkpointing_on | ✅ | 76.36 | 17.60 | 7.70 |
| triton_on_checkpointing_off | ✅ | 78.90 | 18.09 | 7.70 |
| triton_on_lora_quant_4bit | ❌ | N/A | N/A | 7.70 |
| triton_on_lora_quant_8bit | ✅ | 79.05 | 18.18 | 7.70 |
| triton_on_lora_quant_none | ✅ | 44.61 | 30.28 | 7.70 |
| triton_on_profile | ✅ | 54.00 | 27.72 | 7.70 |

## Detailed Results

### triton_on_baseline

- Success: True
- Duration: 1675.2s
- Steps: 20
- Step time: 74.99s (min: 59.62, max: 99.08)
- Throughput: 17.89 tok/s
- VRAM peak: 7.70 GB

### triton_on_current

- Success: True
- Duration: 1753.0s
- Steps: 20
- Step time: 78.29s (min: 60.47, max: 114.21)
- Throughput: 17.90 tok/s
- VRAM peak: 7.70 GB

### triton_off_baseline

- Success: True
- Duration: 3387.3s
- Steps: 20
- Step time: 153.23s (min: 117.21, max: 219.88)
- Throughput: 11.00 tok/s
- VRAM peak: 7.70 GB

### triton_off_current

- Success: True
- Duration: 3295.3s
- Steps: 20
- Step time: 147.02s (min: 114.31, max: 207.88)
- Throughput: 11.29 tok/s
- VRAM peak: 7.70 GB

### triton_on_grad_accum_1

- Success: True
- Duration: 1781.0s
- Steps: 20
- Step time: 80.27s (min: 61.43, max: 110.36)
- Throughput: 17.84 tok/s
- VRAM peak: 7.70 GB

### triton_off_grad_accum_1

- Success: True
- Duration: 3520.7s
- Steps: 20
- Step time: 158.00s (min: 116.96, max: 253.91)
- Throughput: 10.99 tok/s
- VRAM peak: 7.70 GB

### triton_on_grad_accum_2

- Success: True
- Duration: 1797.9s
- Steps: 20
- Step time: 79.24s (min: 59.31, max: 119.68)
- Throughput: 18.37 tok/s
- VRAM peak: 7.70 GB

### triton_off_grad_accum_2

- Success: False
- Duration: 3600.0s
- Steps: 19
- Step time: 161.63s (min: 114.17, max: 251.69)
- Throughput: 11.32 tok/s
- VRAM peak: 7.70 GB

### triton_on_grad_accum_4

- Success: True
- Duration: 1742.7s
- Steps: 20
- Step time: 77.57s (min: 58.91, max: 115.00)
- Throughput: 17.97 tok/s
- VRAM peak: 7.70 GB

### triton_off_grad_accum_4

- Success: True
- Duration: 3446.1s
- Steps: 20
- Step time: 155.32s (min: 117.22, max: 231.23)
- Throughput: 11.04 tok/s
- VRAM peak: 7.70 GB

### triton_on_grad_accum_8

- Success: True
- Duration: 1835.4s
- Steps: 20
- Step time: 82.42s (min: 60.45, max: 121.87)
- Throughput: 18.42 tok/s
- VRAM peak: 7.70 GB

### triton_off_grad_accum_8

- Success: True
- Duration: 3487.0s
- Steps: 20
- Step time: 157.07s (min: 114.76, max: 232.20)
- Throughput: 11.24 tok/s
- VRAM peak: 7.70 GB

### triton_on_grad_accum_16

- Success: True
- Duration: 1769.1s
- Steps: 20
- Step time: 79.27s (min: 59.54, max: 117.83)
- Throughput: 17.98 tok/s
- VRAM peak: 7.70 GB

### triton_off_grad_accum_16

- Success: True
- Duration: 3598.7s
- Steps: 20
- Step time: 161.30s (min: 116.36, max: 236.88)
- Throughput: 10.99 tok/s
- VRAM peak: 7.70 GB

### triton_on_grad_accum_32

- Success: True
- Duration: 1807.7s
- Steps: 20
- Step time: 81.57s (min: 60.95, max: 114.64)
- Throughput: 18.08 tok/s
- VRAM peak: 7.70 GB

### triton_off_grad_accum_32

- Success: True
- Duration: 3545.6s
- Steps: 20
- Step time: 158.76s (min: 120.39, max: 232.11)
- Throughput: 10.99 tok/s
- VRAM peak: 7.70 GB

### triton_on_group_size_2

- Success: True
- Duration: 617.5s
- Steps: 20
- Step time: 26.30s (min: 21.36, max: 36.03)
- Throughput: 24.87 tok/s
- VRAM peak: 7.70 GB

### triton_on_group_size_4

- Success: True
- Duration: 1729.6s
- Steps: 20
- Step time: 77.64s (min: 61.52, max: 110.68)
- Throughput: 18.25 tok/s
- VRAM peak: 7.70 GB

### triton_on_group_size_8

- Success: True
- Duration: 3392.2s
- Steps: 20
- Step time: 152.73s (min: 117.84, max: 218.75)
- Throughput: 18.47 tok/s
- VRAM peak: 7.70 GB

### triton_on_group_size_16

- Success: False
- Duration: 3600.0s
- Steps: 11
- Step time: 261.47s (min: 231.78, max: 316.83)
- Throughput: 18.41 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_rank_4

- Success: True
- Duration: 2433.4s
- Steps: 20
- Step time: 110.00s (min: 85.12, max: 153.33)
- Throughput: 13.01 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_rank_8

- Success: True
- Duration: 2353.6s
- Steps: 20
- Step time: 105.33s (min: 84.04, max: 143.40)
- Throughput: 12.99 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_rank_16

- Success: True
- Duration: 1677.4s
- Steps: 20
- Step time: 74.83s (min: 61.13, max: 107.70)
- Throughput: 17.59 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_rank_32

- Success: True
- Duration: 1783.3s
- Steps: 20
- Step time: 79.31s (min: 60.28, max: 119.96)
- Throughput: 17.63 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_rank_64

- Success: True
- Duration: 1331.2s
- Steps: 20
- Step time: 54.19s (min: 24.52, max: 100.97)
- Throughput: 27.14 tok/s
- VRAM peak: 7.70 GB

### triton_on_response_len_256

- Success: True
- Duration: 340.6s
- Steps: 20
- Step time: 16.44s (min: 15.62, max: 18.47)
- Throughput: 46.88 tok/s
- VRAM peak: 7.70 GB

### triton_on_response_len_512

- Success: True
- Duration: 877.9s
- Steps: 20
- Step time: 40.34s (min: 29.93, max: 45.99)
- Throughput: 29.20 tok/s
- VRAM peak: 7.70 GB

### triton_on_response_len_768

- Success: True
- Duration: 1669.6s
- Steps: 20
- Step time: 74.47s (min: 54.22, max: 96.59)
- Throughput: 17.55 tok/s
- VRAM peak: 7.70 GB

### triton_on_response_len_1024

- Success: True
- Duration: 1713.1s
- Steps: 20
- Step time: 76.50s (min: 60.43, max: 111.14)
- Throughput: 17.72 tok/s
- VRAM peak: 7.70 GB

### triton_on_entropy_mask_on

- Success: True
- Duration: 1740.0s
- Steps: 20
- Step time: 77.71s (min: 61.20, max: 108.75)
- Throughput: 17.92 tok/s
- VRAM peak: 7.70 GB

### triton_on_entropy_mask_off

- Success: True
- Duration: 1272.7s
- Steps: 20
- Step time: 52.19s (min: 25.11, max: 100.46)
- Throughput: 27.43 tok/s
- VRAM peak: 7.70 GB

### triton_on_checkpointing_on

- Success: True
- Duration: 1714.1s
- Steps: 20
- Step time: 76.36s (min: 59.31, max: 110.31)
- Throughput: 17.60 tok/s
- VRAM peak: 7.70 GB

### triton_on_checkpointing_off

- Success: True
- Duration: 1754.6s
- Steps: 20
- Step time: 78.90s (min: 60.82, max: 115.55)
- Throughput: 18.09 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_quant_4bit

- Success: False
- Duration: 7.7s
- Steps: 0
- VRAM peak: 7.70 GB
- Error: ValueError: optimizer got an empty parameter list

### triton_on_lora_quant_8bit

- Success: True
- Duration: 1764.6s
- Steps: 20
- Step time: 79.05s (min: 60.70, max: 110.18)
- Throughput: 18.18 tok/s
- VRAM peak: 7.70 GB

### triton_on_lora_quant_none

- Success: True
- Duration: 1117.9s
- Steps: 20
- Step time: 44.61s (min: 26.05, max: 81.25)
- Throughput: 30.28 tok/s
- VRAM peak: 7.70 GB

### triton_on_profile

- Success: True
- Duration: 1312.5s
- Steps: 20
- Step time: 54.00s (min: 24.06, max: 101.00)
- Throughput: 27.72 tok/s
- VRAM peak: 7.70 GB
