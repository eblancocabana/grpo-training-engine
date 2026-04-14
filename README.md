# GRPO Training Engine for RTX 3060 Ti (8GB VRAM)

A native PyTorch training engine for GRPO (Group Relative Policy Optimization) specifically optimized for the NVIDIA RTX 3060 Ti with 8GB VRAM.

## 🎯 Key Features

- **Native 4-bit Quantization** using `bitsandbytes` (~0.85GB for 1.5B parameters)
- **Manual LoRA Implementation** without PEFT dependencies
- **GRPO from Scratch** without HF Trainer or TRL
- **Selective Backpropagation** based on token entropy
- **Curriculum Learning (SENT)**: Sorts training data by semantic entropy (easy to hard)
- **Surgical Memory Management** designed for 8GB VRAM constraints

## 📋 System Requirements

```
- GPU: NVIDIA RTX 3060 Ti (8GB VRAM)
- Python: 3.10+
- CUDA: 12.1
- PyTorch: 2.10.0+cu121
```

## 🚀 Quick Install

```bash
# 1. Activate conda environment
conda activate grpo-3060ti

# 2. Install dependencies
bash scripts/install_dependencies.sh

# 3. Verify installation
python train.py --dry-run
```

### Manual Installation

```bash
# PyTorch with CUDA 12.1
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install transformers accelerate bitsandbytes datasets scipy numpy tqdm wandb
```

## 📁 Project Structure

```
.
├── src/
│   ├── core/
│   │   ├── model_loader.py      # 4-bit model loading
│   │   ├── lora.py              # Manual LoRA implementation
│   │   └── memory_manager.py    # VRAM management
│   ├── grpo/
│   │   ├── algorithm.py         # GRPO loss & advantage calculation
│   │   ├── trainer.py           # Native training loop
│   │   └── verifier.py          # Response verification
│   ├── selective/
│   │   └── entropy_mask.py      # Entropy-based masking
│   ├── data/
│   │   ├── gsm8k_loader.py      # GSM8K Dataset loader (with Curriculum)
│   │   └── sent_calculator.py   # Semantic Entropy calculation
│   └── utils/
│       ├── config.py            # Configuration management
│       └── checkpoint.py        # Checkpoint save/load
├── train.py                     # Main training script
├── scripts/
│   ├── inference.py             # Post-training inference
│   ├── preprocess_sent_vllm.py  # SENT Preprocessing (vLLM optimized)
│   └── install_dependencies.sh  # Auto-installation script
├── tests/                   # Comprehensive Test Suite
│   ├── test_grpo_algorithm.py   # GRPO logic validation
│   ├── test_lora.py             # LoRA injection tests
│   ├── test_sent.py             # Curriculum Learning tests
│   ├── test_memory_manager.py   # VRAM constraints tests
│   └── ... (see tests/ for full list)
└── requirements.txt             # Dependencies
```

## 💻 Usage

### 1. Verify System

```bash
python train.py --dry-run
```

Runs the setup/configuration validation path without starting training.
Use `pytest tests/` for the full automated test suite.

### 2. Running Tests

The project includes a comprehensive test suite using `pytest`.

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_sent.py        # Test Curriculum Learning
pytest tests/test_grpo_algorithm.py # Validate GRPO Math
```

### 3. Preprocessing (SENT Curriculum)

To enable Curriculum Learning, you must generate the sorted dataset cache. We provide a vLLM-optimized script for speed:

```bash
# 1. Be sure to have vLLM installed (optional but recommended for speed)
pip install vllm

# 2. Run preprocessing (generates data/cache/gsm8k_sent_sorted.pt)
python scripts/preprocess_sent_vllm.py
```

> **Note:** If you skip this, training will proceed without Curriculum Learning (standard shuffling).

### 3. Training

**Basic training:**
```bash
python train.py
```

**With custom options (including WandB & Entropy):**
```bash
python train.py \
    --epochs 3 \
    --group-size 4 \
    --lora-rank 16 \
    --learning-rate 1e-4 \
    --use-entropy-mask \
    --wandb-project "grpo-experiment-1"
```

**Configuration Test (Dry Run):**
```bash
python train.py --dry-run
```

### 4. Inference

```bash
python scripts/inference.py \
    --lora-weights ./outputs/lora_weights_final.pt \
    --prompt "What is 15 + 27?" \
    --max-tokens 200
```

## ⚙️ Configuration for 8GB VRAM

The optimized configuration for RTX 3060 Ti is located in `src/utils/config.py::get_8gb_vram_config()`:

```python
# Model
- Model: DeepSeek-R1-Distill-Qwen-1.5B
- Quantization: 4-bit (NF4)
- Model Memory: ~0.85 GB

# LoRA
- Rank: 16
- Alpha: 32
- Target modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
- Trainable Params: ~30M (<0.05 GB)

# GRPO
- Group size: 4 (Safe default) - Up to 8 possible with strict memory management
- No KL divergence (VRAM saving)
- Clip epsilon: 0.2

# Training
- Batch size: 1
- Gradient accumulation: 4
- Sequence length: 512 (prompt) + 512 (response)
- Gradient checkpointing: Enabled
```

### VRAM Budget (group_size=8)

| Component | Memory (GB) |
|-----------|-------------|
| Model (4-bit) | ~0.85 |
| LoRA (trainable) | ~0.05 |
| Optimizer (AdamW) | ~0.10 |
| KV Cache | ~0.50 |
| Activations (8 responses) | ~4.5 |
| CUDA Overhead | ~0.5 |
| **Peak Total** | **~6.5** |
| **Safety Margin** | **~1.5** |

## 🔧 Key Components

### 1. Manual LoRA (`src/core/lora.py`)

Custom LoRA implementation without PEFT:

```python
Y = W_4bit(x) + B(A(x)) * (alpha / rank)
```

- **Base**: 4-bit quantized weights (frozen)
- **Adapters**: Matrices A and B in BF16 (trainable)
- **Gradients**: Flow only through A and B

### 2. GRPO Algorithm (`src/grpo/algorithm.py`)

Group Relative Policy Optimization:

```python
# No Value Network (VRAM saving)
Advantage_i = (r_i - mean(r_group)) / (std(r_group) + eps)

# GRPO Loss
loss = -E[min(ratio * A, clip(ratio) * A)]
```

**Advantages over PPO:**
- ~40-50% less VRAM (no Critic network)
- Simpler implementation
- Group baseline instead of Value function

### 3. Selective Backpropagation (`src/selective/entropy_mask.py`)

Entropy-based filtering:

```python
H(x) = -sum(p * log(p))  # Entropy per token
mask = H > threshold     # Keep uncertain tokens
loss = (loss * mask).mean()
```

**Benefits:**
- Blocks gradients for trivial tokens
- Accelerates training
- Focuses on "hard" tokens

### 5. Curriculum Learning (`src/data/sent_calculator.py`)

Implementation of **SENT (Semantic Entropy)**:
1. Sample `M` responses for each query (Temperature=1.0)
2. Cluster responses by semantic meaning (exact answer match)
3. Compute Entropy: `H = -sum(P(c) * log P(c))`
4. Sort dataset: Low Entropy (Easy) → High Entropy (Hard)
5. Train in stages (Curriculum)

### 6. Memory Manager (`src/core/memory_manager.py`)

Aggressive VRAM management:

```python
- Periodic torch.cuda.empty_cache()
- Gradient checkpointing
- Cleanup between phases (gen/train)
- Pre-allocated buffers
```

## 📊 Training Flow

```
1. GENERATION (torch.no_grad)
   ├── Load prompt batch
   ├── Expand to group_size (G=4-8)
   ├── Generate responses
   └── Verify responses → Rewards

2. ADVANTAGE CALCULATION
   ├── Group rewards by prompt
   ├── Normalize: (r - mean) / std
   └── Get advantages

3. TRAINING
   ├── Forward pass with generated tokens
   ├── Calculate entropy per token
   ├── Create selection mask
   ├── GRPO Loss + mask
   ├── Backward (only selected tokens)
   └── Optimizer step
```

## 🎛️ Configurable Parameters

### Training (`train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of epochs |
| `--group-size` | 4 | Responses per prompt |
| `--lora-rank` | 16 | LoRA Rank |
| `--learning-rate` | 1e-4 | Learning rate |
| `--use-entropy-mask` | True | Entropy-based filtering |
| `--use-triton` / `--no-triton` | True | Use Triton kernels (falls back to PyTorch if missing) |
| `--wandb` | True | Enable Weights & Biases logging |

### Advanced Configuration

Edit `src/utils/config.py`:

```python
# In get_8gb_vram_config()
config.training.max_response_length = 512  # Adjust based on VRAM
config.entropy.percentile = 0.5  # % tokens to keep
config.grpo.clip_epsilon = 0.2  # PPO clipping
```

### Triton Kernels

The project includes optional Triton kernels for optimized performance. By default, training attempts to use Triton kernels. If Triton is not installed, it falls back to pure PyTorch implementations automatically.

```bash
# Explicitly disable Triton kernels
python train.py --no-triton

# Explicitly enable (default)
python train.py --use-triton
```

> **Note:** If you encounter issues with Triton, use `--no-triton` to fall back to PyTorch.

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `CUDA out of memory` error

**Solutions:**
```python
# 1. Reduce group_size (Minimum recommended: 4)
config.grpo.group_size = 4

# 2. Reduce sequence length
config.training.max_response_length = 256

# 3. Reduce LoRA rank
config.lora.rank = 8

# 4. Increase cache cleanup frequency
config.training.clear_cache_frequency = 2
```

### SLOWNESS

**Symptoms:** Very slow training

**Common Causes:**
1. **Gradient checkpointing**: Saves VRAM but is slower
2. **High Group size**: More generations = more time
3. **CPU bottleneck**: Check GPU usage with `nvidia-smi`

## 📈 Monitoring

The training process uses **Weights & Biases (WandB)** by default.

**Terminal Output:**
```
[Step 100] VRAM: 5.2GB / 8.0GB (65.0%) | Free: 2.8GB
Epoch 1:  45%|████▌     | 450/1000 [12:30<15:20, loss=0.2341, reward=0.523]
```

**Important Metrics:**
- `loss`: GRPO Loss (should decrease)
- `reward`: Correct response rate (0-1)
- `advantage`: Average advantage
- `VRAM`: GPU Memory usage

## 💾 Checkpoints

Automatically saved in `./outputs/checkpoints/`:

```
checkpoints/
├── checkpoint_step_500.pt      # Full checkpoint
├── checkpoint_step_1000.pt
├── best_model.pt               # Best model
├── lora_weights_final.pt       # LoRA weights only
├── latest.json                 # Last checkpoint info
└── data/cache/                 # SENT Cache (generated)
    └── gsm8k_sent_sorted.pt    # Sorted dataset indices
```

**Load checkpoint:**
```python
from src.utils.checkpoint import CheckpointManager

manager = CheckpointManager("./outputs/checkpoints")
info = manager.load_checkpoint("checkpoint_step_1000.pt", model, optimizer)
```

## 🎯 RTX 3060 Ti Specific Optimizations

### Verified Optimal Configuration (group_size=8)

The following configuration has been verified as stable:

```bash
python train.py --no-wandb --epochs 1 --group-size 8
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `group_size` | 8 | Maximum stable without OOM |
| `max_response_length` | 512 | Allows full reasoning chains |
| `max_prompt_length` | 128 | Optimized for GSM8K |
| `lora_rank` | 16 | Quality/Memory balance |
| `gradient_accumulation` | 4 | Effective batch size = 4 |
| `learning_rate` | 1e-4 | Stable for GRPO |

#### Performance Metrics

| Config | Time/Step | Peak VRAM | Status |
|--------|-----------|-----------|--------|
| group_size=4 | ~16s | ~6.3GB | ✅ Stable |
| group_size=8 | ~33s | ~6.5GB | ✅ Stable, NO OOM |

### 1. Ampere Architecture

- **TF32 enabled**: `torch.backends.cuda.matmul.allow_tf32 = True`
- **cuDNN TF32**: `torch.backends.cudnn.allow_tf32 = True`
- **cuDNN Benchmark**: `torch.backends.cudnn.benchmark = True`
- Uses BF16 (natively supported, better performance than FP16)

### 2. 8GB VRAM Strategy

- Double quantization enabled
- No KL divergence (reference model saving)
- Group size up to 8 (verified stable)
- Micro-batching in generation

### 3. Gradient Checkpointing

- Trade compute ↔ memory
- Essential for 8GB
- Enabled by default

### 4. GRPO Algorithm Fix

To prevent `loss=0` when all rewards in a group are identical:
- `min_std = 0.1` clamp in advantage normalization
- Baseline advantage for uniform groups: `(mean_reward - 0.5) * 0.5`

## 📚 References

- **DeepSeek-R1**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **GRPO Paper**: DeepSeekMath (arXiv:2402.03300)
- **LoRA**: Low-Rank Adaptation of Large Language Models
- **bitsandbytes**: 8-bit & 4-bit quantization

## 🤝 Contributions

This project is part of a Double Degree Thesis (TFG):
- **Computer Engineering**: Efficient training engine
- **Data Science & AI**: GRPO algorithm and reasoning

## 📝 License

Academic project for TFG.

---

**Author**: Endika Blanco Cabana
**Hardware**: NVIDIA RTX 3060 Ti (8GB)
**Date**: February 2026
