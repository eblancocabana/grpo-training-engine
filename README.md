# GRPO Training Engine

Native PyTorch training and evaluation stack for GRPO-style reasoning-model
fine-tuning on constrained NVIDIA GPUs.

The project was built around an RTX 3060 Ti with 8 GB of VRAM and
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. The point is to train a small
reasoning model while keeping the awkward parts visible: memory pressure,
rollout cost, curriculum choices, benchmark selection, and recovery after OOMs.
Those details are easy to lose inside a high-level trainer.

## What this repo contains

- A native GRPO training loop without HF Trainer, TRL, or PEFT.
- 4-bit base-model loading through `bitsandbytes`.
- Manual LoRA adapters over the Q/K/V/O projections.
- Entropy-based selective backpropagation.
- SENT curriculum support based on semantic-entropy ordering.
- Split-aware math dataset loading for GSM8K, GSM-Plus, Open-RS,
  DAPO-MATH-17K, and Open-DeepScaler style workflows.
- Optional Triton kernels for selected hot paths.
- Replay-aware OOM recovery and adaptive micro-batching for low-VRAM runs.
- Benchmark harnesses, optimizer tooling, VRAM profiling, and vLLM reasoning
  evaluation for base/final/best-checkpoint comparisons.

## Thesis context

This repository supports an integrated double-degree final project with two
separate readings of the same system.

The full final project reports are available under `docs/thesis/`:
[Computer Engineering](docs/thesis/INF_Endika_Blanco_Cabana.pdf) and
[Data Science and Artificial Intelligence](docs/thesis/CDIA_Endika_Blanco_Cabana.pdf).

### Computer engineering

The computer engineering side is the constrained-system training engine:
explicit VRAM budgeting, 4-bit loading, manual LoRA injection, phase-aware
memory cleanup, checkpoint/retry behavior, Triton/PyTorch execution choices,
profiler support, and benchmark-backed defaults for a consumer 8 GB GPU.

### Data science and artificial intelligence

The data science and artificial intelligence side is the GRPO reasoning
experiment: reward and verifier design, grouped rollout behavior, clipping and
length-control choices, SENT curriculum experiments, dataset selection,
validation methodology, and post-training reasoning evaluation across math
benchmarks.

## Hardware and software

The default configuration is tuned for this environment:

```text
GPU: NVIDIA RTX 3060 Ti, 8 GB VRAM
CUDA: 12.1-class environment
Python: 3.10+
Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Other CUDA-capable NVIDIA GPUs may work, but memory-related defaults may need
adjustment. CPU-only execution is not a primary target.

## Install

```bash
conda activate grpo-3060ti
bash scripts/install_dependencies.sh
python train.py --dry-run
```

Manual install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`vllm` is optional for core training, but required for the vLLM SENT
preprocessing path and the standalone reasoning-evaluation pipeline.

## Quick start

Run a short configuration and system check:

```bash
python train.py --dry-run
```

Run a small smoke training job without Weights & Biases or the initial
benchmark:

```bash
python train.py \
  --no-wandb \
  --no-initial-benchmark \
  --max-steps 5 \
  --group-size 4 \
  --max-response-length 768
```

Run the current 8 GB preset:

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

The benchmark results favor selective Triton use. Keep Triton for GRPO loss,
entropy masking, and LoRA forward paths, but leave Triton generation off.
`group_size=4` and `max_response_length=768` are the safest defaults for the
8 GB target.

## Training options

Common training flags:

```bash
python train.py \
  --dataset-name gsm8k \
  --epochs 3 \
  --group-size 4 \
  --lora-rank 16 \
  --learning-rate 1e-4 \
  --max-response-length 768 \
  --use-entropy-mask \
  --log-metrics-jsonl \
  --output-dir ./outputs/run_name
```

Useful switches:

- `--no-sent`: disable SENT curriculum ordering.
- `--sent-stage N`: train on a fixed curriculum stage.
- `--no-mask-truncated`: keep truncated completions in the loss.
- `--length-penalty-coef X`: apply a response-length penalty before advantage
  calculation.
- `--temperature X` and `--top-p X`: control rollout sampling.
- `--resume` or `--resume-checkpoint PATH`: continue from a checkpoint.
- `--profile`: enable the live profiler server and profiler hooks.
- `--no-triton`: force the pure PyTorch path.

## SENT preprocessing

SENT orders training examples by semantic entropy. The vLLM path is the
faster preprocessing route when available:

```bash
python scripts/preprocess_sent_vllm.py \
  --dataset-name gsm8k \
  --cache-path data/cache/gsm8k_sent_sorted.pt \
  --M 4 \
  --temperature 1.0 \
  --batch-size 32
```

For non-GSM8K datasets, set `--dataset-name` and use a dataset-specific cache
path. Training resolves compatible SENT caches from dataset metadata when
possible.

## Reasoning evaluation

Use the standalone vLLM evaluator to compare the base model, final LoRA
adapters, and validation-selected best checkpoints:

```bash
python scripts/evaluate_reasoning_vllm.py \
  --models-config configs/models.example.yaml \
  --datasets gsm8k,gsm-plus,math500,aime24 \
  --tier strong \
  --protocol sampled \
  --n-samples 8 \
  --max-new-tokens 8192 \
  --output-dir benchmarks/output/reasoning_eval
```

Model entries live in YAML files under `configs/`. Best checkpoints should be
selected from in-training validation, not from the final benchmark suite.

## Project structure

```text
.
├── train.py                         # Main training CLI
├── src/
│   ├── core/                        # Model loading, LoRA, memory manager
│   ├── data/                        # GSM8K and generic math dataset loading
│   ├── grpo/                        # GRPO algorithm, verifier, trainer, benchmark
│   ├── reasoning_eval/              # vLLM evaluation pipeline
│   ├── selective/                   # Entropy mask logic
│   ├── triton_kernels/              # Optional Triton kernels
│   └── utils/                       # Config, checkpoints, logging
├── scripts/                         # Inference, SENT, filtering, evaluation
├── benchmarks/                      # Bounded benchmark suites
├── optimizer/                       # Benchmark-authoritative optimizer loop
├── tools/                           # Profiling and comparison tools
├── tests/                           # Unit, integration, Triton, SENT, optimizer tests
└── configs/                         # Reasoning-evaluation model configs
```

## Core design

The training loop expands each prompt into a group of sampled completions,
verifies each answer with a rule-based verifier, centers rewards inside the
group, and optimizes LoRA adapter weights with a clipped GRPO objective. The
base model remains quantized and frozen.

```text
prompt -> grouped generation -> verification -> centered advantages
       -> old log-probs -> GRPO loss -> optional entropy mask -> LoRA update
```

The low-VRAM setup relies on these choices:

- 4-bit frozen base model.
- BF16 LoRA adapters only on selected projection layers.
- No value network.
- Response-length caps and grouped rollout limits.
- Gradient checkpointing.
- Explicit cleanup between generation, scoring, and training.
- Adaptive micro-batching and retry behavior after recoverable OOMs.

## Outputs

Training runs write artifacts under the configured `--output-dir`:

```text
outputs/
├── checkpoints/                 # Full checkpoints and latest metadata
├── logs/                        # Runtime logs
├── metrics.jsonl                # Optional structured metrics
├── lora_weights_final.pt        # Final adapter weights
└── baseline_benchmark_done.json # Optional initial benchmark marker
```

Benchmark, reasoning-evaluation, optimizer, SENT, and profiler workflows write
their own outputs under `benchmarks/output/`, `optimizer/artifacts/`,
`data/cache/`, and profiler-specific directories.

Large full checkpoints are not suitable for regular Git blobs. Keep them local
or publish them through an artifact store or Git LFS.

## Testing

```bash
pytest tests/
```

Focused subsets:

```bash
pytest tests/test_grpo_algorithm.py
pytest tests/test_sent.py
pytest tests/test_triton_generation_correctness.py
pytest tests/optimizer/
pytest tests/evaluation/
```

## Inference

```bash
python scripts/inference.py \
  --lora-weights ./outputs/lora_weights_final.pt \
  --prompt "What is 15 + 27?" \
  --max-tokens 200 \
  --temperature 0.7
```

## Limitations

- The repository is optimized for NVIDIA CUDA hardware, especially an 8 GB RTX
  3060 Ti. Other GPUs may require different batch, response-length, or
  micro-batch settings.
- vLLM workflows require a compatible vLLM installation and enough memory for
  the selected evaluation configuration.
- SENT and reward-design results are experimental. The strongest claims should
  be made from benchmark outputs, validation-selected checkpoints, and matched
  comparisons rather than from a single final training checkpoint.

## References

- DeepSeek-R1-Distill-Qwen-1.5B:
  <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>
- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- Dr. GRPO / Understanding R1-Zero-like training:
  <https://openreview.net/forum?id=5PAF7PAY2Y>
- LoRA: <https://arxiv.org/abs/2106.09685>
- QLoRA: Efficient Finetuning of Quantized LLMs:
  <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>
- bitsandbytes: <https://github.com/bitsandbytes-foundation/bitsandbytes>

## License

Academic project for a double-degree final thesis.

Author: Endika Blanco Cabana
