#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from src.core.lora import ManualLoRALayer


def _make_layer(
    in_features: int,
    out_features: int,
    rank: int,
    alpha: int,
    use_triton: bool,
) -> ManualLoRALayer:
    import bitsandbytes as bnb

    base = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=True,
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        quant_type="nf4",
        compress_statistics=True,
    )
    layer = ManualLoRALayer(
        base,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        use_triton=use_triton,
    )
    return layer


def _run_benchmark(
    layer: ManualLoRALayer,
    x: torch.Tensor,
    steps: int,
    warmup: int,
) -> float:
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out = layer(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = layer(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / max(1, steps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LoRA layer forward/backward")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x = torch.randn(args.batch, args.seq, args.in_features, device=device, dtype=dtype)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    layer_triton = _make_layer(
        args.in_features,
        args.out_features,
        args.rank,
        args.alpha,
        use_triton=True,
    ).to(device=device, dtype=dtype)

    layer_torch = _make_layer(
        args.in_features,
        args.out_features,
        args.rank,
        args.alpha,
        use_triton=False,
    ).to(device=device, dtype=dtype)

    layer_torch.lora_A.weight.data.copy_(layer_triton.lora_A.weight.data)
    layer_torch.lora_B.weight.data.copy_(layer_triton.lora_B.weight.data)

    triton_time = _run_benchmark(layer_triton, x, args.steps, args.warmup)
    torch_time = _run_benchmark(layer_torch, x, args.steps, args.warmup)

    print(f"Triton avg step: {triton_time:.6f}s")
    print(f"Torch avg step:  {torch_time:.6f}s")
    if torch_time > 0:
        print(f"Speedup: {(torch_time / triton_time):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
