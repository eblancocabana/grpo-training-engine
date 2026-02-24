from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportUnknownArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportUnreachable=false
# pyright: reportIndexIssue=false

import torch
from typing import cast

try:
    import triton
    import triton.language as tl

    triton_available = True
except Exception:
    triton_available = False
    triton = None
    tl = None


if triton_available:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_xb,
        stride_xd,
        stride_w,
        stride_yb,
        stride_yd,
        n_cols,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        sum_sq = tl.zeros((), tl.float32)
        for offs in range(0, n_cols, BLOCK_D):
            cols = offs + tl.arange(0, BLOCK_D)
            mask = cols < n_cols
            x = tl.load(
                x_ptr + pid * stride_xb + cols * stride_xd,
                mask=mask,
                other=0.0,
            )
            x = tl.cast(x, tl.float32)
            sum_sq += tl.sum(x * x, axis=0)

        mean_sq = sum_sq / tl.cast(n_cols, tl.float32)
        inv_rms = tl.rsqrt(mean_sq + tl.cast(eps, tl.float32))

        for offs in range(0, n_cols, BLOCK_D):
            cols = offs + tl.arange(0, BLOCK_D)
            mask = cols < n_cols
            x = tl.load(
                x_ptr + pid * stride_xb + cols * stride_xd,
                mask=mask,
                other=0.0,
            )
            w = tl.load(w_ptr + cols * stride_w, mask=mask, other=0.0)
            x = tl.cast(x, tl.float32)
            normed = x * inv_rms
            normed = tl.cast(normed, tl.bfloat16)
            y = normed * w
            tl.store(y_ptr + pid * stride_yb + cols * stride_yd, y, mask=mask)

    @triton.jit
    def _silu_mul_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        stride_xb,
        stride_xd,
        stride_yb,
        stride_yd,
        stride_ob,
        stride_od,
        n_cols,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        for offs in range(0, n_cols, BLOCK_D):
            cols = offs + tl.arange(0, BLOCK_D)
            mask = cols < n_cols
            x = tl.load(
                x_ptr + pid * stride_xb + cols * stride_xd,
                mask=mask,
                other=0.0,
            )
            y = tl.load(
                y_ptr + pid * stride_yb + cols * stride_yd,
                mask=mask,
                other=0.0,
            )
            x = tl.cast(x, tl.float32)
            silu = x * tl.sigmoid(x)
            silu = tl.cast(silu, tl.bfloat16)
            y = tl.cast(y, tl.bfloat16)
            out = silu * y
            tl.store(out_ptr + pid * stride_ob + cols * stride_od, out, mask=mask)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not triton_available:
        raise ImportError("Triton is not available for fused_rmsnorm.")
    if x.ndim != 3:
        raise ValueError("fused_rmsnorm expects [B, S, D] input.")
    batch, seq_len, dim = x.shape
    x_2d = x.reshape(batch * seq_len, dim)
    y = torch.empty_like(x_2d)
    block_d = 2 ** int((dim - 1).bit_length())
    block_d = max(128, min(block_d, 1024))

    grid = (x_2d.shape[0],)
    num_warps = 2 if dim <= 256 else 4
    num_stages = 1
    _rmsnorm_kernel[grid](
        x_2d,
        weight,
        y,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        y.stride(0),
        y.stride(1),
        dim,
        eps,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(batch, seq_len, dim)


def fused_silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not triton_available:
        raise ImportError("Triton is not available for fused_silu_mul.")
    if x.shape != y.shape:
        raise ValueError("fused_silu_mul requires matching shapes.")
    batch, seq_len, dim = x.shape
    x_2d = x.reshape(batch * seq_len, dim)
    y_2d = y.reshape(batch * seq_len, dim)
    out = torch.empty_like(x_2d)
    block_d = 2 ** int((dim - 1).bit_length())
    block_d = max(128, min(block_d, 1024))

    grid = (x_2d.shape[0],)
    num_warps = 2 if dim <= 256 else 4
    num_stages = 1
    _silu_mul_kernel[grid](
        x_2d,
        y_2d,
        out,
        x_2d.stride(0),
        x_2d.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        out.stride(0),
        out.stride(1),
        dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(batch, seq_len, dim)


def fused_mlp(
    x: torch.Tensor,
    gate_proj: torch.nn.Module,
    up_proj: torch.nn.Module,
    down_proj: torch.nn.Module,
) -> torch.Tensor:
    gate = gate_proj(x)
    up = up_proj(x)
    act = fused_silu_mul(gate, up)
    return down_proj(act)




def fused_qkv_rope(
    x: torch.Tensor,
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    v_proj: torch.nn.Module,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)
    batch, seq_len, _ = x.shape
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def fused_logits_sampling(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = lm_head(hidden)
    if not do_sample or temperature <= 0.0:
        next_tokens = torch.argmax(logits, dim=-1)
        return logits, next_tokens

    scaled = logits / max(temperature, 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, -float("inf"))
        scaled = torch.full_like(scaled, -float("inf"))
        scaled = scaled.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    if eos_token_id is not None:
        _ = eos_token_id
    return logits, next_tokens
