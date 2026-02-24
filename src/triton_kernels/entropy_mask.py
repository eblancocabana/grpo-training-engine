# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportRedeclaration=false
# pyright: reportUnreachable=false
# pyright: reportIndexIssue=false
# pyright: reportConstantRedefinition=false
from __future__ import annotations

from typing import Optional, Tuple, cast
import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:

    @triton.jit
    def _entropy_hist_kernel(
        logits_ptr,
        attention_mask_ptr,
        entropy_ptr,
        histogram_ptr,
        max_entropy,
        inv_max_entropy,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        USE_ATTENTION_MASK: tl.constexpr,
        COMPUTE_HIST: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_offset = pid * VOCAB_SIZE

        max_val = tl.full((), -float("inf"), tl.float32)
        for v in range(0, VOCAB_SIZE, BLOCK_VOCAB):
            v_idx = v + tl.arange(0, BLOCK_VOCAB)
            mask = v_idx < VOCAB_SIZE
            logits_chunk = tl.load(
                logits_ptr + row_offset + v_idx, mask=mask, other=-float("inf")
            )
            logits_chunk = tl.cast(logits_chunk, tl.float32)
            block_max = tl.max(logits_chunk, axis=0)
            max_val = tl.maximum(max_val, block_max)

        sum_exp = tl.zeros((), tl.float32)
        entropy_acc = tl.zeros((), tl.float32)
        for v in range(0, VOCAB_SIZE, BLOCK_VOCAB):
            v_idx = v + tl.arange(0, BLOCK_VOCAB)
            mask = v_idx < VOCAB_SIZE
            logits_chunk = tl.load(
                logits_ptr + row_offset + v_idx, mask=mask, other=-float("inf")
            )
            logits_chunk = tl.cast(logits_chunk, tl.float32)
            shifted = logits_chunk - max_val
            shifted = tl.where(mask, shifted, 0.0)
            exp_vals = tl.exp(shifted)
            exp_vals = tl.where(mask, exp_vals, 0.0)
            sum_exp += tl.sum(exp_vals, axis=0)
            entropy_acc += tl.sum(exp_vals * shifted, axis=0)

        log_sum_exp = tl.log(sum_exp)
        entropy = log_sum_exp - (entropy_acc / sum_exp)
        tl.store(entropy_ptr + pid, entropy)

        if COMPUTE_HIST:
            if USE_ATTENTION_MASK:
                attn = tl.load(attention_mask_ptr + pid)
                attn = tl.cast(attn, tl.float32)
                valid = attn > 0
            else:
                valid = tl.full((), 1, tl.int1)

            max_entropy_val = tl.cast(max_entropy, tl.float32)
            inv_max_entropy_val = tl.cast(inv_max_entropy, tl.float32)
            entropy_clamped = tl.maximum(entropy, 0.0)
            entropy_clamped = tl.minimum(entropy_clamped, max_entropy_val)
            bin_float = entropy_clamped * inv_max_entropy_val
            bin_idx = tl.cast(bin_float, tl.int32)
            bin_idx = tl.minimum(bin_idx, 255)
            one = tl.full((), 1, tl.int32)
            tl.atomic_add(histogram_ptr + bin_idx, one, mask=valid)

    @triton.jit
    def _entropy_mask_kernel(
        entropy_ptr,
        attention_mask_ptr,
        mask_ptr,
        threshold,
        USE_ATTENTION_MASK: tl.constexpr,
        STRICT_GREATER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        entropy = tl.load(entropy_ptr + pid)
        threshold_val = tl.cast(threshold, tl.float32)

        if STRICT_GREATER:
            keep = entropy > threshold_val
        else:
            keep = entropy >= threshold_val

        if USE_ATTENTION_MASK:
            attn = tl.load(attention_mask_ptr + pid)
            attn = tl.cast(attn, tl.float32)
            keep = keep & (attn > 0)

        mask_val = tl.cast(keep, tl.float32)
        tl.store(mask_ptr + pid, mask_val)
else:

    def _entropy_hist_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for fused_entropy_mask.")

    def _entropy_mask_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for fused_entropy_mask.")


def _select_block_vocab(vocab_size: int) -> int:
    if vocab_size <= 256:
        return 256
    if vocab_size <= 512:
        return 512
    return 1024


def _validate_inputs(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[int, int, int]:
    if logits.ndim != 3:
        raise ValueError(
            "logits must be [batch, seq_len, vocab_size], "
            f"got {logits.shape}."
        )
    batch_size, seq_len, vocab_size = logits.shape
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "logits must be float16/bfloat16/float32, "
            f"got {logits.dtype}."
        )
    if not logits.is_cuda:
        raise ValueError("logits must be on CUDA device.")
    if attention_mask is not None:
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "attention_mask must be [batch, seq_len], "
                f"got {attention_mask.shape}."
            )
        if not attention_mask.is_cuda:
            raise ValueError("attention_mask must be on CUDA device.")
    return batch_size, seq_len, vocab_size


def _compute_threshold_from_histogram(
    histogram: torch.Tensor,
    max_entropy: float,
    percentile: float,
) -> float:
    histogram = histogram.to(dtype=torch.float32)
    cumulative = torch.cumsum(histogram, dim=0)
    target = cumulative[-1] * (1.0 - percentile)
    target = target.unsqueeze(0)
    bin_idx = torch.searchsorted(cumulative, target)
    bin_idx = torch.clamp(bin_idx, 0, histogram.numel() - 1)
    threshold = (bin_idx.to(torch.float32) / 255.0) * max_entropy
    return float(threshold.item())


def _enforce_min_tokens(
    mask: torch.Tensor,
    entropy: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    min_tokens: int,
) -> torch.Tensor:
    if min_tokens <= 0:
        return mask

    batch_size, seq_len = entropy.shape
    for i in range(batch_size):
        selected = int(mask[i].sum().item())
        if selected >= min_tokens:
            continue

        if attention_mask is not None:
            valid_pos = attention_mask[i].bool()
        else:
            valid_pos = torch.ones(seq_len, dtype=torch.bool, device=entropy.device)

        n_valid = int(valid_pos.sum().item())
        n_select = min(min_tokens, n_valid)
        if n_select <= 0:
            continue

        seq_entropy = entropy[i].clone()
        seq_entropy[~valid_pos] = -float("inf")
        _, top_idx = torch.topk(seq_entropy, n_select)
        mask[i] = 0.0
        mask[i, top_idx] = 1.0

    return mask


@torch.no_grad()
def fused_entropy_mask(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    percentile: float = 0.5,
    threshold: Optional[float] = None,
    min_tokens: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available for fused_entropy_mask.")

    if threshold is None:
        if percentile < 0.0 or percentile > 1.0:
            raise ValueError(f"percentile must be in [0, 1], got {percentile}.")

    batch_size, seq_len, vocab_size = _validate_inputs(logits, attention_mask)
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be > 1, got {vocab_size}.")

    if batch_size == 0 or seq_len == 0:
        entropy = torch.empty(
            (batch_size, seq_len), device=logits.device, dtype=torch.float32
        )
        mask = torch.zeros_like(entropy)
        return entropy, mask

    original_dtype = logits.dtype
    if logits.dtype != torch.bfloat16:
        logits = logits.to(dtype=torch.bfloat16)
    logits = logits.contiguous()
    if attention_mask is not None:
        attention_mask = attention_mask.contiguous()

    entropy = torch.empty(
        (batch_size, seq_len), device=logits.device, dtype=torch.float32
    )

    compute_hist = threshold is None
    histogram = None
    if compute_hist:
        histogram = torch.zeros(256, device=logits.device, dtype=torch.int32)

    block_vocab = _select_block_vocab(vocab_size)
    num_warps = 4 if block_vocab <= 512 else 8

    max_entropy = float(math.log(vocab_size))
    inv_max_entropy = 255.0 / max_entropy

    grid = (batch_size * seq_len,)
    _entropy_hist_kernel[grid](
        logits,
        attention_mask if attention_mask is not None else logits,
        entropy,
        histogram if histogram is not None else entropy,
        max_entropy,
        inv_max_entropy,
        VOCAB_SIZE=vocab_size,
        BLOCK_VOCAB=block_vocab,
        USE_ATTENTION_MASK=attention_mask is not None,
        COMPUTE_HIST=compute_hist,
        num_warps=num_warps,
        num_stages=1,
    )

    threshold_explicit = threshold is not None
    strict_greater = threshold_explicit
    if compute_hist:
        histogram_tensor = cast(torch.Tensor, histogram)
        total_valid = int(histogram_tensor.sum().item())
        if total_valid == 0:
            mask = torch.zeros_like(entropy)
            return entropy, mask
        threshold = _compute_threshold_from_histogram(
            histogram_tensor, max_entropy, float(percentile)
        )

    mask = torch.empty(
        (batch_size, seq_len), device=logits.device, dtype=torch.float32
    )
    _entropy_mask_kernel[grid](
        entropy,
        attention_mask if attention_mask is not None else entropy,
        mask,
        float(threshold),
        USE_ATTENTION_MASK=attention_mask is not None,
        STRICT_GREATER=strict_greater,
        num_warps=4,
        num_stages=1,
    )

    mask = _enforce_min_tokens(mask, entropy, attention_mask, min_tokens)
    if original_dtype != torch.bfloat16:
        entropy = entropy.to(dtype=original_dtype)
    return entropy, mask


__all__ = ["fused_entropy_mask"]
