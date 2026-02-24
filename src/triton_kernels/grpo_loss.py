# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportUnknownMemberType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportRedeclaration=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportMissingTypeStubs=false
# pyright: reportConstantRedefinition=false
# pyright: reportDeprecated=false
# pyright: reportUnreachable=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

from typing import Callable, Dict, Protocol, Tuple, TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

if TYPE_CHECKING:
    import triton as triton  # noqa: F401
    import triton.language as tl  # noqa: F401


class _TritonKernel(Protocol):
    def __getitem__(self, grid: Tuple[int, ...]) -> Callable[..., None]:
        ...


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_grpo_loss_kernel(
        logits_ptr,
        target_ptr,
        old_log_probs_ptr,
        advantages_ptr,
        loss_ptr,
        mask_ptr,
        scale_ptr,
        seq_len,
        clip_epsilon,
        epsilon_high,
        delta,
        inv_group_size,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        USE_MASK: tl.constexpr,
        USE_SCALE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        seq_block = tl.program_id(1)

        seq_offsets = seq_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        token_mask = seq_offsets < seq_len
        token_ids = batch_idx * seq_len + seq_offsets
        row_offsets = token_ids * VOCAB_SIZE

        max_val = tl.full((BLOCK_TOKENS,), -float("inf"), tl.float32)
        sum_exp = tl.zeros((BLOCK_TOKENS,), tl.float32)
        for v in range(0, VOCAB_SIZE, BLOCK_VOCAB):
            v_idx = v + tl.arange(0, BLOCK_VOCAB)
            vocab_mask = v_idx < VOCAB_SIZE
            mask = token_mask[:, None] & vocab_mask[None, :]
            logits_chunk = tl.load(
                logits_ptr + row_offsets[:, None] + v_idx[None, :],
                mask=mask,
                other=-float("inf"),
            )
            logits_chunk = tl.cast(logits_chunk, tl.float32)
            block_max = tl.max(logits_chunk, axis=1)
            block_max = tl.where(token_mask, block_max, 0.0)
            new_max = tl.maximum(max_val, block_max)

            block_sum = tl.sum(tl.exp(logits_chunk - block_max[:, None]), axis=1)
            block_sum = tl.where(token_mask, block_sum, 0.0)

            sum_exp = sum_exp * tl.exp(max_val - new_max) + block_sum * tl.exp(
                block_max - new_max
            )
            max_val = new_max

        log_sum_exp = tl.log(sum_exp) + max_val
        log_sum_exp = tl.where(token_mask, log_sum_exp, 0.0)

        target_token = tl.load(target_ptr + token_ids, mask=token_mask, other=0)
        target_token = tl.cast(target_token, tl.int32)
        target_logit = tl.load(
            logits_ptr + row_offsets + target_token, mask=token_mask, other=0.0
        )
        target_logit = tl.cast(target_logit, tl.float32)
        log_prob_new = target_logit - log_sum_exp

        old_log_prob = tl.load(old_log_probs_ptr + token_ids, mask=token_mask, other=0.0)
        old_log_prob = tl.cast(old_log_prob, tl.float32)
        log_ratio = log_prob_new - old_log_prob
        ratio = tl.exp(log_ratio)
        ratio = tl.minimum(ratio, delta)

        advantage = tl.load(advantages_ptr + batch_idx)
        advantage = tl.cast(advantage, tl.float32)

        clip_low = 1.0 - clip_epsilon
        clip_high = 1.0 + epsilon_high
        clipped_ratio = tl.maximum(tl.minimum(ratio, clip_high), clip_low)

        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        token_loss = -tl.minimum(surr1, surr2)
        token_loss = tl.where(token_mask, token_loss, 0.0)

        if USE_MASK:
            mask_val = tl.load(mask_ptr + token_ids, mask=token_mask, other=0.0)
            mask_val = tl.cast(mask_val, tl.float32)
            token_loss = token_loss * mask_val

        if USE_SCALE:
            scale_val = tl.load(scale_ptr + batch_idx)
            scale_val = tl.cast(scale_val, tl.float32)
            token_loss = token_loss * scale_val

        loss_acc = tl.sum(token_loss, axis=0)
        tl.atomic_add(loss_ptr, loss_acc * inv_group_size)
else:

    def _fused_grpo_loss_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for fused_grpo_loss.")


def _select_block_vocab(vocab_size: int) -> int:
    if vocab_size <= 256:
        return 256
    if vocab_size <= 1024:
        return 512
    return 1024


def _select_block_tokens(total_tokens: int) -> int:
    if total_tokens >= 8192:
        return 8
    if total_tokens >= 2048:
        return 4
    return 2


def _select_num_warps(block_vocab: int, block_tokens: int) -> int:
    tile_size = block_vocab * block_tokens
    if tile_size <= 2048:
        return 2
    return 4


def _validate_inputs(
    policy_logits: torch.Tensor,
    target_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> Tuple[int, int, int]:
    if policy_logits.ndim != 3:
        raise ValueError(
            "policy_logits must be [batch, seq_len, vocab_size], "
            f"got {policy_logits.shape}."
        )
    batch_size, seq_len, vocab_size = policy_logits.shape
    if target_ids.shape != (batch_size, seq_len):
        raise ValueError(
            "target_ids must be [batch, seq_len], "
            f"got {target_ids.shape}."
        )
    if old_log_probs.shape != (batch_size, seq_len):
        raise ValueError(
            "old_log_probs must be [batch, seq_len], "
            f"got {old_log_probs.shape}."
        )
    if advantages.shape != (batch_size,):
        raise ValueError(
            "advantages must be [batch], "
            f"got {advantages.shape}."
        )
    if policy_logits.dtype != torch.bfloat16:
        raise ValueError(
            f"policy_logits must be bfloat16, got {policy_logits.dtype}."
        )
    if target_ids.dtype != torch.long:
        raise ValueError(f"target_ids must be int64, got {target_ids.dtype}.")
    if old_log_probs.dtype != torch.float32:
        raise ValueError(
            f"old_log_probs must be float32, got {old_log_probs.dtype}."
        )
    if advantages.dtype != torch.float32:
        raise ValueError(
            f"advantages must be float32, got {advantages.dtype}."
        )
    if not policy_logits.is_cuda:
        raise ValueError("policy_logits must be on CUDA device.")
    if (
        not target_ids.is_cuda
        or not old_log_probs.is_cuda
        or not advantages.is_cuda
    ):
        raise ValueError("All inputs must be on the same CUDA device.")
    return batch_size, seq_len, vocab_size


def _compute_metrics(
    policy_logits: torch.Tensor,
    target_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    epsilon_high: float,
    delta: float,
    loss: torch.Tensor,
) -> Dict[str, float]:
    batch_size, seq_len, vocab_size = policy_logits.shape

    log_probs_target = -F.cross_entropy(
        policy_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len)

    log_ratio = log_probs_target - old_log_probs
    ratio = torch.exp(log_ratio)
    ratio = torch.clamp(ratio, max=delta)

    advantages_expanded = advantages.unsqueeze(1)
    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + epsilon_high
    ) * advantages_expanded
    policy_loss = -torch.min(surr1, surr2)

    ratio_std = ratio.std().item() if ratio.numel() > 1 else 0.0
    adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
    ratio_capped_pct = (ratio >= delta).float().mean().item()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.mean().item(),
        "kl_loss": 0.0,
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio_std,
        "ratio_capped_pct": ratio_capped_pct,
        "advantage_std": adv_std,
    }


def _validate_masks(
    attention_mask: torch.Tensor | None,
    entropy_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
) -> None:
    if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            "attention_mask must be [batch, seq_len], "
            f"got {attention_mask.shape}."
        )
    if entropy_mask is not None and entropy_mask.shape != (batch_size, seq_len):
        raise ValueError(
            "entropy_mask must be [batch, seq_len], "
            f"got {entropy_mask.shape}."
        )


def _prepare_mask_and_scale(
    attention_mask: torch.Tensor | None,
    entropy_mask: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if attention_mask is None and entropy_mask is None:
        return None, None

    if attention_mask is not None and entropy_mask is not None:
        effective_mask = attention_mask.to(dtype=torch.float32) * entropy_mask.to(
            dtype=torch.float32
        )
    elif attention_mask is not None:
        effective_mask = attention_mask.to(dtype=torch.float32)
    else:
        effective_mask = entropy_mask.to(dtype=torch.float32)  # type: ignore[union-attr]

    scale: torch.Tensor | None = None
    if entropy_mask is not None:
        if attention_mask is not None:
            total_tokens = attention_mask.sum(dim=1).to(dtype=torch.float32)
        else:
            total_tokens = torch.full(
                (effective_mask.shape[0],),
                float(seq_len),
                device=device,
                dtype=torch.float32,
            )
        selected_tokens = effective_mask.sum(dim=1)
        scale = torch.where(
            selected_tokens > 0,
            total_tokens / (selected_tokens + 1e-8),
            torch.zeros_like(selected_tokens),
        )

    return effective_mask.contiguous(), scale


def _torch_grpo_loss(
    policy_logits: torch.Tensor,
    target_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    epsilon_high: float,
    delta: float,
    group_size: int,
    attention_mask: torch.Tensor | None,
    entropy_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = policy_logits.shape
    log_probs_target = -F.cross_entropy(
        policy_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len)

    log_ratio = log_probs_target - old_log_probs
    ratio = torch.exp(log_ratio)
    ratio = torch.clamp(ratio, max=delta)

    advantages_expanded = advantages.unsqueeze(1)
    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + epsilon_high) * advantages_expanded
    policy_loss = -torch.min(surr1, surr2)

    loss_per_token = policy_loss
    if entropy_mask is not None:
        loss_per_token = loss_per_token * entropy_mask
        if attention_mask is not None:
            loss_per_token = loss_per_token * attention_mask
    elif attention_mask is not None:
        loss_per_token = loss_per_token * attention_mask

    per_sample_loss = loss_per_token.sum(dim=1)
    if entropy_mask is not None:
        if attention_mask is not None:
            effective_mask = entropy_mask * attention_mask
            total_tokens = attention_mask.sum(dim=1)
        else:
            effective_mask = entropy_mask
            total_tokens = torch.full_like(effective_mask.sum(dim=1), float(seq_len))
        selected_tokens = effective_mask.sum(dim=1)
        scale = torch.where(
            selected_tokens > 0,
            total_tokens / (selected_tokens + 1e-8),
            torch.zeros_like(selected_tokens),
        )
        per_sample_loss = per_sample_loss * scale

    return per_sample_loss.sum() / float(group_size)


class _TritonGRPOLossFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        policy_logits: torch.Tensor,
        target_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float,
        epsilon_high: float,
        delta: float,
        group_size: int,
        attention_mask: torch.Tensor | None,
        entropy_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention_saved = (
            attention_mask
            if attention_mask is not None
            else policy_logits.new_empty(0)
        )
        entropy_saved = (
            entropy_mask if entropy_mask is not None else policy_logits.new_empty(0)
        )
        meta = torch.tensor(
            [
                float(attention_mask is not None),
                float(entropy_mask is not None),
                float(clip_epsilon),
                float(epsilon_high),
                float(delta),
                float(group_size),
            ],
            device=policy_logits.device,
            dtype=torch.float32,
        )
        ctx.save_for_backward(
            policy_logits,
            target_ids,
            old_log_probs,
            advantages,
            attention_saved,
            entropy_saved,
            meta,
        )

        loss = _fused_grpo_loss_forward(
            policy_logits,
            target_ids,
            old_log_probs,
            advantages,
            clip_epsilon,
            epsilon_high,
            delta,
            group_size,
            attention_mask,
            entropy_mask,
        )
        return loss

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            policy_logits,
            target_ids,
            old_log_probs,
            advantages,
            attention_saved,
            entropy_saved,
            meta,
        ) = ctx.saved_tensors
        has_attention_mask = bool(meta[0].item())
        has_entropy_mask = bool(meta[1].item())
        clip_epsilon = float(meta[2].item())
        epsilon_high = float(meta[3].item())
        delta = float(meta[4].item())
        group_size = int(meta[5].item())
        attention_mask = attention_saved if has_attention_mask else None
        entropy_mask = entropy_saved if has_entropy_mask else None

        policy_logits_detached = policy_logits.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = _torch_grpo_loss(
                policy_logits_detached,
                target_ids,
                old_log_probs,
                advantages,
                clip_epsilon,
                epsilon_high,
                delta,
                group_size,
                attention_mask,
                entropy_mask,
            )
        grad_logits = torch.autograd.grad(
            loss,
            policy_logits_detached,
            grad_outputs=grad_outputs[0],
            retain_graph=False,
            create_graph=False,
        )[0]

        return (
            grad_logits,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _fused_grpo_loss_forward(
    policy_logits: torch.Tensor,
    target_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    epsilon_high: float,
    delta: float,
    group_size: int,
    attention_mask: torch.Tensor | None,
    entropy_mask: torch.Tensor | None,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available for fused_grpo_loss.")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}.")

    batch_size, seq_len, vocab_size = _validate_inputs(
        policy_logits, target_ids, old_log_probs, advantages
    )
    _validate_masks(attention_mask, entropy_mask, batch_size, seq_len)

    policy_logits = policy_logits.contiguous()
    target_ids = target_ids.contiguous()
    old_log_probs = old_log_probs.contiguous()
    advantages = advantages.contiguous()

    effective_mask, scale = _prepare_mask_and_scale(
        attention_mask, entropy_mask, seq_len, policy_logits.device
    )

    loss = torch.zeros((), device=policy_logits.device, dtype=torch.float32)

    block_vocab = _select_block_vocab(vocab_size)
    block_tokens = _select_block_tokens(batch_size * seq_len)
    num_warps = _select_num_warps(block_vocab, block_tokens)
    num_stages = 1
    inv_group_size = 1.0 / float(group_size)

    grid: tuple[int, int] = (batch_size, int(triton.cdiv(seq_len, block_tokens)))
    cast_kernel = cast(_TritonKernel, _fused_grpo_loss_kernel)
    cast_kernel[grid](
        policy_logits,
        target_ids,
        old_log_probs,
        advantages,
        loss,
        effective_mask if effective_mask is not None else policy_logits,
        scale if scale is not None else advantages,
        seq_len,
        clip_epsilon,
        epsilon_high,
        delta,
        inv_group_size,
        VOCAB_SIZE=vocab_size,
        BLOCK_VOCAB=block_vocab,
        BLOCK_TOKENS=block_tokens,
        USE_MASK=effective_mask is not None,
        USE_SCALE=scale is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return loss


def fused_grpo_loss(
    policy_logits: torch.Tensor,
    target_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    epsilon_high: float,
    delta: float,
    group_size: int,
    attention_mask: torch.Tensor | None = None,
    entropy_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if policy_logits.requires_grad:
        loss: torch.Tensor = cast(
            torch.Tensor,
            _TritonGRPOLossFn.apply(
                policy_logits,
                target_ids,
                old_log_probs,
                advantages,
                clip_epsilon,
                epsilon_high,
                delta,
                group_size,
                attention_mask,
                entropy_mask,
            ),
        )
    else:
        loss = _fused_grpo_loss_forward(
            policy_logits,
            target_ids,
            old_log_probs,
            advantages,
            clip_epsilon,
            epsilon_high,
            delta,
            group_size,
            attention_mask,
            entropy_mask,
        )

    with torch.no_grad():
        metrics = _compute_metrics(
            policy_logits,
            target_ids,
            old_log_probs,
            advantages,
            clip_epsilon,
            epsilon_high,
            delta,
            loss,
        )

    return loss, metrics


__all__ = ["fused_grpo_loss"]
