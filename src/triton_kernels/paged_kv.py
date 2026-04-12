# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportUnknownMemberType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnusedVariable=false
# pyright: reportUnusedCallResult=false
# pyright: reportRedeclaration=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportMissingTypeStubs=false
# pyright: reportConstantRedefinition=false
# pyright: reportDeprecated=false
# pyright: reportUnreachable=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TYPE_CHECKING, Protocol, cast
import math

import torch

triton = None
tl = None

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


@dataclass
class PagedKVCacheState:
    """Reusable paged KV prompt cache for grouped decode."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    last_tokens: torch.Tensor
    max_context: int
    block_size: int


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_attention_decode_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_bb,
        stride_bt,
        stride_ob,
        stride_oh,
        stride_od,
        block_size,
        kv_group_size,
        scale,
        D: tl.constexpr,
        BLOCK_CTX: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_CONTEXT: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv_head = pid_h // kv_group_size

        ctx_len = tl.load(context_lens_ptr + pid_b)
        ctx_len = tl.cast(ctx_len, tl.int32)

        max_val = tl.full((), -float("inf"), tl.float32)
        for ctx_start in range(0, MAX_CONTEXT, BLOCK_CTX):
            pos = ctx_start + tl.arange(0, BLOCK_CTX)
            mask_pos = pos < ctx_len
            block_idx = pos // block_size
            block_off = pos - block_idx * block_size
            block_ids = tl.load(
                block_tables_ptr + pid_b * stride_bb + block_idx * stride_bt,
                mask=mask_pos,
                other=0,
            )
            block_ids = tl.cast(block_ids, tl.int32)

            logits = tl.zeros((BLOCK_CTX,), tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d = d_start + tl.arange(0, BLOCK_D)
                mask_d = d < D
                q_ptrs = (
                    q_ptr
                    + pid_b * stride_qb
                    + pid_h * stride_qh
                    + d * stride_qd
                )
                q_block = tl.load(q_ptrs, mask=mask_d, other=0.0)
                q_block = tl.cast(q_block, tl.float32)

                k_ptrs = (
                    k_cache_ptr
                    + block_ids[:, None] * stride_kb
                    + kv_head * stride_kh
                    + block_off[:, None] * stride_kt
                    + d[None, :] * stride_kd
                )
                k_block = tl.load(
                    k_ptrs, mask=mask_pos[:, None] & mask_d[None, :], other=0.0
                )
                k_block = tl.cast(k_block, tl.float32)
                logits += tl.sum(k_block * q_block[None, :], axis=1)

            logits = logits * scale
            masked_logits = tl.where(mask_pos, logits, -float("inf"))
            block_max = tl.max(masked_logits, axis=0)
            max_val = tl.maximum(max_val, block_max)

        sum_exp = tl.zeros((), tl.float32)
        for ctx_start in range(0, MAX_CONTEXT, BLOCK_CTX):
            pos = ctx_start + tl.arange(0, BLOCK_CTX)
            mask_pos = pos < ctx_len
            block_idx = pos // block_size
            block_off = pos - block_idx * block_size
            block_ids = tl.load(
                block_tables_ptr + pid_b * stride_bb + block_idx * stride_bt,
                mask=mask_pos,
                other=0,
            )
            block_ids = tl.cast(block_ids, tl.int32)

            logits = tl.zeros((BLOCK_CTX,), tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d = d_start + tl.arange(0, BLOCK_D)
                mask_d = d < D
                q_ptrs = (
                    q_ptr
                    + pid_b * stride_qb
                    + pid_h * stride_qh
                    + d * stride_qd
                )
                q_block = tl.load(q_ptrs, mask=mask_d, other=0.0)
                q_block = tl.cast(q_block, tl.float32)

                k_ptrs = (
                    k_cache_ptr
                    + block_ids[:, None] * stride_kb
                    + kv_head * stride_kh
                    + block_off[:, None] * stride_kt
                    + d[None, :] * stride_kd
                )
                k_block = tl.load(
                    k_ptrs, mask=mask_pos[:, None] & mask_d[None, :], other=0.0
                )
                k_block = tl.cast(k_block, tl.float32)
                logits += tl.sum(k_block * q_block[None, :], axis=1)

            logits = logits * scale
            weights = tl.exp(logits - max_val)
            weights = tl.where(mask_pos, weights, 0.0)
            sum_exp += tl.sum(weights, axis=0)

        for d_start in range(0, D, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            mask_d = d < D
            out_acc = tl.zeros((BLOCK_D,), tl.float32)
            for ctx_start in range(0, MAX_CONTEXT, BLOCK_CTX):
                pos = ctx_start + tl.arange(0, BLOCK_CTX)
                mask_pos = pos < ctx_len
                block_idx = pos // block_size
                block_off = pos - block_idx * block_size
                block_ids = tl.load(
                    block_tables_ptr + pid_b * stride_bb + block_idx * stride_bt,
                    mask=mask_pos,
                    other=0,
                )
                block_ids = tl.cast(block_ids, tl.int32)

                logits = tl.zeros((BLOCK_CTX,), tl.float32)
                for d_inner in range(0, D, BLOCK_D):
                    dd = d_inner + tl.arange(0, BLOCK_D)
                    mask_dd = dd < D
                    q_ptrs = (
                        q_ptr
                        + pid_b * stride_qb
                        + pid_h * stride_qh
                        + dd * stride_qd
                    )
                    q_block = tl.load(q_ptrs, mask=mask_dd, other=0.0)
                    q_block = tl.cast(q_block, tl.float32)

                    k_ptrs = (
                        k_cache_ptr
                        + block_ids[:, None] * stride_kb
                        + kv_head * stride_kh
                        + block_off[:, None] * stride_kt
                        + dd[None, :] * stride_kd
                    )
                    k_block = tl.load(
                        k_ptrs, mask=mask_pos[:, None] & mask_dd[None, :], other=0.0
                    )
                    k_block = tl.cast(k_block, tl.float32)
                    logits += tl.sum(k_block * q_block[None, :], axis=1)

                logits = logits * scale
                weights = tl.exp(logits - max_val)
                weights = tl.where(mask_pos, weights, 0.0)

                v_ptrs = (
                    v_cache_ptr
                    + block_ids[:, None] * stride_kb
                    + kv_head * stride_kh
                    + block_off[:, None] * stride_kt
                    + d[None, :] * stride_kd
                )
                v_block = tl.load(
                    v_ptrs, mask=mask_pos[:, None] & mask_d[None, :], other=0.0
                )
                v_block = tl.cast(v_block, tl.float32)
                out_acc += tl.sum(v_block * weights[:, None], axis=0)

            denom = tl.maximum(sum_exp, 1e-6)
            out_acc = out_acc / denom
            out_ptrs = (
                out_ptr
                + pid_b * stride_ob
                + pid_h * stride_oh
                + d * stride_od
            )
            tl.store(out_ptrs, tl.cast(out_acc, tl.float32), mask=mask_d)

    @triton.jit
    def _paged_kv_update_kernel(
        k_new_ptr,
        v_new_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        stride_knb,
        stride_knh,
        stride_knd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_bb,
        stride_bt,
        block_size,
        num_kv_heads,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        if pid_h >= num_kv_heads:
            return

        ctx_len = tl.load(context_lens_ptr + pid_b)
        ctx_len = tl.cast(ctx_len, tl.int32)
        block_idx = ctx_len // block_size
        block_off = ctx_len - block_idx * block_size
        block_id = tl.load(
            block_tables_ptr + pid_b * stride_bb + block_idx * stride_bt
        )
        block_id = tl.cast(block_id, tl.int32)

        d = tl.arange(0, BLOCK_D)
        mask_d = d < D

        k_ptrs = (
            k_cache_ptr
            + block_id * stride_kb
            + pid_h * stride_kh
            + block_off * stride_kt
            + d * stride_kd
        )
        v_ptrs = (
            v_cache_ptr
            + block_id * stride_kb
            + pid_h * stride_kh
            + block_off * stride_kt
            + d * stride_kd
        )
        k_new_ptrs = (
            k_new_ptr
            + pid_b * stride_knb
            + pid_h * stride_knh
            + d * stride_knd
        )
        v_new_ptrs = (
            v_new_ptr
            + pid_b * stride_knb
            + pid_h * stride_knh
            + d * stride_knd
        )

        k_new = tl.load(k_new_ptrs, mask=mask_d, other=0.0)
        v_new = tl.load(v_new_ptrs, mask=mask_d, other=0.0)
        tl.store(k_ptrs, k_new, mask=mask_d)
        tl.store(v_ptrs, v_new, mask=mask_d)
else:

    def _paged_attention_decode_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for paged_kv_decode.")

    def _paged_kv_update_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for paged_kv_decode.")


def _select_block_ctx(max_context: int) -> int:
    if max_context <= 32:
        return 16
    if max_context <= 128:
        return 32
    if max_context <= 512:
        return 64
    if max_context <= 2048:
        return 256
    return 128


def _select_block_d(head_dim: int) -> int:
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 96:
        return 64
    return 128


def _select_num_warps_update(head_dim: int) -> int:
    if head_dim <= 128:
        return 2
    return 4


def _select_num_warps_decode(block_d: int, block_ctx: int) -> int:
    tile = block_d * block_ctx
    if tile <= 1024:
        return 2
    if tile <= 4096:
        return 4
    return 8


def _select_num_stages_decode(max_context: int) -> int:
    if max_context <= 128:
        return 3
    if max_context <= 512:
        return 2
    return 1


def _validate_inputs(
    input_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must be [batch, seq_len], "
            f"got {input_ids.shape}."
        )
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache/v_cache must be [num_blocks, heads, block, head_dim]."
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes.")
    if block_tables.ndim != 2:
        raise ValueError(
            "block_tables must be [batch, max_blocks], "
            f"got {block_tables.shape}."
        )
    if context_lens.ndim != 1:
        raise ValueError(
            "context_lens must be [batch], "
            f"got {context_lens.shape}."
        )

    batch_size = input_ids.shape[0]
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "block_tables batch dimension must match input_ids batch."
        )
    if context_lens.shape[0] != batch_size:
        raise ValueError(
            "context_lens batch dimension must match input_ids batch."
        )

    if not (input_ids.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("inputs and caches must be on CUDA device.")
    if not block_tables.is_cuda or not context_lens.is_cuda:
        raise ValueError("block_tables and context_lens must be on CUDA device.")
    if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("k_cache/v_cache must be bfloat16.")

    num_blocks, num_kv_heads, block_size, head_dim = k_cache.shape
    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size
    if torch.any(context_lens < 0):
        raise ValueError("context_lens must be non-negative.")
    if torch.any(context_lens >= max_context):
        raise ValueError("context_lens exceed block table capacity.")
    if block_tables.numel() > 0:
        if torch.any(block_tables < 0):
            raise ValueError("block_tables entries must be non-negative.")
        if torch.any(block_tables >= num_blocks):
            raise ValueError("block_tables entries exceed cache capacity.")
    return num_kv_heads, block_size, head_dim, max_context


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        return torch.full_like(logits, -float("inf"))

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    cutoff = cumulative > top_p
    cutoff[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(cutoff, -float("inf"))

    filtered = torch.full_like(logits, -float("inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def _sample_tokens(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if not do_sample or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = logits / max(temperature, 1e-6)
    filtered = _apply_top_p(scaled, top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _infer_head_dim(num_heads: int, hidden_size: Optional[int]) -> int:
    if hidden_size is None:
        raise ValueError("Unable to infer head_dim without hidden_size.")
    return int(hidden_size) // int(num_heads)


def _get_model_components(model: torch.nn.Module) -> tuple[
    torch.nn.Module,
    list[torch.nn.Module],
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
    int,
    int,
    int,
    int,
]:
    model_root = getattr(model, "model", None)
    if model_root is None:
        model_root = model

    if not hasattr(model_root, "embed_tokens"):
        raise AttributeError("Model missing embed_tokens for paged_kv_decode.")
    if not hasattr(model_root, "layers"):
        raise AttributeError("Model missing layers for paged_kv_decode.")
    if not hasattr(model_root, "norm"):
        raise AttributeError("Model missing final norm for paged_kv_decode.")
    if not hasattr(model_root, "rotary_emb"):
        raise AttributeError("Model missing rotary_emb for paged_kv_decode.")
    if not hasattr(model, "lm_head"):
        raise AttributeError("Model missing lm_head for paged_kv_decode.")

    embeddings = cast(torch.nn.Module, model_root.embed_tokens)
    layers = cast(list[torch.nn.Module], list(model_root.layers))
    norm = cast(torch.nn.Module, model_root.norm)
    rotary_emb = cast(torch.nn.Module, model_root.rotary_emb)
    lm_head = cast(torch.nn.Module, model.lm_head)

    config = getattr(model, "config", None)
    num_heads = (
        getattr(model, "num_heads", None)
        or getattr(model, "n_heads", None)
        or getattr(model, "num_attention_heads", None)
        or getattr(config, "num_attention_heads", None)
    )
    if num_heads is None:
        raise ValueError("Unable to infer num_heads for paged_kv_decode.")
    num_heads = int(num_heads)
    num_kv_heads = getattr(model, "num_kv_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(model, "n_kv_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(model, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        raise ValueError("Unable to infer num_kv_heads for paged_kv_decode.")
    num_kv_heads = int(num_kv_heads)

    hidden_size = getattr(model, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model, "embed_dim", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Unable to infer hidden_size for paged_kv_decode.")
    hidden_size = int(hidden_size)

    head_dim = getattr(model, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(model, "attention_head_size", None)
    if head_dim is None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = _infer_head_dim(num_heads, hidden_size)

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
        )

    return (
        embeddings,
        layers,
        norm,
        rotary_emb,
        lm_head,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_size,
    )


def _prefill_paged_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
) -> None:
    if input_ids.numel() == 0:
        return
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_kv = outputs.past_key_values
    if past_kv is None:
        return

    token_slots = attention_mask.to(dtype=torch.int32).cumsum(dim=1) - 1
    valid_mask = attention_mask.bool()
    block_idx = torch.where(valid_mask, token_slots // block_size, 0)
    block_off = torch.where(valid_mask, token_slots % block_size, 0)
    block_ids = torch.gather(block_tables, 1, block_idx.clamp(min=0))

    for layer_idx in range(len(past_kv)):
        k_layer, v_layer = past_kv[layer_idx]
        k_valid = k_layer.permute(0, 2, 1, 3)[valid_mask].to(dtype=torch.bfloat16)
        v_valid = v_layer.permute(0, 2, 1, 3)[valid_mask].to(dtype=torch.bfloat16)
        valid_block_ids = block_ids[valid_mask]
        valid_block_offs = block_off[valid_mask]

        k_slots = k_cache[layer_idx].permute(0, 2, 1, 3)
        v_slots = v_cache[layer_idx].permute(0, 2, 1, 3)
        k_slots[valid_block_ids, valid_block_offs] = k_valid
        v_slots[valid_block_ids, valid_block_offs] = v_valid


def _build_block_tables(
    batch_size: int, max_blocks: int, device: torch.device
) -> torch.Tensor:
    block_offsets = (
        torch.arange(batch_size, device=device, dtype=torch.int32) * max_blocks
    ).unsqueeze(1)
    return block_offsets + torch.arange(
        max_blocks, device=device, dtype=torch.int32
    ).unsqueeze(0)


def _clone_paged_cache_state(
    state: PagedKVCacheState, repeats: int
) -> PagedKVCacheState:
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0, got {repeats}.")

    base_batch = state.block_tables.shape[0]
    max_blocks = state.block_tables.shape[1]
    total_batch = base_batch * repeats
    num_layers, _, num_kv_heads, block_size, head_dim = state.k_cache.shape
    device = state.k_cache.device

    # The trainer expands one prefetched prompt at a time. In that case we can
    # share immutable full prefix blocks across grouped responses and only
    # allocate private blocks for the mutable tail. This avoids cloning the
    # entire prompt cache for every grouped sample.
    if base_batch == 1 and repeats > 1:
        prefix_ctx = int(state.context_lens[0].item())
        shared_full_blocks = min(prefix_ctx // block_size, max_blocks)
        has_partial_block = (prefix_ctx % block_size) != 0
        private_blocks_per_row = max_blocks - shared_full_blocks

        if private_blocks_per_row <= 0:
            return PagedKVCacheState(
                k_cache=state.k_cache.clone(),
                v_cache=state.v_cache.clone(),
                block_tables=state.block_tables.repeat(repeats, 1).clone(),
                context_lens=state.context_lens.repeat(repeats).clone(),
                last_tokens=state.last_tokens.repeat(repeats).clone(),
                max_context=state.max_context,
                block_size=state.block_size,
            )

        total_blocks = shared_full_blocks + repeats * private_blocks_per_row
        expanded_k = torch.zeros(
            (num_layers, total_blocks, num_kv_heads, block_size, head_dim),
            device=device,
            dtype=state.k_cache.dtype,
        )
        expanded_v = torch.zeros_like(expanded_k)

        block_tables = torch.empty(
            (repeats, max_blocks), device=device, dtype=torch.int32
        )

        if shared_full_blocks > 0:
            shared_source_ids = state.block_tables[0, :shared_full_blocks].to(torch.long)
            expanded_k[:, :shared_full_blocks].copy_(state.k_cache.index_select(1, shared_source_ids))
            expanded_v[:, :shared_full_blocks].copy_(state.v_cache.index_select(1, shared_source_ids))
            shared_ids = torch.arange(
                shared_full_blocks, device=device, dtype=torch.int32
            )
            block_tables[:, :shared_full_blocks] = shared_ids.unsqueeze(0)

        partial_source_block = None
        if has_partial_block:
            partial_source_block = int(state.block_tables[0, shared_full_blocks].item())

        row_private_ids = torch.arange(
            private_blocks_per_row, device=device, dtype=torch.int32
        )
        for row in range(repeats):
            row_private_start = shared_full_blocks + row * private_blocks_per_row
            private_ids = row_private_start + row_private_ids
            block_tables[row, shared_full_blocks:] = private_ids
            if partial_source_block is not None:
                expanded_k[:, row_private_start].copy_(state.k_cache[:, partial_source_block])
                expanded_v[:, row_private_start].copy_(state.v_cache[:, partial_source_block])

        return PagedKVCacheState(
            k_cache=expanded_k,
            v_cache=expanded_v,
            block_tables=block_tables,
            context_lens=state.context_lens.repeat(repeats).clone(),
            last_tokens=state.last_tokens.repeat(repeats).clone(),
            max_context=state.max_context,
            block_size=state.block_size,
        )

    base_k = state.k_cache.view(num_layers, base_batch, max_blocks, num_kv_heads, block_size, head_dim)
    base_v = state.v_cache.view(num_layers, base_batch, max_blocks, num_kv_heads, block_size, head_dim)
    expanded_k = (
        base_k.repeat(1, repeats, 1, 1, 1, 1)
        .reshape(num_layers, total_batch * max_blocks, num_kv_heads, block_size, head_dim)
        .contiguous()
    )
    expanded_v = (
        base_v.repeat(1, repeats, 1, 1, 1, 1)
        .reshape(num_layers, total_batch * max_blocks, num_kv_heads, block_size, head_dim)
        .contiguous()
    )

    return PagedKVCacheState(
        k_cache=expanded_k,
        v_cache=expanded_v,
        block_tables=_build_block_tables(total_batch, max_blocks, state.k_cache.device),
        context_lens=state.context_lens.repeat(repeats).clone(),
        last_tokens=state.last_tokens.repeat(repeats).clone(),
        max_context=state.max_context,
        block_size=state.block_size,
    )


@torch.no_grad()
def prefill_paged_kv_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    block_size: int = 16,
) -> PagedKVCacheState:
    """Prefill a reusable paged prompt cache excluding the prompt's last token."""
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available for paged_kv_decode.")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}.")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be rank-2 tensors.")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape.")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must have at least 1 token.")

    (
        _embeddings,
        layers,
        _norm,
        _rotary_emb,
        _lm_head,
        num_heads,
        num_kv_heads,
        head_dim,
        _hidden_size,
    ) = _get_model_components(model)

    prompt_lens = attention_mask.sum(dim=1).to(dtype=torch.int32)
    if torch.any(prompt_lens <= 0):
        raise ValueError("Each prompt must contain at least one unmasked token.")
    max_prompt_len = int(prompt_lens.max().item())
    max_context_needed = max(max_prompt_len - 1 + max_new_tokens + 1, 1)
    max_blocks = math.ceil(max_context_needed / block_size)
    total_blocks = input_ids.shape[0] * max_blocks

    k_cache = torch.zeros(
        (len(layers), total_blocks, num_kv_heads, block_size, head_dim),
        device=input_ids.device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.zeros_like(k_cache)
    block_tables = _build_block_tables(input_ids.shape[0], max_blocks, input_ids.device)
    context_lens = torch.clamp(prompt_lens - 1, min=0)

    if max_prompt_len > 1:
        _prefill_paged_cache(
            model,
            input_ids[:, :-1],
            attention_mask[:, :-1],
            k_cache,
            v_cache,
            block_tables,
            block_size,
            num_heads,
            num_kv_heads,
        )

    last_token_positions = (
        attention_mask.shape[1]
        - 1
        - attention_mask.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
    )
    last_tokens = input_ids.gather(1, last_token_positions.unsqueeze(1)).squeeze(1)

    return PagedKVCacheState(
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        last_tokens=last_tokens,
        max_context=max_context_needed,
        block_size=block_size,
    )


def expand_paged_kv_cache_state(
    state: PagedKVCacheState, repeats: int
) -> PagedKVCacheState:
    """Expand a single prefetched prompt cache into independent grouped copies."""
    return _clone_paged_cache_state(state, repeats)


def _append_kv_to_paged_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    block_size: int,
) -> None:
    block_idx = torch.div(context_lens, block_size, rounding_mode="floor")
    block_off = torch.remainder(context_lens, block_size)
    block_ids = torch.gather(block_tables, 1, block_idx.unsqueeze(1)).squeeze(1)

    k_slots = k_cache.permute(0, 2, 1, 3)
    v_slots = v_cache.permute(0, 2, 1, 3)
    k_slots[block_ids, block_off] = k_new
    v_slots[block_ids, block_off] = v_new


def _torch_paged_attention_decode(
    attn_module: torch.nn.Module,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    batch_size, _num_heads, head_dim = q.shape
    max_context = int(context_lens.max().item())
    positions = torch.arange(max_context, device=q.device, dtype=torch.int32)
    block_idx = torch.div(positions, block_size, rounding_mode="floor").unsqueeze(0)
    block_idx = block_idx.expand(batch_size, -1)
    block_off = torch.remainder(positions, block_size).unsqueeze(0).expand(batch_size, -1)
    block_ids = torch.gather(block_tables, 1, block_idx)

    k_seq = k_cache[block_ids, :, block_off].permute(0, 2, 1, 3)
    v_seq = v_cache[block_ids, :, block_off].permute(0, 2, 1, 3)
    mask = positions.unsqueeze(0) >= context_lens.unsqueeze(1)
    attention_mask = None
    if mask.any():
        attention_mask = torch.zeros(
            (batch_size, 1, 1, max_context),
            device=q.device,
            dtype=torch.float32,
        )
        attention_mask = attention_mask.masked_fill(
            mask[:, None, None, :],
            torch.finfo(attention_mask.dtype).min,
        )

    attn_impl = getattr(getattr(attn_module, "config", None), "_attn_implementation", None)
    get_interface = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
    if callable(get_interface):
        attention_interface = get_interface(attn_impl, eager_attention_forward)
    elif attn_impl == "eager" or attn_impl is None:
        attention_interface = eager_attention_forward
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get(attn_impl, eager_attention_forward)
    attn_out, _ = attention_interface(
        attn_module,
        q.unsqueeze(2),
        k_seq,
        v_seq,
        attention_mask,
        dropout=0.0,
        scaling=getattr(attn_module, "scaling", 1.0 / math.sqrt(head_dim)),
        is_causal=False,
        sliding_window=getattr(attn_module, "sliding_window", None),
    )
    return attn_out.squeeze(1)


@torch.no_grad()
def decode_from_paged_kv_cache(
    model: torch.nn.Module,
    state: PagedKVCacheState,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: Optional[int],
    seed: Optional[int],
) -> torch.Tensor:
    """Decode from a prefetched paged KV cache state."""
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available for paged_kv_decode.")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}.")

    (
        embeddings,
        layers,
        norm,
        rotary_emb,
        lm_head,
        num_heads,
        num_kv_heads,
        head_dim,
        _hidden_size,
    ) = _get_model_components(model)

    batch_size = state.block_tables.shape[0]
    if torch.any(state.context_lens + max_new_tokens > state.max_context):
        raise ValueError("Requested decode exceeds block table capacity.")

    generated = torch.full(
        (batch_size, max_new_tokens),
        pad_token_id,
        device=state.k_cache.device,
        dtype=state.last_tokens.dtype,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator(device=state.k_cache.device)
        generator.manual_seed(seed)

    context_lens = state.context_lens.clone()
    last_tokens = state.last_tokens.clone()
    finished = torch.zeros(batch_size, device=state.k_cache.device, dtype=torch.bool)
    update_kernel = cast(_TritonKernel, _paged_kv_update_kernel)
    block_d = _select_block_d(head_dim)
    update_warps = _select_num_warps_update(head_dim)

    def _step() -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        hidden_states = cast(torch.Tensor, embeddings(last_tokens))
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1)

        position_ids = context_lens.unsqueeze(1)
        cos, sin = rotary_emb(hidden_states, position_ids)
        new_context_lens = context_lens + 1

        for layer_idx, layer in enumerate(layers):
            layer_typed = cast(torch.nn.Module, layer)
            input_norm = cast(torch.nn.Module, getattr(layer_typed, "input_layernorm"))
            post_norm = cast(torch.nn.Module, getattr(layer_typed, "post_attention_layernorm"))
            attn = cast(torch.nn.Module, getattr(layer_typed, "self_attn"))
            mlp = cast(torch.nn.Module, getattr(layer_typed, "mlp"))

            residual = hidden_states
            hidden_states = cast(torch.Tensor, input_norm(hidden_states))
            q_states = cast(
                torch.Tensor,
                cast(torch.nn.Module, getattr(attn, "q_proj"))(hidden_states),
            )
            k_states = cast(
                torch.Tensor,
                cast(torch.nn.Module, getattr(attn, "k_proj"))(hidden_states),
            )
            v_states = cast(
                torch.Tensor,
                cast(torch.nn.Module, getattr(attn, "v_proj"))(hidden_states),
            )
            q_states = q_states.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
            k_states = k_states.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_states.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
            q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)
            q = q_states[:, :, 0, :].to(dtype=torch.bfloat16).contiguous()
            k_new = k_states[:, :, 0, :].to(dtype=torch.bfloat16).contiguous()
            v_new = v_states[:, :, 0, :].to(dtype=torch.bfloat16).contiguous()

            update_kernel[(batch_size, num_kv_heads)](
                k_new,
                v_new,
                state.k_cache[layer_idx],
                state.v_cache[layer_idx],
                state.block_tables,
                context_lens,
                k_new.stride(0),
                k_new.stride(1),
                k_new.stride(2),
                state.k_cache[layer_idx].stride(0),
                state.k_cache[layer_idx].stride(1),
                state.k_cache[layer_idx].stride(2),
                state.k_cache[layer_idx].stride(3),
                state.block_tables.stride(0),
                state.block_tables.stride(1),
                state.block_size,
                num_kv_heads,
                D=head_dim,
                BLOCK_D=block_d,
                num_warps=update_warps,
                num_stages=1,
            )

            attn_out = _torch_paged_attention_decode(
                attn,
                q,
                state.k_cache[layer_idx],
                state.v_cache[layer_idx],
                state.block_tables,
                new_context_lens,
                state.block_size,
            )
            attn_out = attn_out.to(hidden_states.dtype)
            attn_out = attn_out.reshape(batch_size, 1, num_heads * head_dim)
            attn_out = cast(
                torch.Tensor,
                cast(torch.nn.Module, getattr(attn, "o_proj"))(attn_out),
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = cast(torch.Tensor, post_norm(hidden_states))
            hidden_states = cast(torch.Tensor, mlp(hidden_states))
            hidden_states = residual + hidden_states

        hidden_states = norm(hidden_states)
        logits = cast(torch.Tensor, lm_head(hidden_states[:, -1, :]))
        next_tokens = _sample_tokens(
            logits,
            do_sample,
            temperature,
            top_p,
            generator,
        )
        return next_tokens, new_context_lens

    for step in range(max_new_tokens):
        if finished.all():
            break
        next_tokens, new_context_lens = _step()
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, pad_token_id),
            next_tokens,
        )
        generated[:, step] = next_tokens
        last_tokens = next_tokens
        if eos_token_id is not None and eos_token_id >= 0:
            finished = finished | (next_tokens == eos_token_id)
        context_lens = new_context_lens

    return generated


@torch.no_grad()
def paged_kv_decode_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: Optional[int],
    seed: Optional[int],
) -> torch.Tensor:
    prefix_state = prefill_paged_kv_cache(
        model,
        input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
    )
    return decode_from_paged_kv_cache(
        model,
        prefix_state,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        seed=seed,
    )


@torch.no_grad()
def paged_kv_decode(
    input_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    qkv_proj_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    seed: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available for paged_kv_decode.")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}.")
    update_kernel = cast(_TritonKernel, _paged_kv_update_kernel)
    decode_kernel = cast(_TritonKernel, _paged_attention_decode_kernel)

    num_kv_heads, block_size, head_dim, max_context = _validate_inputs(
        input_ids, k_cache, v_cache, block_tables, context_lens
    )

    if context_lens.max().item() >= max_context:
        raise ValueError("context_lens exceed block table capacity.")
    if torch.any(context_lens + max_new_tokens > max_context):
        raise ValueError("Requested decode exceeds block table capacity.")

    if eos_token_id is None:
        eos_token_id = -1
    if pad_token_id is None:
        pad_token_id = 0

    kv_group_size = 1

    batch_size = input_ids.shape[0]
    generated = torch.full(
        (batch_size, max_new_tokens),
        pad_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed)

    last_tokens = input_ids[:, -1]
    finished = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)

    block_ctx = _select_block_ctx(max_context)
    block_d = _select_block_d(head_dim)
    update_warps = _select_num_warps_update(head_dim)
    decode_warps = _select_num_warps_decode(block_d, block_ctx)
    decode_stages = _select_num_stages_decode(max_context)
    scale = 1.0 / math.sqrt(head_dim)

    for step in range(max_new_tokens):
        if finished.all():
            break

        q, k_new, v_new = qkv_proj_fn(last_tokens)
        if q.ndim != 3 or q.shape[0] != batch_size or q.shape[2] != head_dim:
            raise ValueError("qkv_proj_fn must return [B, H, D] tensors.")
        num_heads = q.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"q heads ({num_heads}) must be divisible by cache kv heads ({num_kv_heads})."
            )
        kv_group_size = num_heads // num_kv_heads
        if k_new.shape != (batch_size, num_kv_heads, head_dim):
            raise ValueError(
                "qkv_proj_fn must return [B, H_kv, D] tensors for keys matching cache heads."
            )
        if v_new.shape != (batch_size, num_kv_heads, head_dim):
            raise ValueError(
                "qkv_proj_fn must return [B, H_kv, D] tensors for values matching cache heads."
            )

        q = q.to(dtype=torch.bfloat16)
        k_new = k_new.to(dtype=torch.bfloat16)
        v_new = v_new.to(dtype=torch.bfloat16)

        update_grid = (batch_size, num_kv_heads)
        update_kernel[update_grid](
            k_new,
            v_new,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_new.stride(0),
            k_new.stride(1),
            k_new.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            block_size,
            num_kv_heads,
            D=head_dim,
            BLOCK_D=block_d,
            num_warps=update_warps,
            num_stages=1,
        )

        context_lens = context_lens + (~finished).to(context_lens.dtype)
        attn_out = torch.empty_like(q, dtype=torch.float32)

        attn_grid = (batch_size, num_heads)
        decode_kernel[attn_grid](
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            attn_out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            attn_out.stride(0),
            attn_out.stride(1),
            attn_out.stride(2),
            block_size,
            kv_group_size,
            scale,
            D=head_dim,
            BLOCK_CTX=block_ctx,
            BLOCK_D=block_d,
            MAX_CONTEXT=max_context,
            num_warps=decode_warps,
            num_stages=decode_stages,
        )

        logits = logits_fn(attn_out)
        if logits.ndim != 2 or logits.shape[0] != batch_size:
            raise ValueError("logits_fn must return [B, vocab] logits.")

        next_tokens = _sample_tokens(
            logits, do_sample, temperature, top_p, generator
        )
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, pad_token_id),
            next_tokens,
        )
        generated[:, step] = next_tokens
        last_tokens = next_tokens

        if eos_token_id >= 0:
            finished = finished | (next_tokens == eos_token_id)

    return generated


__all__ = [
    "PagedKVCacheState",
    "decode_from_paged_kv_cache",
    "expand_paged_kv_cache_state",
    "paged_kv_decode",
    "paged_kv_decode_model",
    "prefill_paged_kv_cache",
]
