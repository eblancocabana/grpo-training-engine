import types
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.grpo.trainer import GRPOTrainerLoop
from src.triton_kernels import TRITON_AVAILABLE
from src.triton_kernels.fused_ops import (
    fused_logits_sampling,
    fused_rmsnorm,
    fused_silu_mul,
)
from src.triton_kernels.paged_kv import (
    PagedKVCacheState,
    expand_paged_kv_cache_state,
    paged_kv_decode_model,
)
from src.utils.config import get_8gb_vram_config


def _skip_if_no_cuda_or_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton generation correctness tests")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton is required for Triton generation correctness tests")


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def batch_decode(tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(map(str, row.tolist())) for row in tokens]


class _DummyMemoryManager:
    @staticmethod
    def clear_cache(*args, **kwargs):
        del args, kwargs
        return None


def _build_loop(model: nn.Module, *, use_triton_generation: bool, group_size: int) -> GRPOTrainerLoop:
    config = get_8gb_vram_config()
    config.training.use_triton_kernels = True
    config.training.use_triton_generation = use_triton_generation
    config.training.use_triton_grpo_loss = False
    config.training.use_triton_entropy_mask = False
    config.training.use_triton_lora = False
    config.training.generation_do_sample = False
    config.training.max_response_length = 6
    config.grpo.group_size = group_size

    loop = GRPOTrainerLoop(config)
    loop.model = model
    loop.tokenizer = _DummyTokenizer()
    loop.memory_manager = _DummyMemoryManager()
    loop.group_sampler = types.SimpleNamespace(group_size=group_size)
    loop._gen_micro_batch = 2
    loop.device = next(model.parameters()).device
    return loop


def _tiny_qwen_model(device: torch.device) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_cache=True,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = Qwen2ForCausalLM(config)
    return cast(Qwen2ForCausalLM, model.to(device=device, dtype=torch.bfloat16).eval())


def test_expand_paged_kv_cache_state_clones_multi_prompt_rows() -> None:
    state = PagedKVCacheState(
        k_cache=torch.arange(2 * 6 * 1 * 2 * 4, dtype=torch.float32).view(2, 6, 1, 2, 4),
        v_cache=torch.arange(2 * 6 * 1 * 2 * 4, dtype=torch.float32).view(2, 6, 1, 2, 4) + 1000,
        block_tables=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
        context_lens=torch.tensor([2, 1], dtype=torch.int32),
        last_tokens=torch.tensor([7, 8], dtype=torch.long),
        max_context=6,
        block_size=2,
    )

    expanded = expand_paged_kv_cache_state(state, repeats=2)

    assert expanded.block_tables.shape == (4, 3)
    assert torch.equal(expanded.context_lens, torch.tensor([2, 1, 2, 1], dtype=torch.int32))
    assert torch.equal(expanded.last_tokens, torch.tensor([7, 8, 7, 8], dtype=torch.long))
    assert torch.equal(expanded.block_tables[0], torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(expanded.block_tables[1], torch.tensor([3, 4, 5], dtype=torch.int32))
    assert torch.equal(expanded.block_tables[2], torch.tensor([6, 7, 8], dtype=torch.int32))
    assert torch.equal(expanded.block_tables[3], torch.tensor([9, 10, 11], dtype=torch.int32))
    assert torch.equal(expanded.k_cache[:, 0:6], state.k_cache)
    assert torch.equal(expanded.k_cache[:, 6:12], state.k_cache)


def test_expand_paged_kv_cache_state_shares_immutable_prefix_blocks() -> None:
    state = PagedKVCacheState(
        k_cache=torch.arange(1 * 4 * 1 * 2 * 2, dtype=torch.float32).view(1, 4, 1, 2, 2),
        v_cache=torch.arange(1 * 4 * 1 * 2 * 2, dtype=torch.float32).view(1, 4, 1, 2, 2) + 100,
        block_tables=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        context_lens=torch.tensor([3], dtype=torch.int32),  # one full block + one partial token
        last_tokens=torch.tensor([9], dtype=torch.long),
        max_context=8,
        block_size=2,
    )

    expanded = expand_paged_kv_cache_state(state, repeats=3)

    assert expanded.block_tables.shape == (3, 4)
    assert torch.equal(expanded.context_lens, torch.tensor([3, 3, 3], dtype=torch.int32))
    assert torch.equal(expanded.last_tokens, torch.tensor([9, 9, 9], dtype=torch.long))
    assert torch.equal(expanded.block_tables[:, 0], torch.tensor([0, 0, 0], dtype=torch.int32))
    assert torch.equal(expanded.k_cache[:, 0], state.k_cache[:, 0])
    assert torch.equal(expanded.v_cache[:, 0], state.v_cache[:, 0])

    # The partially filled block must be copied into each row's private mutable tail.
    assert torch.equal(expanded.k_cache[:, 1], state.k_cache[:, 1])
    assert torch.equal(expanded.k_cache[:, 4], state.k_cache[:, 1])
    assert torch.equal(expanded.k_cache[:, 7], state.k_cache[:, 1])
    assert torch.equal(expanded.v_cache[:, 1], state.v_cache[:, 1])
    assert torch.equal(expanded.v_cache[:, 4], state.v_cache[:, 1])
    assert torch.equal(expanded.v_cache[:, 7], state.v_cache[:, 1])


def test_triton_generation_matches_torch_prefix_cache_on_tiny_qwen_greedy() -> None:
    _skip_if_no_cuda_or_triton()

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = _tiny_qwen_model(device)

    input_ids = torch.tensor(
        [
            [0, 0, 11, 12, 13],
            [0, 21, 22, 23, 24],
        ],
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    torch_loop = _build_loop(model, use_triton_generation=False, group_size=3)
    triton_loop = _build_loop(model, use_triton_generation=True, group_size=3)

    with torch.inference_mode():
        texts_torch, ids_torch, mask_torch = torch_loop._generate_responses_with_tokens(
            input_ids, attention_mask
        )
        texts_triton, ids_triton, mask_triton = triton_loop._generate_responses_with_tokens(
            input_ids, attention_mask
        )

    assert texts_triton == texts_torch
    torch.testing.assert_close(ids_triton, ids_torch)
    assert torch.equal(mask_triton, mask_torch)


def test_paged_kv_decode_model_handles_mixed_prompt_lengths() -> None:
    _skip_if_no_cuda_or_triton()

    device = torch.device("cuda")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model = _tiny_qwen_model(device)

    input_ids = torch.tensor(
        [
            [0, 0, 11, 12, 13],
            [0, 21, 22, 23, 24],
        ],
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    def _normalize_rows(output_ids: torch.Tensor) -> list[list[int]]:
        rows: list[list[int]] = []
        for row in output_ids.detach().cpu():
            valid_len = row.shape[0]
            eos_positions = (row == _DummyTokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                valid_len = int(eos_positions[0].item()) + 1
            else:
                non_pad_positions = (row != _DummyTokenizer.pad_token_id).nonzero(as_tuple=False)
                valid_len = (
                    int(non_pad_positions[-1].item()) + 1 if non_pad_positions.numel() > 0 else 0
                )
            rows.append(row[:valid_len].tolist())
        return rows

    with torch.inference_mode():
        batched = paged_kv_decode_model(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            block_size=16,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=_DummyTokenizer.pad_token_id,
            eos_token_id=_DummyTokenizer.eos_token_id,
            seed=None,
        )
        single_rows = [
            paged_kv_decode_model(
                model=model,
                input_ids=input_ids[row_idx : row_idx + 1],
                attention_mask=attention_mask[row_idx : row_idx + 1],
                max_new_tokens=5,
                block_size=16,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=_DummyTokenizer.pad_token_id,
                eos_token_id=_DummyTokenizer.eos_token_id,
                seed=None,
            )
            for row_idx in range(input_ids.shape[0])
        ]

    assert _normalize_rows(batched) == [
        _normalize_rows(single_row)[0] for single_row in single_rows
    ]


def test_triton_generation_auto_prefers_torch_for_sampled_qwen2_group_size_one() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(
                model_type="qwen2",
                num_attention_heads=12,
                num_key_value_heads=2,
            )

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=1)
    loop.config.training.triton_generation_mode = "auto"
    loop.config.training.generation_do_sample = True

    with patch("src.grpo.trainer.TRITON_AVAILABLE", True):
        enabled, reason = loop._resolve_triton_generation_decision()

    assert enabled is False
    assert "group_size=1" in reason
    assert "torch prefix-cache" in reason


def test_triton_generation_auto_runtime_prefers_torch_when_tail_microbatch_is_one() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(
                model_type="qwen2",
                num_attention_heads=12,
                num_key_value_heads=2,
            )

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=3)
    loop.config.training.triton_generation_mode = "auto"
    loop.config.training.generation_do_sample = True
    loop._gen_micro_batch = 2

    with patch("src.grpo.trainer.TRITON_AVAILABLE", True):
        enabled, reason = loop._resolve_triton_generation_runtime_decision(
            group_size=3,
            micro_batch_size=2,
        )

    assert enabled is False
    assert "microbatch schedule includes a batch of 1" in reason
    assert "torch prefix-cache" in reason


def test_triton_generation_runtime_tail_microbatch_uses_torch_path() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(
                model_type="qwen2",
                num_attention_heads=12,
                num_key_value_heads=2,
            )

        def eval(self):
            return self

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=3)
    loop.config.training.triton_generation_mode = "auto"
    loop.config.training.generation_do_sample = True
    loop._gen_micro_batch = 2
    input_ids = torch.tensor([[0, 11, 12]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)

    with patch.object(
        loop,
        "_prefill_triton_prompt_cache",
        side_effect=AssertionError("runtime auto policy should not prefill Triton"),
    ), patch.object(
        loop,
        "_generate_with_triton_paged_prefix_cache",
        side_effect=AssertionError("runtime auto policy should not decode with Triton"),
    ), patch.object(
        loop,
        "_prefill_prompt_cache",
        return_value=object(),
    ) as torch_prefill, patch.object(
        loop,
        "_generate_with_expanded_prefix_cache",
        side_effect=[
            torch.tensor([[7], [8]], dtype=torch.long),
            torch.tensor([[9]], dtype=torch.long),
        ],
    ) as torch_decode, patch("src.grpo.trainer.TRITON_AVAILABLE", True):
        texts, response_ids, response_mask = loop._generate_responses_with_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    assert torch_prefill.call_count == 1
    assert torch_decode.call_count == 2
    assert texts == ["7", "8", "9"]
    torch.testing.assert_close(response_ids, torch.tensor([[7], [8], [9]], dtype=torch.long))
    assert torch.equal(response_mask, torch.ones_like(response_ids))


def test_triton_generation_auto_prefers_triton_for_sampled_qwen2_group_size_two_or_more() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(
                model_type="qwen2",
                num_attention_heads=12,
                num_key_value_heads=2,
            )

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=4)
    loop.config.training.triton_generation_mode = "auto"
    loop.config.training.generation_do_sample = True

    with patch("src.grpo.trainer.TRITON_AVAILABLE", True):
        enabled, reason = loop._resolve_triton_generation_decision()

    assert enabled is True
    assert "group_size>=2" in reason
    assert "Triton" in reason


def test_triton_generation_auto_runtime_keeps_triton_when_all_microbatches_are_two_or_more() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(
                model_type="qwen2",
                num_attention_heads=12,
                num_key_value_heads=2,
            )

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=4)
    loop.config.training.triton_generation_mode = "auto"
    loop.config.training.generation_do_sample = True
    loop._gen_micro_batch = 2

    with patch("src.grpo.trainer.TRITON_AVAILABLE", True):
        enabled, reason = loop._resolve_triton_generation_runtime_decision(
            group_size=4,
            micro_batch_size=2,
        )

    assert enabled is True
    assert "Triton paged-KV" in reason


def test_triton_generation_mode_on_keeps_supported_qwen2_enabled() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(
                model_type="qwen2",
                num_attention_heads=12,
                num_key_value_heads=2,
            )

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=4)
    loop.config.training.triton_generation_mode = "on"
    loop.config.training.generation_do_sample = True

    with patch("src.grpo.trainer.TRITON_AVAILABLE", True):
        enabled, reason = loop._resolve_triton_generation_decision()

    assert enabled is True
    assert reason == "Triton paged-KV generation enabled"


@pytest.mark.parametrize("top_p", [0.0, -0.5])
def test_fused_logits_sampling_treats_nonpositive_top_p_as_no_filter(top_p: float) -> None:
    hidden = torch.zeros((1, 4), dtype=torch.float32)
    lm_head = nn.Identity()
    generator_no_filter = torch.Generator().manual_seed(0)
    generator_boundary = torch.Generator().manual_seed(0)

    logits_no_filter, tokens_no_filter = fused_logits_sampling(
        hidden,
        lm_head,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        generator=generator_no_filter,
        eos_token_id=None,
    )
    logits_boundary, tokens_boundary = fused_logits_sampling(
        hidden,
        lm_head,
        do_sample=True,
        temperature=1.0,
        top_p=top_p,
        generator=generator_boundary,
        eos_token_id=None,
    )

    torch.testing.assert_close(logits_boundary, logits_no_filter)
    torch.testing.assert_close(tokens_boundary, tokens_no_filter)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 5e-3, 5e-3),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_fused_rmsnorm_preserves_dtype_and_matches_reference(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    _skip_if_no_cuda_or_triton()

    x = torch.randn(2, 3, 32, device="cuda", dtype=dtype)
    weight = torch.randn(32, device="cuda", dtype=dtype)
    out = fused_rmsnorm(x, weight, 1e-6)

    ref = x.float()
    ref = ref * torch.rsqrt(ref.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = (ref * weight.float()).to(dtype)

    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 5e-3, 5e-3),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_fused_silu_mul_preserves_dtype_and_matches_reference(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    _skip_if_no_cuda_or_triton()

    x = torch.randn(2, 3, 32, device="cuda", dtype=dtype)
    y = torch.randn(2, 3, 32, device="cuda", dtype=dtype)
    out = fused_silu_mul(x, y)
    ref = (torch.nn.functional.silu(x.float()) * y.float()).to(dtype)

    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
