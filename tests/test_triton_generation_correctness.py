import types
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.grpo.trainer import GRPOTrainerLoop
from src.triton_kernels import TRITON_AVAILABLE
from src.triton_kernels.paged_kv import PagedKVCacheState, expand_paged_kv_cache_state
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


def test_triton_generation_prefills_once_per_prompt_and_expands_per_microbatch() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

    loop = _build_loop(DummyModel(), use_triton_generation=True, group_size=5)
    input_ids = torch.tensor([[0, 11, 12], [0, 21, 22]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long)

    fake_state = PagedKVCacheState(
        k_cache=torch.zeros((1, 1, 1, 1, 1)),
        v_cache=torch.zeros((1, 1, 1, 1, 1)),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        context_lens=torch.zeros((1,), dtype=torch.int32),
        last_tokens=torch.tensor([12]),
        max_context=8,
        block_size=1,
    )

    def _decode(_model, state, **kwargs):
        del _model, kwargs
        batch = state.last_tokens.shape[0]
        return torch.arange(1, batch + 1, dtype=torch.long).unsqueeze(1)

    with patch("src.grpo.trainer.prefill_paged_kv_cache", return_value=fake_state) as prefill, patch(
        "src.grpo.trainer.expand_paged_kv_cache_state",
        side_effect=lambda state, repeats: PagedKVCacheState(
            k_cache=state.k_cache,
            v_cache=state.v_cache,
            block_tables=torch.zeros((repeats, 1), dtype=torch.int32),
            context_lens=torch.zeros((repeats,), dtype=torch.int32),
            last_tokens=torch.full((repeats,), 5, dtype=torch.long),
            max_context=state.max_context,
            block_size=state.block_size,
        ),
    ) as expand, patch(
        "src.grpo.trainer.decode_from_paged_kv_cache", side_effect=_decode
    ) as decode, patch.object(
        loop, "_prefill_prompt_cache", side_effect=AssertionError("torch prefill should not run")
    ):
        texts, response_ids, response_mask = loop._generate_responses_with_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    assert prefill.call_count == 2
    assert expand.call_count == 6
    assert decode.call_count == 6
    assert response_ids.shape[0] == 10
    assert response_mask.shape == response_ids.shape
    assert len(texts) == 10


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


def test_triton_generation_auto_prefers_torch_for_sampled_qwen2() -> None:
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

    assert enabled is False
    assert "torch prefix-cache" in reason


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
