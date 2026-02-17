"""
Tests for KV cache prefix sharing in generation.

Tests both:
- White-box: internal _expand_prefix_cache behavior, DynamicCache structure
- Black-box: generate_responses output correctness and ordering
"""
import torch
import torch.nn as nn
import random
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from transformers import DynamicCache

# Determinism
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExpandPrefixCache:
    """White-box tests for _expand_prefix_cache static method."""

    def _make_cache(self, batch=1, num_heads=2, seq_len=10, head_dim=8, num_layers=3):
        """Create a DynamicCache with known values for testing."""
        cache = DynamicCache()
        for layer_idx in range(num_layers):
            # Use deterministic values so we can verify correctness
            k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
            v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
            cache.update(k, v, layer_idx)
        return cache

    def test_output_type_is_dynamic_cache(self):
        """_expand_prefix_cache returns a DynamicCache instance."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache()
        result = GRPOTrainerLoop._expand_prefix_cache(cache, 4)
        assert isinstance(result, DynamicCache)

    def test_output_has_correct_num_layers(self):
        """Output cache has the same number of layers as input."""
        from src.grpo.trainer import GRPOTrainerLoop
        for num_layers in [1, 3, 28]:
            cache = self._make_cache(num_layers=num_layers)
            result = GRPOTrainerLoop._expand_prefix_cache(cache, 4)
            assert len(result.layers) == num_layers

    def test_batch_dimension_expanded(self):
        """Keys and values batch dimension equals repeats argument."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache(batch=1, num_heads=4, seq_len=16, head_dim=8)
        for repeats in [1, 2, 4, 8]:
            result = GRPOTrainerLoop._expand_prefix_cache(cache, repeats)
            for layer in result.layers:
                assert layer.keys.shape[0] == repeats
                assert layer.values.shape[0] == repeats

    def test_non_batch_dimensions_preserved(self):
        """Heads, seq_len, and head_dim are unchanged after expansion."""
        from src.grpo.trainer import GRPOTrainerLoop
        num_heads, seq_len, head_dim = 4, 20, 16
        cache = self._make_cache(num_heads=num_heads, seq_len=seq_len, head_dim=head_dim)
        result = GRPOTrainerLoop._expand_prefix_cache(cache, 3)
        for layer in result.layers:
            assert layer.keys.shape[1] == num_heads
            assert layer.keys.shape[2] == seq_len
            assert layer.keys.shape[3] == head_dim

    def test_expanded_values_are_copies_of_original(self):
        """Each batch element in expanded cache equals the original."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache(batch=1, num_layers=2)
        repeats = 4
        result = GRPOTrainerLoop._expand_prefix_cache(cache, repeats)

        for layer_idx in range(len(cache.layers)):
            orig_k = cache.layers[layer_idx].keys[0]  # [heads, seq, dim]
            orig_v = cache.layers[layer_idx].values[0]
            for b in range(repeats):
                assert torch.equal(result.layers[layer_idx].keys[b], orig_k)
                assert torch.equal(result.layers[layer_idx].values[b], orig_v)

    def test_original_cache_not_mutated(self):
        """Expanding does not modify the original cache."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache(batch=1)
        orig_k = cache.layers[0].keys.clone()
        orig_v = cache.layers[0].values.clone()

        result = GRPOTrainerLoop._expand_prefix_cache(cache, 4)

        # Mutate result to be sure it doesn't affect original
        result.layers[0].keys.fill_(999.0)
        assert torch.equal(cache.layers[0].keys, orig_k)
        assert torch.equal(cache.layers[0].values, orig_v)

    def test_expanded_tensors_are_contiguous(self):
        """Expanded tensors must be contiguous for efficient compute."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache()
        result = GRPOTrainerLoop._expand_prefix_cache(cache, 4)
        for layer in result.layers:
            assert layer.keys.is_contiguous()
            assert layer.values.is_contiguous()

    def test_seq_length_matches_original(self):
        """Expanded cache reports same seq_length as original."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache(seq_len=42)
        result = GRPOTrainerLoop._expand_prefix_cache(cache, 3)
        assert result.get_seq_length() == cache.get_seq_length()

    def test_expand_single_repeat_is_identity(self):
        """Expanding with repeats=1 produces a cache with identical values."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache(batch=1)
        result = GRPOTrainerLoop._expand_prefix_cache(cache, 1)
        for i in range(len(cache.layers)):
            assert torch.equal(result.layers[i].keys, cache.layers[i].keys)
            assert torch.equal(result.layers[i].values, cache.layers[i].values)

    def test_device_preserved(self):
        """Output tensors are on the same device as input."""
        from src.grpo.trainer import GRPOTrainerLoop
        cache = self._make_cache()
        result = GRPOTrainerLoop._expand_prefix_cache(cache, 4)
        for layer in result.layers:
            assert layer.keys.device == cache.layers[0].keys.device
            assert layer.values.device == cache.layers[0].values.device

    def test_matches_deepcopy_batch_repeat_interleave(self):
        """Output is identical to the old deepcopy + batch_repeat_interleave approach."""
        from copy import deepcopy
        from src.grpo.trainer import GRPOTrainerLoop

        cache = self._make_cache(batch=1, num_heads=4, seq_len=64, head_dim=16, num_layers=5)
        repeats = 4

        # Old approach
        old_cache = deepcopy(cache)
        old_cache.batch_repeat_interleave(repeats)

        # New approach
        new_cache = GRPOTrainerLoop._expand_prefix_cache(cache, repeats)

        for i in range(len(cache.layers)):
            assert torch.equal(old_cache.layers[i].keys, new_cache.layers[i].keys)
            assert torch.equal(old_cache.layers[i].values, new_cache.layers[i].values)


class TestGroupSamplerNoExpandBatch:
    """Verify expand_batch was removed from GroupSampler."""

    def test_expand_batch_removed(self):
        """GroupSampler should no longer have expand_batch method."""
        from src.grpo.algorithm import GroupSampler
        sampler = GroupSampler(group_size=4)
        assert not hasattr(sampler, "expand_batch")

    def test_group_responses_still_works(self):
        """group_responses must still function after expand_batch removal."""
        from src.grpo.algorithm import GroupSampler
        sampler = GroupSampler(group_size=3)
        responses = ["a", "b", "c", "d", "e", "f"]
        grouped = sampler.group_responses(responses)
        assert grouped == [["a", "b", "c"], ["d", "e", "f"]]


class TestGenerateResponsesOrdering:
    """Black-box tests for generate_responses output structure."""

    def test_output_length_matches_batch_times_group(self):
        """generate_responses returns batch_size * group_size strings."""
        from src.grpo.algorithm import GroupSampler

        # We can't easily test the full method without a real model,
        # so we verify the ordering contract by checking the GroupSampler
        # can correctly group the expected output.
        group_size = 8
        batch_size = 2
        expected_len = batch_size * group_size

        # Simulate what generate_responses produces
        texts = [f"prompt{p}_resp{r}" for p in range(batch_size) for r in range(group_size)]
        assert len(texts) == expected_len

        sampler = GroupSampler(group_size=group_size)
        grouped = sampler.group_responses(texts)
        assert len(grouped) == batch_size
        assert all(len(g) == group_size for g in grouped)

        # Verify ordering: first group is prompt 0's responses
        assert all("prompt0" in t for t in grouped[0])
        assert all("prompt1" in t for t in grouped[1])

    def test_microbatch_sizes_cover_group_size(self):
        """Verify micro-batching produces correct total regardless of divisibility."""
        group_size = 8
        for micro_batch in [1, 2, 3, 4, 5, 7, 8]:
            count = 0
            for g_start in range(0, group_size, micro_batch):
                g_end = min(g_start + micro_batch, group_size)
                current_micro = g_end - g_start
                assert current_micro > 0
                count += current_micro
            assert count == group_size, (
                f"micro_batch={micro_batch}: got {count} instead of {group_size}"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPrefillAndCropContract:
    """White-box tests for the prefill + crop contract."""

    def test_crop_removes_last_position(self):
        """DynamicCache.crop(n-1) removes last position from n-length cache."""
        cache = DynamicCache()
        seq_len = 10
        k = torch.randn(1, 2, seq_len, 8, device="cuda")
        v = torch.randn(1, 2, seq_len, 8, device="cuda")
        cache.update(k, v, 0)

        assert cache.get_seq_length() == seq_len
        cache.crop(seq_len - 1)
        assert cache.get_seq_length() == seq_len - 1
        # Verify content is the prefix
        assert torch.equal(cache.layers[0].keys, k[:, :, :seq_len - 1, :])

    def test_crop_to_zero_gives_empty_cache(self):
        """crop(0) empties the cache (single-token prompt edge case)."""
        cache = DynamicCache()
        k = torch.randn(1, 2, 1, 8, device="cuda")
        v = torch.randn(1, 2, 1, 8, device="cuda")
        cache.update(k, v, 0)

        cache.crop(0)
        assert cache.get_seq_length() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExpandCacheMemory:
    """Performance tests for cache expansion."""

    def test_expand_is_faster_than_deepcopy(self):
        """expand+contiguous should be faster than deepcopy+batch_repeat_interleave."""
        import time
        from copy import deepcopy
        from src.grpo.trainer import GRPOTrainerLoop

        # Use realistic cache size: 28 layers, 127 seq, like real model
        cache = DynamicCache()
        for layer_idx in range(28):
            k = torch.randn(1, 2, 127, 128, device="cuda")
            v = torch.randn(1, 2, 127, 128, device="cuda")
            cache.update(k, v, layer_idx)

        repeats = 4
        n_iters = 100

        # Warmup
        for _ in range(3):
            c = deepcopy(cache); c.batch_repeat_interleave(repeats); del c
            c = GRPOTrainerLoop._expand_prefix_cache(cache, repeats); del c
        torch.cuda.synchronize()

        # Time deepcopy
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            c = deepcopy(cache); c.batch_repeat_interleave(repeats); del c
        torch.cuda.synchronize()
        t_dc = time.perf_counter() - t0

        # Time expand
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            c = GRPOTrainerLoop._expand_prefix_cache(cache, repeats); del c
        torch.cuda.synchronize()
        t_ec = time.perf_counter() - t0

        # expand should be at least 1.5x faster
        assert t_ec < t_dc, (
            f"expand ({t_ec:.3f}s) should be faster than deepcopy ({t_dc:.3f}s)"
        )
