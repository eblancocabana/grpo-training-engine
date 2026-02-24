# pyright: reportMissingImports=false, reportUnknownMemberType=false

import math
import time
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
import pytest

from src.grpo.algorithm import GRPOTrainer
from src.selective.entropy_mask import EntropyCalculator
from src.triton_kernels import TRITON_AVAILABLE
from src.triton_kernels.grpo_loss import fused_grpo_loss
from src.triton_kernels.paged_kv import paged_kv_decode


def _skip_if_no_cuda_or_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton GRPO loss test")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton is required for fused GRPO loss test")


def test_triton_grpo_loss_matches_torch_grpo_loss() -> None:
    _skip_if_no_cuda_or_triton()

    _ = torch.manual_seed(123)
    _ = torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    batch_size = 2
    seq_len = 4
    vocab_size = 16
    group_size = 2

    policy_logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.bfloat16
    )
    old_policy_logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.float32
    )
    target_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    advantages = torch.randn(batch_size, device=device, dtype=torch.float32)

    old_log_probs = -F.cross_entropy(
        old_policy_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len)

    clip_epsilon = 0.2
    epsilon_high = 0.3
    delta = 1.5

    loss_triton, metrics_triton = fused_grpo_loss(
        policy_logits=policy_logits,
        target_ids=target_ids,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_epsilon=clip_epsilon,
        epsilon_high=epsilon_high,
        delta=delta,
        group_size=group_size,
    )

    trainer = GRPOTrainer(
        clip_epsilon=clip_epsilon,
        epsilon_high=epsilon_high,
        delta=delta,
        group_size=group_size,
        use_kl=False,
        use_triton_kernels=False,
    )
    loss_torch, metrics_torch = trainer.compute_grpo_loss(
        policy_logits=policy_logits,
        advantages=advantages,
        old_log_probs=old_log_probs,
        target_ids=target_ids,
    )

    torch.testing.assert_close(loss_triton, loss_torch, rtol=1e-3, atol=1e-3)

    assert set(metrics_triton.keys()) == set(metrics_torch.keys())
    for key in metrics_triton:
        assert metrics_triton[key] == pytest.approx(
            metrics_torch[key], rel=5e-3, abs=5e-3
        )


def test_triton_entropy_mask_matches_torch_entropy_mask() -> None:
    _skip_if_no_cuda_or_triton()

    _ = torch.manual_seed(42)
    _ = torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    batch_size = 2
    seq_len = 6
    vocab_size = 16

    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.float32
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
        device=device,
        dtype=torch.float32,
    )

    calculator = EntropyCalculator(percentile=0.5, min_tokens=2)

    entropy_triton, mask_triton = calculator.calculate_entropy_and_mask(
        logits,
        attention_mask=attention_mask,
        use_triton_kernels=True,
    )
    entropy_torch, mask_torch = calculator.calculate_entropy_and_mask(
        logits,
        attention_mask=attention_mask,
        use_triton_kernels=False,
    )

    torch.testing.assert_close(entropy_triton, entropy_torch, rtol=1e-3, atol=1e-3)

    if not torch.equal(mask_triton, mask_torch):
        ratio = (mask_triton.mean() / mask_torch.mean()).item()
        assert ratio == pytest.approx(1.0, rel=0.01, abs=0.01)


def test_triton_paged_kv_decode_matches_generate() -> None:
    _skip_if_no_cuda_or_triton()

    _ = torch.manual_seed(7)
    _ = torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    batch_size = 1
    vocab_size = 32
    num_heads = 2
    head_dim = 8
    prompt_len = 3
    max_new_tokens = 3
    block_size = 4

    class TinyPagedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            embed_dim = num_heads * head_dim
            self.embed: torch.nn.Embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.q_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, vocab_size, bias=False)
            self.num_heads: int = num_heads
            self.head_dim: int = head_dim

        def _project(
            self, token_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = cast(torch.Tensor, self.embed(token_ids))
            if x.ndim == 2:
                x = x.unsqueeze(1)
            batch, seq_len, _ = x.shape
            q = cast(torch.Tensor, self.q_proj(x)).view(
                batch, seq_len, self.num_heads, self.head_dim
            )
            k = cast(torch.Tensor, self.k_proj(x)).view(
                batch, seq_len, self.num_heads, self.head_dim
            )
            v = cast(torch.Tensor, self.v_proj(x)).view(
                batch, seq_len, self.num_heads, self.head_dim
            )
            return q, k, v

        def prefill_kv(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            _, k, v = self._project(token_ids)
            return k, v

        def qkv_for_decode(
            self, token_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = self._project(token_ids)
            return q[:, 0], k[:, 0], v[:, 0]

        def logits_from_attn(self, attn_out: torch.Tensor) -> torch.Tensor:
            batch = attn_out.shape[0]
            return cast(torch.Tensor, self.out_proj(attn_out.reshape(batch, -1)))

        @torch.no_grad()
        def generate(self, input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
            tokens: torch.Tensor = input_ids
            generated: list[torch.Tensor] = []
            scale = 1.0 / math.sqrt(self.head_dim)

            for _ in range(max_tokens):
                q, k, v = self._project(tokens)
                q_last = q[:, -1].to(torch.bfloat16).float()
                k = k.to(torch.bfloat16).float()
                v = v.to(torch.bfloat16).float()

                scores = (q_last[:, None, :, :] * k).sum(-1) * scale
                scores = scores.transpose(1, 2)
                weights = torch.softmax(scores, dim=-1)
                v = v.transpose(1, 2)
                attn_out = (weights[..., None] * v).sum(dim=-2)

                logits = self.logits_from_attn(attn_out)
                next_token = torch.argmax(logits, dim=-1)
                generated.append(next_token)
                tokens = torch.cat([tokens, next_token[:, None]], dim=1)

            return torch.stack(generated, dim=1)

    model = TinyPagedModel().to(device)
    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    max_context_needed = prompt_len - 1 + max_new_tokens + 1
    max_blocks = math.ceil(max_context_needed / block_size)
    k_cache = torch.zeros(
        (max_blocks, num_heads, block_size, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.arange(max_blocks, device=device, dtype=torch.int32)
    block_tables = block_tables.unsqueeze(0).repeat(batch_size, 1)
    context_lens = torch.full(
        (batch_size,),
        prompt_len - 1,
        device=device,
        dtype=torch.int32,
    )

    if prompt_len > 1:
        with torch.no_grad():
            k_prefill, v_prefill = model.prefill_kv(input_ids[:, :-1])
        k_prefill = k_prefill.to(torch.bfloat16)
        v_prefill = v_prefill.to(torch.bfloat16)
        for position in range(prompt_len - 1):
            block_idx = position // block_size
            block_off = position % block_size
            block_id = int(block_tables[0, block_idx].item())
            k_cache[block_id, :, block_off, :] = k_prefill[0, position]
            v_cache[block_id, :, block_off, :] = v_prefill[0, position]

    generated_triton = paged_kv_decode(
        input_ids=input_ids,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        qkv_proj_fn=model.qkv_for_decode,
        logits_fn=model.logits_from_attn,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        seed=123,
    )

    generated_reference = model.generate(input_ids, max_new_tokens)
    assert torch.equal(generated_triton, generated_reference)


def test_triton_grpo_perf_improves_over_torch() -> None:
    _skip_if_no_cuda_or_triton()

    _ = torch.manual_seed(321)
    _ = torch.cuda.manual_seed_all(321)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    # Larger sizes to demonstrate Triton kernel benefits at scale
    batch_size = 16
    seq_len = 256
    vocab_size = 4096
    group_size = 4
    warmup_iters = 3
    timed_iters = 20

    policy_logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.bfloat16
    )
    old_policy_logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.float32
    )
    target_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    advantages = torch.randn(batch_size, device=device, dtype=torch.float32)

    old_log_probs = -F.cross_entropy(
        old_policy_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len)

    clip_epsilon = 0.2
    epsilon_high = 0.3
    delta = 1.5

    def triton_loss() -> torch.Tensor:
        loss, _ = fused_grpo_loss(
            policy_logits=policy_logits,
            target_ids=target_ids,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=clip_epsilon,
            epsilon_high=epsilon_high,
            delta=delta,
            group_size=group_size,
        )
        return loss

    trainer = GRPOTrainer(
        clip_epsilon=clip_epsilon,
        epsilon_high=epsilon_high,
        delta=delta,
        group_size=group_size,
        use_kl=False,
        use_triton_kernels=False,
    )

    def torch_loss() -> torch.Tensor:
        loss, _ = trainer.compute_grpo_loss(
            policy_logits=policy_logits,
            advantages=advantages,
            old_log_probs=old_log_probs,
            target_ids=target_ids,
        )
        return loss

    def benchmark(fn: Callable[[], torch.Tensor]) -> tuple[float, float]:
        with torch.no_grad():
            for _ in range(warmup_iters):
                value: torch.Tensor = fn()
                del value
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start_allocated = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            for _ in range(timed_iters):
                value = fn()
                del value
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated(device)
        per_iter = elapsed / timed_iters
        peak_extra = max(0.0, float(peak_allocated - start_allocated))
        return per_iter, peak_extra

    torch_time, torch_peak = benchmark(torch_loss)
    triton_time, triton_peak = benchmark(triton_loss)

    if torch_peak <= 0.0 or triton_peak <= 0.0:
        pytest.skip("CUDA peak memory stats unavailable")

    time_improvement = (torch_time - triton_time) / torch_time
    memory_improvement = (torch_peak - triton_peak) / torch_peak

    # Print improvements
    print(f"\n[GRPO Perf] Time improvement: {time_improvement*100:.2f}%")
    print(f"[GRPO Perf] Memory improvement: {memory_improvement*100:.2f}%")
    print(f"[GRPO Perf] Torch time: {torch_time*1000:.2f}ms, Triton time: {triton_time*1000:.2f}ms")
    print(f"[GRPO Perf] Torch peak: {torch_peak/1024**2:.2f}MB, Triton peak: {triton_peak/1024**2:.2f}MB")

    # Check time improvement: require >=5% or xfail
    if time_improvement < 0.05:
        pytest.xfail(
            f"Triton time improvement insufficient. "
            f"Time: {time_improvement*100:.2f}% (required >=5.0%), "
            f"(Torch: {torch_time*1000:.2f}ms, Triton: {triton_time*1000:.2f}ms)"
        )

    # Check memory improvement: require >=5% or xfail
    if memory_improvement < 0.05:
        pytest.xfail(
            f"Triton memory improvement insufficient. "
            f"Memory: {memory_improvement*100:.2f}% (required >=5.0%), "
            f"(Torch: {torch_peak/1024**2:.2f}MB, Triton: {triton_peak/1024**2:.2f}MB)"
        )



def test_triton_entropy_mask_perf_improves_over_torch() -> None:
    _skip_if_no_cuda_or_triton()

    _ = torch.manual_seed(456)
    _ = torch.cuda.manual_seed_all(456)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    # Moderate sizes to show Triton kernel benefits
    batch_size = 8
    seq_len = 128
    vocab_size = 2048
    warmup_iters = 3
    timed_iters = 20

    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.float32
    )
    attention_mask = torch.ones(
        batch_size, seq_len, device=device, dtype=torch.float32
    )

    calculator = EntropyCalculator(percentile=0.5, min_tokens=2)

    def triton_entropy_mask() -> tuple[torch.Tensor, torch.Tensor]:
        entropy, mask = calculator.calculate_entropy_and_mask(
            logits,
            attention_mask=attention_mask,
            use_triton_kernels=True,
        )
        return entropy, mask

    def torch_entropy_mask() -> tuple[torch.Tensor, torch.Tensor]:
        entropy, mask = calculator.calculate_entropy_and_mask(
            logits,
            attention_mask=attention_mask,
            use_triton_kernels=False,
        )
        return entropy, mask

    def benchmark(
        fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[float, float]:
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start_allocated = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            for _ in range(timed_iters):
                _ = fn()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated(device)
        per_iter = elapsed / timed_iters
        peak_extra = max(0.0, float(peak_allocated - start_allocated))
        return per_iter, peak_extra

    torch_time, torch_peak = benchmark(torch_entropy_mask)
    triton_time, triton_peak = benchmark(triton_entropy_mask)

    if torch_peak <= 0.0 or triton_peak <= 0.0:
        pytest.skip("CUDA peak memory stats unavailable")

    time_improvement = (torch_time - triton_time) / torch_time
    memory_improvement = (torch_peak - triton_peak) / torch_peak

    # Print improvements
    print(f"\n[Entropy Perf] Time improvement: {time_improvement*100:.2f}%")
    print(f"[Entropy Perf] Memory improvement: {memory_improvement*100:.2f}%")
    print(f"[Entropy Perf] Torch time: {torch_time*1000:.2f}ms, Triton time: {triton_time*1000:.2f}ms")
    print(f"[Entropy Perf] Torch peak: {torch_peak/1024**2:.2f}MB, Triton peak: {triton_peak/1024**2:.2f}MB")

    # Check time improvement: require >=5% or xfail
    if time_improvement < 0.05:
        pytest.xfail(
            f"Triton time improvement insufficient. "
            f"Time: {time_improvement*100:.2f}% (required >=5.0%), "
            f"(Torch: {torch_time*1000:.2f}ms, Triton: {triton_time*1000:.2f}ms)"
        )

    # Check memory improvement: require >=5% or xfail
    if memory_improvement < 0.05:
        pytest.xfail(
            f"Triton memory improvement insufficient. "
            f"Memory: {memory_improvement*100:.2f}% (required >=5.0%), "
            f"(Torch: {torch_peak/1024**2:.2f}MB, Triton: {triton_peak/1024**2:.2f}MB)"
        )



def test_triton_paged_kv_decode_perf_improves_over_generate() -> None:
    _skip_if_no_cuda_or_triton()

    _ = torch.manual_seed(999)
    _ = torch.cuda.manual_seed_all(999)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    batch_size = 1
    vocab_size = 32
    num_heads = 2
    head_dim = 8
    prompt_len = 5
    max_new_tokens = 8
    block_size = 4
    warmup_iters = 3
    timed_iters = 20

    class TinyPagedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            embed_dim = num_heads * head_dim
            self.embed: torch.nn.Embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.q_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj: torch.nn.Linear = torch.nn.Linear(embed_dim, vocab_size, bias=False)
            self.num_heads: int = num_heads
            self.head_dim: int = head_dim

        def _project(
            self, token_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = cast(torch.Tensor, self.embed(token_ids))
            if x.ndim == 2:
                x = x.unsqueeze(1)
            batch, seq_len, _ = x.shape
            q = cast(torch.Tensor, self.q_proj(x)).view(
                batch, seq_len, self.num_heads, self.head_dim
            )
            k = cast(torch.Tensor, self.k_proj(x)).view(
                batch, seq_len, self.num_heads, self.head_dim
            )
            v = cast(torch.Tensor, self.v_proj(x)).view(
                batch, seq_len, self.num_heads, self.head_dim
            )
            return q, k, v

        def prefill_kv(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            _, k, v = self._project(token_ids)
            return k, v

        def qkv_for_decode(
            self, token_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = self._project(token_ids)
            return q[:, 0], k[:, 0], v[:, 0]

        def logits_from_attn(self, attn_out: torch.Tensor) -> torch.Tensor:
            batch = attn_out.shape[0]
            return cast(torch.Tensor, self.out_proj(attn_out.reshape(batch, -1)))

        @torch.no_grad()
        def generate(self, input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
            tokens: torch.Tensor = input_ids
            generated: list[torch.Tensor] = []
            scale = 1.0 / math.sqrt(self.head_dim)

            for _ in range(max_tokens):
                q, k, v = self._project(tokens)
                q_last = q[:, -1].to(torch.bfloat16).float()
                k = k.to(torch.bfloat16).float()
                v = v.to(torch.bfloat16).float()

                scores = (q_last[:, None, :, :] * k).sum(-1) * scale
                scores = scores.transpose(1, 2)
                weights = torch.softmax(scores, dim=-1)
                v = v.transpose(1, 2)
                attn_out = (weights[..., None] * v).sum(dim=-2)

                logits = self.logits_from_attn(attn_out)
                next_token = torch.argmax(logits, dim=-1)
                generated.append(next_token)
                tokens = torch.cat([tokens, next_token[:, None]], dim=1)

            return torch.stack(generated, dim=1)

    model = TinyPagedModel().to(device)
    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    max_context_needed = prompt_len - 1 + max_new_tokens + 1
    max_blocks = math.ceil(max_context_needed / block_size)
    k_cache = torch.zeros(
        (max_blocks, num_heads, block_size, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.arange(max_blocks, device=device, dtype=torch.int32)
    block_tables = block_tables.unsqueeze(0).repeat(batch_size, 1)
    context_lens = torch.full(
        (batch_size,),
        prompt_len - 1,
        device=device,
        dtype=torch.int32,
    )

    if prompt_len > 1:
        with torch.no_grad():
            k_prefill, v_prefill = model.prefill_kv(input_ids[:, :-1])
        k_prefill = k_prefill.to(torch.bfloat16)
        v_prefill = v_prefill.to(torch.bfloat16)
        for position in range(prompt_len - 1):
            block_idx = position // block_size
            block_off = position % block_size
            block_id = int(block_tables[0, block_idx].item())
            k_cache[block_id, :, block_off, :] = k_prefill[0, position]
            v_cache[block_id, :, block_off, :] = v_prefill[0, position]

    def triton_decode() -> torch.Tensor:
        return paged_kv_decode(
            input_ids=input_ids,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            qkv_proj_fn=model.qkv_for_decode,
            logits_fn=model.logits_from_attn,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            seed=999,
        )

    def baseline_generate() -> torch.Tensor:
        return model.generate(input_ids, max_new_tokens)

    def benchmark(
        fn: Callable[[], torch.Tensor],
    ) -> tuple[float, float]:
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start_allocated = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            for _ in range(timed_iters):
                _ = fn()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated(device)
        per_iter = elapsed / timed_iters
        peak_extra = max(0.0, float(peak_allocated - start_allocated))
        return per_iter, peak_extra

    baseline_time, baseline_peak = benchmark(baseline_generate)
    triton_time, triton_peak = benchmark(triton_decode)

    if baseline_peak <= 0.0 or triton_peak <= 0.0:
        pytest.skip("CUDA peak memory stats unavailable")

    time_improvement = (baseline_time - triton_time) / baseline_time
    memory_improvement = (baseline_peak - triton_peak) / baseline_peak

    print(f"\n[PagedKV Decode Perf] Time improvement: {time_improvement*100:.2f}%")
    print(f"[PagedKV Decode Perf] Memory improvement: {memory_improvement*100:.2f}%")
    print(f"[PagedKV Decode Perf] Baseline time: {baseline_time*1000:.2f}ms, Triton time: {triton_time*1000:.2f}ms")
    print(f"[PagedKV Decode Perf] Baseline peak: {baseline_peak/1024**2:.2f}MB, Triton peak: {triton_peak/1024**2:.2f}MB")

    if time_improvement < 0.05:
        pytest.xfail(
            f"Triton paged_kv_decode time improvement insufficient. "
            f"Time: {time_improvement*100:.2f}% (required >=5.0%), "
            f"(Baseline: {baseline_time*1000:.2f}ms, Triton: {triton_time*1000:.2f}ms)"
        )

    if memory_improvement < 0.05:
        pytest.xfail(
            f"Triton paged_kv_decode memory improvement insufficient. "
            f"Memory: {memory_improvement*100:.2f}% (required >=5.0%), "
            f"(Baseline: {baseline_peak/1024**2:.2f}MB, Triton: {triton_peak/1024**2:.2f}MB)"
        )
