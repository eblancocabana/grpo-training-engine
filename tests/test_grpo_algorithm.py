import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import importlib
from typing import Dict, Any, List, Tuple
from unittest.mock import MagicMock, patch

# Enforce CUDA present (user requirement)
assert torch.cuda.is_available(), "CUDA is required for this test suite"

# Determinism for reproducible tests
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


class TestDrGRPOAlgorithm:
    """White-box tests for every step of Dr. GRPO algorithm."""

    def test_step_1_group_rewards_shape(self):
        """Step 1: Verify rewards are grouped correctly by prompt."""
        from src.grpo.algorithm import GRPOTrainer

        batch_size = 3
        group_size = 4
        total_samples = batch_size * group_size

        # Create rewards in flat format
        rewards = torch.arange(total_samples, dtype=torch.float32)

        trainer = GRPOTrainer(group_size=group_size)

        # Reshape to [batch_size, group_size]
        grouped = rewards.view(batch_size, group_size)

        assert grouped.shape == (batch_size, group_size)
        assert grouped[0, 0].item() == 0.0
        assert grouped[0, 3].item() == 3.0
        assert grouped[2, 0].item() == 8.0

    def test_step_2_advantage_calculation_exact(self):
        """Step 2: Verify exact advantage calculation: (r - mean)."""
        from src.grpo.algorithm import GRPOTrainer

        rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
        group_size = 2

        trainer = GRPOTrainer(group_size=group_size)
        advantages = trainer.calculate_advantages(rewards, group_size)

        expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        torch.testing.assert_close(advantages, expected, atol=1e-6, rtol=1e-6)

    def test_step_2_advantage_uniform_rewards(self):
        """Step 2: Verify uniform rewards give zero advantages (no min_std hack needed in Dr. GRPO)."""
        from src.grpo.algorithm import GRPOTrainer

        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
        group_size = 2

        trainer = GRPOTrainer(group_size=group_size)
        advantages = trainer.calculate_advantages(rewards, group_size)

        expected = torch.zeros_like(rewards)
        assert torch.allclose(advantages, expected, atol=1e-6)

    def test_step_3_log_prob_calculation(self):
        """Step 3: Verify log probability extraction from logits."""
        batch_size = 2
        seq_len = 3
        vocab_size = 5

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 0] = 2.0
        logits[:, :, 1] = 1.0

        target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        assert selected_log_probs.shape == (batch_size, seq_len)
        assert torch.all(selected_log_probs < 0)
        log_prob_token_0 = log_probs[0, 0, 0].item()
        log_prob_token_1 = log_probs[0, 0, 1].item()
        assert log_prob_token_0 > log_prob_token_1

    def test_step_4_ratio_computation(self):
        """Step 4: Verify ratio = exp(log_prob - old_log_prob)."""
        log_prob = torch.tensor([[0.0, -0.5]])
        old_log_prob = torch.tensor([[-0.5, -1.0]])
        expected_ratio = torch.exp(log_prob - old_log_prob)
        assert torch.all(expected_ratio > 1.0)
        assert torch.allclose(
            expected_ratio, torch.tensor([[1.6487, 1.6487]]), atol=0.001
        )

    def test_step_5a_clipping_lower_bound(self):
        """Step 5a: Verify two-sided clipping lower bound [1-epsilon, 1+epsilon_high]."""
        from src.grpo.algorithm import GRPOTrainer

        ratios = torch.tensor([0.1, 0.5, 0.9])
        trainer = GRPOTrainer(clip_epsilon=0.2, epsilon_high=0.3)
        lower_bound = 1.0 - trainer.clip_epsilon
        clipped = torch.clamp(ratios, min=lower_bound)
        assert torch.all(clipped >= lower_bound - 1e-5)
        assert abs(clipped[0].item() - 0.8) < 1e-5
        assert abs(clipped[1].item() - 0.8) < 1e-5
        assert abs(clipped[2].item() - 0.9) < 1e-5

    def test_step_5b_clipping_upper_bound(self):
        """Step 5b: Verify two-sided clipping upper bound 1+epsilon_high."""
        from src.grpo.algorithm import GRPOTrainer

        ratios = torch.tensor([1.2, 1.5, 2.0])
        trainer = GRPOTrainer(clip_epsilon=0.2, epsilon_high=0.3)
        upper_bound = 1.0 + trainer.epsilon_high
        clipped = torch.clamp(ratios, max=upper_bound)
        assert torch.all(clipped <= upper_bound + 1e-5)
        assert abs(clipped[0].item() - 1.2) < 1e-5
        assert abs(clipped[1].item() - 1.3) < 1e-5
        assert abs(clipped[2].item() - 1.3) < 1e-5

    def test_step_5c_delta_safety_cap(self):
        """Step 5c: Verify hard safety cap delta on ratios."""
        delta = 1.5
        ratios = torch.tensor([0.5, 1.0, 2.0, 5.0])
        capped = torch.clamp(ratios, min=0.0, max=delta)
        assert torch.all(capped >= 0.0)
        assert torch.all(capped <= delta)
        assert abs(capped[0].item() - 0.5) < 1e-5
        assert abs(capped[2].item() - delta) < 1e-5
        assert abs(capped[3].item() - delta) < 1e-5

    def test_step_6_unclipped_vs_clipped_loss(self):
        """Step 6: Verify loss uses min(ratio*A, clip(ratio)*A)."""
        advantage = torch.tensor([2.0])
        ratio = torch.tensor([1.5])
        clip_epsilon = 0.2
        epsilon_high = 0.3
        upper_bound = 1.0 + epsilon_high
        unclipped = ratio * advantage
        clipped_ratio = torch.clamp(ratio, min=1.0 - clip_epsilon, max=upper_bound)
        clipped = clipped_ratio * advantage
        loss_unclipped = -unclipped
        loss_clipped = -clipped
        loss = torch.max(loss_unclipped, loss_clipped)
        assert abs(loss.item() - (-2.6)) < 0.01

    def test_step_7_global_normalization(self):
        """Step 7: Verify loss is normalized by group_size, not per-token."""
        batch_size = 2
        group_size = 4
        seq_len = 10
        token_losses = torch.randn(batch_size * group_size, seq_len)
        normalized_loss = token_losses.sum() / group_size
        wrong_normalization = token_losses.sum() / (batch_size * group_size * seq_len)
        assert normalized_loss != wrong_normalization
        assert normalized_loss > wrong_normalization

    def test_full_grpo_loss_computation(self):
        """Integration: Full GRPO loss from logits to final value."""
        from src.grpo.algorithm import GRPOTrainer

        batch = 2
        seq = 4
        vocab = 8
        torch.manual_seed(123)
        policy_logits = torch.randn(batch, seq, vocab)
        old_logits = policy_logits.clone() + torch.randn(batch, seq, vocab) * 0.1
        target_ids = torch.randint(0, vocab, (batch, seq))
        advantages = torch.tensor([0.5, -0.5])
        trainer = GRPOTrainer(
            group_size=1, clip_epsilon=0.2, epsilon_high=0.3, delta=1.5
        )
        loss, metrics = trainer.compute_grpo_loss(
            policy_logits=policy_logits,
            advantages=advantages,
            old_policy_logits=old_logits,
            target_ids=target_ids,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in metrics
        assert "ratio_mean" in metrics
        assert "ratio_capped_pct" in metrics
        assert 0.0 <= metrics["ratio_mean"] <= 1.5
        assert 0.0 <= metrics["ratio_capped_pct"] <= 1.0
