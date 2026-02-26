"""
Test: Entropy mask alignment in training step.

Verifies that the entropy mask produced by EntropyCalculator has the same
shape as the logits/target slices passed to compute_grpo_loss. A bug here
would cause either a shape mismatch crash or — worse — a silent 1-token
misalignment where entropy values are applied to the wrong token positions.
"""
import torch
import torch.nn.functional as F

# Enforce CUDA present
assert torch.cuda.is_available(), "CUDA is required for this test suite"


class TestEntropyMaskAlignment:
    """Verify entropy mask shape matches the loss tensors exactly."""

    def _simulate_training_step(self, prompt_len: int, resp_len: int, batch: int = 2):
        """
        Simulate the shape logic from GRPOTrainerLoop.training_step
        (lines ~718-772 of trainer.py) with synthetic data.

        Returns the shapes that would be passed to compute_grpo_loss.
        """
        from src.selective.entropy_mask import EntropyCalculator

        vocab_size = 32
        total_len = prompt_len + resp_len
        device = "cuda"

        # Simulate model output logits: [B, prompt_len + resp_len, V]
        logits = torch.randn(batch, total_len, vocab_size, device=device)

        # response_only_mask: 0 for prompt tokens, 1 for response tokens
        response_only_mask = torch.cat([
            torch.zeros(batch, prompt_len, dtype=torch.long, device=device),
            torch.ones(batch, resp_len, dtype=torch.long, device=device),
        ], dim=1)

        # --- This mirrors the CURRENT (fixed) trainer code ---
        calculator = EntropyCalculator(percentile=0.5, min_tokens=1)

        # Entropy computed on logits[:, :-1, :] — same slice that goes to loss
        loss_logits = logits[:, :-1, :].contiguous()
        loss_resp_mask = response_only_mask[:, 1:].contiguous()
        entropy, entropy_mask = calculator.calculate_entropy_and_mask(
            loss_logits,
            attention_mask=loss_resp_mask,
        )

        # These are the tensors that go to compute_grpo_loss
        policy_logits_for_loss = logits[:, :-1, :]          # [B, total_len-1, V]
        target_ids_for_loss = torch.randint(0, vocab_size, (batch, total_len), device=device)[:, 1:]
        attention_mask_for_loss = response_only_mask[:, 1:]  # [B, total_len-1]

        return {
            "policy_logits": policy_logits_for_loss,
            "target_ids": target_ids_for_loss,
            "attention_mask": attention_mask_for_loss,
            "entropy_mask": entropy_mask,
            "entropy": entropy,
        }

    def test_entropy_mask_shape_matches_logits(self):
        """entropy_mask must have the same [B, seq] shape as logits[:, :-1, :]."""
        shapes = self._simulate_training_step(prompt_len=10, resp_len=20)
        assert shapes["entropy_mask"].shape == shapes["policy_logits"].shape[:2], (
            f"entropy_mask shape {shapes['entropy_mask'].shape} != "
            f"logits shape {shapes['policy_logits'].shape[:2]}"
        )

    def test_entropy_mask_shape_matches_attention_mask(self):
        """entropy_mask and attention_mask must be identical in shape."""
        shapes = self._simulate_training_step(prompt_len=10, resp_len=20)
        assert shapes["entropy_mask"].shape == shapes["attention_mask"].shape, (
            f"entropy_mask shape {shapes['entropy_mask'].shape} != "
            f"attention_mask shape {shapes['attention_mask'].shape}"
        )

    def test_entropy_mask_shape_matches_target_ids(self):
        """entropy_mask and target_ids must be [B, total_len-1]."""
        shapes = self._simulate_training_step(prompt_len=10, resp_len=20)
        assert shapes["entropy_mask"].shape == shapes["target_ids"].shape, (
            f"entropy_mask shape {shapes['entropy_mask'].shape} != "
            f"target_ids shape {shapes['target_ids'].shape}"
        )

    def test_entropy_mask_zero_on_prompt_positions(self):
        """Entropy mask should be 0 for ALL prompt token positions."""
        prompt_len = 10
        shapes = self._simulate_training_step(prompt_len=prompt_len, resp_len=20)
        # After the [:, 1:] shift, prompt occupies positions [0, prompt_len-1)
        prompt_part = shapes["entropy_mask"][:, :prompt_len - 1]
        assert prompt_part.sum().item() == 0.0, (
            f"Prompt portion of entropy_mask should be all zeros, "
            f"but got sum={prompt_part.sum().item()}"
        )

    def test_entropy_mask_nonzero_on_response_positions(self):
        """Entropy mask should have at least some 1s in the response region."""
        prompt_len = 10
        shapes = self._simulate_training_step(prompt_len=prompt_len, resp_len=20)
        # Response positions start at prompt_len-1 in the shifted mask
        resp_part = shapes["entropy_mask"][:, prompt_len - 1:]
        assert resp_part.sum().item() > 0, (
            f"Response portion should have selected tokens, "
            f"but got sum={resp_part.sum().item()}"
        )

    def test_grpo_loss_accepts_aligned_entropy_mask(self):
        """Full integration: compute_grpo_loss should not crash with aligned entropy_mask."""
        from src.grpo.algorithm import GRPOTrainer

        shapes = self._simulate_training_step(prompt_len=10, resp_len=20, batch=4)
        trainer = GRPOTrainer(group_size=4, clip_epsilon=0.2, epsilon_high=0.3, delta=1.5)

        # Dummy old_log_probs and advantages
        old_log_probs = -F.cross_entropy(
            shapes["policy_logits"].reshape(-1, shapes["policy_logits"].size(-1)),
            shapes["target_ids"].reshape(-1),
            reduction="none",
        ).view(shapes["target_ids"].shape)
        advantages = torch.randn(4, device="cuda")

        loss, metrics = trainer.compute_grpo_loss(
            policy_logits=shapes["policy_logits"],
            advantages=advantages,
            old_log_probs=old_log_probs,
            target_ids=shapes["target_ids"],
            attention_mask=shapes["attention_mask"],
            entropy_mask=shapes["entropy_mask"],
        )

        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert "loss" in metrics

    def test_various_prompt_response_ratios(self):
        """Shape alignment should hold for any prompt/response length combination."""
        for prompt_len, resp_len in [(1, 5), (5, 1), (50, 200), (128, 384)]:
            shapes = self._simulate_training_step(
                prompt_len=prompt_len, resp_len=resp_len, batch=1
            )
            assert shapes["entropy_mask"].shape == shapes["policy_logits"].shape[:2], (
                f"Misalignment at prompt={prompt_len}, resp={resp_len}: "
                f"mask={shapes['entropy_mask'].shape}, logits={shapes['policy_logits'].shape[:2]}"
            )
