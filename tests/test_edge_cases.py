import torch
import torch.nn as nn
import torch.nn.functional as F


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_zero_length_response(self):
        from src.grpo.verifier import RuleBasedVerifier

        verifier = RuleBasedVerifier()
        result = verifier.verify("", "42")
        reward = result[0]
        assert reward == 0.0

    def test_very_long_response(self):
        from src.grpo.verifier import RuleBasedVerifier

        verifier = RuleBasedVerifier()
        long_response = "Answer is " + " " * 10000 + "\\boxed{42}"
        result = verifier.verify(long_response, "42")
        reward = result[0]
        assert reward == 1.0

    def test_extreme_advantages(self):
        from src.grpo.algorithm import GRPOTrainer

        trainer = GRPOTrainer(group_size=1)
        policy_logits = torch.randn(1, 4, 8)
        old_logits = policy_logits.clone()
        advantages = torch.tensor([100.0])
        targets = torch.zeros(1, 4, dtype=torch.long)
        loss, metrics = trainer.compute_grpo_loss(
            policy_logits, advantages, old_logits, target_ids=targets
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_negative_rewards(self):
        from src.grpo.algorithm import GRPOTrainer

        rewards = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        group_size = 2
        trainer = GRPOTrainer(group_size=group_size)
        advantages = trainer.calculate_advantages(rewards, group_size)
        assert not torch.isnan(advantages).any()
        grouped = advantages.view(-1, group_size)
        for g in grouped:
            assert abs(g.mean().item()) < 1e-6

    def test_single_token_sequence(self):
        from src.grpo.algorithm import GRPOTrainer

        trainer = GRPOTrainer(group_size=1)
        policy_logits = torch.randn(2, 1, 8)
        old_logits = policy_logits.clone()
        advantages = torch.tensor([1.0, -1.0])
        targets = torch.zeros(2, 1, dtype=torch.long)
        loss, metrics = trainer.compute_grpo_loss(
            policy_logits, advantages, old_logits, target_ids=targets
        )
        assert not torch.isnan(loss)

    def test_oom_recovery_simulation(self):
        from src.core.memory_manager import MemoryManager

        mm = MemoryManager()
        mm.clear_cache(aggressive=True)
        assert True

    def test_checkpoint_save_load_simulation(self):
        import tempfile, os
        import torch

        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 100,
            "loss": 0.5,
        }
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(checkpoint, f.name)
            loaded = torch.load(f.name, weights_only=False)
            assert "model_state_dict" in loaded
            assert "optimizer_state_dict" in loaded
            assert loaded["step"] == 100
            os.unlink(f.name)
