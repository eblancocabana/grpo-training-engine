import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch


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

    def test_checkpoint_load_moves_optimizer_state_to_model_device(self):
        import tempfile

        from src.utils.checkpoint import CheckpointManager

        device = torch.device("cuda")
        model = nn.Linear(4, 4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(2, 4, device=device)
        y = torch.randn(2, 4, device=device)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            checkpoint_path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                step=3,
                epoch=1,
            )

            new_model = nn.Linear(4, 4).to(device)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
            manager.load_checkpoint(
                checkpoint_path,
                model=new_model,
                optimizer=new_optimizer,
                scheduler=None,
            )

            state_tensors = [
                value
                for state in new_optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            ]
            assert state_tensors
            assert all(t.device.type == "cuda" for t in state_tensors)

    def test_paged_kv_decode_wrapper_falls_back_without_triton(self):
        import src.triton_kernels as triton_kernels

        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3]])

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False):
            generated = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 1]]),
                max_new_tokens=3,
            )

        model.generate.assert_called_once()
        assert torch.equal(generated, torch.tensor([[1, 2, 3]]))
