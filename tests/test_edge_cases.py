import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
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

    def test_move_optimizer_state_to_model_device_uses_state_key_devices(self):
        from src.utils.checkpoint import CheckpointManager

        class FakeStateKey:
            def __init__(self, device: str):
                self.device = torch.device(device)

        model = nn.Linear(1, 1)
        cpu_key = FakeStateKey("cpu")
        meta_key = FakeStateKey("meta")
        optimizer = Mock()
        optimizer.state = {
            cpu_key: {"buf": torch.tensor([1.0])},
            meta_key: {"buf": torch.tensor([2.0])},
        }

        CheckpointManager.move_optimizer_state_to_model_device(optimizer, model)

        assert optimizer.state[cpu_key]["buf"].device.type == "cpu"
        assert optimizer.state[meta_key]["buf"].device.type == "meta"

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

    def test_paged_kv_decode_wrapper_falls_back_without_triton_with_per_row_seeds(self):
        import src.triton_kernels as triton_kernels

        model = Mock()

        def fake_generate(*, input_ids, attention_mask=None, max_new_tokens=128, **kwargs):
            del attention_mask, max_new_tokens
            assert "seed" not in kwargs
            assert "seeds" not in kwargs
            assert "block_size" not in kwargs
            first_token = int(input_ids[0, 0].item())
            return torch.tensor([[first_token, first_token + 10]], dtype=torch.long)

        model.generate.side_effect = fake_generate

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False):
            generated = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                max_new_tokens=2,
                do_sample=True,
                seeds=[11, 22],
                block_size=16,
                pad_token_id=0,
            )

        assert model.generate.call_count == 2
        assert torch.equal(
            generated,
            torch.tensor([[1, 11], [3, 13]], dtype=torch.long),
        )

    def test_paged_kv_decode_wrapper_fallback_uses_model_pad_token_for_seeded_rows(self):
        import src.triton_kernels as triton_kernels

        model = Mock()
        model.generation_config = Mock(pad_token_id=99)
        generated_rows = [
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[3, 4, 5]], dtype=torch.long),
        ]

        def fake_generate(*, input_ids, attention_mask=None, max_new_tokens=128, **kwargs):
            del input_ids, attention_mask, max_new_tokens, kwargs
            return generated_rows.pop(0)

        model.generate.side_effect = fake_generate

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False):
            generated = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                max_new_tokens=3,
                do_sample=True,
                seeds=[11, 22],
            )

        assert torch.equal(
            generated,
            torch.tensor([[1, 2, 99], [3, 4, 5]], dtype=torch.long),
        )

    def test_paged_kv_decode_seeded_fallback_merges_structured_generate_outputs(self):
        import src.triton_kernels as triton_kernels

        model = Mock()
        model.generate.side_effect = [
            {
                "sequences": torch.tensor([[1, 2]], dtype=torch.long),
                "scores": (
                    torch.tensor([[0.1, 0.9]], dtype=torch.float32),
                    torch.tensor([[0.3, 0.7]], dtype=torch.float32),
                ),
            },
            {
                "sequences": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "scores": (
                    torch.tensor([[0.2, 0.8]], dtype=torch.float32),
                    torch.tensor([[0.4, 0.6]], dtype=torch.float32),
                ),
            },
        ]

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False):
            generated = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                max_new_tokens=3,
                do_sample=True,
                seeds=[11, 22],
                pad_token_id=99,
                return_dict_in_generate=True,
                output_scores=True,
            )

        assert isinstance(generated, dict)
        assert torch.equal(
            generated["sequences"],
            torch.tensor([[1, 2, 99], [3, 4, 5]], dtype=torch.long),
        )
        assert len(generated["scores"]) == 2
        assert torch.equal(
            generated["scores"][0],
            torch.tensor([[0.1, 0.9], [0.2, 0.8]], dtype=torch.float32),
        )
        assert torch.equal(
            generated["scores"][1],
            torch.tensor([[0.3, 0.7], [0.4, 0.6]], dtype=torch.float32),
        )

    def test_paged_kv_decode_seeded_fallback_merges_variable_length_scores(self):
        import src.triton_kernels as triton_kernels

        model = Mock()
        model.generate.side_effect = [
            {
                "sequences": torch.tensor([[1, 2]], dtype=torch.long),
                "scores": (
                    torch.tensor([[0.1, 0.9]], dtype=torch.float32),
                ),
            },
            {
                "sequences": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "scores": (
                    torch.tensor([[0.2, 0.8]], dtype=torch.float32),
                    torch.tensor([[0.4, 0.6]], dtype=torch.float32),
                ),
            },
        ]

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False):
            generated = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                max_new_tokens=3,
                do_sample=True,
                seeds=[11, 22],
                pad_token_id=99,
                return_dict_in_generate=True,
                output_scores=True,
            )

        assert isinstance(generated, dict)
        assert len(generated["scores"]) == 2
        assert torch.equal(
            generated["scores"][0],
            torch.tensor([[0.1, 0.9], [0.2, 0.8]], dtype=torch.float32),
        )
        assert torch.equal(
            generated["scores"][1],
            torch.tensor([[0.0, 0.0], [0.4, 0.6]], dtype=torch.float32),
        )

    def test_paged_kv_decode_seeded_fallback_privately_seeds_all_visible_cuda_devices(self):
        import src.triton_kernels as triton_kernels

        class FakeCudaTensor:
            def __init__(self, values):
                self._values = values
                self.device = torch.device("cuda:1")
                self.dtype = values.dtype
                self.shape = values.shape

            def __getitem__(self, item):
                return FakeCudaTensor(self._values[item])

        model = Mock()
        model.generate.side_effect = [
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[3, 4]], dtype=torch.long),
        ]
        manual_seed_calls = []

        @contextmanager
        def fake_fork_rng(*, devices):
            assert devices == [0, 1]
            yield

        entered_devices = []

        @contextmanager
        def fake_cuda_device(device_idx):
            entered_devices.append(int(device_idx))
            yield

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False), patch(
            "torch.cuda.is_available", return_value=True
        ), patch(
            "torch.cuda.device_count", return_value=2
        ), patch(
            "torch.random.fork_rng",
            side_effect=fake_fork_rng,
        ), patch(
            "torch.cuda.device",
            side_effect=fake_cuda_device,
        ), patch(
            "torch.cuda.manual_seed",
            side_effect=lambda seed: manual_seed_calls.append(int(seed)),
        ) as manual_seed, patch(
            "torch.cuda.manual_seed_all",
        ) as manual_seed_all:
            generated = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=FakeCudaTensor(torch.tensor([[1, 2], [3, 4]], dtype=torch.long)),
                attention_mask=FakeCudaTensor(
                    torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
                ),
                max_new_tokens=2,
                do_sample=True,
                seeds=[11, 22],
                pad_token_id=0,
            )

        assert manual_seed.call_count == 4
        manual_seed.assert_any_call(11)
        manual_seed.assert_any_call(22)
        manual_seed_all.assert_not_called()
        assert manual_seed_calls == [11, 11, 22, 22]
        assert entered_devices == [0, 1, 0, 1]
        assert torch.equal(generated, torch.tensor([[1, 2], [3, 4]], dtype=torch.long))

    def test_paged_kv_decode_wrapper_rejects_low_level_fallback_without_triton(self):
        import src.triton_kernels as triton_kernels

        with patch.object(triton_kernels, "TRITON_AVAILABLE", False):
            try:
                triton_kernels.paged_kv_decode(
                    model=Mock(),
                    input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                    k_cache=torch.zeros(1),
                    v_cache=torch.zeros(1),
                    block_tables=torch.zeros(1, 1, dtype=torch.int32),
                    context_lens=torch.ones(1, dtype=torch.int32),
                    qkv_proj_fn=Mock(),
                    logits_fn=Mock(),
                )
            except RuntimeError as exc:
                assert "requires Triton" in str(exc)
            else:
                raise AssertionError("Expected low-level fallback without Triton to fail.")
