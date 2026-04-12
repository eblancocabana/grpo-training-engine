import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import types
from unittest.mock import patch


class TestIntegration:
    """End-to-end integration tests."""

    def setup_fake_bnb(self):
        import sys, types

        fake_bnb = types.ModuleType("bitsandbytes")
        fake_bnb.nn = types.SimpleNamespace()
        fake_bnb.nn.Linear4bit = None
        sys.modules["bitsandbytes"] = fake_bnb

    def test_training_step_end_to_end(self):
        self.setup_fake_bnb()
        from src.grpo.algorithm import GRPOTrainer
        from src.grpo.verifier import RuleBasedVerifier

        device = torch.device("cuda")

        class TinyPolicy(nn.Module):
            def __init__(self, vocab=8, hidden=16):
                super().__init__()
                self.embed = nn.Embedding(vocab, hidden)
                self.fc = nn.Linear(hidden, vocab)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                return self.fc(x)

        model = TinyPolicy().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = GRPOTrainer(group_size=2)
        verifier = RuleBasedVerifier()
        prompt_ids = torch.randint(0, 8, (2, 4), device=device)
        with torch.no_grad():
            logits = model(prompt_ids)
            probs = F.softmax(logits, dim=-1)
            generated = torch.argmax(probs, dim=-1)
        responses = [f"\\boxed{x.item()}" for x in generated[:, -1]]
        rewards_list = []
        for r in responses:
            result = verifier.verify(r, "5")
            rewards_list.append(result[0])
        rewards = torch.tensor(rewards_list, device=device)
        advantages = trainer.calculate_advantages(rewards, group_size=2)
        old_logits = logits.detach()
        for step in range(5):
            optimizer.zero_grad()
            new_logits = model(prompt_ids)
            loss, metrics = trainer.compute_grpo_loss(
                new_logits,
                advantages,
                old_policy_logits=old_logits,
                target_ids=generated,
            )
            loss.backward()
            optimizer.step()
        assert loss.item() is not None
        assert "ratio_mean" in metrics

    def test_overfitting_tiny_dataset(self):
        from src.grpo.algorithm import GRPOTrainer

        device = torch.device("cuda")

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        model = TinyModel().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = GRPOTrainer(group_size=1)
        x = torch.randn(1, 4, device=device)
        target = torch.tensor([0], device=device)
        initial_logits = model(x)
        initial_loss = F.cross_entropy(initial_logits, target)
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        final_logits = model(x)
        final_loss = F.cross_entropy(final_logits, target)
        assert final_loss < initial_loss

    def test_gradient_accumulation_simulation(self):
        device = torch.device("cuda")
        model = nn.Linear(4, 4).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        accumulation_steps = 4
        accumulated_grad = None
        for i in range(accumulation_steps):
            x = torch.randn(1, 4, device=device)
            y = torch.randint(0, 4, (1,), device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y) / accumulation_steps
            loss.backward()
            if accumulated_grad is None:
                accumulated_grad = model.weight.grad.clone()
            else:
                accumulated_grad += model.weight.grad
        assert accumulated_grad is not None
        assert not torch.allclose(accumulated_grad, torch.zeros_like(accumulated_grad))

    def test_triton_generation_path_keeps_response_tokens(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            @staticmethod
            def batch_decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return [" ".join(map(str, row.tolist())) for row in tokens]

        class DummyMemoryManager:
            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            def eval(self):
                return self

        config = get_8gb_vram_config()
        config.training.use_triton_kernels = True
        config.training.generation_do_sample = False
        config.grpo.group_size = 2
        loop = GRPOTrainerLoop(config)
        loop.model = DummyModel()
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.group_sampler = types.SimpleNamespace(group_size=2)
        loop._gen_micro_batch = 2
        loop.device = "cpu"

        fake_outputs = torch.tensor([[101, 102, 0], [201, 202, 0]], dtype=torch.long)
        fake_state = types.SimpleNamespace(
            k_cache=torch.zeros((1, 1, 1, 1, 1)),
            v_cache=torch.zeros((1, 1, 1, 1, 1)),
            block_tables=torch.zeros((1, 1), dtype=torch.int32),
            context_lens=torch.zeros((1,), dtype=torch.int32),
            last_tokens=torch.tensor([12], dtype=torch.long),
            max_context=8,
            block_size=1,
        )

        with patch("src.grpo.trainer.TRITON_AVAILABLE", True), patch(
            "src.grpo.trainer.prefill_paged_kv_cache",
            return_value=fake_state,
        ), patch(
            "src.grpo.trainer.expand_paged_kv_cache_state",
            return_value=fake_state,
        ), patch(
            "src.grpo.trainer.decode_from_paged_kv_cache",
            return_value=fake_outputs,
        ):
            generated = loop.generate_responses(
                input_ids=torch.tensor([[0, 11, 12]], dtype=torch.long),
                attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
            )

        assert generated == ["101 102", "201 202"]

    def test_generate_fallback_strips_prompt_tokens(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            @staticmethod
            def batch_decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return [" ".join(map(str, row.tolist())) for row in tokens]

        class DummyMemoryManager:
            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            def eval(self):
                return self

            def generate(self, input_ids, **kwargs):
                del kwargs
                responses = torch.tensor([[101, 102], [201, 202]], dtype=torch.long)
                return torch.cat([input_ids, responses], dim=1)

        config = get_8gb_vram_config()
        config.training.use_triton_kernels = True
        config.grpo.group_size = 2
        loop = GRPOTrainerLoop(config)
        loop.model = DummyModel()
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.group_sampler = types.SimpleNamespace(group_size=2)
        loop._gen_micro_batch = 2
        loop.device = "cpu"
        loop._prefill_prompt_cache = lambda *args, **kwargs: object()
        loop._expand_prefix_cache = staticmethod(lambda cache, repeats: cache)
        loop._generate_with_expanded_prefix_cache = lambda **kwargs: torch.tensor(
            [[101, 102], [201, 202]], dtype=torch.long
        )

        with patch("src.grpo.trainer.TRITON_AVAILABLE", False):
            generated = loop.generate_responses(
                input_ids=torch.tensor([[0, 11, 12]], dtype=torch.long),
                attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
            )

        assert generated == ["101 102", "201 202"]

    def test_generate_responses_with_tokens_preserves_exact_rollout_ids(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

            @staticmethod
            def batch_decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return [" ".join(map(str, row.tolist())) for row in tokens]

        class DummyMemoryManager:
            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            def eval(self):
                return self

        config = get_8gb_vram_config()
        config.training.use_triton_kernels = True
        config.training.generation_do_sample = False
        config.grpo.group_size = 2
        loop = GRPOTrainerLoop(config)
        loop.model = DummyModel()
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.group_sampler = types.SimpleNamespace(group_size=2)
        loop._gen_micro_batch = 2
        loop.device = "cpu"

        fake_outputs = torch.tensor([[101, 9, 0], [201, 202, 0]], dtype=torch.long)
        fake_state = types.SimpleNamespace(
            k_cache=torch.zeros((1, 1, 1, 1, 1)),
            v_cache=torch.zeros((1, 1, 1, 1, 1)),
            block_tables=torch.zeros((1, 1), dtype=torch.int32),
            context_lens=torch.zeros((1,), dtype=torch.int32),
            last_tokens=torch.tensor([12], dtype=torch.long),
            max_context=8,
            block_size=1,
        )

        with patch("src.grpo.trainer.TRITON_AVAILABLE", True), patch(
            "src.grpo.trainer.prefill_paged_kv_cache",
            return_value=fake_state,
        ), patch(
            "src.grpo.trainer.expand_paged_kv_cache_state",
            return_value=fake_state,
        ), patch(
            "src.grpo.trainer.decode_from_paged_kv_cache",
            return_value=fake_outputs,
        ):
            texts, response_ids, response_mask = loop._generate_responses_with_tokens(
                input_ids=torch.tensor([[0, 11, 12]], dtype=torch.long),
                attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
            )

        assert texts == ["101 9", "201 202"]
        assert torch.equal(
            response_ids.cpu(),
            torch.tensor([[101, 9, 0], [201, 202, 0]], dtype=torch.long),
        )
        assert torch.equal(
            response_mask.cpu(),
            torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.long),
        )

    def test_training_step_uses_lora_disabled_reference_logits_for_kl(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.core.lora import ManualLoRALayer
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

            def __call__(self, texts, **kwargs):
                del texts, kwargs
                return {
                    "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                }

        class DummyMemoryManager:
            @staticmethod
            def reset_peak_stats():
                return None

            @staticmethod
            def maybe_update_checkpointing(*args, **kwargs):
                return None

            @staticmethod
            def optimize_for_inference():
                return None

            @staticmethod
            def optimize_for_training():
                return None

            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

            @staticmethod
            def step():
                return None

            @staticmethod
            def get_memory_stats():
                return {"reserved_gb": 0.0}

        class TinyKLModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.head = ManualLoRALayer(
                    nn.Linear(8, 16, bias=False),
                    rank=2,
                    alpha=4,
                    adapter_quantization="none",
                    use_triton=False,
                )
                with torch.no_grad():
                    self.head.lora_B.weight.fill_(0.2)

            def forward(
                self,
                input_ids,
                attention_mask=None,
                use_cache=False,
                past_key_values=None,
            ):
                del attention_mask, use_cache, past_key_values
                x = self.embed(input_ids)
                logits = self.head(x)
                return types.SimpleNamespace(logits=logits)

        class CaptureTrainer:
            def __init__(self):
                self.captured = {}

            @staticmethod
            def calculate_advantages(rewards):
                return rewards

            def compute_grpo_loss(self, **kwargs):
                self.captured["policy_logits"] = kwargs["policy_logits"].detach()
                self.captured["reference_logits"] = kwargs["reference_logits"]
                loss = kwargs["policy_logits"].float().mean()
                return loss, {"loss": float(loss.item()), "ratio_mean": 1.0}

        config = get_8gb_vram_config()
        config.grpo.group_size = 1
        config.grpo.use_kl = True
        config.entropy.use_entropy_mask = False
        config.grpo.mask_truncated_completions = False
        config.training.gradient_accumulation_steps = 1
        config.training.log_interval = 10_000
        loop = GRPOTrainerLoop(config)
        loop.model = TinyKLModel()
        loop.device = "cpu"
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.verifier = types.SimpleNamespace(
            verify=lambda gen_text, gt: (1.0, {"match": True})
        )
        loop.grpo_trainer = CaptureTrainer()
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.01)
        loop.scheduler = types.SimpleNamespace(
            step=lambda: None, get_last_lr=lambda: [0.01]
        )
        loop._generate_responses_with_tokens = lambda input_ids, attention_mask: (
            ["resp"],
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[1, 1]], dtype=torch.long),
        )
        loop._compute_old_log_probs_with_prompt_cache = lambda **kwargs: torch.zeros(
            1, 3, dtype=torch.float32
        )

        batch = {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "answers": ["5"],
            "questions": ["q"],
        }

        loop.training_step(batch)

        captured = loop.grpo_trainer.captured
        assert captured["reference_logits"] is not None
        assert not torch.allclose(
            captured["policy_logits"], captured["reference_logits"]
        )

    def test_training_step_computes_frozen_policy_passes_in_eval_mode(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

        class DummyMemoryManager:
            @staticmethod
            def reset_peak_stats():
                return None

            @staticmethod
            def maybe_update_checkpointing(*args, **kwargs):
                return None

            @staticmethod
            def get_peak_memory_stats():
                return {"usage_fraction": 0.0, "peak_usage_fraction": 0.0}

            @staticmethod
            def optimize_for_inference():
                return None

            @staticmethod
            def optimize_for_training():
                return None

            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

            @staticmethod
            def step():
                return None

            @staticmethod
            def get_memory_stats():
                return {"reserved_gb": 0.0}

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.head = nn.Linear(8, 16, bias=False)
                self.forward_training_states = []

            def forward(
                self,
                input_ids,
                attention_mask=None,
                use_cache=False,
                past_key_values=None,
            ):
                del attention_mask, use_cache, past_key_values
                self.forward_training_states.append(self.training)
                x = self.embed(input_ids)
                return types.SimpleNamespace(logits=self.head(x))

        class CaptureTrainer:
            @staticmethod
            def calculate_advantages(rewards):
                return rewards

            @staticmethod
            def compute_grpo_loss(**kwargs):
                loss = kwargs["policy_logits"].float().mean()
                return loss, {"loss": float(loss.item()), "ratio_mean": 1.0}

        config = get_8gb_vram_config()
        config.grpo.group_size = 1
        config.grpo.use_kl = False
        config.entropy.use_entropy_mask = False
        config.grpo.mask_truncated_completions = False
        config.training.gradient_accumulation_steps = 1
        loop = GRPOTrainerLoop(config)
        loop.model = TinyModel()
        loop.device = "cpu"
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.verifier = types.SimpleNamespace(
            verify=lambda gen_text, gt: (1.0, {"match": True})
        )
        loop.grpo_trainer = CaptureTrainer()
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.01)
        loop.scheduler = types.SimpleNamespace(
            step=lambda: None, get_last_lr=lambda: [0.01]
        )
        observed_train_flags = []
        loop._generate_responses_with_tokens = lambda input_ids, attention_mask: (
            ["resp"],
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[1, 1]], dtype=torch.long),
        )

        def capture_old_log_probs(**kwargs):
            del kwargs
            observed_train_flags.append(loop.model.training)
            return torch.zeros(1, 3, dtype=torch.float32)

        loop._compute_old_log_probs_with_prompt_cache = capture_old_log_probs

        batch = {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "answers": ["5"],
            "questions": ["q"],
        }

        loop.training_step(batch)

        assert observed_train_flags == [False]
        assert loop.model.forward_training_states[-1] is True

    def test_train_epoch_flushes_partial_gradient_accumulation(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.gradient_accumulation_steps = 4
        config.training.log_interval = 10_000
        config.training.save_interval = 10_000
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.scheduler = types.SimpleNamespace(
            step=lambda: None, get_last_lr=lambda: [0.1]
        )
        loop.memory_manager = types.SimpleNamespace(
            get_memory_stats=lambda: {"reserved_gb": 0.0},
            print_memory_stats=lambda *args, **kwargs: None,
            clear_cache=lambda *args, **kwargs: None,
        )
        loop.benchmark = types.SimpleNamespace(run=lambda step: None)
        loop.save_checkpoint = lambda *args, **kwargs: None

        call_count = {"value": 0}
        real_step = loop.optimizer.step

        def counted_step():
            call_count["value"] += 1
            return real_step()

        loop.optimizer.step = counted_step
        loop.training_step = lambda batch: {
            "loss": 1.0,
            "avg_reward": 0.0,
            "tokens_per_sec": 1.0,
        }
        loop._accumulation_batches = 1

        reached = loop.train_epoch_with_skip([{"dummy": 1}], epoch=0)

        assert reached is False
        assert call_count["value"] == 1
        assert loop._accumulation_batches == 0

    def test_flush_accumulated_gradients_rescales_partial_window(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.gradient_accumulation_steps = 4
        config.training.max_grad_norm = 10.0
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            loop.model.weight.zero_()
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=1.0)
        loop.scheduler = types.SimpleNamespace(step=lambda: None)
        loop._accumulation_batches = 2
        loop.model.weight.grad = torch.ones_like(loop.model.weight)

        loop._flush_accumulated_gradients()

        torch.testing.assert_close(loop.model.weight.detach(), torch.tensor([[-2.0]]))

    def test_masked_group_advantages_exclude_truncated_samples(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        loop = GRPOTrainerLoop(get_8gb_vram_config())
        loop.config.grpo.group_size = 2
        rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 1.0])

        advantages = loop._calculate_masked_group_advantages(rewards, mask)

        torch.testing.assert_close(
            advantages,
            torch.tensor([0.0, 0.0, -1.5, 1.5]),
        )

    def test_scheduler_budget_accounts_for_epoch_end_flushes(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.gradient_accumulation_steps = 4
        loop = GRPOTrainerLoop(config)

        assert loop._estimate_total_optimizer_steps(steps_per_epoch=5, num_epochs=3) == 6

    def test_generate_responses_restores_model_train_mode(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

            @staticmethod
            def batch_decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return [" ".join(map(str, row.tolist())) for row in tokens]

        class DummyMemoryManager:
            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

        config = get_8gb_vram_config()
        config.training.use_triton_kernels = False
        config.grpo.group_size = 1
        loop = GRPOTrainerLoop(config)
        loop.model = DummyModel()
        loop.model.train()
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.group_sampler = types.SimpleNamespace(group_size=1)
        loop._gen_micro_batch = 1
        loop.device = "cpu"
        loop._prefill_prompt_cache = lambda *args, **kwargs: object()
        loop._expand_prefix_cache = staticmethod(lambda cache, repeats: cache)
        loop._generate_with_expanded_prefix_cache = lambda **kwargs: torch.tensor(
            [[5, 9]], dtype=torch.long
        )

        loop._generate_responses_with_tokens(
            input_ids=torch.tensor([[7, 8]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        )

        assert loop.model.training is True

    def test_generate_responses_uses_triton_when_sampling_enabled(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

            @staticmethod
            def batch_decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return [" ".join(map(str, row.tolist())) for row in tokens]

        class DummyMemoryManager:
            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.generate_calls = 0

            def generate(self, input_ids, **kwargs):
                del kwargs
                self.generate_calls += 1
                return torch.cat(
                    [input_ids, torch.tensor([[5, 9]], dtype=torch.long)], dim=1
                )

        config = get_8gb_vram_config()
        config.training.use_triton_kernels = True
        config.training.generation_do_sample = True
        config.grpo.group_size = 1
        loop = GRPOTrainerLoop(config)
        loop.model = DummyModel()
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.group_sampler = types.SimpleNamespace(group_size=1)
        loop._gen_micro_batch = 1
        loop.device = "cpu"
        loop._prefill_prompt_cache = lambda *args, **kwargs: object()
        loop._expand_prefix_cache = staticmethod(lambda cache, repeats: cache)
        loop._generate_with_expanded_prefix_cache = lambda **kwargs: torch.tensor(
            [[5, 9]], dtype=torch.long
        )

        with patch("src.grpo.trainer.paged_kv_decode") as paged_decode:
            paged_decode.return_value = torch.tensor([[5, 9]], dtype=torch.long)
            generated = loop.generate_responses(
                input_ids=torch.tensor([[7, 8]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            )

        paged_decode.assert_called_once()
        assert generated == ["5 9"]

    def test_load_checkpoint_discards_partial_accumulation_state(self, tmp_path):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.checkpoint_dir = str(tmp_path / "checkpoints")
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.scheduler = types.SimpleNamespace(
            state_dict=lambda: {},
            load_state_dict=lambda state: None,
        )
        loop.checkpoint_manager = __import__(
            "src.utils.checkpoint", fromlist=["CheckpointManager"]
        ).CheckpointManager(config.training.checkpoint_dir)
        loop.optimizer_step = 3
        loop._accumulation_batches = 2
        loop._dataloader_seed = 123
        loop.save_checkpoint()

        restored = GRPOTrainerLoop(config)
        restored.model = nn.Linear(1, 1)
        restored.optimizer = optim.SGD(restored.model.parameters(), lr=0.1)
        restored.scheduler = types.SimpleNamespace(
            state_dict=lambda: {},
            load_state_dict=lambda state: None,
        )
        restored.checkpoint_manager = loop.checkpoint_manager

        checkpoint_path = str(
            tmp_path / "checkpoints" / f"checkpoint_step_{loop.global_step}.pt"
        )
        restored.load_checkpoint(checkpoint_path)

        assert restored._accumulation_batches == 0
        assert restored._dataloader_seed == 123

    def test_train_epoch_retries_oom_without_leaking_grads(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.log_interval = 10_000
        config.training.save_interval = 10_000
        config.training.max_steps = None
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.scheduler = types.SimpleNamespace(
            step=lambda: None, get_last_lr=lambda: [0.1]
        )
        loop.memory_manager = types.SimpleNamespace(
            get_memory_stats=lambda: {"reserved_gb": 0.0},
            print_memory_stats=lambda *args, **kwargs: None,
            clear_cache=lambda *args, **kwargs: None,
        )
        loop.benchmark = types.SimpleNamespace(run=lambda step: None)
        loop.save_checkpoint = lambda: None
        loop._gen_micro_batch = 4
        loop._train_micro_batch = 4
        loop.global_step = 1

        pre_grad = torch.tensor([[3.0]])
        loop.model.weight.grad = pre_grad.clone()
        call_count = {"value": 0}

        def fake_training_step(batch):
            del batch
            call_count["value"] += 1
            if call_count["value"] == 1:
                loop.model.weight.grad.add_(5.0)
                raise RuntimeError("CUDA out of memory")
            assert torch.allclose(loop.model.weight.grad, pre_grad)
            loop.global_step += 1
            return {"loss": 1.0, "avg_reward": 0.0, "tokens_per_sec": 1.0}

        loop.training_step = fake_training_step

        reached = loop.train_epoch_with_skip([{"dummy": 1}], epoch=0)

        assert reached is False
        assert call_count["value"] == 2
        assert loop._gen_micro_batch == 2
        assert loop._train_micro_batch == 2
