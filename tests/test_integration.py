from contextlib import contextmanager

import pytest
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

    def test_train_py_forwards_fixed_sent_stage(self, tmp_path):
        import train as train_module

        captured = {}

        class DummyTrainer:
            def __init__(self, config):
                captured["config"] = config
                self.checkpoint_manager = None

            def setup(self):
                captured["setup_called"] = True

            def train(self, sent_stage=None):
                captured["sent_stage"] = sent_stage

        with patch.object(train_module, "GRPOTrainerLoop", DummyTrainer), patch.object(
            train_module, "setup_logging"
        ), patch.object(train_module, "kill_stale_train_processes"), patch.object(
            train_module, "clear_gpu_memory"
        ), patch.object(train_module, "check_system", return_value=True), patch(
            "sys.argv",
            [
                "train.py",
                "--output-dir",
                str(tmp_path),
                "--sent-stage",
                "2",
            ],
        ):
            train_module.main()

        assert captured["setup_called"] is True
        assert captured["sent_stage"] == 2

    def test_train_py_forwards_fixed_sent_stage_when_resuming(self, tmp_path):
        import train as train_module

        captured = {}

        class DummyCheckpointManager:
            @staticmethod
            def get_latest_checkpoint():
                return "/tmp/fake_checkpoint.pt"

        class DummyTrainer:
            def __init__(self, config):
                captured["config"] = config
                self.checkpoint_manager = DummyCheckpointManager()

            def setup(self):
                captured["setup_called"] = True

            def load_checkpoint(self, checkpoint_path):
                captured["checkpoint_path"] = checkpoint_path
                return {"step": 12, "epoch": 1}

            def train(self, sent_stage=None):
                captured["sent_stage"] = sent_stage

        with patch.object(train_module, "GRPOTrainerLoop", DummyTrainer), patch.object(
            train_module, "setup_logging"
        ), patch.object(train_module, "kill_stale_train_processes"), patch.object(
            train_module, "clear_gpu_memory"
        ), patch.object(train_module, "check_system", return_value=True), patch(
            "sys.argv",
            [
                "train.py",
                "--output-dir",
                str(tmp_path),
                "--resume",
                "--sent-stage",
                "2",
            ],
        ):
            train_module.main()

        assert captured["setup_called"] is True
        assert captured["checkpoint_path"] == "/tmp/fake_checkpoint.pt"
        assert captured["sent_stage"] == 2

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

    def test_generate_fallback_singleton_sampled_response_uses_preserved_seed(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

        class DummyMemoryManager:
            @staticmethod
            def clear_cache(*args, **kwargs):
                return None

        active_seed = {"value": None}
        observed_seeds = []

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.generate_seed_history = []

            def generate(self, input_ids, **kwargs):
                del kwargs
                self.generate_seed_history.append(active_seed["value"])
                response = torch.tensor([[101, 102]], dtype=torch.long)
                return torch.cat([input_ids, response], dim=1)

        @contextmanager
        def fake_fork(seed):
            observed_seeds.append(seed)
            active_seed["value"] = seed
            try:
                yield
            finally:
                active_seed["value"] = None

        config = get_8gb_vram_config()
        config.training.generation_do_sample = True
        loop = GRPOTrainerLoop(config)
        loop.model = DummyModel()
        loop.tokenizer = DummyTokenizer()
        loop.memory_manager = DummyMemoryManager()
        loop.device = "cpu"
        loop._fork_local_sampling_rng = fake_fork

        output = loop._generate_with_model_generate(
            real_ids=torch.tensor([[11, 12]], dtype=torch.long),
            real_mask=torch.tensor([[1, 1]], dtype=torch.long),
            current_micro=1,
            sample_seeds=[12345],
        )

        assert observed_seeds == [12345]
        assert loop.model.generate_seed_history == [12345]
        torch.testing.assert_close(
            output,
            torch.tensor([[11, 12, 101, 102]], dtype=torch.long),
        )

    def test_triton_sampled_generation_keeps_grouped_decode(self):
        import sys

        sys.modules.pop("bitsandbytes", None)
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

        config = get_8gb_vram_config()
        config.training.generation_do_sample = True
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.tokenizer = DummyTokenizer()

        decode_calls = []

        def fake_expand(prefix_state, repeats):
            return ("expanded", prefix_state, repeats)

        def fake_decode(model, state, **kwargs):
            del model
            decode_calls.append((state, kwargs))
            return torch.tensor([[5, 6], [7, 8]], dtype=torch.long)

        with patch("src.grpo.trainer.expand_paged_kv_cache_state", side_effect=fake_expand), patch(
            "src.grpo.trainer.decode_from_paged_kv_cache", side_effect=fake_decode
        ):
            output = loop._generate_with_triton_paged_prefix_cache(
                prefix_state="prefix",
                current_micro=2,
                sample_seeds=[11, 22],
            )

        assert len(decode_calls) == 1
        state, kwargs = decode_calls[0]
        assert state == ("expanded", "prefix", 2)
        assert kwargs["seed"] is None
        assert kwargs["seeds"] == [11, 22]
        torch.testing.assert_close(output, torch.tensor([[5, 6], [7, 8]], dtype=torch.long))

    def test_public_triton_decode_wrappers_forward_per_row_seeds(self):
        import src.triton_kernels as triton_kernels
        from src.triton_kernels.paged_kv import paged_kv_decode_model

        model = nn.Linear(1, 1)
        input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        decode_calls = []
        wrapper_calls = []

        def fake_prefill(*args, **kwargs):
            del args, kwargs
            return "prefix"

        def fake_decode(*args, **kwargs):
            del args
            decode_calls.append(kwargs)
            return torch.tensor([[5, 6], [7, 8]], dtype=torch.long)

        def fake_model_decode(**kwargs):
            wrapper_calls.append(kwargs)
            return torch.tensor([[9, 10], [11, 12]], dtype=torch.long)

        with patch(
            "src.triton_kernels.paged_kv.prefill_paged_kv_cache",
            side_effect=fake_prefill,
        ), patch(
            "src.triton_kernels.paged_kv.decode_from_paged_kv_cache",
            side_effect=fake_decode,
        ):
            direct_output = paged_kv_decode_model(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                block_size=16,
                do_sample=True,
                temperature=0.9,
                top_p=0.8,
                pad_token_id=0,
                eos_token_id=2,
                seed=None,
                seeds=[11, 22],
            )

        with patch("src.triton_kernels._get_triton_kernel", return_value=object()), patch(
            "src.triton_kernels.paged_kv.paged_kv_decode_model",
            side_effect=fake_model_decode,
        ):
            wrapper_output = triton_kernels.paged_kv_decode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                do_sample=True,
                temperature=0.9,
                top_p=0.8,
                pad_token_id=0,
                eos_token_id=2,
                seeds=[33, 44],
            )

        assert len(decode_calls) == 1
        assert decode_calls[0]["seed"] is None
        assert decode_calls[0]["seeds"] == [11, 22]
        torch.testing.assert_close(
            direct_output,
            torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
        )

        assert len(wrapper_calls) == 1
        assert wrapper_calls[0]["seed"] is None
        assert wrapper_calls[0]["seeds"] == [33, 44]
        torch.testing.assert_close(
            wrapper_output,
            torch.tensor([[9, 10], [11, 12]], dtype=torch.long),
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
        loop._generate_responses_with_tokens = lambda input_ids, attention_mask, sample_seeds=None: (
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
        loop._generate_responses_with_tokens = lambda input_ids, attention_mask, sample_seeds=None: (
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

    def test_training_step_preserves_post_flush_peak_for_vram_auto(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.head = nn.Linear(8, 16, bias=False)

            def forward(
                self,
                input_ids,
                attention_mask=None,
                use_cache=False,
                past_key_values=None,
            ):
                del attention_mask, use_cache, past_key_values
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

        peak_phase = {"value": "training"}
        captured = {}

        def get_peak_memory_stats():
            if peak_phase["value"] == "training":
                return {
                    "usage_fraction": 0.45,
                    "peak_usage_fraction": 0.5,
                    "peak_allocated_gb": 3.0,
                    "peak_reserved_gb": 4.0,
                }
            return {
                "usage_fraction": 0.4,
                "peak_usage_fraction": 0.9,
                "peak_allocated_gb": 5.5,
                "peak_reserved_gb": 6.5,
            }

        def maybe_update_checkpointing(model, step, memory_stats=None):
            del model, step
            captured["memory_stats"] = memory_stats

        config = get_8gb_vram_config()
        config.grpo.group_size = 1
        config.grpo.use_kl = False
        config.entropy.use_entropy_mask = False
        config.grpo.mask_truncated_completions = False
        config.training.gradient_accumulation_steps = 1
        config.training.micro_batch_probe_interval = 1
        config.training.micro_batch_probe_cooldown = 0
        config.training.micro_batch_probe_usage_threshold = 0.8
        loop = GRPOTrainerLoop(config)
        loop.model = TinyModel()
        loop.device = "cpu"
        loop.tokenizer = DummyTokenizer()
        loop._gen_micro_batch = 1
        loop._max_gen_micro_batch = 1
        loop._train_micro_batch = 2
        loop._max_train_micro_batch = 4
        loop._refresh_oom_backoff_count()
        loop.memory_manager = types.SimpleNamespace(
            reset_peak_stats=lambda: None,
            get_peak_memory_stats=get_peak_memory_stats,
            get_memory_stats=lambda: {
                "allocated_gb": 0.1,
                "reserved_gb": 0.2,
                "max_allocated_gb": 0.3,
                "free_gb": 7.5,
                "usage_fraction": 0.1,
            },
            maybe_update_checkpointing=maybe_update_checkpointing,
            optimize_for_inference=lambda: None,
            optimize_for_training=lambda: None,
            clear_cache=lambda *args, **kwargs: None,
            step=lambda: None,
        )
        loop.verifier = types.SimpleNamespace(
            verify=lambda gen_text, gt: (1.0, {"match": True})
        )
        loop.grpo_trainer = CaptureTrainer()
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.01)
        loop.scheduler = types.SimpleNamespace(
            step=lambda: None, get_last_lr=lambda: [0.01]
        )
        real_flush = loop._flush_accumulated_gradients

        def flush_with_post_peak():
            peak_phase["value"] = "post_flush"
            return real_flush()

        loop._flush_accumulated_gradients = flush_with_post_peak
        loop._generate_responses_with_tokens = (
            lambda input_ids, attention_mask, sample_seeds=None: (
                ["resp"],
                torch.tensor([[1, 2]], dtype=torch.long),
                torch.tensor([[1, 1]], dtype=torch.long),
            )
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

        assert captured["memory_stats"]["peak_usage_fraction"] == 0.9
        assert captured["memory_stats"]["peak_reserved_gb"] == 6.5
        assert loop._phase_probe_usage["training"] == 0.9

        loop._record_successful_recovery_batch()

        assert loop._train_micro_batch == 2

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
        loop._run_training_step_with_oom_recovery = lambda batch, epoch: {
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
        loop.global_step = 5
        loop.current_step = 5
        loop.current_epoch = 1
        loop.optimizer_step = 3
        loop._accumulation_batches = 2
        loop._dataloader_seed = 123
        loop._gen_micro_batch = 2
        loop._train_micro_batch = 1
        loop._gen_probe_success_batches = 5
        loop._train_probe_success_batches = 7
        loop._gen_probe_cooldown = 3
        loop._train_probe_cooldown = 4
        loop._refresh_oom_backoff_count()
        loop._partial_accumulation_replay_step = 3
        loop._partial_accumulation_recovery_state = (
            loop._serialize_adaptive_recovery_state()
        )
        loop._gen_micro_batch = 3
        loop._train_micro_batch = 2
        loop._gen_probe_success_batches = 1
        loop._train_probe_success_batches = 2
        loop._gen_probe_cooldown = 1
        loop._train_probe_cooldown = 0
        loop._refresh_oom_backoff_count()
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
        assert restored.global_step == 3
        assert restored.current_step == 3
        assert restored._resume_step == 3
        assert restored._gen_micro_batch == 2
        assert restored._train_micro_batch == 1
        assert restored._gen_probe_success_batches == 5
        assert restored._train_probe_success_batches == 7
        assert restored._gen_probe_cooldown == 3
        assert restored._train_probe_cooldown == 4
        assert restored._oom_backoff_count == 2
        assert restored._partial_accumulation_recovery_state == {
            "gen_micro_batch": 2,
            "train_micro_batch": 1,
            "gen_probe_success_batches": 5,
            "train_probe_success_batches": 7,
            "gen_probe_cooldown": 3,
            "train_probe_cooldown": 4,
        }

    def test_load_checkpoint_restores_rng_state_at_step_boundary(self, tmp_path):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        original_torch_state = torch.get_rng_state()
        original_cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
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

            torch.manual_seed(12345)
            expected_rng_state = loop._serialize_rng_state()
            torch.set_rng_state(expected_rng_state["torch_rng_state"].clone())
            expected_sample_seeds = loop._create_rollout_sample_seeds(batch_size=2)
            torch.set_rng_state(expected_rng_state["torch_rng_state"].clone())
            loop._partial_accumulation_replay_step = 2

            loop.save_checkpoint()

            checkpoint_path = str(
                tmp_path / "checkpoints" / f"checkpoint_step_{loop.global_step}.pt"
            )
            raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            assert raw_checkpoint["partial_accumulation_replay_pending"] is False
            assert raw_checkpoint["partial_accumulation_replay_step"] == loop.global_step

            restored = GRPOTrainerLoop(config)
            restored.model = nn.Linear(1, 1)
            restored.optimizer = optim.SGD(restored.model.parameters(), lr=0.1)
            restored.scheduler = types.SimpleNamespace(
                state_dict=lambda: {},
                load_state_dict=lambda state: None,
            )
            restored.checkpoint_manager = loop.checkpoint_manager

            restored.load_checkpoint(checkpoint_path)

            actual_sample_seeds = restored._create_rollout_sample_seeds(batch_size=2)

            torch.testing.assert_close(actual_sample_seeds, expected_sample_seeds)
        finally:
            torch.set_rng_state(original_torch_state)
            if original_cuda_state is not None:
                torch.cuda.set_rng_state_all(original_cuda_state)

    def test_load_checkpoint_restores_replay_rng_state_for_partial_accumulation(
        self, tmp_path
    ):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        original_torch_state = torch.get_rng_state()
        original_cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
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
            loop.global_step = 5
            loop.current_step = 5
            loop.current_epoch = 1
            loop._accumulation_batches = 2
            loop._partial_accumulation_replay_step = 3

            torch.manual_seed(24680)
            replay_rng_state = loop._serialize_rng_state()
            torch.set_rng_state(replay_rng_state["torch_rng_state"].clone())
            expected_sample_seeds = loop._create_rollout_sample_seeds(batch_size=2)
            loop._partial_accumulation_rng_state = replay_rng_state

            _ = torch.randint(low=0, high=2**31 - 1, size=(8,), dtype=torch.int64)

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

            actual_sample_seeds = restored._create_rollout_sample_seeds(batch_size=2)

            torch.testing.assert_close(actual_sample_seeds, expected_sample_seeds)
        finally:
            torch.set_rng_state(original_torch_state)
            if original_cuda_state is not None:
                torch.cuda.set_rng_state_all(original_cuda_state)

    def test_load_checkpoint_replays_first_in_progress_batch_from_window_start(
        self, tmp_path
    ):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        original_torch_state = torch.get_rng_state()
        original_cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
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
            loop.global_step = 5
            loop.current_step = 5
            loop.current_epoch = 1
            loop._training_step_in_progress = True
            loop._accumulation_batches = 0
            loop._partial_accumulation_replay_step = 5

            torch.manual_seed(13579)
            replay_rng_state = loop._serialize_rng_state()
            torch.set_rng_state(replay_rng_state["torch_rng_state"].clone())
            expected_sample_seeds = loop._create_rollout_sample_seeds(batch_size=2)
            loop._partial_accumulation_rng_state = replay_rng_state

            _ = torch.randint(low=0, high=2**31 - 1, size=(8,), dtype=torch.int64)

            loop.save_checkpoint(suffix="_interrupted")

            restored = GRPOTrainerLoop(config)
            restored.model = nn.Linear(1, 1)
            restored.optimizer = optim.SGD(restored.model.parameters(), lr=0.1)
            restored.scheduler = types.SimpleNamespace(
                state_dict=lambda: {},
                load_state_dict=lambda state: None,
            )
            restored.checkpoint_manager = loop.checkpoint_manager

            checkpoint_path = str(
                tmp_path / "checkpoints" / f"checkpoint_step_{loop.global_step}_interrupted.pt"
            )
            restored.load_checkpoint(checkpoint_path)

            actual_sample_seeds = restored._create_rollout_sample_seeds(batch_size=2)

            assert restored.global_step == 5
            assert restored.current_step == 5
            assert restored._resume_step == 5
            torch.testing.assert_close(actual_sample_seeds, expected_sample_seeds)
        finally:
            torch.set_rng_state(original_torch_state)
            if original_cuda_state is not None:
                torch.cuda.set_rng_state_all(original_cuda_state)

    def test_save_checkpoint_after_in_step_flush_resumes_after_committed_batch(
        self, tmp_path
    ):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.checkpoint_dir = str(tmp_path / "checkpoints")
        config.training.gradient_accumulation_steps = 1

        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.scheduler = types.SimpleNamespace(
            step=lambda: None,
            state_dict=lambda: {},
            load_state_dict=lambda state: None,
        )
        loop.checkpoint_manager = __import__(
            "src.utils.checkpoint", fromlist=["CheckpointManager"]
        ).CheckpointManager(config.training.checkpoint_dir)

        loop.global_step = 5
        loop.current_step = 5
        loop.current_epoch = 1
        loop._training_step_in_progress = True
        loop._accumulation_batches = 1
        loop._partial_accumulation_replay_step = 5

        param = next(loop.model.parameters())
        param.grad = torch.ones_like(param)

        assert loop._flush_accumulated_gradients() is True
        assert loop._partial_accumulation_replay_step == 6

        loop.save_checkpoint(suffix="_interrupted")

        checkpoint_path = (
            tmp_path / "checkpoints" / f"checkpoint_step_{loop.global_step}_interrupted.pt"
        )
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert raw_checkpoint["partial_accumulation_replay_pending"] is True
        assert raw_checkpoint["partial_accumulation_replay_step"] == 6

        restored = GRPOTrainerLoop(config)
        restored.model = nn.Linear(1, 1)
        restored.optimizer = optim.SGD(restored.model.parameters(), lr=0.1)
        restored.scheduler = types.SimpleNamespace(
            step=lambda: None,
            state_dict=lambda: {},
            load_state_dict=lambda state: None,
        )
        restored.checkpoint_manager = loop.checkpoint_manager

        restored.load_checkpoint(str(checkpoint_path))

        assert restored.global_step == 6
        assert restored.current_step == 6
        assert restored._resume_step == 6
        assert restored._accumulation_batches == 0

    def test_prepare_training_state_keeps_rollout_tensors_on_cpu(self):
        from src.grpo.trainer import GRPOTrainerLoop, RolloutStepState
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.grpo.group_size = 2
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.device = "cpu"
        loop.memory_manager = types.SimpleNamespace(
            optimize_for_inference=lambda: None,
            optimize_for_training=lambda: None,
        )
        loop._compute_old_log_probs_with_prompt_cache = lambda **kwargs: torch.zeros(
            (2, 3), dtype=torch.float32
        )

        rollout_state = RolloutStepState(
            generated_texts=["a", "b"],
            response_ids_cpu=torch.tensor([[5, 6], [7, 0]], dtype=torch.long),
            response_mask_cpu=torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            rewards_cpu=torch.tensor([1.0, 0.0], dtype=torch.float32),
            response_lengths_cpu=torch.tensor([2.0, 1.0], dtype=torch.float32),
            truncation_mask_cpu=torch.tensor([1.0, 1.0], dtype=torch.float32),
            advantages_cpu=torch.tensor([0.5, -0.5], dtype=torch.float32),
            debug_infos=[{}, {}],
        )

        prepared = loop._prepare_training_state(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            },
            rollout_state,
        )

        assert prepared.all_input_ids.device.type == "cpu"
        assert prepared.all_attention_mask.device.type == "cpu"
        assert prepared.response_only_mask.device.type == "cpu"
        assert prepared.advantages.device.type == "cpu"
        assert prepared.all_old_log_probs.device.type == "cpu"

    def test_setup_disables_checkpoint_manager_when_checkpoint_dir_is_empty(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.checkpoint_dir = ""
        config.sent.enabled = False
        config.wandb.enabled = False

        loop = GRPOTrainerLoop(config)

        fake_model = nn.Linear(1, 1)
        fake_tokenizer = object()

        with patch("src.grpo.trainer.load_4bit_engine", return_value=(fake_model, fake_tokenizer)), patch(
            "src.grpo.trainer.inject_lora_layers"
        ), patch(
            "src.grpo.trainer.print_model_memory_usage"
        ), patch(
            "src.grpo.trainer.get_lora_parameters",
            return_value=list(fake_model.parameters()),
        ), patch(
            "src.grpo.trainer.MemoryManager"
        ) as memory_manager_cls, patch(
            "src.grpo.trainer.GSM8KBenchmark"
        ):
            memory_manager_cls.return_value = types.SimpleNamespace(
                enable_checkpointing=lambda model: None,
                print_memory_stats=lambda prefix: None,
            )
            loop.setup()

        assert loop.checkpoint_manager is None

    def test_setup_passes_model_loading_knobs_from_config(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.sent.enabled = False
        config.wandb.enabled = False
        config.model.model_id = "custom-model"
        config.model.load_in_4bit = False
        config.model.bnb_4bit_compute_dtype = "float16"
        config.model.bnb_4bit_quant_type = "fp4"
        config.model.bnb_4bit_use_double_quant = False
        config.model.attn_implementation = "eager"
        config.model.device_map = "cpu"

        loop = GRPOTrainerLoop(config)
        fake_model = nn.Linear(1, 1)
        fake_tokenizer = object()

        with patch("src.grpo.trainer.load_4bit_engine", return_value=(fake_model, fake_tokenizer)) as load_engine, patch(
            "src.grpo.trainer.inject_lora_layers"
        ), patch(
            "src.grpo.trainer.print_model_memory_usage"
        ), patch(
            "src.grpo.trainer.get_lora_parameters",
            return_value=list(fake_model.parameters()),
        ), patch(
            "src.grpo.trainer.MemoryManager"
        ) as memory_manager_cls, patch(
            "src.grpo.trainer.GSM8KBenchmark"
        ):
            memory_manager_cls.return_value = types.SimpleNamespace(
                enable_checkpointing=lambda model: None,
                print_memory_stats=lambda prefix: None,
            )
            loop.setup()

        load_engine.assert_called_once_with(
            "custom-model",
            load_in_4bit=False,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            attn_implementation="eager",
            device_map="cpu",
        )

    def test_secondary_trainer_cli_debug_flag_sets_verbosity(self):
        import src.grpo.trainer as trainer_module

        captured = {}

        class DummyTrainer:
            def __init__(self, config):
                captured["config"] = config

            def setup(self):
                captured["setup_called"] = True

            def train(self, sent_stage=None):
                captured["sent_stage"] = sent_stage

        with patch.object(trainer_module, "GRPOTrainerLoop", DummyTrainer), patch(
            "sys.argv", ["trainer.py", "--debug"]
        ):
            trainer_module.main()

        assert captured["config"].training.verbosity >= 1
        assert captured["setup_called"] is True
        assert captured["sent_stage"] is None

    def test_save_lora_weights_writes_final_artifact_without_checkpoint_manager(self, tmp_path):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class TinyLoRAModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.ones(2, 2))
                self.lora_B = nn.Parameter(torch.ones(2, 2) * 2)

        config = get_8gb_vram_config()
        config.training.output_dir = str(tmp_path)
        config.training.checkpoint_dir = None
        loop = GRPOTrainerLoop(config)
        loop.model = TinyLoRAModule()
        loop.global_step = 7
        loop.current_epoch = 2
        loop.checkpoint_manager = None

        loop.save_lora_weights(suffix="_final")

        saved = torch.load(tmp_path / "lora_weights_final.pt", map_location="cpu")
        assert "lora_weights" in saved
        assert "lora_A" in saved["lora_weights"]
        assert "lora_B" in saved["lora_weights"]
        assert saved["metadata"]["step"] == 7
        assert saved["metadata"]["epoch"] == 2

    def test_train_auto_advances_sent_stage(self, tmp_path):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class FakeSentDataset:
            def __init__(self):
                self.use_sent = True
                self.num_stages = 2
                self.current_stage = 0
                self.stage_history = []

            def set_stage(self, stage):
                self.current_stage = stage
                self.stage_history.append(stage)

            def get_stage_info(self):
                return {
                    "num_stages": self.num_stages,
                    "stage_start_idx": (self.current_stage - 1) * 10,
                    "stage_end_idx": self.current_stage * 10,
                }

        class FakeLoader(list):
            pass

        config = get_8gb_vram_config()
        config.training.output_dir = str(tmp_path)
        config.training.checkpoint_dir = None
        config.training.num_epochs = 3
        config.training.skip_initial_benchmark = True
        config.training.max_steps = None
        config.sent.enabled = True
        config.sent.curriculum_stages = 2
        config.wandb.enabled = False

        loop = GRPOTrainerLoop(config)
        loop.tokenizer = object()

        dataset = FakeSentDataset()
        loader = FakeLoader([{"dummy": 1}])
        loader.dataset = dataset
        loader.uses_sent_curriculum = True
        loader.sent_num_stages = 2

        with patch("src.grpo.trainer.create_grpo_dataloader", return_value=loader):
            loop._configure_scheduler = lambda *args, **kwargs: None
            loop.train_epoch_with_skip = lambda *args, **kwargs: False
            loop.save_checkpoint = lambda *args, **kwargs: None
            loop.save_lora_weights = lambda *args, **kwargs: None
            loop._finish_wandb = lambda *args, **kwargs: None
            loop.train()

        assert dataset.stage_history == [1, 2]

    def test_train_uses_plain_epoch_semantics_when_sent_falls_back(self, tmp_path):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        class FakeLoader(list):
            pass

        config = get_8gb_vram_config()
        config.training.output_dir = str(tmp_path)
        config.training.checkpoint_dir = None
        config.training.num_epochs = 2
        config.training.skip_initial_benchmark = True
        config.training.max_steps = None
        config.sent.enabled = True
        config.wandb.enabled = False

        loop = GRPOTrainerLoop(config)
        loop.tokenizer = object()

        created_epochs = []

        def fake_create(sent_stage, epoch):
            created_epochs.append(epoch)
            loader = FakeLoader([{"dummy": 1}])
            loader.dataset = object()
            loader.uses_sent_curriculum = False
            loader.sent_num_stages = 1
            return loader

        loop._create_train_dataloader = fake_create
        loop._configure_scheduler = lambda *args, **kwargs: None
        loop.train_epoch_with_skip = lambda *args, **kwargs: False
        loop.save_checkpoint = lambda *args, **kwargs: None
        loop.save_lora_weights = lambda *args, **kwargs: None
        loop._finish_wandb = lambda *args, **kwargs: None

        loop.train()

        assert created_epochs == [0, 0, 1]

    def test_retry_reuses_rollouts_on_training_oom_without_leaking_grads(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.memory_manager = types.SimpleNamespace(
            clear_cache=lambda *args, **kwargs: None,
            enable_checkpointing=lambda *args, **kwargs: None,
            get_memory_stats=lambda: {"reserved_gb": 0.0},
        )
        loop._gen_micro_batch = 4
        loop._train_micro_batch = 4
        loop._max_gen_micro_batch = 4
        loop._max_train_micro_batch = 4
        loop._refresh_oom_backoff_count()
        loop._begin_training_step = lambda: None

        pre_grad = torch.tensor([[3.0]])
        loop.model.weight.grad = pre_grad.clone()
        call_count = {"rollout": 0, "prepare": 0, "train": 0}

        def fake_prepare_rollout(batch, sample_seeds=None):
            del batch, sample_seeds
            call_count["rollout"] += 1
            return object()

        def fake_prepare_training(batch, rollout_state):
            del batch, rollout_state
            call_count["prepare"] += 1
            return object()

        def fake_execute_training(prepared_state):
            del prepared_state
            call_count["train"] += 1
            if call_count["train"] == 1:
                loop.model.weight.grad.add_(5.0)
                raise RuntimeError("CUDA out of memory")
            assert torch.allclose(loop.model.weight.grad, pre_grad)
            return {"loss": 1.0, "avg_reward": 0.0, "tokens_per_sec": 1.0}

        loop._prepare_rollout_state = fake_prepare_rollout
        loop._prepare_training_state = fake_prepare_training
        loop._execute_training_from_state = fake_execute_training

        metrics = loop._run_training_step_with_oom_recovery(
            {"input_ids": torch.tensor([[1]], dtype=torch.long)}, epoch=0
        )

        assert metrics["loss"] == 1.0
        assert call_count == {"rollout": 1, "prepare": 1, "train": 2}
        assert loop._gen_micro_batch == 4
        assert loop._train_micro_batch == 2

    def test_training_retry_restores_torch_rng_state(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.memory_manager = types.SimpleNamespace(
            clear_cache=lambda *args, **kwargs: None,
            enable_checkpointing=lambda *args, **kwargs: None,
            get_memory_stats=lambda: {"reserved_gb": 0.0},
        )
        loop._gen_micro_batch = 4
        loop._train_micro_batch = 4
        loop._max_gen_micro_batch = 4
        loop._max_train_micro_batch = 4
        loop._refresh_oom_backoff_count()
        loop._begin_training_step = lambda: None

        call_count = {"rollout": 0, "prepare": 0, "train": 0}
        observed_draws = []
        initial_torch_state = torch.get_rng_state()

        def fake_prepare_rollout(batch, sample_seeds=None):
            del batch, sample_seeds
            call_count["rollout"] += 1
            return object()

        def fake_prepare_training(batch, rollout_state):
            del batch, rollout_state
            call_count["prepare"] += 1
            return object()

        def fake_execute_training(prepared_state):
            del prepared_state
            call_count["train"] += 1
            draw = torch.rand(8)
            observed_draws.append(draw)
            if call_count["train"] == 1:
                raise RuntimeError("CUDA out of memory")
            torch.testing.assert_close(observed_draws[1], observed_draws[0])
            return {"loss": 1.0, "avg_reward": 0.0, "tokens_per_sec": 1.0}

        loop._prepare_rollout_state = fake_prepare_rollout
        loop._prepare_training_state = fake_prepare_training
        loop._execute_training_from_state = fake_execute_training

        try:
            metrics = loop._run_training_step_with_oom_recovery(
                {"input_ids": torch.tensor([[1]], dtype=torch.long)}, epoch=0
            )
        finally:
            torch.set_rng_state(initial_torch_state)

        assert metrics["loss"] == 1.0
        assert call_count == {"rollout": 1, "prepare": 1, "train": 2}
        assert len(observed_draws) == 2
        assert loop._train_micro_batch == 2

    def test_retry_rng_snapshot_captures_all_cuda_devices(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        loop = GRPOTrainerLoop(get_8gb_vram_config())

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=2
        ), patch(
            "torch.cuda.get_rng_state",
            side_effect=lambda device_idx: torch.tensor(
                [device_idx], dtype=torch.uint8
            ),
        ):
            _, cuda_state = loop._capture_retry_rng_state()

        assert cuda_state is not None
        assert sorted(cuda_state) == [0, 1]
        torch.testing.assert_close(cuda_state[0], torch.tensor([0], dtype=torch.uint8))
        torch.testing.assert_close(cuda_state[1], torch.tensor([1], dtype=torch.uint8))

    def test_fork_local_sampling_rng_privately_seeds_all_visible_cuda_devices(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        loop = GRPOTrainerLoop(get_8gb_vram_config())
        loop.device = "cuda:1"
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

        with patch("torch.cuda.is_available", return_value=True), patch(
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
            with loop._fork_local_sampling_rng(123):
                pass

        assert manual_seed.call_count == 2
        manual_seed.assert_any_call(123)
        manual_seed_all.assert_not_called()
        assert manual_seed_calls == [123, 123]
        assert entered_devices == [0, 1]

    def test_restore_serialized_rng_state_skips_missing_cuda_devices(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        loop = GRPOTrainerLoop(get_8gb_vram_config())
        restored_devices = []

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), patch(
            "torch.cuda.set_rng_state",
            side_effect=lambda state, device: restored_devices.append(
                (int(device), state.clone())
            ),
        ):
            restored = loop._restore_serialized_rng_state(
                {
                    "torch_rng_state": torch.get_rng_state().clone(),
                    "cuda_rng_state": {
                        0: torch.tensor([0], dtype=torch.uint8),
                        1: torch.tensor([1], dtype=torch.uint8),
                    },
                }
            )

        assert restored is True
        assert len(restored_devices) == 1
        assert restored_devices[0][0] == 0
        torch.testing.assert_close(
            restored_devices[0][1], torch.tensor([0], dtype=torch.uint8)
        )

    def test_training_flush_oom_restores_accumulation_batches_before_retry(self):
        from src.grpo.trainer import GRPOTrainerLoop, PreparedTrainingState
        from src.utils.config import get_8gb_vram_config

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 9

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.head = nn.Linear(8, 16, bias=False)

            def forward(
                self,
                input_ids,
                attention_mask=None,
                use_cache=False,
                past_key_values=None,
            ):
                del attention_mask, use_cache, past_key_values
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

        class TrackingScheduler:
            def __init__(self):
                self.step_calls = 0

            def step(self):
                self.step_calls += 1

            def get_last_lr(self):
                return [0.1]

            def state_dict(self):
                return {"step_calls": self.step_calls}

            def load_state_dict(self, state):
                self.step_calls = int(state["step_calls"])

        config = get_8gb_vram_config()
        config.grpo.group_size = 1
        config.grpo.use_kl = False
        config.grpo.mask_truncated_completions = False
        config.entropy.use_entropy_mask = False
        config.training.gradient_accumulation_steps = 1

        loop = GRPOTrainerLoop(config)
        loop.model = TinyModel()
        loop.device = "cpu"
        loop.tokenizer = DummyTokenizer()
        loop.optimizer = optim.AdamW(loop.model.parameters(), lr=0.1)
        loop._initialize_optimizer_state()
        loop.scheduler = TrackingScheduler()
        loop.memory_manager = types.SimpleNamespace(
            clear_cache=lambda *args, **kwargs: None,
            enable_checkpointing=lambda *args, **kwargs: None,
            maybe_update_checkpointing=lambda *args, **kwargs: None,
            get_peak_memory_stats=lambda: {
                "usage_fraction": 0.0,
                "peak_usage_fraction": 0.0,
            },
            get_memory_stats=lambda: {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "free_gb": 0.0,
                "usage_fraction": 0.0,
            },
            optimize_for_inference=lambda: None,
            optimize_for_training=lambda: None,
            reset_peak_stats=lambda: None,
            step=lambda: None,
        )
        loop.verifier = types.SimpleNamespace(
            verify=lambda gen_text, gt: (1.0, {"match": True})
        )
        loop.grpo_trainer = CaptureTrainer()

        prepared_state = PreparedTrainingState(
            all_input_ids=torch.tensor([[3, 4, 5]], dtype=torch.long),
            all_attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
            response_only_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
            advantages=torch.tensor([1.0], dtype=torch.float32),
            rewards=torch.tensor([1.0], dtype=torch.float32),
            response_lengths=torch.tensor([2.0], dtype=torch.float32),
            truncation_mask=torch.tensor([1.0], dtype=torch.float32),
            all_old_log_probs=torch.zeros((1, 2), dtype=torch.float32),
        )

        call_count = {"rollout": 0, "prepare": 0, "flush": 0}
        loop._prepare_rollout_state = lambda batch, sample_seeds=None: (
            call_count.__setitem__("rollout", call_count["rollout"] + 1) or object()
        )
        loop._prepare_training_state = lambda batch, rollout_state: (
            call_count.__setitem__("prepare", call_count["prepare"] + 1)
            or prepared_state
        )

        real_flush = loop._flush_accumulated_gradients
        baseline_param = next(loop.model.parameters()).detach().clone()
        baseline_state_step = loop.optimizer.state[next(loop.model.parameters())][
            "step"
        ].detach().clone()

        def flush_with_oom():
            call_count["flush"] += 1
            if call_count["flush"] == 1:
                param = next(loop.model.parameters())
                param.data.add_(10.0)
                loop.optimizer.state[param]["step"].fill_(99.0)
                loop.scheduler.step_calls = 7
                loop.optimizer_step = 5
                raise RuntimeError("CUDA out of memory")
            assert loop._accumulation_batches == 1
            param = next(loop.model.parameters())
            assert torch.allclose(param, baseline_param)
            assert torch.allclose(
                loop.optimizer.state[param]["step"], baseline_state_step
            )
            assert loop.scheduler.step_calls == 0
            assert loop.optimizer_step == 0
            return real_flush()

        loop._flush_accumulated_gradients = flush_with_oom

        metrics = loop._run_training_step_with_oom_recovery(
            {
                "input_ids": torch.tensor([[3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
                "answers": ["5"],
                "questions": ["q"],
            },
            epoch=0,
        )

        assert metrics["loss"] != 0.0
        assert call_count == {"rollout": 1, "prepare": 1, "flush": 2}
        assert loop._accumulation_batches == 0
        assert loop._train_micro_batch == 2
        assert loop.scheduler.step_calls == 1
        assert loop.optimizer_step == 1
        assert torch.allclose(next(loop.model.parameters()), baseline_param) is False

    def test_training_retry_snapshot_skips_param_cpu_clone_before_flush(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.gradient_accumulation_steps = 4

        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(2, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)

        for param in loop.model.parameters():
            param.grad = torch.ones_like(param)

        cpu_calls = {"value": 0}
        original_cpu = torch.Tensor.cpu

        def counting_cpu(tensor, *args, **kwargs):
            cpu_calls["value"] += 1
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(
            torch.Tensor,
            "cpu",
            autospec=True,
            side_effect=counting_cpu,
        ):
            snapshot = loop._capture_optimizer_grad_snapshot()

        assert snapshot.trainable_param_snapshot is None
        assert snapshot.optimizer_state is None
        assert snapshot.scheduler_state is None
        assert len(snapshot.grad_snapshot) == len(list(loop.model.parameters()))
        assert cpu_calls["value"] == 0

    def test_retry_reuses_rollouts_on_old_log_prob_oom(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        loop = GRPOTrainerLoop(config)
        loop.model = nn.Linear(1, 1)
        loop.optimizer = optim.SGD(loop.model.parameters(), lr=0.1)
        loop.memory_manager = types.SimpleNamespace(
            clear_cache=lambda *args, **kwargs: None,
            enable_checkpointing=lambda *args, **kwargs: None,
            get_memory_stats=lambda: {"reserved_gb": 0.0},
        )
        loop._gen_micro_batch = 4
        loop._train_micro_batch = 4
        loop._max_gen_micro_batch = 4
        loop._max_train_micro_batch = 4
        loop._refresh_oom_backoff_count()
        loop._begin_training_step = lambda: None

        call_count = {"rollout": 0, "prepare": 0, "train": 0}

        def fake_prepare_rollout(batch, sample_seeds=None):
            del batch, sample_seeds
            call_count["rollout"] += 1
            return object()

        def fake_prepare_training(batch, rollout_state):
            del batch, rollout_state
            call_count["prepare"] += 1
            if call_count["prepare"] == 1:
                raise RuntimeError("CUDA out of memory")
            return object()

        def fake_execute_training(prepared_state):
            del prepared_state
            call_count["train"] += 1
            return {"loss": 1.0, "avg_reward": 0.0, "tokens_per_sec": 1.0}

        loop._prepare_rollout_state = fake_prepare_rollout
        loop._prepare_training_state = fake_prepare_training
        loop._execute_training_from_state = fake_execute_training

        metrics = loop._run_training_step_with_oom_recovery(
            {"input_ids": torch.tensor([[1]], dtype=torch.long)}, epoch=0
        )

        assert metrics["loss"] == 1.0
        assert call_count == {"rollout": 1, "prepare": 2, "train": 1}
        assert loop._gen_micro_batch == 2
        assert loop._train_micro_batch == 4

    def test_recovery_probe_only_grows_backed_off_phase(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.micro_batch_probe_interval = 2
        config.training.micro_batch_probe_cooldown = 0
        config.training.micro_batch_probe_usage_threshold = 0.8

        loop = GRPOTrainerLoop(config)
        loop._gen_micro_batch = 2
        loop._train_micro_batch = 4
        loop._max_gen_micro_batch = 4
        loop._max_train_micro_batch = 4
        loop._refresh_oom_backoff_count()
        loop.memory_manager = types.SimpleNamespace(
            get_peak_memory_stats=lambda: {
                "usage_fraction": 0.5,
                "peak_usage_fraction": 0.5,
            },
            get_memory_stats=lambda: {"usage_fraction": 0.5},
        )

        loop._record_successful_recovery_batch()
        assert loop._gen_micro_batch == 2
        assert loop._train_micro_batch == 4

        loop._record_successful_recovery_batch()
        assert loop._gen_micro_batch == 3
        assert loop._train_micro_batch == 4

    def test_recovery_probe_uses_phase_specific_headroom(self):
        from src.grpo.trainer import GRPOTrainerLoop
        from src.utils.config import get_8gb_vram_config

        config = get_8gb_vram_config()
        config.training.micro_batch_probe_interval = 1
        config.training.micro_batch_probe_cooldown = 0
        config.training.micro_batch_probe_usage_threshold = 0.8

        loop = GRPOTrainerLoop(config)
        loop._gen_micro_batch = 2
        loop._train_micro_batch = 4
        loop._max_gen_micro_batch = 4
        loop._max_train_micro_batch = 4
        loop._refresh_oom_backoff_count()
        loop._phase_probe_usage = {
            "generation": 0.5,
            "training": 0.95,
        }
        loop.memory_manager = types.SimpleNamespace(
            get_peak_memory_stats=lambda: {
                "usage_fraction": 0.95,
                "peak_usage_fraction": 0.95,
            },
            get_memory_stats=lambda: {"usage_fraction": 0.95},
        )

        loop._record_successful_recovery_batch()

        assert loop._gen_micro_batch == 3
        assert loop._train_micro_batch == 4


def test_actual_truncation_detection_is_independent_of_loss_masking():
    from src.grpo.trainer import GRPOTrainerLoop
    from src.utils.config import get_8gb_vram_config

    config = get_8gb_vram_config()
    config.training.max_response_length = 4
    config.grpo.mask_truncated_completions = False

    loop = GRPOTrainerLoop(config)
    loop.tokenizer = types.SimpleNamespace(eos_token_id=9)

    response_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 9, 0],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    actual_truncation_mask = loop._compute_actual_truncation_mask(
        response_ids, response_mask
    )
    loss_truncation_mask = loop._compute_truncation_mask(response_ids, response_mask)

    torch.testing.assert_close(
        actual_truncation_mask, torch.tensor([0.0, 1.0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        loss_truncation_mask, torch.tensor([1.0, 1.0], dtype=torch.float32)
    )


def test_truncation_observability_metrics_reflect_actual_ratio_and_policy():
    from src.grpo.trainer import GRPOTrainerLoop
    from src.utils.config import get_8gb_vram_config

    config = get_8gb_vram_config()
    config.grpo.mask_truncated_completions = False

    loop = GRPOTrainerLoop(config)
    metrics = loop._build_truncation_observability_metrics(
        torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    )

    assert metrics["actual_truncated_completions_ratio"] == pytest.approx(0.5)
    assert metrics["truncation_masking_active"] == pytest.approx(0.0)
    assert metrics["truncated_completions_masked_out_of_loss_ratio"] == pytest.approx(
        0.0
    )

    loop.config.grpo.mask_truncated_completions = True
    metrics = loop._build_truncation_observability_metrics(
        torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    )

    assert metrics["actual_truncated_completions_ratio"] == pytest.approx(0.5)
    assert metrics["truncation_masking_active"] == pytest.approx(1.0)
    assert metrics["truncated_completions_masked_out_of_loss_ratio"] == pytest.approx(
        0.5
    )
