"""
Main GRPO Training Loop - Optimized for 8GB VRAM (RTX 3060 Ti).
Native PyTorch implementation without HF Trainer.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
from typing import Optional, Dict, List, Any
import gc
import time
import math
import argparse
import logging
import json
import hashlib
from contextlib import contextmanager

from src.utils.logging_utils import get_logger, TRACE, log_tensor_meta

# Module-level logger
logger = get_logger("trainer")

from src.core.model_loader import load_4bit_engine
from src.core.lora import inject_lora_layers, get_lora_parameters, lora_disabled
from src.core.memory_manager import MemoryManager, print_model_memory_usage
from src.grpo.algorithm import GRPOTrainer, GroupSampler
from src.grpo.verifier import RuleBasedVerifier

from src.selective.entropy_mask import EntropyCalculator
from src.triton_kernels import TRITON_AVAILABLE
from src.triton_kernels.paged_kv import (
    decode_from_paged_kv_cache,
    expand_paged_kv_cache_state,
    prefill_paged_kv_cache,
)

from src.data.gsm8k_loader import create_grpo_dataloader
from src.grpo.benchmark import GSM8KBenchmark
from src.utils.checkpoint import CheckpointManager, save_training_config
from src.utils.config import Config, get_8gb_vram_config
from tools.vram_profiler.profiler_hooks import ProfilerHooks, ProfilerState

# Enable TF32 for faster matrix operations on Ampere (RTX 3060 Ti)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
logger.info("[Setup] TF32 and cudnn.benchmark enabled for faster operations")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GRPOTrainerLoop:
    """
    Complete GRPO training loop for 8GB VRAM systems.

    Features:
    - Manual LoRA on 4-bit quantized model
    - Group sampling for GRPO
    - Entropy-based selective backpropagation
    - Aggressive memory management
    - Native PyTorch (no HF Trainer)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration (uses 8GB config if None)
        """
        self.config = config or get_8gb_vram_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("=" * 60)
        logger.info("GRPO Training Engine - 8GB VRAM Optimized")
        logger.info("=" * 60)

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.memory_manager = None
        self.grpo_trainer = None
        self.verifier = None
        self.entropy_calculator = None
        self.group_sampler = None
        self.checkpoint_manager = None
        self.generator = None
        self.benchmark = None

        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.global_step = 0
        self.optimizer_step = 0
        self._accumulation_batches = 0
        self._gen_micro_batch = 4
        self._train_micro_batch = 4
        self._oom_backoff_count = 0
        self._post_oom_success_batches = 0
        self._wandb_run = None
        self._step_start_time = None
        self._resume_step = None
        self._resume_epoch = None
        self._dataloader_seed = int(torch.initial_seed() % (2**31))
        self._profiler_hooks: ProfilerHooks | None = None
        self._metrics_jsonl_path: Optional[str] = None
        self._triton_generation_policy_logged = False

    def setup(self):
        """Setup model, tokenizer, and training components."""
        logger.info("\n[Setup] Loading model and tokenizer...")

        # Load model in 4-bit
        self.model, self.tokenizer = load_4bit_engine(self.config.model.model_id)

        if self.model is None:
            raise RuntimeError("Failed to load model")

        # Inject LoRA layers
        logger.info("\n[Setup] Injecting LoRA layers...")
        inject_lora_layers(
            self.model,
            target_modules=self.config.lora.target_modules,
            rank=self.config.lora.rank,
            alpha=self.config.lora.alpha,
            dropout=self.config.lora.dropout,
            use_triton=(
                self.config.training.use_triton_kernels
                and self.config.training.use_triton_lora
            ),
            prefer_base_layer=self.config.training.triton_lora_prefer_base,
            adapter_quantization=self.config.lora.adapter_quantization,
        )

        # Print memory usage
        print_model_memory_usage(self.model, "Model with LoRA")

        # Setup optimizer (only LoRA parameters)
        lora_params = get_lora_parameters(self.model)
        logger.info("\n[Setup] Optimizing %s LoRA parameter tensors", len(lora_params))

        self.optimizer = AdamW(
            lora_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
        )
        self._initialize_optimizer_state()

        # Setup scheduler with a placeholder budget; it is recalibrated once the
        # dataloader size is known so the cosine schedule matches the real run.
        self.scheduler = self._create_scheduler(total_optimizer_steps=1)

        # Setup memory manager
        self.memory_manager = MemoryManager(
            device=self.device,
            enable_gradient_checkpointing=self.config.training.enable_gradient_checkpointing,
            clear_cache_frequency=self.config.training.clear_cache_frequency,
            checkpointing_strategy=self.config.training.checkpointing_strategy,
            checkpointing_layer_name_patterns=(
                self.config.training.checkpointing_layer_name_patterns
            ),
            checkpointing_layer_types=self.config.training.checkpointing_layer_types,
            checkpointing_vram_enable_threshold=(
                self.config.training.checkpointing_vram_enable_threshold
            ),
            checkpointing_vram_disable_threshold=(
                self.config.training.checkpointing_vram_disable_threshold
            ),
            checkpointing_update_interval_steps=(
                self.config.training.checkpointing_update_interval_steps
            ),
        )
        self.memory_manager.enable_checkpointing(self.model)

        if getattr(self.config.training, "profile_enabled", False):
            state = ProfilerState.get_instance()
            state.set_config_snapshot(self.config.to_dict())
            self._profiler_hooks = ProfilerHooks(
                state=state, memory_manager=self.memory_manager
            )

        # Setup GRPO components
        self.grpo_trainer = GRPOTrainer(
            clip_epsilon=self.config.grpo.clip_epsilon,
            epsilon_high=self.config.grpo.epsilon_high,
            delta=self.config.grpo.delta,
            kl_coef=self.config.grpo.kl_coef,
            group_size=self.config.grpo.group_size,
            use_kl=self.config.grpo.use_kl,
            use_triton_kernels=(
                self.config.training.use_triton_kernels
                and self.config.training.use_triton_grpo_loss
            ),
        )

        # QLoRA: Keep LayerNorms frozen - only LoRA adapters are trainable
        # This saves VRAM and improves stability in RL training
        # LayerNorm unfreezing is not needed for GSM8K with rank-16 LoRA

        self.verifier = RuleBasedVerifier()
        self.entropy_calculator = EntropyCalculator(
            threshold=self.config.entropy.threshold,
            percentile=self.config.entropy.percentile,
            min_tokens=self.config.entropy.min_tokens,
        )
        self.group_sampler = GroupSampler(group_size=self.config.grpo.group_size)

        # Setup checkpoint manager
        checkpoint_dir = self.config.training.checkpoint_dir
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = checkpoint_dir.strip() or None

        if checkpoint_dir is not None:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir
            )
        else:
            self.checkpoint_manager = None

        # Initialize benchmark
        self.benchmark = GSM8KBenchmark(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_manager=self.memory_manager,
            dataset_split="test",
            num_samples=50,
            device=self.device,
            generate_fn=self.generate_benchmark_responses,
            max_new_tokens=self.config.training.max_response_length,
            max_prompt_length=self.config.training.max_prompt_length,
            do_sample=self.config.training.generation_do_sample,
            temperature=self.config.training.generation_temperature,
            top_p=self.config.training.generation_top_p,
        )

        # Setup WandB
        self._setup_wandb()

        logger.info("\n[Setup] Complete!")
        self.memory_manager.print_memory_stats("[Setup]")

    def _setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        if not self.config.wandb.enabled:
            logger.info("[WandB] Disabled by configuration")
            return

        if not WANDB_AVAILABLE:
            logger.info("[WandB] Not installed. Run: pip install wandb")
            return

        try:
            run_name = self.config.wandb.run_name
            if not run_name:
                run_name = f"python-grpo-{time.strftime('%Y%m%d-%H%M%S')}"

            self._wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity if self.config.wandb.entity else None,
                name=run_name,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config={
                    "implementation": "python",
                    "model": self.config.model.__dict__,
                    "lora": self.config.lora.__dict__,
                    "grpo": self.config.grpo.__dict__,
                    "entropy": self.config.entropy.__dict__,
                    "training": self.config.training.__dict__,
                },
                reinit=True,
            )

            if self.config.wandb.log_model and self.model is not None:
                wandb.watch(
                    self.model,
                    log="gradients" if self.config.wandb.log_gradients else None,
                    log_freq=100,
                )

            logger.info("[WandB] Initialized run: %s", run_name)
            logger.info("[WandB] Project: %s", self.config.wandb.project)
            logger.info("[WandB] URL: %s", self._wandb_run.get_url())

        except Exception as e:
            logger.info("[WandB] Failed to initialize: %s", e)
            self._wandb_run = None

    def _build_metrics_log_dict(
        self, metrics: Dict[str, float], prefix: str = "train"
    ) -> Dict[str, float]:
        """Build the canonical metrics payload shared by WandB and JSONL logs."""
        log_dict: Dict[str, float] = {
            f"{prefix}/epoch": self.current_epoch,
        }

        for key, value in metrics.items():
            log_dict[f"{prefix}/{key}"] = value

        vram_stats = self.memory_manager.get_memory_stats()
        if vram_stats:
            log_dict["memory/vram_used_gb"] = vram_stats.get("reserved_gb", 0)
            log_dict["memory/vram_allocated_gb"] = vram_stats.get("allocated_gb", 0)

        if self.scheduler is not None:
            log_dict["train/learning_rate"] = self.scheduler.get_last_lr()[0]
        else:
            log_dict["train/learning_rate"] = self.config.training.learning_rate
        log_dict["train/gen_micro_batch"] = self._gen_micro_batch
        log_dict["train/train_micro_batch"] = self._train_micro_batch
        log_dict["train/oom_backoff_count"] = self._oom_backoff_count

        if self._step_start_time:
            step_time = time.time() - self._step_start_time
            log_dict["perf/step_time_s"] = step_time

        return log_dict

    def _resolve_metrics_jsonl_path(self) -> Optional[str]:
        if not getattr(self.config.training, "log_metrics_jsonl", False):
            return None

        if self.config.training.metrics_jsonl_path:
            return self.config.training.metrics_jsonl_path

        return os.path.join(self.config.training.output_dir, "metrics.jsonl")

    def _prepare_metrics_jsonl(self):
        self._metrics_jsonl_path = self._resolve_metrics_jsonl_path()
        if self._metrics_jsonl_path is None:
            return

        metrics_dir = os.path.dirname(self._metrics_jsonl_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(self._metrics_jsonl_path, "w", encoding="utf-8"):
            pass

    def _append_metrics_jsonl_entry(self, event: str, payload: Dict[str, Any]):
        if self._metrics_jsonl_path is None:
            return

        entry = {
            "event": event,
            "step": self.global_step,
            "epoch": self.current_epoch,
            "timestamp": time.time(),
            **payload,
        }
        with open(self._metrics_jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def _log_wandb_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to WandB and optional local JSONL."""
        log_dict = self._build_metrics_log_dict(metrics, prefix=prefix)
        self._append_metrics_jsonl_entry("train_metrics", log_dict)

        if self._wandb_run is None:
            return

        if self.global_step % self.config.wandb.log_frequency != 0:
            return

        wandb.log(log_dict, step=self.global_step)

    def _finish_wandb(self):
        """Finish WandB run and upload final artifacts."""
        if self._wandb_run is None:
            return

        try:
            final_lora_path = os.path.join(
                self.config.training.output_dir, "lora_weights_final.pt"
            )
            if os.path.exists(final_lora_path):
                artifact = wandb.Artifact(
                    name=f"lora-weights-{self._wandb_run.id}",
                    type="model",
                    description="Final LoRA weights",
                )
                artifact.add_file(final_lora_path)
                self._wandb_run.log_artifact(artifact)

            wandb.finish()
            logger.info("[WandB] Run finished successfully")
        except Exception as e:
            logger.info("[WandB] Error finishing run: %s", e)

    def _prefill_prompt_cache(self, real_ids: torch.Tensor, real_mask: torch.Tensor):
        """Run a single forward pass on the prompt to build the KV cache.

        Returns a DynamicCache with ``prompt_len - 1`` positions filled
        (i.e. the last token is *not* cached).  This cache must come from a
        true prefix-only forward on ``real_ids[:, :-1]``. Cropping a cache that
        was built from the full prompt is not equivalent on the Qwen2/DeepSeek
        decode path and can shift the next-token logits.

        Args:
            real_ids:  [1, prompt_len] – token ids (no padding)
            real_mask: [1, prompt_len] – attention mask (all 1s)

        Returns:
            past_key_values: DynamicCache for the prompt (prompt_len-1 positions).
        """
        prompt_len = real_ids.shape[1]
        if prompt_len <= 1:
            outputs = self.model(
                input_ids=real_ids,
                attention_mask=real_mask,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            past_kv.crop(0)
            del outputs
            return past_kv

        outputs = self.model(
            input_ids=real_ids[:, :-1],
            attention_mask=real_mask[:, :-1],
            use_cache=True,
        )
        past_kv = outputs.past_key_values  # DynamicCache, seq_len = prompt_len - 1
        del outputs
        return past_kv

    def _sample_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample the next token using the configured generation policy."""
        if not self.config.training.generation_do_sample:
            return logits.argmax(dim=-1)

        temperature = max(
            float(self.config.training.generation_temperature),
            torch.finfo(logits.dtype).eps,
        )
        if temperature != 1.0:
            logits = logits / temperature

        top_p = self.config.training.generation_top_p
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(
                sorted_remove, torch.finfo(sorted_logits.dtype).min
            )
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(dim=-1, index=sampled_sorted).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _create_scheduler(self, total_optimizer_steps: int) -> LambdaLR:
        """Create the LR scheduler using the real optimizer-step budget."""
        warmup_steps = self.config.training.warmup_steps
        total_steps = max(1, total_optimizer_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = min(
                1.0,
                (step - warmup_steps) / max(1, total_steps - warmup_steps),
            )
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _estimate_total_optimizer_steps(
        self, steps_per_epoch: int, num_epochs: int
    ) -> int:
        """Estimate optimizer steps when partial accumulation flushes at epoch end."""
        accum_steps = max(1, self.config.training.gradient_accumulation_steps)
        max_batches = self.config.training.max_steps
        remaining_batches = (
            max_batches if max_batches is not None else steps_per_epoch * num_epochs
        )
        total_optimizer_steps = 0

        for _ in range(num_epochs):
            if remaining_batches <= 0:
                break
            epoch_batches = min(steps_per_epoch, remaining_batches)
            total_optimizer_steps += math.ceil(epoch_batches / accum_steps)
            remaining_batches -= epoch_batches

        return max(1, total_optimizer_steps)

    def _configure_scheduler(self, steps_per_epoch: int, num_epochs: int) -> None:
        """Rebuild the scheduler once the true run length is known."""
        if self.optimizer is None:
            return

        total_optimizer_steps = self._estimate_total_optimizer_steps(
            steps_per_epoch, num_epochs
        )
        previous_state = self.scheduler.state_dict() if self.scheduler is not None else None
        self.scheduler = self._create_scheduler(total_optimizer_steps=total_optimizer_steps)
        if previous_state is not None:
            self.scheduler.load_state_dict(previous_state)

    def _optimizer_step_due(self) -> bool:
        return (
            self._accumulation_batches
            >= self.config.training.gradient_accumulation_steps
        )

    def _flush_accumulated_gradients(self) -> bool:
        """Apply any pending gradients and advance the optimizer once."""
        if self.optimizer is None or self._accumulation_batches <= 0:
            return False

        grad_scale = (
            self.config.training.gradient_accumulation_steps
            / max(1, self._accumulation_batches)
        )
        if grad_scale != 1.0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(grad_scale)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step += 1
        self._accumulation_batches = 0
        return True

    @contextmanager
    def _temporary_eval_mode(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    def _get_triton_generation_mode(self) -> str:
        mode = getattr(self.config.training, "triton_generation_mode", None)
        if mode in {"auto", "on", "off"}:
            return mode
        return "on" if getattr(self.config.training, "use_triton_generation", True) else "off"

    def _is_qwen_family_generation_model(self) -> bool:
        model_config = getattr(self.model, "config", None)
        model_type = str(getattr(model_config, "model_type", "") or "").lower()
        model_id = str(getattr(self.config.model, "model_id", "") or "").lower()
        return (
            "qwen" in model_type
            or "qwen" in model_id
            or "deepseek-r1-distill-qwen" in model_id
        )

    def _resolve_triton_generation_decision(self) -> tuple[bool, str]:
        if not getattr(self.config.training, "use_triton_kernels", False):
            return False, "global Triton kernels disabled"
        mode = self._get_triton_generation_mode()
        if mode == "off" or not getattr(self.config.training, "use_triton_generation", True):
            return False, "generation mode is off"
        if not TRITON_AVAILABLE:
            return False, "Triton runtime unavailable"

        model_config = getattr(self.model, "config", None)
        num_heads = getattr(model_config, "num_attention_heads", None)
        num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_heads % num_kv_heads != 0
        ):
            return False, "attention head layout is not divisible for GQA paged decode"

        if mode == "auto":
            if self.config.training.generation_do_sample and self._is_qwen_family_generation_model():
                group_size = int(getattr(self.config.grpo, "group_size", 1) or 1)
                if group_size <= 1:
                    return (
                        False,
                        "auto policy routes sampled DeepSeek/Qwen generation with group_size=1 to the torch prefix-cache path because it remains closer to the reference and faster in elapsed time on the RTX 3060 Ti target",
                    )
                return (
                    True,
                    "auto policy routes sampled DeepSeek/Qwen generation with group_size>=2 to Triton paged-KV because it is exact on greedy parity, passes sampled validation for those group sizes, and benchmarks faster on the RTX 3060 Ti target",
                )

        return True, "Triton paged-KV generation enabled"

    def _resolve_triton_generation_runtime_decision(
        self, *, group_size: int, micro_batch_size: int
    ) -> tuple[bool, str]:
        enabled, reason = self._resolve_triton_generation_decision()
        if not enabled:
            return enabled, reason

        mode = self._get_triton_generation_mode()
        if (
            mode == "auto"
            and self.config.training.generation_do_sample
            and self._is_qwen_family_generation_model()
        ):
            chunk_sizes = [
                min(g_start + micro_batch_size, group_size) - g_start
                for g_start in range(0, group_size, micro_batch_size)
            ]
            if any(chunk_size == 1 for chunk_size in chunk_sizes):
                return (
                    False,
                    "auto policy routes sampled DeepSeek/Qwen generation to the torch prefix-cache path because the effective generation microbatch schedule includes a batch of 1, which remains the unresolved sampled case on the RTX 3060 Ti target",
                )

        return enabled, reason

    def _should_use_triton_generation(self) -> bool:
        enabled, reason = self._resolve_triton_generation_decision()
        if not self._triton_generation_policy_logged:
            logger.info(
                "[TritonGen] mode=%s enabled=%s reason=%s",
                self._get_triton_generation_mode(),
                enabled,
                reason,
            )
            self._triton_generation_policy_logged = True
        return enabled

    def _create_train_dataloader(self, sent_stage: int, epoch: int):
        requested_use_sent = self.config.sent.enabled
        do_shuffle = not requested_use_sent
        generator = torch.Generator()
        generator.manual_seed(self._dataloader_seed + epoch)

        dataloader = create_grpo_dataloader(
            tokenizer=self.tokenizer,
            split="train",
            batch_size=self.config.training.batch_size,
            max_prompt_length=self.config.training.max_prompt_length,
            shuffle=do_shuffle,
            use_sent=requested_use_sent,
            sent_config=self.config.sent,
            num_stages=self.config.sent.curriculum_stages,
            cache_path=self.config.sent.cache_path,
            model_id=self.config.model.model_id,
            generator=generator,
        )

        if self._dataloader_uses_sent(dataloader):
            self._apply_sent_stage(dataloader, sent_stage)

        return dataloader

    @staticmethod
    def _dataloader_uses_sent(dataloader: Any) -> bool:
        return bool(getattr(dataloader, "uses_sent_curriculum", False))

    def _resolve_sent_stage_for_epoch(
        self, epoch: int, sent_stage: Optional[int], use_sent: bool
    ) -> int:
        if not use_sent:
            return 1
        if sent_stage is not None:
            return sent_stage
        return min(epoch + 1, self.config.sent.curriculum_stages)

    def _apply_sent_stage(self, dataloader: Any, sent_stage: int) -> None:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None or not hasattr(dataset, "set_stage"):
            return
        current_stage = getattr(dataset, "current_stage", None)
        if current_stage == sent_stage:
            return
        dataset.set_stage(sent_stage)
        dataloader.current_sent_stage = sent_stage
        if hasattr(dataset, "get_stage_info"):
            stage_info = dataset.get_stage_info()
            logger.info(
                "[SENT] Training on stage %d/%d (samples %d-%d)",
                sent_stage,
                stage_info["num_stages"],
                stage_info["stage_start_idx"],
                stage_info["stage_end_idx"],
            )
        else:
            logger.info("[SENT] Training on stage %d", sent_stage)

    def _initialize_optimizer_state(self) -> None:
        """Pre-allocate AdamW state so the first real optimizer step is not a surprise OOM."""
        if not isinstance(self.optimizer, AdamW):
            return

        for group in self.optimizer.param_groups:
            amsgrad = bool(group.get("amsgrad", False))
            for param in group["params"]:
                if param is None or not param.requires_grad:
                    continue
                state = self.optimizer.state[param]
                if state:
                    continue
                state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

    def _compute_truncation_mask(
        self, response_ids: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mark completions that should contribute gradient signal."""
        truncation_mask = torch.ones(
            response_ids.shape[0], dtype=torch.float32, device=self.device
        )
        if not self.config.grpo.mask_truncated_completions:
            return truncation_mask

        eos_id = self.tokenizer.eos_token_id
        for idx in range(response_ids.shape[0]):
            resp_tokens = response_ids[idx][response_mask[idx].bool()]
            if resp_tokens.numel() >= self.config.training.max_response_length:
                if eos_id not in resp_tokens:
                    truncation_mask[idx] = 0.0
        return truncation_mask

    def _calculate_masked_group_advantages(
        self, rewards: torch.Tensor, sample_mask: torch.Tensor
    ) -> torch.Tensor:
        """Center rewards within each prompt group while excluding masked samples."""
        group_size = self.config.grpo.group_size
        grouped_rewards = rewards.view(-1, group_size)
        grouped_mask = sample_mask.view(-1, group_size)
        valid_counts = grouped_mask.sum(dim=1, keepdim=True)
        safe_counts = torch.clamp(valid_counts, min=1.0)
        mean_rewards = (grouped_rewards * grouped_mask).sum(dim=1, keepdim=True) / safe_counts
        advantages = (grouped_rewards - mean_rewards) * grouped_mask
        advantages = torch.where(valid_counts > 0, advantages, torch.zeros_like(advantages))
        return advantages.view(-1)

    def _responses_from_output_tokens(
        self, output_ids: torch.Tensor
    ) -> tuple[list[str], torch.Tensor]:
        """Decode response-only token tensors and build an exact rollout mask."""
        if output_ids.ndim != 2:
            raise ValueError(
                f"Expected response token tensor of shape [batch, seq], got {output_ids.shape}."
            )

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        response_mask = torch.zeros_like(output_ids, dtype=torch.long)
        decoded_rows: list[torch.Tensor] = []

        for row_idx in range(output_ids.shape[0]):
            row = output_ids[row_idx]
            valid_len = row.shape[0]
            found_eos = False

            if eos_token_id is not None:
                eos_positions = (row == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    valid_len = int(eos_positions[0].item()) + 1
                    found_eos = True
            if pad_token_id is not None and not found_eos:
                non_pad_positions = (row != pad_token_id).nonzero(as_tuple=False)
                valid_len = (
                    int(non_pad_positions[-1].item()) + 1
                    if non_pad_positions.numel() > 0
                    else 0
                )

            if valid_len > 0:
                response_mask[row_idx, :valid_len] = 1
                decoded_rows.append(row[:valid_len].detach().cpu())
            else:
                decoded_rows.append(row[:0].detach().cpu())

        generated_texts = self.tokenizer.batch_decode(
            decoded_rows,
            skip_special_tokens=True,
        )
        return generated_texts, response_mask

    def _prefill_triton_prompt_cache(
        self, real_ids: torch.Tensor, real_mask: torch.Tensor
    ):
        """Build a reusable Triton paged-KV prompt cache for grouped responses."""
        return prefill_paged_kv_cache(
            self.model,
            real_ids,
            real_mask,
            max_new_tokens=self.config.training.max_response_length,
            block_size=16,
        )

    def _generate_with_triton_paged_prefix_cache(
        self,
        prefix_state,
        current_micro: int,
    ) -> torch.Tensor:
        """Decode grouped responses from a shared Triton paged-KV prompt cache."""
        decode_state = expand_paged_kv_cache_state(prefix_state, current_micro)
        return decode_from_paged_kv_cache(
            self.model,
            decode_state,
            max_new_tokens=self.config.training.max_response_length,
            do_sample=self.config.training.generation_do_sample,
            temperature=self.config.training.generation_temperature,
            top_p=self.config.training.generation_top_p,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else (self.tokenizer.eos_token_id or 0)
            ),
            eos_token_id=self.tokenizer.eos_token_id,
            seed=None,
        )

    def _generate_with_expanded_prefix_cache(
        self,
        real_ids: torch.Tensor,
        real_mask: torch.Tensor,
        prefix_cache,
        current_micro: int,
    ) -> torch.Tensor:
        """Decode responses directly from a prefetched prompt cache."""
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = pad_token_id
        max_new_tokens = self.config.training.max_response_length
        prompt_len = real_ids.shape[1]
        mb_cache = self._expand_prefix_cache(prefix_cache, current_micro)

        generated_ids = torch.full(
            (current_micro, max_new_tokens),
            pad_token_id,
            dtype=real_ids.dtype,
            device=self.device,
        )
        current_input_ids = real_ids[:, -1:].expand(current_micro, -1)
        current_attention_mask = torch.ones(
            current_micro,
            prompt_len,
            dtype=real_mask.dtype,
            device=self.device,
        )
        unfinished = torch.ones(current_micro, dtype=torch.bool, device=self.device)
        generated_steps = 0
        pad_tokens = torch.full(
            (current_micro,), pad_token_id, dtype=real_ids.dtype, device=self.device
        )

        for step_idx in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=mb_cache,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            sampled_tokens = self._sample_next_tokens(next_logits)
            next_tokens = torch.where(unfinished, sampled_tokens, pad_tokens)

            generated_ids[:, step_idx] = next_tokens
            generated_steps = step_idx + 1
            mb_cache = outputs.past_key_values
            unfinished = unfinished & next_tokens.ne(eos_token_id)

            del outputs, next_logits, sampled_tokens

            if not unfinished.any():
                break

            current_input_ids = next_tokens.unsqueeze(1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        current_micro,
                        1,
                        dtype=real_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=1,
            )

        del mb_cache, current_input_ids, current_attention_mask, unfinished, pad_tokens
        return generated_ids[:, :generated_steps]

    def _generate_responses_with_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_size_override: int | None = None,
    ) -> tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate responses for GRPO group sampling with KV-cache prefix sharing.

        For each prompt:
          1. Run a single prefill forward to build the prompt KV cache
             (cropped to prompt_len-1 positions).
          2. For each micro-batch of group_size, expand the cached prefix
             tensors via ``expand().contiguous()`` into a fresh DynamicCache,
             then call ``generate()`` with the full ``input_ids``.
             ``generate()`` sees that the cache already covers positions
             [0..prompt_len-2] and only needs to process the last prompt
             token before starting autoregressive decoding — saving a full
             prompt re-computation per micro-batch.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Tuple of decoded strings, response token ids, and response mask.
        """
        was_training = self.model.training
        self.model.eval()

        generated_texts = []
        response_batches: list[torch.Tensor] = []
        group_size = (
            group_size_override
            if group_size_override is not None
            else self.group_sampler.group_size
        )
        batch_size = input_ids.shape[0]
        micro_batch_size = self._gen_micro_batch
        use_triton_generation, triton_generation_reason = (
            self._resolve_triton_generation_runtime_decision(
                group_size=group_size,
                micro_batch_size=micro_batch_size,
            )
        )
        if not self._triton_generation_policy_logged:
            logger.info(
                "[TritonGen] mode=%s enabled=%s reason=%s",
                self._get_triton_generation_mode(),
                use_triton_generation,
                triton_generation_reason,
            )
            self._triton_generation_policy_logged = True

        with torch.inference_mode():
            for prompt_idx in range(batch_size):
                single_ids = input_ids[prompt_idx : prompt_idx + 1]
                single_mask = attention_mask[prompt_idx : prompt_idx + 1]

                # Strip left-padding: only process real tokens
                first_real = single_mask[0].argmax().item()
                real_ids = single_ids[:, first_real:]
                real_mask = single_mask[:, first_real:]
                prompt_len = real_ids.shape[1]
                use_triton_kernels = use_triton_generation

                if use_triton_kernels:
                    try:
                        prefix_cache = self._prefill_triton_prompt_cache(
                            real_ids, real_mask
                        )
                    except ImportError:
                        prefix_cache = None
                        use_triton_kernels = False
                    except (AttributeError, RuntimeError, ValueError) as exc:
                        if "out of memory" in str(exc).lower():
                            raise
                        logger.warning(
                            "Falling back from Triton paged-KV prefill to torch prefix cache: %s",
                            exc,
                        )
                        prefix_cache = None
                        use_triton_kernels = False
                        prefix_cache = self._prefill_prompt_cache(real_ids, real_mask)
                else:
                    # --- Prefill: compute prompt KV cache once ---
                    prefix_cache = self._prefill_prompt_cache(real_ids, real_mask)

                # Generate group_size responses via micro-batching
                for g_start in range(0, group_size, micro_batch_size):
                    g_end = min(g_start + micro_batch_size, group_size)
                    current_micro = g_end - g_start
                    outputs_are_response_only = True

                    if use_triton_kernels:
                        if TRITON_AVAILABLE:
                            try:
                                outputs = self._generate_with_triton_paged_prefix_cache(
                                    prefix_state=prefix_cache,
                                    current_micro=current_micro,
                                )
                            except ImportError:
                                outputs = self.model.generate(
                                    input_ids=real_ids.expand(current_micro, -1),
                                    attention_mask=real_mask.expand(current_micro, -1),
                                    max_new_tokens=self.config.training.max_response_length,
                                    do_sample=self.config.training.generation_do_sample,
                                    temperature=self.config.training.generation_temperature,
                                    top_p=self.config.training.generation_top_p,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    use_cache=True,
                                )
                                outputs_are_response_only = False
                            except (RuntimeError, ValueError) as exc:
                                if "out of memory" in str(exc).lower():
                                    raise
                                logger.warning(
                                    "Falling back from Triton paged_kv_decode to model.generate(): %s",
                                    exc,
                                )
                                outputs = self.model.generate(
                                    input_ids=real_ids.expand(current_micro, -1),
                                    attention_mask=real_mask.expand(current_micro, -1),
                                    max_new_tokens=self.config.training.max_response_length,
                                    do_sample=self.config.training.generation_do_sample,
                                    temperature=self.config.training.generation_temperature,
                                    top_p=self.config.training.generation_top_p,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    use_cache=True,
                                )
                                outputs_are_response_only = False
                        else:
                            outputs = self.model.generate(
                                input_ids=real_ids.expand(current_micro, -1),
                                attention_mask=real_mask.expand(current_micro, -1),
                                max_new_tokens=self.config.training.max_response_length,
                                do_sample=self.config.training.generation_do_sample,
                                temperature=self.config.training.generation_temperature,
                                top_p=self.config.training.generation_top_p,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                use_cache=True,
                            )
                            outputs_are_response_only = False
                    else:
                        outputs = self._generate_with_expanded_prefix_cache(
                            real_ids=real_ids,
                            real_mask=real_mask,
                            prefix_cache=prefix_cache,
                            current_micro=current_micro,
                        )

                    if not outputs_are_response_only:
                        outputs = outputs[:, prompt_len:]

                    outputs_cpu = outputs.detach().cpu()
                    decoded_batch, _ = self._responses_from_output_tokens(outputs_cpu)
                    generated_texts.extend(decoded_batch)
                    response_batches.append(outputs_cpu)

                    if use_triton_kernels:
                        del outputs
                    else:
                        del outputs

                    if (g_start // micro_batch_size) % 6 == 0:
                        self.memory_manager.clear_cache()

                if use_triton_kernels:
                    del single_ids, single_mask, real_ids, real_mask, prefix_cache
                else:
                    del single_ids, single_mask, real_ids, real_mask, prefix_cache

        try:
            if response_batches:
                max_response_width = max(batch.shape[1] for batch in response_batches)
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id
                if pad_token_id is None:
                    pad_token_id = 0
                padded_batches = []
                for batch in response_batches:
                    if batch.shape[1] == max_response_width:
                        padded_batches.append(batch)
                        continue
                    padded = torch.full(
                        (batch.shape[0], max_response_width),
                        pad_token_id,
                        dtype=batch.dtype,
                    )
                    padded[:, : batch.shape[1]] = batch
                    padded_batches.append(padded)
                response_ids = torch.cat(padded_batches, dim=0)
                _, response_mask = self._responses_from_output_tokens(response_ids)
            else:
                response_ids = torch.empty((0, 0), dtype=input_ids.dtype)
                response_mask = torch.empty((0, 0), dtype=torch.long)
            return generated_texts, response_ids.to(self.device), response_mask.to(
                self.device
            )
        finally:
            self.model.train(was_training)

    def generate_responses(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[str]:
        generated_texts, _, _ = self._generate_responses_with_tokens(
            input_ids, attention_mask
        )
        return generated_texts

    def generate_benchmark_responses(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[str]:
        generated_texts, _, _ = self._generate_responses_with_tokens(
            input_ids, attention_mask, group_size_override=1
        )
        return generated_texts

    @staticmethod
    def _expand_prefix_cache(cache, repeats: int):
        """Create a new DynamicCache with tensors expanded to ``repeats`` copies.

        Instead of ``deepcopy`` + ``batch_repeat_interleave``, this builds a
        fresh cache using ``expand().contiguous()`` per layer.  This is more
        efficient because:
          • It skips Python-level deep-copy of the cache object hierarchy.
          • Each layer's keys/values go from [1, H, S, D] → [repeats, H, S, D]
            in one GPU operation.
          • The original cache is not mutated.
        """
        from transformers import DynamicCache

        expanded = DynamicCache()
        for layer in cache.layers:
            k = layer.keys.expand(repeats, -1, -1, -1).contiguous()
            v = layer.values.expand(repeats, -1, -1, -1).contiguous()
            expanded.update(k, v, len(expanded.layers))
        return expanded

    def _compute_old_log_probs_with_prompt_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        response_only_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        group_size = self.config.grpo.group_size
        gen_micro_batch = self._gen_micro_batch
        prompt_width = input_ids.shape[1]
        response_width = response_ids.shape[1]
        num_samples = response_ids.shape[0]

        all_old_log_probs = torch.zeros(
            num_samples,
            prompt_width + response_width - 1,
            dtype=torch.float32,
            device=self.device,
        )

        for prompt_idx in range(batch_size):
            single_ids = input_ids[prompt_idx : prompt_idx + 1]
            single_mask = attention_mask[prompt_idx : prompt_idx + 1]

            first_real = single_mask[0].argmax().item()
            real_ids = single_ids[:, first_real:]
            real_mask = single_mask[:, first_real:]
            real_prompt_len = real_ids.shape[1]
            prefix_cache = self._prefill_prompt_cache(real_ids, real_mask)

            group_start = prompt_idx * group_size
            group_end = group_start + group_size
            group_response_ids = response_ids[group_start:group_end]
            group_response_mask = response_mask[group_start:group_end]

            for g_start in range(0, group_size, gen_micro_batch):
                g_end = min(g_start + gen_micro_batch, group_size)
                current_micro = g_end - g_start

                mb_response_ids = group_response_ids[g_start:g_end]
                mb_response_mask = group_response_mask[g_start:g_end]
                mb_cache = self._expand_prefix_cache(prefix_cache, current_micro)

                anchor_ids = real_ids[:, -1:].expand(current_micro, -1)
                scorer_input_ids = torch.cat([anchor_ids, mb_response_ids], dim=1)
                scorer_attention_mask = torch.cat(
                    [
                        torch.ones(
                            current_micro,
                            real_prompt_len,
                            dtype=real_mask.dtype,
                            device=self.device,
                        ),
                        mb_response_mask,
                    ],
                    dim=1,
                )

                outputs = self.model(
                    input_ids=scorer_input_ids,
                    attention_mask=scorer_attention_mask,
                    past_key_values=mb_cache,
                    use_cache=True,
                )

                compact_logits = outputs.logits[:, :-1, :]
                compact_targets = mb_response_ids
                compact_log_probs = -F.cross_entropy(
                    compact_logits.reshape(-1, compact_logits.size(-1)),
                    compact_targets.reshape(-1),
                    reduction="none",
                ).view(compact_targets.shape)
                compact_log_probs = compact_log_probs * mb_response_mask

                all_old_log_probs[
                    group_start + g_start : group_start + g_end,
                    prompt_width - 1 : prompt_width - 1 + response_width,
                ] = compact_log_probs

                del outputs, compact_logits, mb_cache

            del prefix_cache
            self.memory_manager.clear_cache()

        return all_old_log_probs * response_only_mask[:, 1:]

    def training_step(self, batch: Dict) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Batch of data with prompts and answers

        Returns:
            Dictionary of metrics
        """
        self._step_start_time = time.time()
        self.model.train()
        if hasattr(self.memory_manager, "reset_peak_stats"):
            self.memory_manager.reset_peak_stats()

        if self._profiler_hooks:
            self._profiler_hooks.on_step_start(self.global_step, self.current_epoch)

        if self._accumulation_batches == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Get batch data
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        ground_truths = batch["answers"]

        # Phase 1: Generation (no gradients)
        self.memory_manager.optimize_for_inference()

        if self._profiler_hooks:
            self._profiler_hooks.on_phase_start("generation", step=self.global_step)

        generated_texts, response_ids, response_mask = self._generate_responses_with_tokens(
            input_ids, attention_mask
        )

        if self._profiler_hooks:
            self._profiler_hooks.on_phase_end("generation", step=self.global_step)

        # Calculate lengths for penalty
        response_lengths = response_mask.sum(dim=1).float()

        # Compute rewards
        rewards_list = []
        debug_infos = []
        group_size = self.config.grpo.group_size

        # Expand ground truths
        expanded_ground_truths = []
        for gt in ground_truths:
            expanded_ground_truths.extend([gt] * group_size)

        for i, (gen_text, gt) in enumerate(
            zip(generated_texts, expanded_ground_truths)
        ):
            reward, info = self.verifier.verify(gen_text, gt)

            # Apply length penalty
            if self.config.grpo.length_penalty_coef > 0:
                penalty = response_lengths[i] * self.config.grpo.length_penalty_coef
                reward -= penalty.item()

            rewards_list.append(reward)
            debug_infos.append(info)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # TRACE-level logging for tensor metadata
        log_tensor_meta(logger, "Rewards", rewards, level=TRACE)

        # --- DEBUG PRINT ---
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n[DEBUG] Step %s", self.global_step)
            # Only print first item in batch
            b = 0
            if b < len(batch["questions"]):
                q_text = batch["questions"][b]
                gt_text = ground_truths[b]
                logger.debug("Question: %s", q_text)
                logger.debug("Ground Truth: %s", gt_text)
                logger.debug("-" * 20)

                start_idx = b * group_size
                end_idx = start_idx + group_size
                batch_responses = generated_texts[start_idx:end_idx]
                batch_rewards = rewards_list[start_idx:end_idx]
                batch_infos = debug_infos[start_idx:end_idx]

                # Calculate statistics
                correct_count = sum(
                    1 for info in batch_infos if info.get("match", False)
                )

                total_count = len(batch_rewards)
                failed_count = total_count - correct_count

                logger.debug(
                    "Responses: %d total | %d correct | %d failed",
                    total_count,
                    correct_count,
                    failed_count,
                )
                logger.debug("")

                # Only show failed responses
                failed_shown = 0
                for i, (resp, rew, info) in enumerate(
                    zip(batch_responses, batch_rewards, batch_infos)
                ):
                    if info.get("match", False):
                        continue  # Skip correct responses

                    failed_shown += 1
                    logger.debug("[FAILED] Response %d (Reward: %.2f):", i + 1, rew)
                    clean_resp = resp.replace("\n", "\\n")
                    logger.debug("  -> Text: %s", clean_resp)
                    logger.debug("  -> Extracted: %s", info.get("extracted_answer"))
                    logger.debug("  -> GT: %s", info.get("ground_truth_answer"))
                    logger.debug("  -> Match: %s", info.get("match"))
                    logger.debug("-" * 10)

                if failed_shown == 0:
                    logger.debug("All responses correct!")
            logger.debug("=" * 40)
        # -------------------

        truncation_mask = self._compute_truncation_mask(response_ids, response_mask)
        advantages = self._calculate_masked_group_advantages(rewards, truncation_mask)

        # TRACE-level logging for tensor metadata
        log_tensor_meta(logger, "Advantages", advantages, level=TRACE)
        self.memory_manager.optimize_for_training()

        total_loss = 0.0
        all_metrics = []

        # Get expanded prompts (each prompt repeated group_size times)
        group_size = self.config.grpo.group_size
        prompt_ids_expanded = input_ids.repeat_interleave(
            group_size, dim=0
        )  # [batch*G, prompt_len]
        prompt_mask_expanded = attention_mask.repeat_interleave(group_size, dim=0)

        num_samples = response_ids.shape[0]

        del generated_texts

        # Concatenate prompt + response for each sample
        # This matches grpo_zero: batch_token_ids = prefix_token_ids + generated_token_ids
        all_input_ids = torch.cat(
            [prompt_ids_expanded, response_ids], dim=1
        )  # [batch*G, prompt_len + resp_len]
        all_attention_mask = torch.cat([prompt_mask_expanded, response_mask], dim=1)

        # Create response-only mask: 0 for prompt tokens, 1 for response tokens
        # This ensures only response tokens contribute to loss (like grpo_zero's batch_masks)
        prompt_len = prompt_ids_expanded.shape[1]
        num_samples = all_input_ids.shape[0]

        # response_only_mask: [batch*G, prompt_len + resp_len]
        # First prompt_len positions are 0, remaining resp_len positions use response_mask
        response_only_mask = torch.cat(
            [
                torch.zeros(
                    num_samples, prompt_len, dtype=torch.long, device=self.device
                ),
                response_mask,
            ],
            dim=1,
        )
        if self.config.grpo.mask_truncated_completions:
            response_only_mask = response_only_mask * truncation_mask.unsqueeze(1)

        del prompt_ids_expanded, prompt_mask_expanded, response_ids, response_mask

        # Pre-compute old log probabilities (Phase 1.5)
        # This is CRITICAL: We need the log probs of the generated text *before* updates start.
        # Otherwise, if we compute them inside the loop, they change as the model updates,
        # leading to ratio ~ 1.0 and zero gradients (or pure policy gradient without clipping).
        all_old_log_probs = []

        self.memory_manager.optimize_for_inference()

        if self._profiler_hooks:
            self._profiler_hooks.on_phase_start("old_log_probs", step=self.global_step)

        with self._temporary_eval_mode(), torch.no_grad():
            all_old_log_probs = self._compute_old_log_probs_with_prompt_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_ids=all_input_ids[:, prompt_len:],
                response_mask=response_only_mask[:, prompt_len:],
                response_only_mask=response_only_mask,
            )

        if self._profiler_hooks:
            self._profiler_hooks.on_phase_end("old_log_probs", step=self.global_step)

        # Re-enable gradients for training phase
        self.memory_manager.optimize_for_training()
        self.model.train()

        training_micro_batch = self._train_micro_batch

        if self._profiler_hooks:
            self._profiler_hooks.on_phase_start("training", step=self.global_step)

        for start_idx in range(0, num_samples, training_micro_batch):
            end_idx = min(start_idx + training_micro_batch, num_samples)

            batch_input_ids = all_input_ids[start_idx:end_idx]
            batch_attention_mask = all_attention_mask[start_idx:end_idx]
            batch_response_mask = response_only_mask[
                start_idx:end_idx
            ]  # Response-only mask
            batch_advantages = advantages[start_idx:end_idx]
            batch_old_log_probs = all_old_log_probs[start_idx:end_idx]

            reference_logits = None
            if self.config.grpo.use_kl:
                with self._temporary_eval_mode(), torch.no_grad(), lora_disabled(
                    self.model
                ):
                    reference_outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        use_cache=False,
                    )
                reference_logits = reference_outputs.logits[:, :-1, :].detach()
                del reference_outputs

            # Forward pass (model sees full prompt+response context)
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,  # Full attention for context
                use_cache=False,  # CRITICAL: Disable cache for gradient checkpointing
            )

            logits = outputs.logits

            # TRACE-level logging for tensor metadata
            log_tensor_meta(logger, "Logits", logits, level=TRACE)
            entropy_mask = None
            if self.config.entropy.use_entropy_mask:
                # Compute entropy on logits[:, :-1, :] — exactly the same slice passed to
                # compute_grpo_loss — so the mask is always in perfect token alignment.
                # batch_response_mask[:, 1:] selects response-only positions in that slice.
                loss_logits = logits[:, :-1, :].contiguous()
                loss_resp_mask = batch_response_mask[:, 1:].contiguous()
                entropy, entropy_mask = (
                    self.entropy_calculator.calculate_entropy_and_mask(
                        loss_logits,
                        attention_mask=loss_resp_mask,
                        use_triton_kernels=(
                            self.config.training.use_triton_kernels
                            and self.config.training.use_triton_entropy_mask
                        ),
                    )
                )

            # Compute loss for batch
            # CRITICAL: Use response_only_mask (not attention_mask) so only response tokens
            # contribute to loss.

            seq_len = logits.shape[1] - 1
            if batch_old_log_probs.shape[1] != seq_len:
                raise ValueError(
                    f"Shape mismatch: old_log_probs {batch_old_log_probs.shape} "
                    f"vs logits {logits.shape} (expected seq_len={seq_len})"
                )

            loss, metrics = self.grpo_trainer.compute_grpo_loss(
                policy_logits=logits[:, :-1, :],
                advantages=batch_advantages,
                old_log_probs=batch_old_log_probs,
                target_ids=batch_input_ids[:, 1:],
                attention_mask=batch_response_mask[:, 1:],
                entropy_mask=entropy_mask if entropy_mask is not None else None,
                reference_logits=reference_logits,
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item() * (end_idx - start_idx)
            all_metrics.append(metrics)

            # Clear memory between micro-batches
            del outputs, logits

        del all_input_ids, all_attention_mask, response_only_mask, all_old_log_probs

        self._accumulation_batches += 1
        if self._optimizer_step_due():
            self._flush_accumulated_gradients()
        if hasattr(self.memory_manager, "maybe_update_checkpointing"):
            peak_stats = None
            if hasattr(self.memory_manager, "get_peak_memory_stats"):
                peak_stats = self.memory_manager.get_peak_memory_stats()
            self.memory_manager.maybe_update_checkpointing(
                self.model,
                step=self.global_step,
                memory_stats=peak_stats,
            )

        # Aggregate metrics
        avg_metrics = {
            "loss": total_loss / num_samples,
            "avg_reward": rewards.mean().item(),
            "avg_response_length": response_lengths.mean().item(),
        }

        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        # Add entropy stats
        if self.config.entropy.use_entropy_mask:
            avg_metrics["entropy_masked_ratio"] = avg_metrics.get(
                "selected_tokens_ratio", 0.0
            )

        # Add reward distribution stats
        avg_metrics["reward_std"] = rewards.std().item()
        avg_metrics["reward_max"] = rewards.max().item()
        avg_metrics["reward_min"] = rewards.min().item()

        avg_metrics["positive_advantages_ratio"] = (
            (advantages > 0).float().mean().item()
        )

        if self.config.grpo.mask_truncated_completions:
            avg_metrics["truncated_completions_ratio"] = (
                1.0 - truncation_mask.mean().item()
            )

        step_time_s = None
        if self._step_start_time:
            step_time_s = time.time() - self._step_start_time
            avg_metrics["step_time_s"] = step_time_s
        total_tokens = response_lengths.sum().item()
        avg_metrics["tokens_per_sec"] = (
            total_tokens / step_time_s if step_time_s and step_time_s > 0 else 0.0
        )

        vram_stats = self.memory_manager.get_memory_stats()
        if "error" not in vram_stats:
            avg_metrics["vram_allocated"] = vram_stats.get("allocated_gb", 0.0)
            avg_metrics["vram_reserved"] = vram_stats.get("reserved_gb", 0.0)
            avg_metrics["vram_max_allocated"] = vram_stats.get("max_allocated_gb", 0.0)
            avg_metrics["vram_free"] = vram_stats.get("free_gb", 0.0)
            avg_metrics["vram_usage_fraction"] = vram_stats.get("usage_fraction", 0.0)

        avg_metrics["gen_micro_batch"] = float(self._gen_micro_batch)
        avg_metrics["train_micro_batch"] = float(self._train_micro_batch)
        avg_metrics["oom_backoff_count"] = float(self._oom_backoff_count)
        avg_metrics["optimizer_step"] = float(self.optimizer_step)

        self.global_step += 1
        self.memory_manager.step()

        self._log_wandb_metrics(avg_metrics)

        if self._profiler_hooks:
            self._profiler_hooks.on_phase_end("training", step=self.global_step - 1)
            self._profiler_hooks.on_step_end(
                self.global_step - 1, avg_metrics, self.current_epoch
            )

        return avg_metrics

    def train_epoch(self, dataloader, epoch: int):
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        """
        return self.train_epoch_with_skip(dataloader, epoch, skip_steps=0)

    def _max_steps_reached(self) -> bool:
        max_steps = self.config.training.max_steps
        return max_steps is not None and self.global_step >= max_steps

    def _capture_optimizer_grad_snapshot(self) -> Dict[int, torch.Tensor]:
        """Clone current gradients so an OOM retry can restore accumulation state."""
        if self.optimizer is None:
            return {}

        snapshot: Dict[int, torch.Tensor] = {}
        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                if param.grad is not None:
                    snapshot[id(param)] = param.grad.detach().clone()
        return snapshot

    def _restore_optimizer_grad_snapshot(
        self, snapshot: Dict[int, torch.Tensor]
    ) -> None:
        """Restore gradients captured before a failed training-step attempt."""
        if self.optimizer is None:
            return

        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                grad = snapshot.get(id(param))
                if grad is None:
                    param.grad = None
                else:
                    param.grad = grad.to(device=param.device, dtype=param.dtype)

    def train_epoch_with_skip(self, dataloader, epoch: int, skip_steps: int = 0):
        self.model.train()
        epoch_metrics = []
        reached_max_steps = False

        dataloader_iter = iter(dataloader)
        total_steps = len(dataloader)
        pbar = tqdm(range(total_steps), desc=f"Epoch {epoch + 1}")

        if skip_steps >= len(dataloader):
            logger.info(
                "[Resume] Skip steps (%d) >= steps per epoch (%d); skipping epoch %d.",
                skip_steps,
                len(dataloader),
                epoch + 1,
            )
            return

        for batch_idx in pbar:
            if self._profiler_hooks:
                self._profiler_hooks.on_data_start(step=self.global_step)
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                if self._profiler_hooks:
                    self._profiler_hooks.on_data_end(step=self.global_step)
                break
            if self._profiler_hooks:
                self._profiler_hooks.on_data_end(step=self.global_step)
            if skip_steps > 0 and batch_idx < skip_steps:
                if batch_idx == 0:
                    logger.info(
                        "[Resume] Skipping first %d steps of epoch %d...",
                        skip_steps,
                        epoch + 1,
                    )
                continue
            if self._max_steps_reached():
                logger.info("Reached max steps (%s); stopping epoch.", self.global_step)
                reached_max_steps = True
                break
            while True:
                grad_snapshot = self._capture_optimizer_grad_snapshot()
                try:
                    metrics = self.training_step(batch)
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    self._restore_optimizer_grad_snapshot(grad_snapshot)
                    if self._profiler_hooks:
                        self._profiler_hooks.on_oom(
                            self.global_step,
                            {
                                "gen_micro_batch": self._gen_micro_batch,
                                "train_micro_batch": self._train_micro_batch,
                                "epoch": epoch,
                            },
                        )
                    next_gen_micro = max(1, self._gen_micro_batch // 2)
                    next_train_micro = max(1, self._train_micro_batch // 2)
                    if (
                        next_gen_micro == self._gen_micro_batch
                        and next_train_micro == self._train_micro_batch
                    ):
                        logger.info(
                            "\n[OOM] Unable to recover at minimum micro-batches (gen=%s, train=%s).",
                            self._gen_micro_batch,
                            self._train_micro_batch,
                        )
                        self.memory_manager.clear_cache(aggressive=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                        raise
                    self._oom_backoff_count += 1
                    self._post_oom_success_batches = 0
                    self._gen_micro_batch = next_gen_micro
                    self._train_micro_batch = next_train_micro
                    if hasattr(self.memory_manager, "enable_checkpointing"):
                        self.memory_manager.enable_checkpointing(self.model)
                    logger.info(
                        f"\n[OOM] Backoff #{self._oom_backoff_count}: gen_batch={self._gen_micro_batch}, train_batch={self._train_micro_batch}"
                    )
                    self.memory_manager.clear_cache(aggressive=True)
                    torch.cuda.empty_cache()
                    gc.collect()

            epoch_metrics.append(metrics)

            if self._oom_backoff_count > 0:
                self._post_oom_success_batches += 1
                if self._post_oom_success_batches >= 50:
                    self._gen_micro_batch = min(4, self._gen_micro_batch + 1)
                    self._train_micro_batch = min(4, self._train_micro_batch + 1)
                    self._oom_backoff_count = max(0, self._oom_backoff_count - 1)
                    self._post_oom_success_batches = 0

            vram_info = self.memory_manager.get_memory_stats()
            vram_str = (
                f"{vram_info['reserved_gb']:.1f}GB"
                if "reserved_gb" in vram_info
                else "N/A"
            )

            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['avg_reward']:.3f}",
                    "tokens_per_sec": f"{metrics['tokens_per_sec']:.3f}",
                    "vram": vram_str,
                    "step": self.global_step,
                }
            )

            # Logging
            if self.global_step % self.config.training.log_interval == 0:
                self.memory_manager.print_memory_stats(f"[Step {self.global_step}]")

            # Benchmark evaluation every 100 steps
            if self.global_step % 100 == 0:
                try:
                    if self._profiler_hooks:
                        self._profiler_hooks.annotate_step(
                            self.global_step, "benchmark"
                        )
                    self.benchmark.run(self.global_step)
                except Exception as e:
                    logger.info(
                        "[Benchmark] Failed at step %s: %s", self.global_step, e
                    )

            # Save checkpoint
            if (
                self.global_step % self.config.training.save_interval == 0
                and self._accumulation_batches == 0
            ):
                if self._profiler_hooks:
                    self._profiler_hooks.annotate_step(
                        self.global_step,
                        "checkpoint",
                        {"suffix": ""},
                    )
                self.save_checkpoint()

        if self._accumulation_batches > 0:
            logger.info(
                "[Train] Flushing partial accumulation window (%d/%d batches).",
                self._accumulation_batches,
                self.config.training.gradient_accumulation_steps,
            )
            self._flush_accumulated_gradients()

        # Epoch summary
        if not epoch_metrics:
            logger.info(
                "\n[Epoch %s] No steps completed; skipping summary.",
                epoch + 1,
            )
            return reached_max_steps

        avg_loss = sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics)
        avg_reward = sum(m["avg_reward"] for m in epoch_metrics) / len(epoch_metrics)

        logger.info(
            f"\n[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.3f}"
        )

        epoch_summary = {
            "epoch/loss": avg_loss,
            "epoch/avg_reward": avg_reward,
            "epoch/steps": len(epoch_metrics),
        }
        if epoch_metrics:
            for key in epoch_metrics[0].keys():
                if key not in ["loss", "avg_reward"]:
                    epoch_summary[f"epoch/{key}"] = sum(
                        m.get(key, 0) for m in epoch_metrics
                    ) / len(epoch_metrics)

        self._append_metrics_jsonl_entry("epoch_summary", epoch_summary)

        if self._wandb_run is not None:
            wandb.log(epoch_summary, step=self.global_step)

        return reached_max_steps

    def train(self, num_epochs: Optional[int] = None, sent_stage: Optional[int] = None):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs (uses config if None)
            sent_stage: Fixed curriculum stage to train on. If None, advance
                automatically from stage 1 toward the configured maximum.
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        logger.info("\n[Train] Creating dataloader...")
        initial_sent_stage = self._resolve_sent_stage_for_epoch(
            epoch=0, sent_stage=sent_stage, use_sent=self.config.sent.enabled
        )
        dataloader = self._create_train_dataloader(sent_stage=initial_sent_stage, epoch=0)
        use_sent = self._dataloader_uses_sent(dataloader)
        if self.config.sent.enabled and not use_sent:
            logger.warning(
                "[SENT] Requested but inactive for this run; trainer will use standard GSM8K epoch semantics."
            )

        logger.info("[Train] Starting training for %s epochs...", num_epochs)
        logger.info("[Train] Steps per epoch: ~%s", len(dataloader))
        logger.info(
            f"[Train] Gradient accumulation: {self.config.training.gradient_accumulation_steps}"
        )
        logger.info(
            f"[Train] Effective batch size: {self.config.training.batch_size * self.config.training.gradient_accumulation_steps}"
        )
        self._configure_scheduler(len(dataloader), num_epochs)

        if (
            self.config.training.max_steps is not None
            and self.global_step >= self.config.training.max_steps
        ):
            logger.info(
                "[Resume] Max steps (%d) already reached at resume; stopping.",
                self.config.training.max_steps,
            )
            return

        # Save initial config
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        save_training_config(
            self.config, os.path.join(self.config.training.output_dir, "config.json")
        )
        self._prepare_metrics_jsonl()
        self._append_metrics_jsonl_entry(
            "run_info",
            {
                "effective_batch": (
                    self.config.training.batch_size
                    * self.config.training.gradient_accumulation_steps
                ),
                "group_size": self.config.grpo.group_size,
                "max_steps": self.config.training.max_steps,
                "steps_per_epoch": len(dataloader),
                "output_dir": self.config.training.output_dir,
                "use_triton": self.config.training.use_triton_kernels,
            },
        )

        # Run initial benchmark once per model/config (sentinel)
        baseline_marker = os.path.join(
            self.config.training.output_dir, "baseline_benchmark_done.json"
        )
        force = getattr(self.config.training, "force_initial_benchmark", False)
        skip_initial = getattr(self.config.training, "skip_initial_benchmark", False)

        # Initialize run_initial
        run_initial = True
        
        if skip_initial:
            logger.info("[Benchmark] Initial benchmark skipped by configuration.")
            run_initial = False

        def _has_checkpoints() -> bool:
            ckpt_dir = getattr(self.config.training, "checkpoint_dir", None)
            if not ckpt_dir:
                return False
            if not os.path.isdir(ckpt_dir):
                return False
            for f in os.listdir(ckpt_dir):
                if f.startswith("checkpoint_step_") or f.endswith(".pt"):
                    return True
            return False
        if not force:
            if _has_checkpoints():
                logger.info(
                    "[Benchmark] Checkpoints detected; skipping initial benchmark."
                )
                run_initial = False
            elif os.path.exists(baseline_marker):
                try:
                    with open(baseline_marker, "r") as fh:
                        meta = json.load(fh)
                    if meta.get("model_id") == self.config.model.model_id:
                        logger.info(
                            "[Benchmark] Baseline benchmark already present for this model; skipping."
                        )
                        run_initial = False
                    else:
                        logger.info(
                            "[Benchmark] Baseline marker exists but model_id differs; re-running benchmark."
                        )
                        run_initial = True
                except Exception:
                    run_initial = True

        if run_initial:
            try:
                logger.info(
                    "[Train] Running initial benchmark before training start..."
                )
                metrics = self.benchmark.run(self.global_step)
                # write marker with minimal metadata
                try:
                    model_repr = repr(self.config.model.__dict__)
                except Exception:
                    model_repr = str(self.config.model.model_id)

                checksum_src = (
                    model_repr + "|" + str(getattr(self.tokenizer, "vocab_size", ""))
                )
                model_checksum = hashlib.sha256(checksum_src.encode()).hexdigest()

                meta = {
                    "model_id": self.config.model.model_id,
                    "model_checksum": model_checksum,
                    "tokenizer_vocab_size": getattr(self.tokenizer, "vocab_size", None),
                    "time": time.time(),
                    "metrics": metrics,
                    "generation": {
                        "max_new_tokens": self.config.training.max_response_length,
                        "do_sample": self.config.training.generation_do_sample,
                    },
                }
                os.makedirs(self.config.training.output_dir, exist_ok=True)
                with open(baseline_marker, "w") as fh:
                    json.dump(meta, fh)
                logger.info(
                    "[Benchmark] Initial benchmark complete; marker saved at %s.",
                    baseline_marker,
                )

                # Log baseline metrics to WandB if available
                if self._wandb_run is not None and metrics:
                    try:
                        wandb.log(
                            {f"baseline/{k}": v for k, v in metrics.items()},
                            step=self.global_step,
                        )
                        logger.info("[WandB] Baseline metrics logged to WandB.")
                    except Exception as e:
                        logger.info("[WandB] Failed to log baseline metrics: %s", e)

            except Exception as e:
                logger.info("[Benchmark] Initial benchmark failed: %s", e)

        # Training loop
        resume_epoch = 0
        resume_skip_steps = 0
        if self._resume_step is not None:
            steps_per_epoch = len(dataloader)
            if steps_per_epoch > 0:
                resume_epoch = self._resume_step // steps_per_epoch
                resume_skip_steps = self._resume_step % steps_per_epoch
                if (
                    self._resume_epoch is not None
                    and self._resume_epoch != resume_epoch
                ):
                    logger.info(
                        "[Resume] Adjusting resume epoch from %d to %d based on global_step.",
                        self._resume_epoch,
                        resume_epoch,
                    )
                logger.info(
                    "[Resume] Resuming at global_step=%d (epoch=%d, skip=%d).",
                    self._resume_step,
                    resume_epoch,
                    resume_skip_steps,
                )
            else:
                resume_epoch = self._resume_epoch or 0
                logger.info(
                    "[Resume] Resuming at epoch=%d (global_step=%d).",
                    resume_epoch,
                    self._resume_step,
                )

        if self._profiler_hooks:
            self._profiler_hooks.on_training_start(self.global_step, self.current_epoch)

        for epoch in range(resume_epoch, num_epochs):
            if self._max_steps_reached():
                logger.info(
                    "[Train] Reached max steps (%d); stopping training loop.",
                    self.global_step,
                )
                break
            self.current_epoch = epoch
            current_sent_stage = self._resolve_sent_stage_for_epoch(
                epoch=epoch,
                sent_stage=sent_stage,
                use_sent=use_sent,
            )
            if not use_sent:
                dataloader = self._create_train_dataloader(
                    sent_stage=current_sent_stage,
                    epoch=epoch,
                )
            else:
                self._apply_sent_stage(dataloader, current_sent_stage)
            epoch_skip = resume_skip_steps if epoch == resume_epoch else 0
            if self._profiler_hooks:
                self._profiler_hooks.annotate_step(
                    self.global_step,
                    "sent_stage",
                    {
                        "stage": current_sent_stage,
                        "epoch": epoch,
                        "num_stages": self.config.sent.curriculum_stages,
                        "enabled": use_sent,
                    },
                )
            should_stop_training = self.train_epoch_with_skip(
                dataloader, epoch, skip_steps=epoch_skip
            )

            # Save epoch checkpoint
            self.save_checkpoint(suffix=f"_epoch_{epoch + 1}")

            if should_stop_training:
                logger.info(
                    "[Train] Reached max steps (%d); stopping training loop.",
                    self.global_step,
                )
                break

        logger.info("\n[Train] Training complete!")

        # Save final checkpoint
        self.save_checkpoint(suffix="_final")
        self.save_lora_weights(suffix="_final")

        self._finish_wandb()

        if self._profiler_hooks:
            self._profiler_hooks.on_training_end(self.global_step, self.current_epoch)

    def load_checkpoint(
        self, checkpoint_path: str, strict: bool = True
    ) -> Dict[str, Any]:
        if self.checkpoint_manager is None:
            raise RuntimeError("CheckpointManager not initialized. Call setup() first.")

        info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=strict,
        )
        if self.optimizer is not None:
            self.checkpoint_manager.move_optimizer_state_to_model_device(
                self.optimizer,
                self.model,
            )

        loaded_step = info.get("step", 0)
        self.global_step = loaded_step
        self.current_step = self.global_step
        self.current_epoch = info.get("epoch", 0)
        self.optimizer_step = info.get("optimizer_step", 0)
        saved_accumulation_batches = info.get("accumulation_batches", 0)
        self._accumulation_batches = 0
        self._dataloader_seed = info.get("dataloader_seed", self._dataloader_seed)
        self._resume_step = self.global_step
        self._resume_epoch = self.current_epoch
        if saved_accumulation_batches:
            replay_step = max(0, loaded_step - saved_accumulation_batches)
            self.global_step = replay_step
            self.current_step = replay_step
            self._resume_step = replay_step
            logger.warning(
                "[Resume] Discarding %d partially accumulated batches because gradient buffers are not checkpointed; rewinding replay step from %d to %d.",
                saved_accumulation_batches,
                loaded_step,
                replay_step,
            )
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)

        logger.info(
            "[Resume] Loaded checkpoint at step %d, epoch %d.",
            self.global_step,
            self.current_epoch,
        )
        return info

    def save_checkpoint(self, suffix: str = ""):
        """Save training checkpoint."""
        if self.checkpoint_manager is None:
            return
        
        checkpoint_name = f"checkpoint_step_{self.global_step}{suffix}.pt"

        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            checkpoint_name=checkpoint_name,
            extra_state={
                "optimizer_step": self.optimizer_step,
                "accumulation_batches": self._accumulation_batches,
                "dataloader_seed": self._dataloader_seed,
            },
        )

    def save_lora_weights(self, suffix: str = ""):
        """Save only LoRa weights."""
        save_path = os.path.join(
            self.config.training.output_dir, f"lora_weights{suffix}.pt"
        )
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        manager = self.checkpoint_manager or CheckpointManager(
            self.config.training.output_dir
        )

        manager.save_lora_weights(
            model=self.model,
            save_path=save_path,
            metadata={
                "step": self.global_step,
                "epoch": self.current_epoch,
                "config": self.config.to_dict(),
            },
        )


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="GRPO Training with Entropy-Aware Selective Backpropagation"
    )

    # Basic training args
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (uses config default if not set)",
    )

    parser.add_argument(
        "--group-size", type=int, default=None, help="Group size for GRPO sampling"
    )

    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for training"
    )

    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")

    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--sent-stage",
        type=int,
        default=1,
        help="Curriculum stage to train on (1=easiest, default: 1)",
    )

    parser.add_argument(
        "--epsilon-high",
        type=float,
        default=None,
        help="Upper clip bound for two-sided clipping",
    )

    parser.add_argument(
        "--delta", type=float, default=None, help="Hard safety cap on ratio"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )

    parser.add_argument(
        "--force-initial-benchmark",
        action="store_true",
        help="Force running the initial benchmark even if sentinel/checkpoints exist",
    )
    parser.add_argument(
        "--log-metrics-jsonl",
        action="store_true",
        help="Write structured training metrics to JSONL",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Custom JSONL metrics path; defaults to output_dir/metrics.jsonl when enabled",
    )

    args = parser.parse_args()

    # Get base config
    config = get_8gb_vram_config()

    # Override with command-line arguments if provided
    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    if args.group_size is not None:
        config.grpo.group_size = args.group_size

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    if args.lora_rank is not None:
        config.lora.rank = args.lora_rank

    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate

    if args.debug:
        config.training.debug = True

    if args.epsilon_high is not None:
        config.grpo.epsilon_high = args.epsilon_high

    if args.delta is not None:
        config.grpo.delta = args.delta

    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
        config.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.training.log_dir = os.path.join(args.output_dir, "logs")
    config.training.log_metrics_jsonl = args.log_metrics_jsonl
    config.training.metrics_jsonl_path = args.metrics_path

    # Force initial benchmark via CLI flag
    if getattr(args, "force_initial_benchmark", None) is not None:
        config.training.force_initial_benchmark = args.force_initial_benchmark

    # Print configuration
    logger.info("\n" + "=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info("Epochs: %s", config.training.num_epochs)
    logger.info("Group Size: %s", config.grpo.group_size)
    logger.info("Clip Epsilon: %s", config.grpo.clip_epsilon)
    logger.info("Epsilon High: %s", config.grpo.epsilon_high)
    logger.info("Delta: %s", config.grpo.delta)
    logger.info("Mask Truncated: %s", config.grpo.mask_truncated_completions)
    logger.info("Batch Size: %s", config.training.batch_size)
    logger.info(
        "Gradient Accumulation: %s", config.training.gradient_accumulation_steps
    )
    logger.info("LoRA Rank: %s", config.lora.rank)
    logger.info("Learning Rate: %s", config.training.learning_rate)
    logger.info("Output Dir: %s", config.training.output_dir)
    logger.info("Metrics JSONL: %s", config.training.log_metrics_jsonl)
    if config.training.log_metrics_jsonl:
        logger.info(
            "Metrics Path: %s",
            config.training.metrics_jsonl_path
            or os.path.join(config.training.output_dir, "metrics.jsonl"),
        )
    logger.info("=" * 60 + "\n")

    # Create trainer
    trainer = GRPOTrainerLoop(config)

    # Setup
    trainer.setup()

    # Train
    trainer.train(sent_stage=args.sent_stage)


if __name__ == "__main__":
    main()
