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
from typing import Optional, Dict, List
import gc
import time
import math
import argparse
import logging
import json
import hashlib

from src.utils.logging_utils import get_logger, TRACE, log_tensor_meta

# Module-level logger
logger = get_logger("trainer")

from src.core.model_loader import load_4bit_engine
from src.core.lora import inject_lora_layers, get_lora_parameters
from src.core.memory_manager import MemoryManager, print_model_memory_usage
from src.grpo.algorithm import GRPOTrainer, GroupSampler
from src.grpo.verifier import RuleBasedVerifier

from src.selective.entropy_mask import EntropyCalculator

from src.data.gsm8k_loader import create_grpo_dataloader
from src.grpo.benchmark import GSM8KBenchmark
from src.utils.checkpoint import CheckpointManager, save_training_config
from src.utils.config import Config, get_8gb_vram_config

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
        self._gen_micro_batch = 4
        self._train_micro_batch = 4
        self._oom_backoff_count = 0
        self._wandb_run = None
        self._step_start_time = None

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

        # Setup scheduler: linear warmup then cosine decay
        total_steps = self.config.training.num_epochs * 1000
        warmup_steps = self.config.training.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Setup memory manager
        self.memory_manager = MemoryManager(
            device=self.device,
            enable_gradient_checkpointing=self.config.training.enable_gradient_checkpointing,
            clear_cache_frequency=self.config.training.clear_cache_frequency,
        )
        self.memory_manager.enable_checkpointing(self.model)

        # Setup GRPO components
        self.grpo_trainer = GRPOTrainer(
            clip_epsilon=self.config.grpo.clip_epsilon,
            epsilon_high=self.config.grpo.epsilon_high,
            delta=self.config.grpo.delta,
            kl_coef=self.config.grpo.kl_coef,
            group_size=self.config.grpo.group_size,
            use_kl=self.config.grpo.use_kl,
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
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.training.checkpoint_dir
        )

        # Initialize benchmark
        self.benchmark = GSM8KBenchmark(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_manager=self.memory_manager,
            dataset_split="test",
            num_samples=50,
            device=self.device,
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

    def _log_wandb_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to WandB."""
        if self._wandb_run is None:
            return

        if self.global_step % self.config.wandb.log_frequency != 0:
            return

        log_dict = {
            f"{prefix}/epoch": self.current_epoch,
        }

        for key, value in metrics.items():
            log_dict[f"{prefix}/{key}"] = value

        vram_stats = self.memory_manager.get_memory_stats()
        if vram_stats:
            log_dict["memory/vram_used_gb"] = vram_stats.get("reserved_gb", 0)
            log_dict["memory/vram_allocated_gb"] = vram_stats.get("allocated_gb", 0)


        log_dict["train/learning_rate"] = self.scheduler.get_last_lr()[0]
        log_dict["train/gen_micro_batch"] = self._gen_micro_batch
        log_dict["train/train_micro_batch"] = self._train_micro_batch
        log_dict["train/oom_backoff_count"] = self._oom_backoff_count

        if self._step_start_time:
            step_time = time.time() - self._step_start_time
            log_dict["perf/step_time_s"] = step_time

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

    def generate_responses(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[str]:
        """
        Generate responses for GRPO group sampling.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            List of generated text strings
        """
        self.model.eval()

        input_ids_expanded, attention_mask_expanded = self.group_sampler.expand_batch(
            input_ids, attention_mask
        )

        generated_texts = []

        micro_batch_size = self._gen_micro_batch
        total_samples = input_ids_expanded.shape[0]

        with torch.no_grad():
            for i in range(0, total_samples, micro_batch_size):
                end_idx = min(i + micro_batch_size, total_samples)

                batch_input_ids = input_ids_expanded[i:end_idx].to(self.device)
                batch_attention_mask = attention_mask_expanded[i:end_idx].to(
                    self.device
                )

                # Standard generation loop
                outputs = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=self.config.training.max_response_length,
                    do_sample=self.config.training.generation_do_sample,
                    temperature=self.config.training.generation_temperature,
                    top_p=self.config.training.generation_top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

                for j in range(outputs.shape[0]):
                    response_ids = outputs[j, batch_input_ids.shape[1] :]
                    response_text = self.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    )
                    generated_texts.append(response_text)

                del outputs, batch_input_ids, batch_attention_mask

                if (i // micro_batch_size) % 2 == 0:
                    self.memory_manager.clear_cache()

        return generated_texts

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

        if self.global_step % self.config.training.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Get batch data
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        ground_truths = batch["answers"]

        # Phase 1: Generation (no gradients)
        self.memory_manager.optimize_for_inference()

        generated_texts = self.generate_responses(input_ids, attention_mask)

        # Tokenize responses (without prompt) early for length penalty
        gen_encodings = self.tokenizer(
            generated_texts,
            truncation=True,
            max_length=self.config.training.max_response_length,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )
        response_ids = gen_encodings["input_ids"].to(self.device)
        response_mask = gen_encodings["attention_mask"].to(self.device)
        
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

        for i, (gen_text, gt) in enumerate(zip(generated_texts, expanded_ground_truths)):
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
                correct_count = sum(1 for info in batch_infos if info.get("match", False))
                
                total_count = len(batch_rewards)
                failed_count = total_count - correct_count

                logger.debug(
                    "Responses: %d total | %d correct | %d failed",
                    total_count, correct_count, failed_count
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

        # Calculate advantages
        advantages = self.grpo_trainer.calculate_advantages(rewards)

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

        # Mask truncated completions: zero out loss for responses that hit max_response_length
        # without producing an EOS token (truncated mid-thought → noisy gradient signal)
        if self.config.grpo.mask_truncated_completions:
            eos_id = self.tokenizer.eos_token_id
            truncation_mask = torch.ones(
                num_samples, dtype=torch.float32, device=self.device
            )
            for idx in range(num_samples):
                resp_tokens = response_ids[idx][response_mask[idx].bool()]
                if resp_tokens.numel() >= self.config.training.max_response_length:
                    if eos_id not in resp_tokens:
                        truncation_mask[idx] = 0.0

            if truncation_mask.sum() == 0:
                truncation_mask[:] = 1.0

            advantages = advantages * truncation_mask

        del generated_texts, gen_encodings

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

        with torch.no_grad():
            gen_micro_batch = self._gen_micro_batch
            for start_idx in range(0, num_samples, gen_micro_batch):
                end_idx = min(start_idx + gen_micro_batch, num_samples)

                batch_ids = all_input_ids[start_idx:end_idx]
                batch_mask = all_attention_mask[start_idx:end_idx]

                outputs = self.model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    use_cache=False,
                )

                # Logits for next-token prediction
                logits = outputs.logits[:, :-1, :]
                targets = batch_ids[:, 1:]

                # Compute log probs
                log_probs = F.log_softmax(logits, dim=-1)

                # Gather log probs for target tokens
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=targets.unsqueeze(-1)
                ).squeeze(-1)
                token_log_probs = (
                    token_log_probs * response_only_mask[start_idx:end_idx, 1:]
                )

                all_old_log_probs.append(token_log_probs)

                del outputs, logits, log_probs
                self.memory_manager.clear_cache()

        all_old_log_probs = torch.cat(all_old_log_probs, dim=0)

        # Re-enable gradients for training phase
        self.memory_manager.optimize_for_training()

        training_micro_batch = self._train_micro_batch

        for start_idx in range(0, num_samples, training_micro_batch):
            end_idx = min(start_idx + training_micro_batch, num_samples)

            batch_input_ids = all_input_ids[start_idx:end_idx]
            batch_attention_mask = all_attention_mask[start_idx:end_idx]
            batch_response_mask = response_only_mask[
                start_idx:end_idx
            ]  # Response-only mask
            batch_advantages = advantages[start_idx:end_idx]
            batch_old_log_probs = all_old_log_probs[start_idx:end_idx]

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
                entropy = self.entropy_calculator.calculate_entropy(logits)
                # Combine entropy mask with response_only_mask
                base_entropy_mask = self.entropy_calculator.create_mask(
                    entropy, batch_attention_mask
                )
                # Only keep entropy mask where response_only_mask is 1
                entropy_mask = base_entropy_mask * batch_response_mask

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
                entropy_mask=entropy_mask[:, 1:] if entropy_mask is not None else None,
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

        # Optimizer step (only after accumulation)
        if (
            self.global_step + 1
        ) % self.config.training.gradient_accumulation_steps == 0:
            # Clip gradients only at the accumulation boundary, not every step.
            # Clipping every step destroys accumulated gradients by repeatedly
            # capping them to max_norm before the full effective batch is gathered.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

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
            avg_metrics["entropy_masked_ratio"] = avg_metrics.get("selected_tokens_ratio", 0.0)

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

        self.global_step += 1
        self.memory_manager.step()

        self._log_wandb_metrics(avg_metrics)

        return avg_metrics

    def train_epoch(self, dataloader, epoch: int):
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        """
        self.model.train()
        epoch_metrics = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            try:
                metrics = self.training_step(batch)
                epoch_metrics.append(metrics)

                if self._oom_backoff_count > 0 and batch_idx % 50 == 0:
                    self._gen_micro_batch = min(4, self._gen_micro_batch + 1)
                    self._train_micro_batch = min(4, self._train_micro_batch + 1)
                    self._oom_backoff_count = max(0, self._oom_backoff_count - 1)

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
                        self.benchmark.run(self.global_step)
                    except Exception as e:
                        logger.info("[Benchmark] Failed at step %s: %s", self.global_step, e)

                # Save checkpoint
                if self.global_step % self.config.training.save_interval == 0:
                    self.save_checkpoint()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._oom_backoff_count += 1
                    self._gen_micro_batch = max(1, self._gen_micro_batch // 2)
                    self._train_micro_batch = max(1, self._train_micro_batch // 2)
                    logger.info(
                        f"\n[OOM] Backoff #{self._oom_backoff_count}: gen_batch={self._gen_micro_batch}, train_batch={self._train_micro_batch}"
                    )
                    self.memory_manager.clear_cache(aggressive=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

        # Epoch summary
        avg_loss = sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics)
        avg_reward = sum(m["avg_reward"] for m in epoch_metrics) / len(epoch_metrics)

        logger.info(
            f"\n[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.3f}"
        )

        if self._wandb_run is not None:
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
            wandb.log(epoch_summary, step=self.global_step)

    def train(self, num_epochs: Optional[int] = None, sent_stage: int = 1):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs (uses config if None)
            sent_stage: Curriculum stage to train on (1-indexed, default=1 = easiest)
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        # Create dataloader
        logger.info("\n[Train] Creating dataloader...")
        # Determine shuffle based on SENT
        # If SENT is enabled, we MUST NOT shuffle (curriculum depends on order)
        use_sent = self.config.sent.enabled
        do_shuffle = not use_sent
        
        dataloader = create_grpo_dataloader(
            tokenizer=self.tokenizer,
            split="train",
            batch_size=self.config.training.batch_size,
            max_prompt_length=self.config.training.max_prompt_length,
            shuffle=do_shuffle,
            use_sent=use_sent,
            sent_config=self.config.sent,
            num_stages=self.config.sent.curriculum_stages,
            cache_path=self.config.sent.cache_path,
        )

        # Set curriculum stage if SENT is enabled
        if use_sent and hasattr(dataloader.dataset, 'set_stage'):
            dataloader.dataset.set_stage(sent_stage)
            stage_info = dataloader.dataset.get_stage_info()
            logger.info("[SENT] Training on stage %d/%d (samples %d-%d)",
                        sent_stage, stage_info["num_stages"],
                        stage_info["stage_start_idx"], stage_info["stage_end_idx"])

        logger.info("[Train] Starting training for %s epochs...", num_epochs)
        logger.info("[Train] Steps per epoch: ~%s", len(dataloader))
        logger.info(
            f"[Train] Gradient accumulation: {self.config.training.gradient_accumulation_steps}"
        )
        logger.info(
            f"[Train] Effective batch size: {self.config.training.batch_size * self.config.training.gradient_accumulation_steps}"
        )

        # Save initial config
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        save_training_config(
            self.config, os.path.join(self.config.training.output_dir, "config.json")
        )

        # Run initial benchmark once per model/config (sentinel)
        baseline_marker = os.path.join(self.config.training.output_dir, "baseline_benchmark_done.json")
        force = getattr(self.config.training, "force_initial_benchmark", False)

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

        run_initial = True
        if not force:
            if _has_checkpoints():
                logger.info("[Benchmark] Checkpoints detected; skipping initial benchmark.")
                run_initial = False
            elif os.path.exists(baseline_marker):
                try:
                    with open(baseline_marker, "r") as fh:
                        meta = json.load(fh)
                    if meta.get("model_id") == self.config.model.model_id:
                        logger.info("[Benchmark] Baseline benchmark already present for this model; skipping.")
                        run_initial = False
                    else:
                        logger.info("[Benchmark] Baseline marker exists but model_id differs; re-running benchmark.")
                        run_initial = True
                except Exception:
                    run_initial = True

        if run_initial:
            try:
                logger.info("[Train] Running initial benchmark before training start...")
                metrics = self.benchmark.run(self.global_step)
                # write marker with minimal metadata
                try:
                    model_repr = repr(self.config.model.__dict__)
                except Exception:
                    model_repr = str(self.config.model.model_id)

                checksum_src = model_repr + "|" + str(getattr(self.tokenizer, "vocab_size", ""))
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
                logger.info("[Benchmark] Initial benchmark complete; marker saved at %s.", baseline_marker)

                # Log baseline metrics to WandB if available
                if self._wandb_run is not None and metrics:
                    try:
                        wandb.log({f"baseline/{k}": v for k, v in metrics.items()}, step=self.global_step)
                        logger.info("[WandB] Baseline metrics logged to WandB.")
                    except Exception as e:
                        logger.info("[WandB] Failed to log baseline metrics: %s", e)

            except Exception as e:
                logger.info("[Benchmark] Initial benchmark failed: %s", e)

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.train_epoch(dataloader, epoch)

            # Save epoch checkpoint
            self.save_checkpoint(suffix=f"_epoch_{epoch + 1}")

        logger.info("\n[Train] Training complete!")

        # Save final checkpoint
        self.save_checkpoint(suffix="_final")
        self.save_lora_weights(suffix="_final")

        self._finish_wandb()

    def save_checkpoint(self, suffix: str = ""):
        """Save training checkpoint."""
        checkpoint_name = f"checkpoint_step_{self.global_step}{suffix}.pt"

        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            checkpoint_name=checkpoint_name,
        )

    def save_lora_weights(self, suffix: str = ""):
        """Save only LoRA weights."""
        save_path = os.path.join(
            self.config.training.output_dir, f"lora_weights{suffix}.pt"
        )

        self.checkpoint_manager.save_lora_weights(
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
    logger.info("=" * 60 + "\n")

    # Create trainer
    trainer = GRPOTrainerLoop(config)

    # Setup
    trainer.setup()

    # Train
    trainer.train(sent_stage=args.sent_stage)


if __name__ == "__main__":
    main()
