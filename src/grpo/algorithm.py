"""
Dr. GRPO (Group Relative Policy Optimization) Algorithm Implementation.
Implements Dr. GRPO loss with two-sided clipping, advantage calculation, and KL divergence.
"""
from __future__ import annotations

import importlib
from typing import Callable, cast
import torch
import torch.nn.functional as F


class GRPOTrainer:
    """
    Dr. GRPO trainer with two-sided clipping.
    
    Key differences from standard GRPO:
    - Global normalization: divides by group_size instead of per-sequence token count
    - Two-sided clipping: asymmetric bounds (epsilon, epsilon_high) for the PPO ratio
    - Hard safety cap (delta): clamps ratio to prevent small-batch explosion
    """

    clip_epsilon: float
    epsilon_high: float
    delta: float
    kl_coef: float
    group_size: int
    use_kl: bool
    use_triton_kernels: bool
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        epsilon_high: float = 0.3,
        delta: float = 1.5,
        kl_coef: float = 0.1,
        group_size: int = 4,
        use_kl: bool = False,
        use_triton_kernels: bool = False,
    ):
        self.clip_epsilon = clip_epsilon
        self.epsilon_high = epsilon_high
        self.delta = delta
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.use_kl = use_kl
        self.use_triton_kernels = use_triton_kernels
    
    def calculate_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int | None = None
    ) -> torch.Tensor:
        """
        Calculate Dr. GRPO group-relative advantages using centered rewards.
        
        A_i = r_i - mean(r_group)
        
        Dr. GRPO removes the standard deviation normalization from standard GRPO
        to eliminate question-level difficulty bias. When all rewards in a group
        are identical (common in RLVR), advantages are naturally zero — no
        artificial baseline or min_std hack needed.
        
        Args:
            rewards: Tensor of rewards [batch_size * group_size]
            group_size: Group size G (uses self.group_size if None)
            
        Returns:
            Advantages tensor of same shape as rewards
        """
        if group_size is None:
            group_size = self.group_size
        
        batch_size = rewards.shape[0] // group_size
        
        rewards_grouped = rewards.view(batch_size, group_size)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        advantages = rewards_grouped - mean_rewards
        
        return advantages.view(-1)
    
    def compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model.
        
        Uses the Schulman approximation:
        D_KL = exp(log_policy - log_ref) - (log_policy - log_ref) - 1
        
        Args:
            policy_logits: Policy model logits [batch, seq_len, vocab_size]
            reference_logits: Reference model logits [batch, seq_len, vocab_size]
            
        Returns:
            KL divergence per token [batch, seq_len]
        """
        # Convert to log probabilities
        log_policy = F.log_softmax(policy_logits, dim=-1)
        log_ref = F.log_softmax(reference_logits, dim=-1)
        
        # Get probabilities from policy
        policy_probs = torch.exp(log_policy)
        
        # KL = sum(p_policy * (log_policy - log_ref))
        kl = torch.sum(
            policy_probs * (log_policy - log_ref),
            dim=-1
        )
        
        return kl
    
    def compute_grpo_loss(
        self,
        policy_logits: torch.Tensor,
        advantages: torch.Tensor,
        old_policy_logits: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        reference_logits: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        entropy_mask: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute Dr. GRPO loss with two-sided clipping.
        
        Loss = -(1/G) * sum_i[ sum_t[ min(ratio_t * A_i, clip(ratio_t) * A_i) ] ]
        
        Two-sided clipping:
          - ratio clamped to [1-epsilon, 1+epsilon_high] in surrogate
          - ratio hard-capped at delta before any computation
        
        Global normalization: divides by group_size (G) not per-sequence token count.
        """
        batch_size, seq_len, vocab_size = policy_logits.shape
        
        if target_ids is None:
            raise ValueError(
                "target_ids must be provided: GRPO requires the actually generated tokens from rollout, not inferred tokens from logits."
            )
        target_tokens = target_ids
        
        # OPTIMIZED: Use cross_entropy to get log_probs of target tokens
        # This avoids materializing the full [batch, seq, vocab] log_probs tensor
        # -F.cross_entropy = log(p_target)
        if old_log_probs is not None:
            old_log_probs_target = old_log_probs
        elif old_policy_logits is not None:
            # OPTIMIZED: Same optimization for old policy
            old_log_probs_target = -F.cross_entropy(
                old_policy_logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1),
                reduction='none'
            ).view(batch_size, seq_len)
        else:
            raise ValueError("Either old_log_probs or old_policy_logits must be provided")

        use_triton_loss = self.use_triton_kernels and not self.use_kl
        if use_triton_loss:
            triton_module = importlib.import_module("src.triton_kernels")
            fused_grpo_loss = cast(
                Callable[..., object],
                getattr(triton_module, "fused_grpo_loss"),
            )
            return cast(
                tuple[torch.Tensor, dict[str, float]],
                fused_grpo_loss(
                    policy_logits=policy_logits,
                    target_ids=target_tokens,
                    old_log_probs=old_log_probs_target,
                    advantages=advantages,
                    clip_epsilon=self.clip_epsilon,
                    epsilon_high=self.epsilon_high,
                    delta=self.delta,
                    group_size=self.group_size,
                    attention_mask=attention_mask,
                    entropy_mask=entropy_mask,
                ),
            )

        log_probs_target = -F.cross_entropy(
            policy_logits.reshape(-1, vocab_size),
            target_tokens.reshape(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        
        log_ratio = log_probs_target - old_log_probs_target
        ratio = torch.exp(log_ratio)
        
        # Hard safety cap: clamp ratio to [0, delta] before clipping
        ratio = torch.clamp(ratio, max=self.delta)
        
        advantages_expanded = advantages.unsqueeze(1).expand(-1, seq_len)
        
        # Two-sided clipping: asymmetric bounds [1 - epsilon, 1 + epsilon_high]
        surr1 = ratio * advantages_expanded
        surr2 = torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.epsilon_high
        ) * advantages_expanded
        
        policy_loss = -torch.min(surr1, surr2)
        
        kl_loss = torch.zeros_like(policy_loss)
        if self.use_kl and reference_logits is not None and self.kl_coef > 0:
            kl_per_token = self.compute_kl_divergence(policy_logits, reference_logits)
            kl_loss = self.kl_coef * kl_per_token
        
        loss_per_token = policy_loss + kl_loss
        
        if entropy_mask is not None:
            loss_per_token = loss_per_token * entropy_mask
            if attention_mask is not None:
                loss_per_token = loss_per_token * attention_mask
        elif attention_mask is not None:
            loss_per_token = loss_per_token * attention_mask
        
        per_sample_loss = loss_per_token.sum(dim=1)
        if entropy_mask is not None:
            if attention_mask is not None:
                effective_mask = entropy_mask * attention_mask
                total_tokens = attention_mask.sum(dim=1)
            else:
                effective_mask = entropy_mask
                total_tokens = torch.full_like(effective_mask.sum(dim=1), float(seq_len))
            selected_tokens = effective_mask.sum(dim=1)
            scale = torch.where(
                selected_tokens > 0,
                total_tokens / (selected_tokens + 1e-8),
                torch.zeros_like(selected_tokens),
            )
            per_sample_loss = per_sample_loss * scale

        loss = per_sample_loss.sum() / self.group_size
        
        with torch.no_grad():
            adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
            ratio_std = ratio.std().item() if ratio.numel() > 1 else 0.0
            ratio_capped = (ratio >= self.delta).float().mean().item()
            metrics = {
                'loss': loss.item(),
                'policy_loss': policy_loss.mean().item(),
                'kl_loss': kl_loss.mean().item() if self.use_kl else 0.0,
                'ratio_mean': ratio.mean().item(),
                'ratio_std': ratio_std,
                'ratio_capped_pct': ratio_capped,
                'advantage_std': adv_std,
            }
            
            if entropy_mask is not None:
                metrics['selected_tokens_ratio'] = entropy_mask.mean().item()
        
        return loss, metrics
    
    def compute_log_probs(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute per-sample sum of log probabilities for given tokens."""
        log_probs = F.log_softmax(logits, dim=-1)
        
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask
            return token_log_probs.sum(dim=1)
        else:
            return token_log_probs.sum(dim=1)


class GroupSampler:
    """
    Samples multiple responses (group) for each prompt.
    """

    group_size: int
    
    def __init__(self, group_size: int = 4):
        """
        Args:
            group_size: Number of responses per prompt (G)
        """
        self.group_size = group_size

    def group_responses(
        self,
        responses: list[str],
        group_size: int | None = None
    ) -> list[list[str]]:
        """
        Group responses by prompt.
        
        Args:
            responses: Flat list of all responses
            group_size: Group size (uses self.group_size if None)
            
        Returns:
            Grouped responses: [num_prompts, group_size]
        """
        if group_size is None:
            group_size = self.group_size
        
        num_prompts = len(responses) // group_size
        grouped: list[list[str]] = []
        
        for i in range(num_prompts):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            grouped.append(responses[start_idx:end_idx])
        
        return grouped
