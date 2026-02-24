"""
Entropy-Aware Selective Backpropagation.
Calculates token-level entropy and creates gradient masks.
"""
import importlib
from typing import Callable, cast

import torch
import torch.nn.functional as F


class EntropyCalculator:
    """
    Calculates per-token entropy from logits for selective backpropagation.
    
    High entropy = uncertain/confused tokens (keep gradients)
    Low entropy = confident tokens (block gradients)
    
    Formula: H(p) = -sum(p * log(p))
    """
    
    def __init__(
        self,
        threshold: float | None = None,
        percentile: float = 0.5,
        min_tokens: int = 10
    ):
        """
        Args:
            threshold: Fixed entropy threshold (if None, uses percentile)
            percentile: Percentile of tokens to keep (0.0-1.0)
            min_tokens: Minimum number of tokens to keep per sequence
        """
        self.threshold: float | None = threshold
        self.percentile: float = percentile
        self.min_tokens: int = min_tokens
    
    def calculate_entropy(
        self,
        logits: torch.Tensor,
        chunk_size: int = 256
    ) -> torch.Tensor:
        """
        Calculate entropy for each token position using chunked processing to save memory.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            chunk_size: Number of tokens to process at once (default: 256)
            
        Returns:
            Entropy values [batch, seq_len]
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten batch and seq dimensions for easier chunking
        # reshaping to (-1, vocab_size) is usually a view if contiguous
        flat_logits = logits.reshape(-1, vocab_size)
        total_tokens = flat_logits.size(0)
        
        entropy_list: list[torch.Tensor] = []
        
        with torch.no_grad():
            # Process in chunks to avoid materializing full probabilities tensor
            for i in range(0, total_tokens, chunk_size):
                chunk = flat_logits[i:i + chunk_size]

                # Standard entropy calculation on small chunk
                log_probs = F.log_softmax(chunk, dim=-1)
                probs = torch.exp(log_probs)

                # H = -sum(p * log(p))
                chunk_entropy = -(probs * log_probs).sum(dim=-1)

                entropy_list.append(chunk_entropy)
            
        # Concatenate and restore shape
        entropy = torch.cat(entropy_list).view(batch_size, seq_len)
        
        return entropy

    def calculate_entropy_and_mask(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_triton_kernels: bool = False,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if use_triton_kernels:
                triton_module = importlib.import_module("src.triton_kernels")
                fused_entropy_mask = cast(
                    Callable[..., tuple[torch.Tensor, torch.Tensor]],
                    getattr(triton_module, "fused_entropy_mask"),
                )
                entropy, mask = fused_entropy_mask(
                    logits,
                    attention_mask=attention_mask,
                    threshold=self.threshold,
                    percentile=self.percentile,
                    min_tokens=self.min_tokens,
                )
                return entropy, mask

            entropy = self.calculate_entropy(logits, chunk_size=chunk_size)
            mask = self.create_mask(entropy, attention_mask)
            return entropy, mask
    
    def create_mask_from_threshold(
        self,
        entropy: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Create binary mask from fixed threshold.
        
        Args:
            entropy: Entropy values [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Binary mask [batch, seq_len] (1 = keep, 0 = mask)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set")
        
        mask = (entropy > self.threshold).float()
        
        # Apply attention mask
        if attention_mask is not None:
            mask = mask * attention_mask
        
        return mask
    
    def create_mask_from_percentile(
        self,
        entropy: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Create binary mask keeping top percentile by entropy.
        
        Uses torch.quantile on all valid (non-padding) tokens to compute a
        single global threshold, following the TRL reference approach. Then
        enforces a per-sequence minimum token guarantee so no sequence is
        left with fewer than min_tokens selected.
        
        Args:
            entropy: Entropy values [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Binary mask [batch, seq_len]
        """
        batch_size, seq_len = entropy.shape
        
        # Collect valid (non-padding) entropy values across the batch
        if attention_mask is not None:
            valid_entropy = entropy[attention_mask.bool()].float()
        else:
            valid_entropy = entropy.reshape(-1).float()
        
        if valid_entropy.numel() == 0:
            return torch.zeros_like(entropy)
        
        # Compute global threshold via quantile (matches TRL's get_high_entropy_mask).
        # self.percentile is the fraction to KEEP (e.g. 0.5 = top 50%).
        # quantile(1 - 0.5) = median → tokens >= median are the top 50%.
        entropy_threshold = torch.quantile(valid_entropy, 1.0 - self.percentile)
        
        # Apply threshold globally
        mask = (entropy >= entropy_threshold).float()
        
        # Zero out padding tokens
        if attention_mask is not None:
            mask = mask * attention_mask
        
        # Per-sequence min_tokens guarantee:
        # If any sequence has fewer than min_tokens selected, force-select
        # its top-min_tokens valid tokens by entropy.
        for i in range(batch_size):
            selected = int(mask[i].sum().item())
            if selected >= self.min_tokens:
                continue
            
            # Determine valid positions for this sequence
            if attention_mask is not None:
                valid_pos = attention_mask[i].bool()
            else:
                valid_pos = torch.ones(seq_len, dtype=torch.bool, device=entropy.device)
            
            n_valid = int(valid_pos.sum().item())
            n_select = min(self.min_tokens, n_valid)
            
            if n_select > 0:
                # Mask invalid positions so they can't be selected
                seq_entropy = entropy[i].clone()
                seq_entropy[~valid_pos] = -float('inf')
                _, top_idx = torch.topk(seq_entropy, n_select)
                mask[i] = 0.0
                mask[i, top_idx] = 1.0
        
        return mask
    
    def create_mask(
        self,
        entropy: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Create gradient mask based on entropy.
        
        Args:
            entropy: Entropy values [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Binary mask [batch, seq_len]
        """
        if self.threshold is not None:
            return self.create_mask_from_threshold(entropy, attention_mask)
        else:
            return self.create_mask_from_percentile(entropy, attention_mask)
    
    def apply_entropy_mask(
        self,
        loss: torch.Tensor,
        entropy_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply entropy mask to loss tensor.
        
        Args:
            loss: Per-token loss [batch, seq_len]
            entropy_mask: Binary mask [batch, seq_len]
            
        Returns:
            Masked loss scalar
        """
        masked_loss = loss * entropy_mask
        
        # Normalize by number of selected tokens
        mask_sum = entropy_mask.sum() + 1e-8
        return masked_loss.sum() / mask_sum
    
    def get_entropy_stats(
        self,
        entropy: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> dict[str, float]:
        """
        Get statistics about entropy distribution.
        
        Args:
            entropy: Entropy values [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Dictionary of statistics
        """
        if attention_mask is not None:
            valid_entropy = entropy[attention_mask == 1]
        else:
            valid_entropy = entropy.view(-1)
        
        return {
            'entropy_mean': valid_entropy.mean().item(),
            'entropy_std': valid_entropy.std().item(),
            'entropy_min': valid_entropy.min().item(),
            'entropy_max': valid_entropy.max().item(),
            'entropy_median': valid_entropy.median().item(),
        }
