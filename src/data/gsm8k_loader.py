"""
GSM8K Dataset Loader for GRPO Training.
Handles loading, formatting, and batching of GSM8K math problems.
"""
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any, Optional, List
import logging
import os
import torch

from ..utils.logging_utils import get_logger
from ..utils.config import SENTConfig

logger = get_logger("data.gsm8k")


def _validate_cache(cache_path: str, config: Optional[SENTConfig] = None) -> tuple[bool, str]:
    """Validate cache file exists and has valid metadata."""
    if not os.path.exists(cache_path):
        return False, "Cache file does not exist"
    
    try:
        import json
        import hashlib
        import torch
        
        if cache_path.endswith('.json'):
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            data = torch.load(cache_path, weights_only=False)
        
        metadata = data.get("metadata", {})
        if metadata.get("status") != "complete":
            return False, f"Cache status is '{metadata.get('status')}', expected 'complete'"
        
        if config is not None:
            cfg_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            cfg_json = json.dumps(cfg_dict, sort_keys=True)
            cfg_hash = hashlib.sha256(cfg_json.encode()).hexdigest()
            cached_hash = metadata.get("config_hash", "")
            if cached_hash and cfg_hash != cached_hash:
                logger.warning(f"Config hash mismatch: cache={cached_hash}, current={cfg_hash}")
        
        return True, "Cache valid"
    except Exception as e:
        return False, f"Cache validation error: {e}"


class GRPOGSM8KDataset(Dataset):
    """
    GSM8K dataset optimized for GRPO training.
    Only returns prompts (questions) for generation phase.
    """
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_prompt_length: int = 512
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            split: Dataset split
            max_prompt_length: Maximum prompt length
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        
        # Load dataset
        logger.info("Loading GSM8K %s split for GRPO...", split)
        self.dataset = load_dataset("gsm8k", "main", split=split)
        logger.info("Loaded %d examples", len(self.dataset))
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example for GRPO.

        Returns:
            Dictionary with prompt and ground truth answer
        """
        item = self.dataset[idx]
        question = item["question"]
        answer_text = item["answer"]
        
        # Extract final answer
        if "####" in answer_text:
            final_answer = answer_text.rsplit("####", 1)[1].strip()
        else:
            final_answer = ""
        
        # Format prompt using model's native chat template
        # Reference: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/tokenizer_config.json
        # DeepSeek-R1-Distill-Qwen uses <｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Adds "<｜Assistant｜><think>\n"
        )
        
        # Tokenize prompt only
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "question": question,
            "answer": final_answer,
        }


class SENTGSM8KDataset(GRPOGSM8KDataset):
    """GSM8K dataset with SENT (Semantic Entropy) curriculum ordering.
    
    Loads sorted indices from cache and supports curriculum staging.
    """
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_prompt_length: int = 512,
        use_sent: bool = True,
        cache_path: str = "data/cache/gsm8k_sent_sorted.pt",
        sent_config: Optional[SENTConfig] = None,
        num_stages: int = 1
    ):
        super().__init__(tokenizer, split, max_prompt_length)
        
        self.use_sent = use_sent
        self.cache_path = cache_path
        self.sent_config = sent_config or SENTConfig()
        self.num_stages = max(1, num_stages)
        
        self.sorted_indices: List[int] = []
        self.entropies: List[float] = []
        self.current_stage = 0
        self.current_stage_indices: List[int] = []
        
        if self.use_sent:
            is_valid, msg = _validate_cache(self.cache_path, self.sent_config)
            if not is_valid:
                raise ValueError(
                    f"SENT cache invalid: {msg}. "
                    f"Run 'python scripts/preprocess_sent.py' to generate cache."
                )
            
            import torch
            import json
            
            if self.cache_path.endswith('.json'):
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)
            else:
                cache_data = torch.load(self.cache_path, weights_only=False)
            self.sorted_indices = cache_data.get("indices", [])
            self.entropies = cache_data.get("entropies", [])
            
            self._compute_stage_boundaries()
            self.set_stage(1)

            logger.info("[SENT] Curriculum Learning enabled")
            logger.info("[SENT] Cache loaded from: %s", self.cache_path)
            logger.info("[SENT] Total sorted samples: %d", len(self.sorted_indices))
            logger.info("[SENT] Curriculum stages: %d (Current: %d)", self.num_stages, self.current_stage)
    
    def _compute_stage_boundaries(self):
        """Compute stage boundaries for curriculum learning."""
        total = len(self.sorted_indices)
        stage_size = total // self.num_stages
        self.stage_boundaries = [
            i * stage_size for i in range(self.num_stages)
        ] + [total]
    
    def set_stage(self, stage_idx: int):
        """Set current curriculum stage (1-indexed)."""
        if stage_idx < 1 or stage_idx > self.num_stages:
            raise ValueError(f"Stage must be 1-{self.num_stages}, got {stage_idx}")
        
        self.current_stage = stage_idx
        start = self.stage_boundaries[stage_idx - 1]
        end = self.stage_boundaries[stage_idx]
        self.current_stage_indices = list(range(start, end))
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about current curriculum stage."""
        return {
            "current_stage": self.current_stage,
            "num_stages": self.num_stages,
            "stage_start_idx": self.stage_boundaries[self.current_stage - 1],
            "stage_end_idx": self.stage_boundaries[self.current_stage],
            "total_samples": len(self.sorted_indices)
        }
    
    def get_entropy(self, idx: int) -> float:
        """Get entropy for a given index in the sorted order."""
        if 0 <= idx < len(self.entropies):
            return self.entropies[idx]
        return float('nan')
    
    def __len__(self) -> int:
        if self.use_sent:
            return len(self.current_stage_indices)
        return super().__len__()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.use_sent:
            actual_idx = self.sorted_indices[self.current_stage_indices[idx]]
            return super().__getitem__(actual_idx)
        return super().__getitem__(idx)


def create_grpo_dataloader(
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    max_prompt_length: int = 512,
    shuffle: bool = False,
    use_sent: bool = True,
    sent_config: Optional[SENTConfig] = None,
    num_stages: int = 1,
    cache_path: str = "data/cache/gsm8k_sent_sorted.pt"
) -> DataLoader:
    """
    Create DataLoader for GRPO training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split
        batch_size: Batch size (number of unique prompts)
        max_prompt_length: Max prompt length
        shuffle: Whether to shuffle (should be False for SENT)
        use_sent: Whether to use SENT curriculum ordering
        sent_config: SENT configuration
        num_stages: Number of curriculum stages
        cache_path: Path to SENT cache file
        
    Returns:
        DataLoader instance
    """
    if use_sent:
        dataset = SENTGSM8KDataset(
            tokenizer=tokenizer,
            split=split,
            max_prompt_length=max_prompt_length,
            use_sent=True,
            cache_path=cache_path,
            sent_config=sent_config,
            num_stages=num_stages
        )
        if shuffle:
            logger.warning("Shuffle=True is not recommended with SENT (order matters for curriculum)")
    else:
        dataset = GRPOGSM8KDataset(
            tokenizer=tokenizer,
            split=split,
            max_prompt_length=max_prompt_length
        )
    
    def grpo_collate(batch):
        # Determine max length in the batch
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_size = len(batch)
        
        # Get padding strategy
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padding_side = getattr(tokenizer, "padding_side", "right")
        
        # Pre-allocate tensors
        input_ids_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            seq_len = len(input_ids)
            
            if padding_side == "left":
                input_ids_padded[i, -seq_len:] = torch.tensor(input_ids, dtype=torch.long)
                attention_mask_padded[i, -seq_len:] = torch.tensor(attention_mask, dtype=torch.long)
            else:
                input_ids_padded[i, :seq_len] = torch.tensor(input_ids, dtype=torch.long)
                attention_mask_padded[i, :seq_len] = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch],
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=grpo_collate,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2
    )
    
    return dataloader
