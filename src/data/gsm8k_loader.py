"""
GSM8K Dataset Loader for GRPO Training.
Handles loading, formatting, and batching of GSM8K math problems.
"""

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any, Optional, List
import hashlib
import json
import logging
import os
import torch

from ..utils.logging_utils import get_logger
from ..utils.config import SENTConfig
from .math_dataset import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    GRPOMathDataset,
    PROMPT_FORMAT_VERSION,
    canonical_dataset_name,
    collate_grpo_math_batch,
    prompt_format_hash,
    safe_dataset_cache_stem,
    split_ratios_dict,
)

logger = get_logger("data.gsm8k")


def format_grpo_prompt(tokenizer, question: str) -> str:
    """Format a GSM8K question exactly as GRPO training will see it."""
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.debug("Falling back to raw prompt because chat template rendering failed.")
    return question


def _make_sent_cache_key(
    sent_config: Optional[object] = None,
    tokenizer: Optional[object] = None,
    max_prompt_length: Optional[int] = None,
    model_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    split_seed: Optional[int] = None,
    split_ratios: Optional[object] = None,
) -> str:
    """Build a compatibility hash for SENT cache reuse."""
    sent_dict = sent_config.to_dict() if hasattr(sent_config, "to_dict") else {}
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        chat_template = None
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    if not isinstance(tokenizer_name, str):
        tokenizer_name = None
    chat_template_hash = (
        hashlib.sha256(chat_template.encode()).hexdigest()
        if chat_template is not None
        else None
    )
    payload = {
        "sent": sent_dict,
        "model_id": model_id,
        "max_prompt_length": max_prompt_length,
        "tokenizer_name": tokenizer_name,
        "tokenizer_class": tokenizer.__class__.__name__ if tokenizer is not None else None,
        "chat_template_hash": chat_template_hash,
        "dataset_name": dataset_name,
        "split": split,
        "split_seed": split_seed,
        "split_ratios": (
            split_ratios_dict(split_ratios)
            if split_ratios is not None
            else None
        ),
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "prompt_format_hash": prompt_format_hash(tokenizer),
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_json.encode()).hexdigest()


def _compute_question_length_percentile(
    dataset, tokenizer, percentile: float = 95.0, batch_size: int = 32
) -> int:
    """Estimate a prompt length cap from templated prompt lengths."""
    if not dataset:
        return 0

    lengths: List[int] = []
    for start in range(0, len(dataset), max(1, batch_size)):
        batch = dataset[start : start + max(1, batch_size)]
        if isinstance(batch, dict):
            questions = batch.get("question", [])
        else:
            questions = [item["question"] for item in batch]

        if not questions:
            return 0

        prompts = []
        for question in questions:
            prompts.append(format_grpo_prompt(tokenizer, question))

        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        if not isinstance(encoded, dict) or "input_ids" not in encoded:
            return 0

        input_ids_batch = encoded["input_ids"]
        if not isinstance(input_ids_batch, list):
            return 0

        if input_ids_batch and isinstance(input_ids_batch[0], int):
            lengths.append(len(input_ids_batch))
        else:
            lengths.extend(len(input_ids) for input_ids in input_ids_batch)

    if len(lengths) == 1:
        return lengths[0]

    lengths = sorted(lengths)
    rank = (len(lengths) - 1) * (percentile / 100.0)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(lengths) - 1)
    weight = rank - lower_idx
    interpolated = (
        lengths[lower_idx] + (lengths[upper_idx] - lengths[lower_idx]) * weight
    )
    return int(interpolated)


def _validate_cache(
    cache_path: str,
    config: Optional[SENTConfig] = None,
    tokenizer: Optional[object] = None,
    max_prompt_length: Optional[int] = None,
    model_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    split_seed: Optional[int] = None,
    split_ratios: Optional[object] = None,
) -> tuple[bool, str]:
    """Validate cache file exists and has valid metadata."""
    if not os.path.exists(cache_path):
        return False, "Cache file does not exist"

    try:
        import torch

        if cache_path.endswith(".json"):
            with open(cache_path, "r") as f:
                data = json.load(f)
        else:
            data = torch.load(cache_path, weights_only=False)

        metadata = data.get("metadata", {})
        if metadata.get("status") != "complete":
            return (
                False,
                f"Cache status is '{metadata.get('status')}', expected 'complete'",
            )

        if config is not None:
            effective_dataset_name = dataset_name
            effective_split = split
            effective_split_seed = split_seed
            effective_split_ratios = split_ratios
            if effective_dataset_name is None and metadata.get("dataset_name") is not None:
                effective_dataset_name = metadata.get("dataset_name")
            if effective_split is None and metadata.get("split") is not None:
                effective_split = metadata.get("split")
            if effective_split_seed is None and metadata.get("split_seed") is not None:
                effective_split_seed = metadata.get("split_seed")
            if effective_split_ratios is None and metadata.get("split_ratios") is not None:
                ratio_meta = metadata.get("split_ratios")
                if isinstance(ratio_meta, dict):
                    effective_split_ratios = (
                        ratio_meta.get("train", 0.90),
                        ratio_meta.get("validation", 0.05),
                        ratio_meta.get("test", 0.05),
                    )

            expected_model_id = model_id
            if expected_model_id is None:
                tokenizer_model_id = getattr(tokenizer, "name_or_path", None)
                if isinstance(tokenizer_model_id, str):
                    expected_model_id = tokenizer_model_id
            expected_key = _make_sent_cache_key(
                config,
                tokenizer=tokenizer,
                max_prompt_length=max_prompt_length,
                model_id=expected_model_id,
                dataset_name=effective_dataset_name,
                split=effective_split,
                split_seed=effective_split_seed,
                split_ratios=effective_split_ratios,
            )
            cached_key = metadata.get("sent_cache_key")
            if cached_key:
                if cached_key != expected_key:
                    return False, "SENT cache compatibility mismatch"
            elif metadata.get("config_hash"):
                return (
                    False,
                    "Legacy SENT cache metadata lacks compatibility key; regenerate cache.",
                )

        if dataset_name is not None and metadata.get("dataset_name") not in (None, dataset_name):
            return False, "SENT cache dataset_name mismatch"
        if split is not None and metadata.get("split") not in (None, split):
            return False, "SENT cache split mismatch"
        if split_seed is not None and metadata.get("split_seed") not in (None, split_seed):
            return False, "SENT cache split_seed mismatch"
        if split_ratios is not None and metadata.get("split_ratios") not in (
            None,
            split_ratios_dict(split_ratios),
        ):
            return False, "SENT cache split ratios mismatch"

        return True, "Cache valid"
    except Exception as e:
        return False, f"Cache validation error: {e}"


class GRPOGSM8KDataset(Dataset):
    """
    GSM8K dataset optimized for GRPO training.
    Only returns prompts (questions) for generation phase.
    """

    def __init__(self, tokenizer, split: str = "train", max_prompt_length: int = 512):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            split: Dataset split
            max_prompt_length: Maximum prompt length
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.requested_max_prompt_length = max_prompt_length
        self.use_sent = False
        self.num_stages = 1

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
        prompt = format_grpo_prompt(self.tokenizer, question)

        # Tokenize prompt only
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
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
        num_stages: int = 1,
        model_id: Optional[str] = None,
    ):
        super().__init__(tokenizer, split, max_prompt_length)

        self.use_sent = use_sent
        self.cache_path = cache_path
        self.sent_config = sent_config or SENTConfig()
        self.num_stages = max(1, num_stages)
        tokenizer_model_id = getattr(tokenizer, "name_or_path", None)
        self.model_id = (
            model_id
            if model_id is not None
            else tokenizer_model_id if isinstance(tokenizer_model_id, str) else None
        )

        self.sorted_indices: List[int] = []
        self.entropies: List[float] = []
        self.current_stage = 0
        self.current_stage_indices: List[int] = []

        if self.use_sent:
            is_valid, msg = _validate_cache(
                self.cache_path,
                self.sent_config,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                model_id=self.model_id,
            )
            if not is_valid:
                raise ValueError(
                    f"SENT cache invalid: {msg}. "
                    f"Run 'python scripts/preprocess_sent.py' to generate cache."
                )

            import torch
            import json

            if self.cache_path.endswith(".json"):
                with open(self.cache_path, "r") as f:
                    cache_data = json.load(f)
            else:
                cache_data = torch.load(self.cache_path, weights_only=False)
            self.sorted_indices = cache_data.get("indices", [])
            self.entropies = cache_data.get("entropies", [])
            if any(
                not isinstance(idx, int) or idx < 0 or idx >= len(self.dataset)
                for idx in self.sorted_indices
            ):
                raise ValueError(
                    "SENT cache indices must be positional dataset indices. "
                    "Regenerate the cache with the current preprocessing scripts."
                )

            self._compute_stage_boundaries()
            self.set_stage(1)

            logger.info("[SENT] Curriculum Learning enabled")
            logger.info("[SENT] Cache loaded from: %s", self.cache_path)
            logger.info("[SENT] Total sorted samples: %d", len(self.sorted_indices))
            logger.info(
                "[SENT] Curriculum stages: %d (Current: %d)",
                self.num_stages,
                self.current_stage,
            )

    def _compute_stage_boundaries(self):
        """Compute stage boundaries for curriculum learning."""
        total = len(self.sorted_indices)
        stage_size = total // self.num_stages
        self.stage_boundaries = [i * stage_size for i in range(self.num_stages)] + [
            total
        ]

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
            "total_samples": len(self.sorted_indices),
        }

    def get_entropy(self, idx: int) -> float:
        """Get entropy for a given index in the sorted order."""
        if 0 <= idx < len(self.entropies):
            return self.entropies[idx]
        return float("nan")

    @staticmethod
    def _sent_rank_fraction(rank: int, total_size: int) -> float:
        """Map a SENT sorted position to a normalized [0, 1] rank fraction."""
        if total_size <= 1:
            return 0.0
        return float(rank) / float(total_size - 1)

    def __len__(self) -> int:
        if self.use_sent:
            return len(self.current_stage_indices)
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.use_sent:
            sent_rank = self.current_stage_indices[idx]
            actual_idx = self.sorted_indices[sent_rank]
            item = super().__getitem__(actual_idx)
            item["sent_rank"] = sent_rank
            item["sent_rank_fraction"] = self._sent_rank_fraction(
                sent_rank, len(self.sorted_indices)
            )
            item["sent_rank_total"] = len(self.sorted_indices)
            return item
        return super().__getitem__(idx)


class SENTMathDataset(GRPOMathDataset):
    """Generic normalized math dataset with SENT curriculum ordering."""

    def __init__(
        self,
        tokenizer,
        *,
        dataset_name: str,
        split: str = "train",
        max_prompt_length: int = 512,
        use_sent: bool = True,
        cache_path: str = "data/cache/gsm8k_sent_sorted.pt",
        sent_config: Optional[SENTConfig] = None,
        num_stages: int = 2,
        model_id: Optional[str] = None,
        split_seed: int = DEFAULT_SPLIT_SEED,
        split_ratios: object = DEFAULT_SPLIT_RATIOS,
        strict_filter_invalid: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split=split,
            max_prompt_length=max_prompt_length,
            split_seed=split_seed,
            split_ratios=split_ratios,
            strict_filter_invalid=strict_filter_invalid,
        )
        self.use_sent = bool(use_sent)
        self.cache_path = cache_path
        self.sent_config = sent_config or SENTConfig()
        self.num_stages = max(1, int(num_stages))
        tokenizer_model_id = getattr(tokenizer, "name_or_path", None)
        self.model_id = (
            model_id
            if model_id is not None
            else tokenizer_model_id if isinstance(tokenizer_model_id, str) else None
        )
        self.sorted_indices: List[int] = []
        self.entropies: List[float] = []
        self.current_stage = 0
        self.current_stage_indices: List[int] = []

        if not self.use_sent:
            return
        if split != "train":
            raise ValueError("SENTMathDataset only supports SENT ordering on train split.")

        is_valid, msg = _validate_cache(
            self.cache_path,
            self.sent_config,
            tokenizer=self.tokenizer,
            max_prompt_length=self.max_prompt_length,
            model_id=self.model_id,
            dataset_name=self.dataset_name,
            split=self.split,
            split_seed=self.split_seed,
            split_ratios=self.split_ratios,
        )
        if not is_valid:
            raise ValueError(
                f"SENT cache invalid: {msg}. Regenerate cache for "
                f"{self.dataset_name}/{self.split} or disable SENT."
            )

        if self.cache_path.endswith(".json"):
            with open(self.cache_path, "r") as f:
                cache_data = json.load(f)
        else:
            cache_data = torch.load(self.cache_path, weights_only=False)
        self.sorted_indices = cache_data.get("indices", [])
        self.entropies = cache_data.get("entropies", [])
        if any(
            not isinstance(idx, int) or idx < 0 or idx >= len(self.rows)
            for idx in self.sorted_indices
        ):
            raise ValueError(
                "SENT cache indices must be positional indices inside the normalized "
                "training split. Regenerate the cache for the current split metadata."
            )

        self._compute_stage_boundaries()
        self.set_stage(1)
        logger.info("[SENT] Curriculum Learning enabled for %s", self.dataset_name)
        logger.info("[SENT] Cache loaded from: %s", self.cache_path)
        logger.info("[SENT] Total sorted samples: %d", len(self.sorted_indices))

    def _compute_stage_boundaries(self):
        total = len(self.sorted_indices)
        stage_size = total // self.num_stages
        self.stage_boundaries = [i * stage_size for i in range(self.num_stages)] + [
            total
        ]

    def set_stage(self, stage_idx: int):
        if stage_idx < 1 or stage_idx > self.num_stages:
            raise ValueError(f"Stage must be 1-{self.num_stages}, got {stage_idx}")
        self.current_stage = stage_idx
        start = self.stage_boundaries[stage_idx - 1]
        end = self.stage_boundaries[stage_idx]
        self.current_stage_indices = list(range(start, end))

    def get_stage_info(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "num_stages": self.num_stages,
            "stage_start_idx": self.stage_boundaries[self.current_stage - 1],
            "stage_end_idx": self.stage_boundaries[self.current_stage],
            "total_samples": len(self.sorted_indices),
        }

    @staticmethod
    def _sent_rank_fraction(rank: int, total_size: int) -> float:
        if total_size <= 1:
            return 0.0
        return float(rank) / float(total_size - 1)

    def __len__(self) -> int:
        if self.use_sent:
            return len(self.current_stage_indices)
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.use_sent:
            sent_rank = self.current_stage_indices[idx]
            actual_idx = self.sorted_indices[sent_rank]
            item = super().__getitem__(actual_idx)
            item["sent_rank"] = sent_rank
            item["sent_rank_fraction"] = self._sent_rank_fraction(
                sent_rank, len(self.sorted_indices)
            )
            item["sent_rank_total"] = len(self.sorted_indices)
            return item
        return super().__getitem__(idx)


def resolve_sent_cache_path(dataset_name: Optional[str], cache_path: str) -> str:
    """Use dataset-specific SENT cache paths when the default GSM8K path is unchanged."""
    if dataset_name is None:
        return cache_path
    dataset_name = canonical_dataset_name(dataset_name)
    if dataset_name == "gsm8k":
        return cache_path
    if cache_path == "data/cache/gsm8k_sent_sorted.pt":
        return f"data/cache/{safe_dataset_cache_stem(dataset_name)}_sent_sorted.pt"
    return cache_path


def create_grpo_dataloader(
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    max_prompt_length: int = 512,
    shuffle: bool = False,
    use_sent: bool = True,
    sent_config: Optional[SENTConfig] = None,
    num_stages: int = 1,
    cache_path: str = "data/cache/gsm8k_sent_sorted.pt",
    num_workers: Optional[int] = None,
    prefetch_factor: int = 2,
    model_id: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    dataset_name: Optional[str] = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_ratios: object = DEFAULT_SPLIT_RATIOS,
    strict_filter_invalid: bool = False,
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
    tokenizer_model_id = getattr(tokenizer, "name_or_path", None)
    resolved_model_id = (
        model_id
        if model_id is not None
        else tokenizer_model_id if isinstance(tokenizer_model_id, str) else None
    )

    dataset_name = canonical_dataset_name(dataset_name) if dataset_name is not None else None
    cache_path = resolve_sent_cache_path(dataset_name, cache_path)
    requested_use_sent = use_sent
    if split != "train" and use_sent:
        logger.info("[SENT] Disabled for %s split; SENT ordering is train-only.", split)
        use_sent = False

    if dataset_name is not None:
        if use_sent:
            is_valid, msg = _validate_cache(
                cache_path,
                sent_config,
                tokenizer=tokenizer,
                max_prompt_length=max_prompt_length,
                model_id=resolved_model_id,
                dataset_name=dataset_name,
                split=split,
                split_seed=split_seed,
                split_ratios=split_ratios,
            )
            if is_valid:
                dataset = SENTMathDataset(
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    split=split,
                    max_prompt_length=max_prompt_length,
                    use_sent=True,
                    cache_path=cache_path,
                    sent_config=sent_config,
                    num_stages=num_stages,
                    model_id=resolved_model_id,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    strict_filter_invalid=strict_filter_invalid,
                )
            else:
                if not os.path.exists(cache_path):
                    logger.warning(
                        "SENT cache unavailable (%s); falling back to standard %s ordering.",
                        msg,
                        dataset_name,
                    )
                    dataset = GRPOMathDataset(
                        tokenizer=tokenizer,
                        dataset_name=dataset_name,
                        split=split,
                        max_prompt_length=max_prompt_length,
                        split_seed=split_seed,
                        split_ratios=split_ratios,
                        strict_filter_invalid=strict_filter_invalid,
                    )
                    use_sent = False
                else:
                    raise ValueError(
                        f"SENT cache invalid: {msg}. Regenerate the cache or disable SENT explicitly."
                    )
        else:
            dataset = GRPOMathDataset(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                split=split,
                max_prompt_length=max_prompt_length,
                split_seed=split_seed,
                split_ratios=split_ratios,
                strict_filter_invalid=strict_filter_invalid,
            )
    elif use_sent:
        is_valid, msg = _validate_cache(
            cache_path,
            sent_config,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            model_id=resolved_model_id,
        )
        if is_valid:
            dataset = SENTGSM8KDataset(
                tokenizer=tokenizer,
                split=split,
                max_prompt_length=max_prompt_length,
                use_sent=True,
                cache_path=cache_path,
                sent_config=sent_config,
                num_stages=num_stages,
                model_id=resolved_model_id,
            )
            if shuffle:
                logger.warning(
                    "Shuffle=True is not recommended with SENT (order matters for curriculum)"
                )
        else:
            if not os.path.exists(cache_path):
                logger.warning(
                    "SENT cache unavailable (%s); falling back to standard GSM8K ordering.",
                    msg,
                )
                dataset = GRPOGSM8KDataset(
                    tokenizer=tokenizer, split=split, max_prompt_length=max_prompt_length
                )
                use_sent = False
            else:
                raise ValueError(
                    f"SENT cache invalid: {msg}. "
                    "Regenerate the cache or disable SENT explicitly."
                )
    else:
        dataset = GRPOGSM8KDataset(
            tokenizer=tokenizer, split=split, max_prompt_length=max_prompt_length
        )

    def grpo_collate(batch):
        if dataset_name is not None:
            return collate_grpo_math_batch(batch, tokenizer)

        # Determine max length in the batch
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_size = len(batch)

        # Get padding strategy
        # Get padding strategy with robust fallback for Mock tokenizers
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None or not isinstance(pad_token_id, int):
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None or not isinstance(pad_token_id, int):
            pad_token_id = 0
        padding_side = getattr(tokenizer, "padding_side", "right")

        # Pre-allocate tensors
        input_ids_padded = torch.full(
            (batch_size, max_len), pad_token_id, dtype=torch.long
        )
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.long)

        # Fill tensors
        for i, item in enumerate(batch):
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            seq_len = len(input_ids)

            if padding_side == "left":
                input_ids_padded[i, -seq_len:] = torch.tensor(
                    input_ids, dtype=torch.long
                )
                attention_mask_padded[i, -seq_len:] = torch.tensor(
                    attention_mask, dtype=torch.long
                )
            else:
                input_ids_padded[i, :seq_len] = torch.tensor(
                    input_ids, dtype=torch.long
                )
                attention_mask_padded[i, :seq_len] = torch.tensor(
                    attention_mask, dtype=torch.long
                )

        collated = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch],
        }
        sent_rank_values = [item.get("sent_rank") for item in batch]
        sent_rank_fraction_values = [item.get("sent_rank_fraction") for item in batch]
        sent_rank_totals = [item.get("sent_rank_total") for item in batch]
        if (
            all(value is not None for value in sent_rank_values)
            and all(value is not None for value in sent_rank_fraction_values)
            and all(value is not None for value in sent_rank_totals)
        ):
            collated["sent_rank"] = torch.tensor(sent_rank_values, dtype=torch.long)
            collated["sent_rank_fraction"] = torch.tensor(
                sent_rank_fraction_values, dtype=torch.float32
            )
            collated["sent_rank_total"] = int(sent_rank_totals[0])

        return collated

    worker_count: int = 4
    if num_workers is not None:
        worker_count = max(0, int(num_workers))
    if worker_count > 0:
        try:
            import multiprocessing as mp

            start_method = mp.get_start_method(allow_none=True)
            if start_method in {"forkserver", "spawn"}:
                worker_count = 0
            else:
                if not torch.cuda.is_available():
                    worker_count = 0
        except RuntimeError:
            worker_count = 0
    if worker_count > 0 and "unittest.mock" in type(tokenizer).__module__:
        worker_count = 0

    effective_shuffle = shuffle
    if requested_use_sent and not use_sent:
        effective_shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        collate_fn=grpo_collate,
        generator=generator,
        pin_memory=True,
        num_workers=worker_count,
        prefetch_factor=prefetch_factor if worker_count > 0 else None,
    )

    dataloader.uses_sent_curriculum = bool(
        use_sent and getattr(dataset, "use_sent", False) and hasattr(dataset, "set_stage")
    )
    dataloader.sent_num_stages = (
        int(getattr(dataset, "num_stages", 1)) if dataloader.uses_sent_curriculum else 1
    )
    dataloader.dataset_name = dataset_name or "gsm8k"
    dataloader.split = split

    return dataloader
