"""
GSM8K Dataset Loader for GRPO Training.
Handles loading, formatting, and batching of GSM8K math problems.
"""
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any
import logging

from ..utils.logging_utils import get_logger

logger = get_logger("data.gsm8k")


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


def create_grpo_dataloader(
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    max_prompt_length: int = 512,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader for GRPO training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split
        batch_size: Batch size (number of unique prompts)
        max_prompt_length: Max prompt length
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader instance
    """
    dataset = GRPOGSM8KDataset(
        tokenizer=tokenizer,
        split=split,
        max_prompt_length=max_prompt_length
    )
    
    def grpo_collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        
        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="longest",
            return_tensors="pt"
        )
        input_ids_padded = padded["input_ids"]
        attention_mask_padded = padded["attention_mask"]
        
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
