import torch
import random
from typing import Dict, Any
from tqdm import tqdm
import wandb

from src.utils.logging_utils import get_logger
from src.core.memory_manager import MemoryManager
from src.grpo.verifier import RuleBasedVerifier
from src.data.gsm8k_loader import GRPOGSM8KDataset

logger = get_logger("grpo.benchmark")


class GSM8KBenchmark:
    """
    Benchmark for GSM8K reasoning tasks during GRPO training.
    Evaluates Pass@1 accuracy, format compliance, and generation length.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        memory_manager: MemoryManager,
        dataset_split: str = "test",
        num_samples: int = 50,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_manager = memory_manager
        self.device = device
        self.num_samples = num_samples

        # Initialize verifier
        self.verifier = RuleBasedVerifier()

        # Load dataset
        logger.info(f"Loading GSM8K {dataset_split} split for benchmark...")
        full_dataset = GRPOGSM8KDataset(tokenizer=tokenizer, split=dataset_split)

        # Select fixed subset
        random.seed(42)
        if len(full_dataset) > num_samples:
            self.indices = random.sample(range(len(full_dataset)), num_samples)
        else:
            self.indices = list(range(len(full_dataset)))

        self.dataset = [full_dataset[i] for i in self.indices]

        logger.info(
            f"Initialized GSM8K Benchmark with {len(self.dataset)} samples from {dataset_split} split"
        )

    def run(self, step: int) -> Dict[str, float]:
        """
        Run benchmark evaluation.
        """
        logger.info(f"Running GSM8K Benchmark at step {step}...")

        # Switch to inference mode
        self.model.eval()
        self.memory_manager.optimize_for_inference()

        metrics = {
            "correct_count": 0,
            "format_compliant_count": 0,
            "total_len": 0,
            "think_len": 0,
        }

        samples_to_log = []

        pbar = tqdm(self.dataset, desc="Benchmarking", leave=False)

        for item in pbar:
            input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(self.device)
            attention_mask = (
                torch.tensor(item["attention_mask"]).unsqueeze(0).to(self.device)
            )
            ground_truth = item["answer"]
            question = item["question"]

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response_ids = generated_ids[0, input_ids.shape[1] :]
            full_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            reward, info = self.verifier.verify(full_text, ground_truth)

            is_correct = reward == 1.0
            metrics["correct_count"] += int(is_correct)
            metrics["total_len"] += len(generated_ids[0]) - len(input_ids[0])

            # Format check
            has_think = "<think>" in full_text and "</think>" in full_text
            has_answer = "<answer>" in full_text or "\\boxed{" in full_text
            format_ok = has_think and has_answer

            if format_ok:
                metrics["format_compliant_count"] += 1

            if len(samples_to_log) < 5:
                samples_to_log.append(
                    [
                        step,
                        question[:100],
                        full_text[-500:] if len(full_text) > 500 else full_text,
                        ground_truth,
                        is_correct,
                        format_ok,
                    ]
                )

            self.memory_manager.clear_cache()

        n = len(self.dataset)
        final_metrics = {
            "val/acc": metrics["correct_count"] / n,
            "val/format_compliance": metrics["format_compliant_count"] / n,
            "val/avg_len": metrics["total_len"] / n,
        }

        logger.info(f"Benchmark Results: Acc={final_metrics['val/acc']:.2f}")

        if wandb.run is not None:
            wandb.log(final_metrics, step=step)
            columns = [
                "Step",
                "Prompt",
                "Generated",
                "Ground Truth",
                "Correct",
                "Format OK",
            ]
            results_table = wandb.Table(columns=columns, data=samples_to_log)
            wandb.log({"val/samples": results_table}, step=step)

        self.model.train()
        self.memory_manager.optimize_for_training()

        return final_metrics
