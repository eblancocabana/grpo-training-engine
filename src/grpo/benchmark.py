import torch
import random
from contextlib import contextmanager
from typing import Callable, Dict, Any, Optional
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from src.utils.logging_utils import get_logger
from src.core.memory_manager import MemoryManager
from src.grpo.verifier import RuleBasedVerifier
from src.data.gsm8k_loader import GRPOGSM8KDataset
from src.data.math_dataset import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    GRPOMathDataset,
    supported_dataset_names,
)

logger = get_logger("grpo.benchmark")


@contextmanager
def _preserve_rng_state():
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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
        generate_fn: Optional[Callable[[torch.Tensor, torch.Tensor], list[str]]] = None,
        max_new_tokens: int = 512,
        max_prompt_length: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        dataset_name: str = "gsm8k",
        split_seed: int = DEFAULT_SPLIT_SEED,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        use_split_abstraction: bool = False,
        strict_filter_invalid: bool = False,
        eval_seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_manager = memory_manager
        self.device = device
        self.num_samples = num_samples
        self.generate_fn = generate_fn
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.eval_seed = int(eval_seed)

        # Initialize verifier
        self.verifier = RuleBasedVerifier()

        # Load dataset
        logger.info(
            "Loading %s %s split for benchmark...",
            dataset_name,
            dataset_split,
        )
        if use_split_abstraction or dataset_name != "gsm8k":
            full_dataset = GRPOMathDataset(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                split=dataset_split,
                max_prompt_length=max_prompt_length,
                split_seed=split_seed,
                split_ratios=split_ratios,
                strict_filter_invalid=strict_filter_invalid,
            )
        else:
            full_dataset = GRPOGSM8KDataset(
                tokenizer=tokenizer,
                split=dataset_split,
                max_prompt_length=max_prompt_length,
            )

        # Select fixed subset
        subset_rng = random.Random(self.eval_seed)
        if len(full_dataset) > num_samples:
            self.indices = subset_rng.sample(range(len(full_dataset)), num_samples)
        else:
            self.indices = list(range(len(full_dataset)))

        self.dataset = [full_dataset[i] for i in self.indices]

        logger.info(
            "Initialized %s Benchmark with %d samples from %s split",
            dataset_name,
            len(self.dataset),
            dataset_split,
        )

    def run(self, step: int) -> Dict[str, float]:
        """
        Run benchmark evaluation.
        """
        logger.info(
            "Running %s Benchmark at step %s (split=%s samples=%d do_sample=%s seed=%d max_response_length=%d)...",
            self.dataset_name,
            step,
            self.dataset_split,
            len(self.dataset),
            self.do_sample,
            self.eval_seed,
            self.max_new_tokens,
        )

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

        with _preserve_rng_state():
            for item in pbar:
                input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = (
                    torch.tensor(item["attention_mask"]).unsqueeze(0).to(self.device)
                )
                ground_truth = item["answer"]
                question = item["question"]

                if self.generate_fn is not None:
                    generated = self.generate_fn(input_ids, attention_mask)
                    if isinstance(generated, str):
                        full_text = generated
                    else:
                        if len(generated) != 1:
                            raise ValueError(
                                "Benchmark generate_fn must return exactly one response per prompt."
                            )
                        full_text = generated[0]
                    generated_len = len(
                        self.tokenizer(
                            full_text,
                            add_special_tokens=False,
                            truncation=False,
                        )["input_ids"]
                    )
                else:
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=self.do_sample,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    response_ids = generated_ids[0, input_ids.shape[1] :]
                    full_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    generated_len = len(response_ids)
                reward, info = self.verifier.verify(full_text, ground_truth)

                is_correct = reward == 1.0
                metrics["correct_count"] += int(is_correct)
                metrics["total_len"] += generated_len

                # Format check
                has_think = "</think>" in full_text
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
            "exact_answer_accuracy": metrics["correct_count"] / n,
            "avg_response_length": metrics["total_len"] / n,
            "eval_dataset_name": self.dataset_name,
            "eval_split": self.dataset_split,
            "eval_sample_count": n,
            "eval_do_sample": float(bool(self.do_sample)),
            "eval_seed": self.eval_seed,
            "max_response_length": self.max_new_tokens,
        }

        logger.info(
            "Benchmark Results [%s/%s]: Acc=%.2f",
            self.dataset_name,
            self.dataset_split,
            final_metrics["val/acc"],
        )

        if WANDB_AVAILABLE and wandb.run is not None:
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


class MathBenchmark(GSM8KBenchmark):
    """Split-aware benchmark for all supported math RL datasets."""

    def __init__(self, *args, **kwargs):
        dataset_name = kwargs.get("dataset_name", "gsm8k")
        if dataset_name not in supported_dataset_names():
            raise ValueError(
                f"Unsupported eval dataset_name='{dataset_name}'. "
                f"Allowed: {', '.join(supported_dataset_names())}"
            )
        kwargs["use_split_abstraction"] = True
        super().__init__(*args, **kwargs)
