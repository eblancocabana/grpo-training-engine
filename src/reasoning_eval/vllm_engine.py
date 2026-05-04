"""vLLM generation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from src.reasoning_eval.schema import Generation


@dataclass(frozen=True)
class GenerationProtocol:
    name: str
    temperature: float
    top_p: float
    n: int
    max_new_tokens: int
    seed: int


def make_sampling_params(SamplingParams: Any, protocol: GenerationProtocol) -> Any:
    kwargs = {
        "n": protocol.n,
        "temperature": protocol.temperature,
        "top_p": protocol.top_p,
        "max_tokens": protocol.max_new_tokens,
        "seed": protocol.seed,
    }
    if protocol.temperature <= 0:
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


class VLLMGenerator:
    """Thin wrapper that keeps one vLLM engine loaded for one base model."""

    def __init__(
        self,
        *,
        model_id: str,
        tokenizer: Any,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str,
        quantization: str | None,
        seed: int,
        enable_lora: bool,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.tokenizer = tokenizer
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "seed": seed,
            "trust_remote_code": True,
            "enable_lora": enable_lora,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
        }
        if quantization and quantization.lower() not in {"none", "null"}:
            kwargs["quantization"] = quantization
        self.llm = LLM(**kwargs)

    def lora_request(self, model_key: str, adapter_path: str | None, adapter_id: int) -> Any | None:
        if not adapter_path:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest(model_key, adapter_id, adapter_path)

    def generate(
        self,
        *,
        prompts: Sequence[str],
        model_key: str,
        dataset: str,
        protocol: GenerationProtocol,
        example_ids: Sequence[str],
        adapter_path: str | None,
        adapter_id: int,
    ) -> list[Generation]:
        sampling_params = make_sampling_params(self.SamplingParams, protocol)
        lora_request = self.lora_request(model_key, adapter_path, adapter_id)
        started = time.time()
        outputs = self.llm.generate(list(prompts), sampling_params, lora_request=lora_request)
        elapsed = time.time() - started
        per_prompt_latency = elapsed / max(1, len(prompts))
        generations: list[Generation] = []
        for prompt, example_id, request_output in zip(prompts, example_ids, outputs):
            for sample_index, completion in enumerate(request_output.outputs):
                token_ids = getattr(completion, "token_ids", None) or []
                finish_reason = getattr(completion, "finish_reason", None)
                generations.append(
                    Generation(
                        model_key=model_key,
                        dataset=dataset,
                        protocol=protocol.name,
                        example_id=example_id,
                        sample_index=sample_index,
                        prompt=prompt,
                        text=getattr(completion, "text", ""),
                        token_count=len(token_ids),
                        finish_reason=str(finish_reason) if finish_reason is not None else None,
                        latency_s=per_prompt_latency,
                    )
                )
        return generations


class MockGenerator:
    """Deterministic test generator used by smoke tests only."""

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer

    def generate(
        self,
        *,
        prompts: Sequence[str],
        model_key: str,
        dataset: str,
        protocol: GenerationProtocol,
        example_ids: Sequence[str],
        adapter_path: str | None = None,
        adapter_id: int = 0,
    ) -> list[Generation]:
        generations: list[Generation] = []
        for prompt, example_id in zip(prompts, example_ids):
            for sample_index in range(protocol.n):
                generations.append(
                    Generation(
                        model_key=model_key,
                        dataset=dataset,
                        protocol=protocol.name,
                        example_id=example_id,
                        sample_index=sample_index,
                        prompt=prompt,
                        text="Reasoning omitted. Final answer: \\boxed{42}",
                        token_count=8,
                        finish_reason="stop",
                        latency_s=0.001,
                    )
                )
        return generations

