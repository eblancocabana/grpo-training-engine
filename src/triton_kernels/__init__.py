from __future__ import annotations

# pyright: reportMissingTypeStubs=false

from importlib import import_module
import math
from typing import Callable, Protocol, cast

import torch

from src.utils.logging_utils import get_logger


logger = get_logger("triton_kernels")


try:
    import triton as _triton  # type: ignore[import-not-found]

    _triton_available = True
    del _triton
except Exception:
    _triton_available = False

TRITON_AVAILABLE = _triton_available


_warned_no_triton = False
_triton_kernel_cache: dict[str, Callable[..., object] | None] = {}


class _GeneratorLike(Protocol):
    def generate(
        self,
        *,
        input_ids: object,
        attention_mask: object | None = None,
        max_new_tokens: int = 128,
        **kwargs: object,
    ) -> object:
        ...


class _EntropyCalculatorLike(Protocol):
    def calculate_entropy(self, logits: object) -> object:
        ...

    def create_mask(self, entropy: object, attention_mask: object | None) -> object:
        ...


class _GrpoTrainerLike(Protocol):
    def compute_grpo_loss(self, *args: object, **kwargs: object) -> object:
        ...


class _ModuleLike(Protocol):
    def __call__(self, x: object) -> object:
        ...


class _WeightedModuleLike(_ModuleLike, Protocol):
    weight: _TensorLike


class _TensorLike(Protocol):
    requires_grad: bool
    dtype: object

    def clone(self) -> object:
        ...

    def to(self, dtype: object) -> object:
        ...

    def __mul__(self, other: float) -> object:
        ...

    def __add__(self, other: object) -> object:
        ...


def _warn_fallback(kernel_name: str) -> None:
    global _warned_no_triton
    if _warned_no_triton:
        return
    if TRITON_AVAILABLE:
        logger.warning(
            "Triton kernel not found; falling back to PyTorch for %s.", kernel_name
        )
    else:
        logger.warning(
            "Triton not available; falling back to PyTorch for %s.", kernel_name
        )
    _warned_no_triton = True


def _get_triton_kernel(name: str) -> Callable[..., object] | None:
    if not TRITON_AVAILABLE:
        return None
    if name in _triton_kernel_cache:
        return _triton_kernel_cache[name]

    module_candidates = [
        "src.triton_kernels.kernels",
    ]
    explicit_modules = {
        "paged_kv_decode": "src.triton_kernels.paged_kv",
        "fused_grpo_loss": "src.triton_kernels.grpo_loss",
        "fused_entropy_mask": "src.triton_kernels.entropy_mask",
        "lora_fused_forward": "src.triton_kernels.lora_forward",
    }
    if name in explicit_modules:
        module_candidates.append(explicit_modules[name])
    module_candidates.append(f"src.triton_kernels.{name}")
    kernel = None
    for module_name in module_candidates:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        kernel = getattr(module, name, None)
        if kernel is not None:
            break

    _triton_kernel_cache[name] = kernel
    return kernel


def _paged_kv_decode_torch(
    model: object,
    input_ids: object,
    attention_mask: object | None = None,
    max_new_tokens: int = 128,
    **kwargs: object,
) -> object:
    model_typed = cast(_GeneratorLike, model)
    return model_typed.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )


def _fused_grpo_loss_torch(*args: object, **kwargs: object) -> object:
    from src.grpo.algorithm import GRPOTrainer

    trainer = kwargs.pop("trainer", None)
    if trainer is None:
        trainer = GRPOTrainer(
            clip_epsilon=cast(float, kwargs.pop("clip_epsilon", 0.2)),
            epsilon_high=cast(float, kwargs.pop("epsilon_high", 0.3)),
            delta=cast(float, kwargs.pop("delta", 1.5)),
            kl_coef=cast(float, kwargs.pop("kl_coef", 0.1)),
            group_size=cast(int, kwargs.pop("group_size", 4)),
            use_kl=cast(bool, kwargs.pop("use_kl", False)),
        )
    trainer_typed = cast(_GrpoTrainerLike, trainer)
    return trainer_typed.compute_grpo_loss(*args, **kwargs)


def _fused_entropy_mask_torch(
    logits: object,
    attention_mask: object | None = None,
    **kwargs: object,
) -> tuple[object, object]:
    from src.selective.entropy_mask import EntropyCalculator

    calculator = kwargs.pop("entropy_calculator", None)
    if calculator is None:
        calculator = EntropyCalculator(
            threshold=cast(float | None, kwargs.pop("threshold", None)),
            percentile=cast(float, kwargs.pop("percentile", 0.5)),
            min_tokens=cast(int, kwargs.pop("min_tokens", 10)),
        )
    calculator_typed = cast(_EntropyCalculatorLike, calculator)
    entropy = calculator_typed.calculate_entropy(logits)
    mask = calculator_typed.create_mask(entropy, attention_mask)
    return entropy, mask


def _lora_fused_forward_torch(
    x: object,
    base_layer: object | None = None,
    lora_A: object | None = None,
    lora_B: object | None = None,
    scaling: float | None = None,
    dropout: object | None = None,
    lora_layer: object | None = None,
) -> object:
    if lora_layer is not None:
        lora_layer_typed = cast(_ModuleLike, lora_layer)
        return lora_layer_typed(x)
    if base_layer is None or lora_A is None or lora_B is None or scaling is None:
        raise ValueError(
            "Provide lora_layer or base_layer/lora_A/lora_B/scaling for fallback."
        )
    base_layer_typed = cast(_ModuleLike, base_layer)
    lora_a_typed = cast(_WeightedModuleLike, lora_A)
    lora_b_typed = cast(_WeightedModuleLike, lora_B)
    base_output = cast(_TensorLike, base_layer_typed(x))
    if base_output.requires_grad:
        base_output = cast(_TensorLike, base_output.clone())
    if dropout is not None:
        dropout_typed = cast(_ModuleLike, dropout)
        x = dropout_typed(x)
    x_adapt = cast(_TensorLike, cast(_TensorLike, x).to(lora_a_typed.weight.dtype))
    lora_output = cast(
        _TensorLike,
        cast(_TensorLike, lora_b_typed(lora_a_typed(x_adapt))) * scaling,
    )
    if lora_output.dtype != base_output.dtype:
        lora_output = cast(_TensorLike, lora_output.to(base_output.dtype))
    return cast(_TensorLike, base_output + lora_output)


def paged_kv_decode(
    model: object,
    input_ids: object,
    attention_mask: object | None = None,
    max_new_tokens: int = 128,
    **kwargs: object,
) -> object:
    kernel = _get_triton_kernel("paged_kv_decode")
    input_ids_t = cast(torch.Tensor, input_ids)
    attention_mask_t = (
        cast(torch.Tensor, attention_mask) if attention_mask is not None else None
    )

    def _getattr(obj: object, name: str) -> object | None:
        return cast(object | None, getattr(obj, name, None))

    if kernel is not None:
        low_level_args = [
            "k_cache",
            "v_cache",
            "block_tables",
            "context_lens",
            "qkv_proj_fn",
            "logits_fn",
        ]
        has_low_level = all(arg in kwargs for arg in low_level_args)
        if has_low_level:
            return kernel(
                input_ids_t,
                k_cache=kwargs["k_cache"],
                v_cache=kwargs["v_cache"],
                block_tables=kwargs["block_tables"],
                context_lens=kwargs["context_lens"],
                qkv_proj_fn=kwargs["qkv_proj_fn"],
                logits_fn=kwargs["logits_fn"],
                max_new_tokens=max_new_tokens,
                do_sample=cast(bool, kwargs.pop("do_sample", True)),
                temperature=cast(float, kwargs.pop("temperature", 1.0)),
                top_p=cast(float, kwargs.pop("top_p", 1.0)),
                pad_token_id=cast(int | None, kwargs.pop("pad_token_id", None)),
                eos_token_id=cast(int | None, kwargs.pop("eos_token_id", None)),
                attention_mask=attention_mask_t,
                use_cache=cast(bool | None, kwargs.pop("use_cache", None)),
                **kwargs,
            )

        from src.triton_kernels.paged_kv import paged_kv_decode_model

        if attention_mask_t is None:
            prompt_lens = torch.full(
                (input_ids_t.shape[0],),
                input_ids_t.shape[1],
                device=input_ids_t.device,
                dtype=torch.int32,
            )
        else:
            prompt_lens = attention_mask_t.sum(dim=-1).to(dtype=torch.int32)

        max_prompt_len = int(prompt_lens.max().item()) if prompt_lens.numel() > 0 else 0
        if attention_mask_t is not None and input_ids_t.shape[1] > max_prompt_len:
            input_ids_t = input_ids_t[:, -max_prompt_len:]
            attention_mask_t = attention_mask_t[:, -max_prompt_len:]

        if attention_mask_t is None:
            attention_mask_t = torch.ones_like(input_ids_t, dtype=torch.long)
        block_size = int(cast(int, kwargs.pop("block_size", 16)))
        return paged_kv_decode_model(
            model=cast(torch.nn.Module, model),
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            do_sample=cast(bool, kwargs.pop("do_sample", True)),
            temperature=cast(float, kwargs.pop("temperature", 1.0)),
            top_p=cast(float, kwargs.pop("top_p", 1.0)),
            pad_token_id=cast(int, kwargs.pop("pad_token_id", 0)),
            eos_token_id=cast(int | None, kwargs.pop("eos_token_id", None)),
            seed=cast(int | None, kwargs.pop("seed", None)),
        )
    raise ImportError("Triton is not available for paged_kv_decode.")


def fused_grpo_loss(*args: object, **kwargs: object) -> object:
    kernel = _get_triton_kernel("fused_grpo_loss")
    if kernel is not None:
        return kernel(*args, **kwargs)
    _warn_fallback("fused_grpo_loss")
    return _fused_grpo_loss_torch(*args, **kwargs)


def fused_entropy_mask(
    logits: object,
    attention_mask: object | None = None,
    **kwargs: object,
) -> tuple[object, object]:
    kernel = _get_triton_kernel("fused_entropy_mask")
    if kernel is not None:
        return cast(
            tuple[object, object],
            kernel(logits, attention_mask=attention_mask, **kwargs),
        )
    _warn_fallback("fused_entropy_mask")
    return _fused_entropy_mask_torch(logits, attention_mask=attention_mask, **kwargs)


def lora_fused_forward(
    x: object,
    base_layer: object | None = None,
    lora_A: object | None = None,
    lora_B: object | None = None,
    scaling: float | None = None,
    dropout: object | None = None,
    lora_layer: object | None = None,
    prefer_base_layer: bool = False,
) -> object:
    kernel = _get_triton_kernel("lora_fused_forward")
    if kernel is not None:
        return kernel(
            x,
            base_layer=base_layer,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            dropout=dropout,
            lora_layer=lora_layer,
            prefer_base_layer=prefer_base_layer,
        )
    _warn_fallback("lora_fused_forward")
    return _lora_fused_forward_torch(
        x,
        base_layer=base_layer,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        dropout=dropout,
        lora_layer=lora_layer,
    )


__all__ = [
    "TRITON_AVAILABLE",
    "paged_kv_decode",
    "fused_grpo_loss",
    "fused_entropy_mask",
    "lora_fused_forward",
]
