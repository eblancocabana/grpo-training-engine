"""
Manual LoRA implementation for 4-bit quantized models.
Implements low-rank adaptation without using PEFT library.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, List, Protocol, cast

import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
from src.triton_kernels import lora_fused_forward
from src.utils.logging_utils import get_logger


logger = get_logger("core.lora")


class _Linear4bitLike(Protocol):
    weight: torch.nn.Parameter
    in_features: int
    out_features: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class _LoRALinearLike(Protocol):
    weight: torch.nn.Parameter

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def _get_bnb_linear4bit() -> Any:
    if bnb is None:
        return None
    nn_namespace = getattr(bnb, "nn", None)
    return getattr(nn_namespace, "Linear4bit", None)


def _get_bnb_linear8bit() -> Any:
    if bnb is None:
        return None
    nn_namespace = getattr(bnb, "nn", None)
    return getattr(nn_namespace, "Linear8bitLt", None)


def _is_linear4bit(module: object) -> bool:
    linear4bit = _get_bnb_linear4bit()
    return linear4bit is not None and isinstance(module, linear4bit)


def _is_supported_base_layer(module: nn.Module) -> bool:
    if _is_linear4bit(module):
        return True
    if isinstance(module, nn.Linear):
        return True
    return False


def _get_lora_compute_dtype(device: torch.device | None = None) -> torch.dtype:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _create_lora_linear(
    in_features: int,
    out_features: int,
    adapter_quantization: str,
    compute_dtype: torch.dtype,
) -> tuple[nn.Module, bool]:
    def _dense_fallback(reason: str) -> tuple[nn.Module, bool]:
        logger.warning("%s Falling back to dense LoRA adapter.", reason)
        linear = nn.Linear(in_features, out_features, bias=False)
        return linear.to(dtype=compute_dtype), False

    def _adapter_is_trainable(module: nn.Module) -> bool:
        params = list(module.parameters())
        if not params:
            return False
        return any(param.requires_grad and torch.is_floating_point(param) for param in params)

    quantization = adapter_quantization.lower()
    if quantization == "8bit":
        linear8bit = _get_bnb_linear8bit()
        if linear8bit is not None:
            try:
                layer = linear8bit(in_features, out_features, bias=False)
                if _adapter_is_trainable(layer):
                    return layer, True
                return _dense_fallback("8-bit LoRA adapter is not trainable in this environment.")
            except Exception as exc:
                return _dense_fallback(f"8-bit LoRA adapter unavailable ({exc}).")
        return _dense_fallback("bitsandbytes unavailable for 8-bit LoRA adapters.")
    elif quantization == "4bit":
        linear4bit = _get_bnb_linear4bit()
        if linear4bit is not None:
            try:
                layer = linear4bit(in_features, out_features, bias=False)
                if _adapter_is_trainable(layer):
                    return layer, True
                return _dense_fallback("4-bit LoRA adapter is not trainable in this environment.")
            except Exception as exc:
                return _dense_fallback(f"4-bit LoRA adapter unavailable ({exc}).")
        return _dense_fallback("bitsandbytes unavailable for 4-bit LoRA adapters.")
    elif quantization != "none":
        message = (
            f"Unsupported LoRA adapter quantization: {adapter_quantization}. "
            "Expected '8bit', '4bit', or 'none'."
        )
        raise ValueError(message)

    linear = nn.Linear(in_features, out_features, bias=False)
    return linear.to(dtype=compute_dtype), False


def _dequantize_4bit(weight: torch.Tensor) -> torch.Tensor:
    functional = getattr(bnb, "functional", None)
    if functional is None or not hasattr(functional, "dequantize_4bit"):
        raise RuntimeError("bitsandbytes.functional.dequantize_4bit is unavailable.")
    quant_state = cast(object | None, getattr(weight, "quant_state", None))
    if quant_state is None:
        raise RuntimeError("Quantization state missing from weight tensor.")
    dequant = cast(Any, functional).dequantize_4bit
    return cast(torch.Tensor, dequant(weight, quant_state))


class ManualLoRALayer(nn.Module):
    """
    Manual LoRA layer that wraps a Linear4bit layer.

    Forward pass: Y = W_4bit(x) + B(A(x)) * (alpha / rank)

    Args:
        base_layer: The Linear4bit layer to wrap
        rank: LoRA rank (r)
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        base_layer: _Linear4bitLike,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        use_triton: bool = True,
        prefer_base_layer: bool = False,
        adapter_quantization: str = "none",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_triton = use_triton
        self.prefer_base_layer = prefer_base_layer
        self.adapter_quantization = adapter_quantization.lower()
        self.adapters_enabled = True

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        base_bias = getattr(self.base_layer, "bias", None)
        if base_bias is not None:
            base_bias.requires_grad = False

        # Get dimensions
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Create LoRA matrices A and B
        # A: input_dim -> rank
        # B: rank -> output_dim
        base_device = cast(torch.Tensor, self.base_layer.weight).device
        self.lora_compute_dtype = _get_lora_compute_dtype(base_device)
        lora_a, _ = _create_lora_linear(
            in_features, rank, adapter_quantization, self.lora_compute_dtype
        )
        lora_b, _ = _create_lora_linear(
            rank, out_features, adapter_quantization, self.lora_compute_dtype
        )
        self.lora_A = cast(_LoRALinearLike, cast(object, lora_a))
        self.lora_B = cast(_LoRALinearLike, cast(object, lora_b))

        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        # A: Kaiming uniform (He initialization)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        # B: Zero initialization (start with identity behavior)
        nn.init.zeros_(self.lora_B.weight)

    @staticmethod
    def _run_projection(module: _LoRALinearLike | _Linear4bitLike, x: torch.Tensor) -> torch.Tensor:
        module_weight = cast(torch.Tensor, module.weight)
        proj_input = x
        if (
            not _is_linear4bit(cast(object, module))
            and torch.is_floating_point(proj_input)
            and proj_input.dtype != module_weight.dtype
        ):
            proj_input = proj_input.to(module_weight.dtype)
        return cast(Callable[[torch.Tensor], torch.Tensor], module)(proj_input)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self._run_projection(self.base_layer, x)
        if not self.adapters_enabled:
            return base_output
        if base_output.requires_grad:
            base_output = base_output.clone()

        x_adapt = x.to(self.lora_compute_dtype)
        lora_hidden = self._run_projection(
            cast(_LoRALinearLike, self.lora_A), self.lora_dropout(x_adapt)
        )
        lora_output = self._run_projection(cast(_LoRALinearLike, self.lora_B), lora_hidden)
        lora_output = lora_output * self.scaling

        if lora_output.dtype != base_output.dtype:
            lora_output = lora_output.to(base_output.dtype)

        return base_output + lora_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base layer and LoRA adaptation.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        if (
            self.use_triton
            and self.adapters_enabled
            and not self.prefer_base_layer
            and self.adapter_quantization == "none"
        ):
            try:
                return cast(
                    torch.Tensor,
                    lora_fused_forward(
                        x,
                        base_layer=self.base_layer,
                        lora_A=self.lora_A,
                        lora_B=self.lora_B,
                        scaling=self.scaling,
                        dropout=self.lora_dropout,
                        prefer_base_layer=self.prefer_base_layer,
                    ),
                )
            except (ImportError, RuntimeError, ValueError):
                pass

        return self._forward_torch(x)

    def merge_weights(self) -> torch.Tensor:
        """
        Compute the effective weight matrix W = W_base + B*A*scaling.
        Note: This dequantizes the base weights for merging.

        Returns:
            Merged weight matrix
        """
        if _is_linear4bit(self.base_layer):
            base_weight = _dequantize_4bit(self.base_layer.weight)
        else:
            base_weight = self.base_layer.weight

        # Compute LoRA weight
        lora_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling
        if lora_weight.dtype != base_weight.dtype:
            lora_weight = lora_weight.to(base_weight.dtype)

        return base_weight + lora_weight


def inject_lora_layers(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    use_triton: bool = True,
    prefer_base_layer: bool = False,
    adapter_quantization: str = "none",
    verbose: bool = True,
) -> int:
    """
    Inject ManualLoRALayer into target modules of a 4-bit quantized model.

    Args:
        model: The quantized model
        target_modules: List of module names to target (e.g., ["q_proj", "v_proj"])
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        verbose: Print injection progress

    Returns:
        Number of layers injected
    """
    injected_count = 0

    for name, module in model.named_modules():
        # Check if this is a target module
        module_short_name = name.split(".")[-1]

        if module_short_name in target_modules and _is_supported_base_layer(
            cast(nn.Module, module)
        ):
            # Navigate to parent module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Create LoRA wrapper
            lora_layer = ManualLoRALayer(
                base_layer=cast(_Linear4bitLike, module),
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_triton=use_triton,
                prefer_base_layer=prefer_base_layer,
                adapter_quantization=adapter_quantization,
            )

            # Move to same device/dtype as base layer
            device = cast(torch.device, cast(_Linear4bitLike, module).weight.device)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            lora_layer = lora_layer.to(device=device, dtype=dtype)

            # Replace the module
            setattr(parent, child_name, lora_layer)

            injected_count += 1
            if verbose:
                logger.debug("Injected into: %s", name)

    if verbose:
        logger.info("Total LoRA layers injected: %d", injected_count)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    return injected_count


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_params.append(param)
    return lora_params


def iter_lora_layers(model: nn.Module) -> Iterator[ManualLoRALayer]:
    """Yield all ManualLoRALayer instances in a model."""
    for module in model.modules():
        if isinstance(module, ManualLoRALayer):
            yield module


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable or disable LoRA adapters in-place."""
    for layer in iter_lora_layers(model):
        layer.adapters_enabled = enabled


@contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily disable LoRA adapters for base-policy forwards."""
    layers = list(iter_lora_layers(model))
    previous_states = [layer.adapters_enabled for layer in layers]
    try:
        for layer in layers:
            layer.adapters_enabled = False
        yield
    finally:
        for layer, was_enabled in zip(layers, previous_states):
            layer.adapters_enabled = was_enabled
