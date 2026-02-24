"""
Manual LoRA implementation for 4-bit quantized models.
Implements low-rank adaptation without using PEFT library.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Protocol, cast

import torch
import torch.nn as nn
import bitsandbytes as bnb
from src.triton_kernels import lora_fused_forward
from src.utils.logging_utils import get_logger


logger = get_logger("core.lora")


class _Linear4bitLike(Protocol):
    weight: torch.nn.Parameter
    in_features: int
    out_features: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def _get_bnb_linear4bit() -> Any:
    nn_namespace = getattr(bnb, "nn", None)
    return getattr(nn_namespace, "Linear4bit", None)


def _is_linear4bit(module: nn.Module) -> bool:
    linear4bit = _get_bnb_linear4bit()
    return linear4bit is not None and isinstance(module, linear4bit)


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
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_triton = use_triton
        self.prefer_base_layer = prefer_base_layer

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False

        # Get dimensions
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Create LoRA matrices A and B
        # A: input_dim -> rank
        # B: rank -> output_dim
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        # A: Kaiming uniform (He initialization)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        # B: Zero initialization (start with identity behavior)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base layer and LoRA adaptation.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        if self.use_triton:
            try:
                return cast(
                    torch.Tensor,
                    lora_fused_forward(
                        x,
                        lora_layer=self,
                        prefer_base_layer=self.prefer_base_layer,
                    ),
                )
            except (ImportError, RuntimeError, ValueError):
                pass

        try:
            base_output = cast(Callable[[torch.Tensor], torch.Tensor], self.base_layer)(x)
            if base_output.requires_grad:
                base_output = base_output.clone()

            x_adapt = x.to(self.lora_A.weight.dtype)
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x_adapt)))
            lora_output = lora_output * self.scaling

            if lora_output.dtype != base_output.dtype:
                lora_output = lora_output.to(base_output.dtype)

            return base_output + lora_output
        except Exception:
            if not self.use_triton:
                raise
            return cast(
                torch.Tensor,
                lora_fused_forward(
                    x,
                    lora_layer=self,
                ),
            )

    def merge_weights(self) -> torch.Tensor:
        """
        Compute the effective weight matrix W = W_base + B*A*scaling.
        Note: This dequantizes the base weights for merging.

        Returns:
            Merged weight matrix
        """
        # Get base weight (dequantized)
        base_weight = _dequantize_4bit(self.base_layer.weight)

        # Compute LoRA weight
        lora_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling

        return base_weight + lora_weight


def inject_lora_layers(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    use_triton: bool = True,
    prefer_base_layer: bool = False,
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

        if module_short_name in target_modules and _is_linear4bit(cast(nn.Module, module)):
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
