"""
Manual LoRA implementation for 4-bit quantized models.
Implements low-rank adaptation without using PEFT library.
"""

import torch
import torch.nn as nn
import bitsandbytes as bnb
from typing import List
from src.utils.logging_utils import get_logger

logger = get_logger("core.lora")


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
        base_layer: bnb.nn.Linear4bit,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

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
        # Base layer output (quantized, frozen)
        # Clone needed for autograd compatibility with quantized layers
        base_output = self.base_layer(x)
        if base_output.requires_grad:
            base_output = base_output.clone()

        # LoRA path (trainable)
        # Cast to adapter dtype to avoid type mismatches
        x_adapt = x.to(self.lora_A.weight.dtype)

        # LoRA computation: B(A(x)) * scaling
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x_adapt)))
        lora_output = lora_output * self.scaling

        # Cast back to base output dtype if needed
        if lora_output.dtype != base_output.dtype:
            lora_output = lora_output.to(base_output.dtype)

        return base_output + lora_output

    def merge_weights(self) -> torch.Tensor:
        """
        Compute the effective weight matrix W = W_base + B*A*scaling.
        Note: This dequantizes the base weights for merging.

        Returns:
            Merged weight matrix
        """
        # Get base weight (dequantized)
        base_weight = bnb.functional.dequantize_4bit(
            self.base_layer.weight, self.base_layer.weight.quant_state
        )

        # Compute LoRA weight
        lora_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling

        return base_weight + lora_weight


def inject_lora_layers(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
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

        if module_short_name in target_modules and isinstance(
            module, bnb.nn.Linear4bit
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
                base_layer=module, rank=rank, alpha=alpha, dropout=dropout
            )

            # Move to same device/dtype as base layer
            device = module.weight.device
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
