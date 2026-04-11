"""
Memory Management for VRAM Optimization.
Handles gradient checkpointing, cache clearing, and memory monitoring.
"""

import torch
import gc
from typing import Any, Iterable, List, Tuple
from src.utils.logging_utils import get_logger, TRACE

logger = get_logger("core.memory_manager")


class MemoryManager:
    def __init__(
        self,
        device: str = "cuda",
        enable_gradient_checkpointing: bool = True,
        clear_cache_frequency: int = 10,
        memory_fraction_warning: float = 0.85,
        checkpointing_strategy: str = "all",
        checkpointing_layer_name_patterns: List[str] | None = None,
        checkpointing_layer_types: List[str] | None = None,
        checkpointing_vram_enable_threshold: float = 0.82,
        checkpointing_vram_disable_threshold: float = 0.72,
        checkpointing_update_interval_steps: int = 10,
    ):
        self.device = device
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.clear_cache_frequency = clear_cache_frequency
        self.memory_fraction_warning = memory_fraction_warning
        self.checkpointing_strategy = checkpointing_strategy
        self.checkpointing_layer_name_patterns = checkpointing_layer_name_patterns or []
        self.checkpointing_layer_types = checkpointing_layer_types or []
        self.checkpointing_vram_enable_threshold = checkpointing_vram_enable_threshold
        self.checkpointing_vram_disable_threshold = checkpointing_vram_disable_threshold
        self.checkpointing_update_interval_steps = max(
            1, checkpointing_update_interval_steps
        )
        self._checkpointing_global_enabled = False
        self._checkpointing_active = False
        self._checkpointing_last_update_step = -1
        self._input_grad_hook_handle: Any | None = None
        self._input_grad_helper_enabled = False
        self.step_count = 0

    def clear_cache(self, aggressive: bool = False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if aggressive:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def enable_checkpointing(self, model: torch.nn.Module):
        if not self.enable_gradient_checkpointing:
            self._apply_checkpointing(model, enable=False, reason="disabled")
            return

        strategy = (self.checkpointing_strategy or "all").lower()
        if strategy == "off":
            self._apply_checkpointing(model, enable=False, reason="off")
            return

        if strategy == "vram_auto":
            # Safety-first: start with checkpointing enabled, then disable only after
            # observed step peaks show enough VRAM headroom.
            self._apply_checkpointing(model, enable=True, selective=True)
            self._checkpointing_active = True
            self._checkpointing_last_update_step = 0
            self.reset_peak_stats()
            return

        if strategy == "subset":
            self._apply_checkpointing(model, enable=True, selective=True)
            self._checkpointing_active = True
            return

        if strategy == "all":
            self._apply_checkpointing(model, enable=True, selective=False)
            self._checkpointing_active = True
            return

        logger.warning(
            "Unknown checkpointing strategy '%s'; defaulting to 'all'.",
            self.checkpointing_strategy,
        )
        self._apply_checkpointing(model, enable=True, selective=False)
        self._checkpointing_active = True

    def maybe_update_checkpointing(
        self,
        model: torch.nn.Module,
        step: int,
        memory_stats: dict | None = None,
    ):
        if not self.enable_gradient_checkpointing:
            return

        if (self.checkpointing_strategy or "all").lower() != "vram_auto":
            return

        if (
            step - self._checkpointing_last_update_step
            < self.checkpointing_update_interval_steps
        ):
            return

        stats = memory_stats or self.get_peak_memory_stats()
        if "error" in stats:
            return

        usage = max(
            stats.get("usage_fraction", 0.0),
            stats.get("peak_usage_fraction", 0.0),
        )
        current_usage = stats.get("usage_fraction", usage)
        peak_usage = stats.get("peak_usage_fraction", usage)
        if (
            not self._checkpointing_active
            and usage >= self.checkpointing_vram_enable_threshold
        ):
            self._apply_checkpointing(model, enable=True, selective=True)
            self._checkpointing_active = True
        elif (
            self._checkpointing_active
            and current_usage <= self.checkpointing_vram_disable_threshold
            and peak_usage <= self.checkpointing_vram_disable_threshold
        ):
            self._apply_checkpointing(model, enable=False, reason="vram_auto")
            self._checkpointing_active = False

        self._checkpointing_last_update_step = step
        self.reset_peak_stats()

    def _apply_checkpointing(
        self,
        model: torch.nn.Module,
        enable: bool,
        selective: bool = False,
        reason: str | None = None,
    ):
        if enable:
            self._enable_checkpointing_hooks(model)

            if selective:
                checkpointable = list(self._iter_checkpointable_modules(model))
                selected = self._select_checkpointing_modules(checkpointable)
                self._apply_checkpointing_flags(checkpointable, enabled=False)
                self._apply_checkpointing_flags(selected, enabled=True)
                logger.info(
                    "Selective gradient checkpointing enabled (%d/%d modules).",
                    len(selected),
                    len(checkpointable),
                )
            else:
                checkpointable = list(self._iter_checkpointable_modules(model))
                self._apply_checkpointing_flags(checkpointable, enabled=True)
                logger.info("Gradient checkpointing enabled for all modules.")
        else:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            self._checkpointing_global_enabled = False
            if (
                hasattr(model, "disable_input_require_grads")
                and self._input_grad_helper_enabled
            ):
                model.disable_input_require_grads()
            self._input_grad_helper_enabled = False
            if self._input_grad_hook_handle is not None:
                self._input_grad_hook_handle.remove()
                self._input_grad_hook_handle = None
            checkpointable = list(self._iter_checkpointable_modules(model))
            self._apply_checkpointing_flags(checkpointable, enabled=False)
            if reason:
                logger.info("Gradient checkpointing disabled (%s).", reason)
            else:
                logger.info("Gradient checkpointing disabled.")

    def _enable_checkpointing_hooks(self, model: torch.nn.Module):
        if (
            hasattr(model, "gradient_checkpointing_enable")
            and not self._checkpointing_global_enabled
        ):
            model.gradient_checkpointing_enable()
            self._checkpointing_global_enabled = True

        # CRITICAL: Enable input gradients to support checkpointing with frozen base layers
        # This prevents "element 0 of tensors does not require grad" error
        if hasattr(model, "enable_input_require_grads"):
            if not self._input_grad_helper_enabled:
                model.enable_input_require_grads()
                self._input_grad_helper_enabled = True
        else:
            if self._input_grad_hook_handle is not None:
                return

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self._input_grad_hook_handle = model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    def _iter_checkpointable_modules(
        self, model: torch.nn.Module
    ) -> Iterable[Tuple[str, torch.nn.Module]]:
        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                yield name, module

    def _select_checkpointing_modules(
        self, modules: List[Tuple[str, torch.nn.Module]]
    ) -> List[Tuple[str, torch.nn.Module]]:
        patterns = self.checkpointing_layer_name_patterns
        types = self.checkpointing_layer_types
        if not patterns and not types:
            return modules

        selected: List[Tuple[str, torch.nn.Module]] = []
        for name, module in modules:
            name_match = any(pattern in name for pattern in patterns)
            type_match = module.__class__.__name__ in types
            if name_match or type_match:
                selected.append((name, module))

        if not selected:
            return modules

        return selected

    @staticmethod
    def _apply_checkpointing_flags(
        modules: List[Tuple[str, torch.nn.Module]], enabled: bool
    ):
        for _, module in modules:
            module.gradient_checkpointing = enabled

    def get_memory_stats(self) -> dict:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        device_id = 0 if self.device == "cuda" else int(self.device.split(":")[-1])

        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024**3)

        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        total = total_mem / (1024**3)
        actually_free = free_mem / (1024**3)

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "total_gb": round(total, 2),
            "free_gb": round(actually_free, 2),
            "usage_fraction": (total - actually_free) / total,
        }

    def get_peak_memory_stats(self) -> dict:
        stats = self.get_memory_stats()
        if "error" in stats:
            return stats

        if not torch.cuda.is_available():
            return stats

        device_id = 0 if self.device == "cuda" else int(self.device.split(":")[-1])
        total_mem = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        peak_allocated = torch.cuda.max_memory_allocated(device_id) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device_id) / (1024**3)
        stats["peak_allocated_gb"] = round(peak_allocated, 2)
        stats["peak_reserved_gb"] = round(peak_reserved, 2)
        stats["peak_usage_fraction"] = min(1.0, peak_reserved / max(total_mem, 1e-9))
        return stats

    def get_available_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        device_id = 0 if self.device == "cuda" else int(self.device.split(":")[-1])
        free_mem, _ = torch.cuda.mem_get_info(device_id)
        return free_mem / (1024**3)

    def print_memory_stats(self, prefix: str = ""):
        stats = self.get_memory_stats()

        if "error" in stats:
            logger.info("%s %s", prefix, stats["error"])
            return

        logger.info(
            "%s VRAM: %.2fGB / %.2fGB (%.1f%%) | Free: %.2fGB",
            prefix,
            stats["reserved_gb"],
            stats["total_gb"],
            stats["usage_fraction"] * 100,
            stats["free_gb"],
        )

    def print_detailed_stats(self):
        """Log detailed memory info at TRACE level.

        Includes allocation breakdown, peaks, and usage details.
        Only executes if logger is enabled for TRACE level.
        """
        if not logger.isEnabledFor(TRACE):
            return

        stats = self.get_memory_stats()

        if "error" in stats:
            logger.log(TRACE, "[MemoryManager] %s", stats["error"])
            return

        logger.log(TRACE, "[MemoryManager] Detailed Memory Stats:")
        logger.log(TRACE, "  Allocated: %.2f GB", stats["allocated_gb"])
        logger.log(TRACE, "  Reserved: %.2f GB", stats["reserved_gb"])
        logger.log(TRACE, "  Max Allocated: %.2f GB", stats["max_allocated_gb"])
        logger.log(TRACE, "  Total GPU: %.2f GB", stats["total_gb"])
        logger.log(TRACE, "  Free: %.2f GB", stats["free_gb"])
        logger.log(TRACE, "  Usage Fraction: %.1f%%", stats["usage_fraction"] * 100)

    def check_memory_warning(self) -> bool:
        stats = self.get_memory_stats()
        if "error" in stats:
            return False

        return stats["usage_fraction"] > self.memory_fraction_warning

    def step(self):
        self.step_count += 1

        if self.step_count % self.clear_cache_frequency == 0:
            self.clear_cache(aggressive=False)

        # Log detailed stats at TRACE level
        self.print_detailed_stats()

        if self.check_memory_warning():
            logger.warning("High VRAM usage detected!")
            self.print_memory_stats(prefix="")
            self.clear_cache(aggressive=True)

    def reset_peak_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def optimize_for_inference(self):
        """Switch to inference mode with gradients disabled."""
        torch.set_grad_enabled(False)
        self.clear_cache(aggressive=False)

    def optimize_for_training(self):
        """Switch to training mode with gradients enabled."""
        torch.set_grad_enabled(True)


def print_model_memory_usage(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print memory usage of model parameters.

    Args:
        model: The model to analyze
        model_name: Name to display
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory (4 bytes per parameter for FP32, 2 for FP16/BF16, 0.5 for 4-bit)
    param_memory_bytes = sum(
        p.numel()
        * (0.5 if not p.requires_grad else 2)  # Assume 4-bit frozen, BF16 trainable
        for p in model.parameters()
    )
    param_memory_gb = param_memory_bytes / (1024**3)

    logger.info("%s Memory Usage:", model_name)
    logger.info("  Total parameters: %s", f"{total_params:,}")
    logger.info(
        "  Trainable parameters: %s (%.2f%%)",
        f"{trainable_params:,}",
        trainable_params / total_params * 100,
    )
    logger.info("  Estimated memory: %.2f GB", param_memory_gb)
