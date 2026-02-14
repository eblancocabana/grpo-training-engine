"""
Memory Management for VRAM Optimization.
Handles gradient checkpointing, cache clearing, and memory monitoring.
"""
import torch
import gc
from src.utils.logging_utils import get_logger, TRACE

logger = get_logger("core.memory_manager")


class MemoryManager:
    
    def __init__(
        self,
        device: str = "cuda",
        enable_gradient_checkpointing: bool = True,
        clear_cache_frequency: int = 10,
        memory_fraction_warning: float = 0.85
    ):
        self.device = device
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.clear_cache_frequency = clear_cache_frequency
        self.memory_fraction_warning = memory_fraction_warning
        self.step_count = 0
        
    def clear_cache(self, aggressive: bool = False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if aggressive:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def enable_checkpointing(self, model: torch.nn.Module):
        if self.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
            # CRITICAL: Enable input gradients to support checkpointing with frozen base layers
            # This prevents "element 0 of tensors does not require grad" error
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                
            logger.info("Gradient checkpointing enabled (with input grads)")
    
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
    
    def get_available_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        device_id = 0 if self.device == "cuda" else int(self.device.split(":")[-1])
        free_mem, _ = torch.cuda.mem_get_info(device_id)
        return free_mem / (1024**3)
    
    def print_memory_stats(self, prefix: str = ""):
        stats = self.get_memory_stats()
        
        if "error" in stats:
            logger.info("%s %s", prefix, stats['error'])
            return
        
        logger.info(
            "%s VRAM: %.2fGB / %.2fGB (%.1f%%) | Free: %.2fGB",
            prefix,
            stats['reserved_gb'],
            stats['total_gb'],
            stats['usage_fraction'] * 100,
            stats['free_gb']
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
            logger.log(TRACE, "[MemoryManager] %s", stats['error'])
            return
        
        logger.log(TRACE, "[MemoryManager] Detailed Memory Stats:")
        logger.log(TRACE, "  Allocated: %.2f GB", stats['allocated_gb'])
        logger.log(TRACE, "  Reserved: %.2f GB", stats['reserved_gb'])
        logger.log(TRACE, "  Max Allocated: %.2f GB", stats['max_allocated_gb'])
        logger.log(TRACE, "  Total GPU: %.2f GB", stats['total_gb'])
        logger.log(TRACE, "  Free: %.2f GB", stats['free_gb'])
        logger.log(TRACE, "  Usage Fraction: %.1f%%", stats['usage_fraction'] * 100)
    
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
        self.clear_cache(aggressive=True)
        torch.set_grad_enabled(False)
    
    def optimize_for_training(self):
        """Switch to training mode with gradients enabled."""
        torch.set_grad_enabled(True)
        self.clear_cache(aggressive=True)


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
        p.numel() * (0.5 if not p.requires_grad else 2)  # Assume 4-bit frozen, BF16 trainable
        for p in model.parameters()
    )
    param_memory_gb = param_memory_bytes / (1024**3)
    
    logger.info("%s Memory Usage:", model_name)
    logger.info("  Total parameters: %s", f"{total_params:,}")
    logger.info("  Trainable parameters: %s (%.2f%%)", f"{trainable_params:,}", trainable_params/total_params*100)
    logger.info("  Estimated memory: %.2f GB", param_memory_gb)

