"""
Configuration management for GRPO training.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List


class VerbosityLevel(IntEnum):
    """Verbosity levels for training output."""
    SILENT = 0   # Default: setup info, progress bar, epoch summaries (INFO)
    DEBUG = 1    # Current --debug: generations, rewards (DEBUG)
    VERBOSE = 2  # Training diagnostics: entropy, grads, timing (TRACE)
    TRACE = 3    # Full trace: tensor shapes, per-token stats, memory dumps (TRACE)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: str = "sdpa"


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class GRPOConfig:
    """Dr. GRPO algorithm configuration with two-sided clipping.
    
    Dr. GRPO eliminates length bias by normalizing loss against the group size
    (a global constant) instead of per-sequence token counts. Two-sided clipping
    uses asymmetric bounds (epsilon, epsilon_high) plus a hard safety cap (delta)
    to stabilize training in low-batch regimes.
    """
    group_size: int = 4
    clip_epsilon: float = 0.2
    epsilon_high: float = 0.3  # Upper clip bound (asymmetric, allows more positive updates)
    delta: float = 1.5  # Hard safety cap on ratio (prevents small-batch explosion)
    kl_coef: float = 0.1
    use_kl: bool = False  # Set to False for 8GB VRAM
    mask_truncated_completions: bool = True  # Zero out loss for truncated generations


@dataclass
class EntropyConfig:
    """Entropy-based selective backpropagation configuration."""
    use_entropy_mask: bool = True
    threshold: Optional[float] = None  # If None, use percentile
    percentile: float = 0.5  # Keep top 50% by entropy
    min_tokens: int = 10


@dataclass
class WandBConfig:
    """Weights & Biases configuration for experiment tracking."""
    enabled: bool = True
    project: str = "grpo-training"
    entity: Optional[str] = None  # Your wandb username or team
    run_name: Optional[str] = None  # Auto-generated if None
    tags: List[str] = field(default_factory=lambda: ["grpo", "deepseek-r1", "8gb-vram"])
    notes: str = "GRPO training with entropy-aware selective backpropagation"
    
    # Logging frequency
    log_frequency: int = 1  # Log every N steps
    log_gradients: bool = False  # Log gradient histograms (expensive)
    log_model: bool = True  # Log model architecture
    
    # Implementation identifier
    implementation: str = "python"  # "python" or "cpp" - distinguishes runs


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    dataset_name: str = "gsm8k"
    max_prompt_length: int = 512
    max_response_length: int = 512
    verbosity: int = 0
    
    # Training loop
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    
    # Memory
    enable_gradient_checkpointing: bool = True
    clear_cache_frequency: int = 10
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Generation
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


# Backward-compatible debug property
TrainingConfig.debug = property(lambda self: self.verbosity >= 1)


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        lora_config = LoRAConfig(**config_dict.get("lora", {}))
        grpo_config = GRPOConfig(**config_dict.get("grpo", {}))
        entropy_config = EntropyConfig(**config_dict.get("entropy", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        wandb_config = WandBConfig(**config_dict.get("wandb", {}))
        
        return cls(
            model=model_config,
            lora=lora_config,
            grpo=grpo_config,
            entropy=entropy_config,
            training=training_config,
            wandb=wandb_config
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "grpo": self.grpo.__dict__,
            "entropy": self.entropy.__dict__,
            "training": self.training.__dict__,
            "wandb": self.wandb.__dict__,
        }


# Preset configurations for different VRAM levels
def get_8gb_vram_config() -> Config:
    """Get optimized config for 8GB VRAM (e.g., RTX 3060 Ti)."""
    config = Config()
    
    # Model settings
    config.model.load_in_4bit = True
    config.model.bnb_4bit_use_double_quant = True
    
    config.lora.rank = 16
    config.lora.alpha = 32
    config.lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    config.grpo.group_size = 4
    config.grpo.clip_epsilon = 0.2
    config.grpo.epsilon_high = 0.3
    config.grpo.delta = 1.5
    config.grpo.use_kl = False
    config.grpo.mask_truncated_completions = True
    
    config.training.batch_size = 1
    config.training.gradient_accumulation_steps = 16
    config.training.enable_gradient_checkpointing = True
    config.training.max_prompt_length = 128
    config.training.max_response_length = 512
    config.training.clear_cache_frequency = 50
    
    # WandB settings
    config.wandb.enabled = True
    config.wandb.project = "grpo-training"
    config.wandb.tags = ["grpo", "deepseek-r1", "8gb-vram", "python"]
    config.wandb.implementation = "python"
    
    return config

