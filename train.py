#!/usr/bin/env python3
"""
Main training script for GRPO on RTX 3060 Ti (8GB VRAM).
Optimized for your specific hardware setup.
"""
import os
import sys
import torch
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.grpo.trainer import GRPOTrainerLoop
from src.utils.config import get_8gb_vram_config, VerbosityLevel
from src.utils.logging_utils import setup_logging, get_logger

# Module-level logger
logger = get_logger("main")


def check_system():
    """Verify system meets requirements."""
    logger.info("=" * 60)
    logger.info("System Check for RTX 3060 Ti")
    logger.info("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False
    
    # Check GPU
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info("GPU: %s", gpu_name)
    logger.info("VRAM: %.1f GB", vram_gb)
    logger.info("CUDA Version: %s", torch.version.cuda)
    logger.info("PyTorch Version: %s", torch.__version__)
    
    # Verify RTX 3060 Ti
    if "3060 Ti" not in gpu_name:
        logger.warning("Expected RTX 3060 Ti, found %s", gpu_name)
        logger.warning("Config will still work but may need adjustment.")
    
    # Check VRAM
    if vram_gb < 7.5:
        logger.warning("Less than 8GB VRAM detected (%.1fGB)", vram_gb)
    
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepSeek-R1-Distill-Qwen-1.5B with GRPO on RTX 3060 Ti"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Group size for GRPO (responses per prompt)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more parameters)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--use-entropy-mask",
        action="store_true",
        default=True,
        help="Use entropy-based selective backpropagation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level: -v (DEBUG), -vv (VERBOSE), -vvv (TRACE)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="[DEPRECATED] Use -v instead. Enable debug mode to print generations"
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=128,
        help="Maximum tokens for prompt"
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=1024,
        help="Maximum tokens for response"
    )
    parser.add_argument(
        "--epsilon-high",
        type=float,
        default=None,
        help="Upper clip bound for two-sided clipping (default: 0.3)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Hard safety cap on ratio (default: 1.5)"
    )
    parser.add_argument(
        "--no-mask-truncated",
        action="store_true",
        help="Disable masking of truncated completions"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 16)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test setup without training"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable WandB logging (default: enabled)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="grpo-training",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="WandB tags (space-separated)"
    )
    
    args = parser.parse_args()
    
    # Handle backward compatibility: --debug is alias for -v
    if args.debug:
        args.verbose = max(args.verbose, 1)
    
    # Create output directories early so logs can be written
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    # Setup logging EARLY (before config loading)
    setup_logging(args.verbose, os.path.join(args.output_dir, "logs"))
    
    # Log startup
    if args.verbose > 0:
        logger.debug("Verbose mode enabled (level=%d)", args.verbose)
    if args.debug:
        logger.debug("Debug flag used (deprecated, use -v instead)")
    
    # System check
    if not check_system():
        sys.exit(1)
    
    # Get optimized config for 8GB VRAM
    logger.info("Loading 8GB VRAM optimized configuration...")
    config = get_8gb_vram_config()
    
    # Override with command line args
    config.training.output_dir = args.output_dir
    config.training.num_epochs = args.epochs
    config.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    config.training.log_dir = os.path.join(args.output_dir, "logs")
    config.grpo.group_size = args.group_size
    config.lora.rank = args.lora_rank
    config.training.learning_rate = args.learning_rate
    config.entropy.use_entropy_mask = args.use_entropy_mask
    config.training.max_prompt_length = args.max_prompt_length
    config.training.max_response_length = args.max_response_length
    
    if args.epsilon_high is not None:
        config.grpo.epsilon_high = args.epsilon_high
    if args.delta is not None:
        config.grpo.delta = args.delta
    if args.no_mask_truncated:
        config.grpo.mask_truncated_completions = False
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # WandB configuration
    config.wandb.enabled = args.wandb and not args.no_wandb
    config.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    if args.wandb_tags:
        config.wandb.tags = args.wandb_tags
    else:
        config.wandb.tags = ["grpo", "deepseek-r1", "8gb-vram", "python"]
    config.wandb.implementation = "python"
    
    # Create checkpoint directory
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    # Print configuration
    logger.info("Training Configuration:")
    logger.info("  Model: %s", config.model.model_id)
    logger.info("  LoRA Rank: %s", config.lora.rank)
    logger.info("  LoRA Alpha: %s", config.lora.alpha)
    logger.info("  Group Size: %s", config.grpo.group_size)
    logger.info("  Clip Epsilon: %s", config.grpo.clip_epsilon)
    logger.info("  Epsilon High: %s", config.grpo.epsilon_high)
    logger.info("  Delta (safety cap): %s", config.grpo.delta)
    logger.info("  Mask Truncated: %s", config.grpo.mask_truncated_completions)
    logger.info("  Learning Rate: %s", config.training.learning_rate)
    logger.info("  Epochs: %s", config.training.num_epochs)
    logger.info("  Gradient Accumulation: %s", config.training.gradient_accumulation_steps)
    logger.info("  Entropy Mask: %s", config.entropy.use_entropy_mask)
    logger.info("  Max Prompt Length: %s", config.training.max_prompt_length)
    logger.info("  Max Response Length: %s", config.training.max_response_length)
    logger.info("  Output Directory: %s", config.training.output_dir)
    logger.info("  WandB Enabled: %s", config.wandb.enabled)
    if config.wandb.enabled:
        logger.info("  WandB Project: %s", config.wandb.project)
        logger.info("  WandB Implementation: %s", config.wandb.implementation)
    
    if args.dry_run:
        logger.info("Setup complete! Exiting without training.")
        logger.info("Run without --dry-run to start training.")
        return
    
    # Create trainer and start training
    logger.info("Initializing trainer...")
    trainer = GRPOTrainerLoop(config)
    
    try:
        trainer.setup()
        trainer.train()
        logger.info("Training completed successfully!")
        logger.info("Checkpoints saved to: %s", config.training.checkpoint_dir)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(suffix="_interrupted")
        logger.info("Checkpoint saved. You can resume later.")
    except Exception as e:
        logger.error("Training failed with error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
