#!/usr/bin/env python3
"""
Main training script for GRPO on RTX 3060 Ti (8GB VRAM).
Optimized for your specific hardware setup.
"""

import os
import sys
import torch
import argparse
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grpo.trainer import GRPOTrainerLoop
from src.utils.config import get_8gb_vram_config
from src.utils.logging_utils import setup_logging, get_logger

# Module-level logger
logger = get_logger("main")


def _read_cmdline(pid_path: Path) -> list[str]:
    try:
        raw = (pid_path / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return []
    if not raw:
        return []
    return [part.decode(errors="ignore") for part in raw.split(b"\0") if part]


def _read_cwd(pid_path: Path) -> Path | None:
    try:
        return (pid_path / "cwd").resolve()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _is_under_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root)
    except (ValueError, OSError):
        return False
    return True


def _find_stale_train_pids(train_script: Path, repo_root: Path) -> list[int]:
    pids: list[int] = []
    proc_root = Path("/proc")
    train_name = train_script.name
    train_path = str(train_script.resolve())
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline = _read_cmdline(entry)
        if not cmdline:
            continue
        if not any(train_name in arg for arg in cmdline):
            continue
        cwd = _read_cwd(entry)
        if cwd is None or not _is_under_repo(cwd, repo_root):
            continue
        if any(train_path == arg for arg in cmdline):
            pids.append(int(entry.name))
            continue
        if any(arg.endswith(train_name) for arg in cmdline):
            pids.append(int(entry.name))
    return pids


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_stale_train_processes(train_script: Path, repo_root: Path) -> None:
    current_pid = os.getpid()
    candidate_pids = [
        pid for pid in _find_stale_train_pids(train_script, repo_root) if pid != current_pid
    ]
    if not candidate_pids:
        return

    logger.warning("Found stale train.py processes: %s", ", ".join(map(str, candidate_pids)))
    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            logger.warning("No permission to terminate pid %s: %s", pid, exc)
            continue

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if all(not _pid_is_alive(pid) for pid in candidate_pids):
            return
        time.sleep(0.1)

    for pid in candidate_pids:
        if not _pid_is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning("Force-killed stale train.py process pid=%s", pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            logger.warning("No permission to force-kill pid %s: %s", pid, exc)


def clear_gpu_memory(force_reset: bool) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")

    if not force_reset:
        return

    if not shutil.which("nvidia-smi"):
        logger.warning("nvidia-smi not available; skipping GPU reset")
        return

    result = subprocess.run(
        ["nvidia-smi", "--gpu-reset"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("GPU reset failed: %s", (result.stderr or "").strip())
    else:
        logger.warning("GPU reset requested via nvidia-smi")


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
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Group size for GRPO (responses per prompt)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16, help="LoRA rank (higher = more parameters)"
    )
    parser.add_argument(
        "--lora-adapter-quant",
        type=str,
        choices=["8bit", "4bit", "none"],
        default="none",
        help="Quantization for LoRA adapters (default: none)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompt batch size before GRPO group expansion",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=None,
        help="Lower PPO/GRPO clip bound epsilon (default: config value)",
    )
    parser.add_argument(
        "--use-entropy-mask",
        action="store_true",
        default=True,
        help="Use entropy-based selective backpropagation",
    )
    parser.add_argument(
        "--no-sent",
        action="store_true",
        help="Disable SENT curriculum loading and use plain GSM8K ordering",
    )
    parser.add_argument(
        "--use-triton",
        action="store_true",
        default=True,
        help="Enable Triton kernels (default: enabled)",
    )
    parser.add_argument(
        "--no-triton", action="store_true", help="Disable Triton kernels"
    )
    parser.add_argument(
        "--triton-lora-prefer-base",
        action="store_true",
        help="Prefer base-layer matmul in Triton LoRA forward",
    )
    parser.add_argument(
        "--triton-generation",
        action="store_true",
        default=None,
        help="Force-enable Triton generation when Triton kernels are enabled",
    )
    parser.add_argument(
        "--triton-generation-mode",
        type=str,
        choices=["auto", "on", "off"],
        default=None,
        help="Set Triton generation routing policy: auto, on, or off",
    )
    parser.add_argument(
        "--no-triton-generation",
        action="store_true",
        help="Disable Triton generation while keeping other Triton kernels available",
    )
    parser.add_argument(
        "--triton-grpo-loss",
        action="store_true",
        default=None,
        help="Force-enable Triton GRPO loss when Triton kernels are enabled",
    )
    parser.add_argument(
        "--no-triton-grpo-loss",
        action="store_true",
        help="Disable Triton GRPO loss while keeping other Triton kernels available",
    )
    parser.add_argument(
        "--triton-entropy-mask",
        action="store_true",
        default=None,
        help="Force-enable Triton entropy masking when Triton kernels are enabled",
    )
    parser.add_argument(
        "--no-triton-entropy-mask",
        action="store_true",
        help="Disable Triton entropy masking while keeping other Triton kernels available",
    )
    parser.add_argument(
        "--triton-lora",
        action="store_true",
        default=None,
        help="Force-enable Triton LoRA forward when Triton kernels are enabled",
    )
    parser.add_argument(
        "--no-triton-lora",
        action="store_true",
        help="Disable Triton LoRA forward while keeping other Triton kernels available",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level: -v (DEBUG), -vv (VERBOSE), -vvv (TRACE)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="[DEPRECATED] Use -v instead. Enable debug mode to print generations",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Maximum tokens for prompt (default: preset value)",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=None,
        help="Maximum tokens for response (default: preset value)",
    )
    parser.add_argument(
        "--epsilon-high",
        type=float,
        default=None,
        help="Upper clip bound for two-sided clipping (default: 0.3)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Hard safety cap on ratio (default: 1.5)",
    )
    parser.add_argument(
        "--no-mask-truncated",
        action="store_true",
        help="Disable masking of truncated completions",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 16)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop training after this many steps",
    )
    parser.add_argument(
        "--log-metrics-jsonl",
        action="store_true",
        help="Write structured training metrics to JSONL",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Custom JSONL metrics path; defaults to output_dir/metrics.jsonl when enabled",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Test setup without training"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable WandB logging (default: enabled)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--wandb-project", type=str, default="grpo-training", help="WandB project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="WandB tags (space-separated)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable live profiler server + Tier 1 hooks",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature used during GRPO generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling threshold used during GRPO generation",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling and use greedy decoding during GRPO generation",
    )
    parser.add_argument(
        "--entropy-percentile",
        type=float,
        default=None,
        help="Fraction of highest-entropy tokens to keep when entropy masking is enabled",
    )
    parser.add_argument(
        "--entropy-min-tokens",
        type=int,
        default=None,
        help="Minimum masked-in tokens per sequence for entropy masking",
    )
    parser.add_argument(
        "--length-penalty-coef",
        type=float,
        default=None,
        help="Penalty applied per generated token before GRPO advantage calculation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for Python, NumPy, and PyTorch",
    )
    parser.add_argument(
        "--no-initial-benchmark",
        action="store_true",
        help="Skip the initial GSM8K benchmark before training",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable checkpoint saving during training",
    )
    parser.add_argument(
        "--gpu-reset",
        action="store_true",
        help="Reset GPU via nvidia-smi before training (aggressive)",
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

    train_script = Path(__file__).resolve()
    repo_root = train_script.parent
    logger.info("Clearing stale training processes from %s", repo_root)
    kill_stale_train_processes(train_script, repo_root)
    clear_gpu_memory(force_reset=args.gpu_reset)

    # System check
    if not check_system():
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info("Global seed set to %s", args.seed)

    # Get optimized config for 8GB VRAM
    logger.info("Loading 8GB VRAM optimized configuration...")
    config = get_8gb_vram_config()

    # Override with command line args
    config.training.output_dir = args.output_dir
    config.training.num_epochs = args.epochs
    config.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    config.training.log_dir = os.path.join(args.output_dir, "logs")
    config.grpo.group_size = args.group_size
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    config.lora.rank = args.lora_rank
    config.lora.adapter_quantization = args.lora_adapter_quant
    config.training.learning_rate = args.learning_rate
    config.entropy.use_entropy_mask = args.use_entropy_mask
    if args.max_prompt_length is not None:
        config.training.max_prompt_length = args.max_prompt_length
    if args.max_response_length is not None:
        config.training.max_response_length = args.max_response_length
    config.training.use_triton_kernels = args.use_triton and not args.no_triton
    default_triton_enabled = config.training.use_triton_kernels
    config.training.triton_generation_mode = (
        "auto" if default_triton_enabled else "off"
    )
    config.training.use_triton_generation = default_triton_enabled
    config.training.use_triton_grpo_loss = default_triton_enabled
    config.training.use_triton_entropy_mask = default_triton_enabled
    config.training.use_triton_lora = default_triton_enabled
    if args.triton_generation_mode is not None:
        config.training.triton_generation_mode = (
            args.triton_generation_mode if default_triton_enabled else "off"
        )
    if args.triton_generation is not None:
        config.training.triton_generation_mode = (
            "on" if default_triton_enabled and args.triton_generation else "off"
        )
    if args.no_triton_generation:
        config.training.triton_generation_mode = "off"
    config.training.use_triton_generation = (
        default_triton_enabled
        and config.training.triton_generation_mode != "off"
    )
    if args.triton_grpo_loss is not None:
        config.training.use_triton_grpo_loss = (
            default_triton_enabled and args.triton_grpo_loss
        )
    if args.no_triton_grpo_loss:
        config.training.use_triton_grpo_loss = False
    if args.triton_entropy_mask is not None:
        config.training.use_triton_entropy_mask = (
            default_triton_enabled and args.triton_entropy_mask
        )
    if args.no_triton_entropy_mask:
        config.training.use_triton_entropy_mask = False
    if args.triton_lora is not None:
        config.training.use_triton_lora = (
            default_triton_enabled and args.triton_lora
        )
    if args.no_triton_lora:
        config.training.use_triton_lora = False
    config.training.triton_lora_prefer_base = args.triton_lora_prefer_base
    config.training.profile_enabled = args.profile
    if args.clip_epsilon is not None:
        config.grpo.clip_epsilon = args.clip_epsilon

    if args.epsilon_high is not None:
        config.grpo.epsilon_high = args.epsilon_high
    if args.delta is not None:
        config.grpo.delta = args.delta
    if args.length_penalty_coef is not None:
        config.grpo.length_penalty_coef = args.length_penalty_coef
    if args.no_mask_truncated:
        config.grpo.mask_truncated_completions = False
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.temperature is not None:
        config.training.generation_temperature = args.temperature
    if args.top_p is not None:
        config.training.generation_top_p = args.top_p
    if args.greedy:
        config.training.generation_do_sample = False
    if args.entropy_percentile is not None:
        config.entropy.percentile = args.entropy_percentile
    if args.entropy_min_tokens is not None:
        config.entropy.min_tokens = args.entropy_min_tokens
    if args.no_sent:
        config.sent.enabled = False

    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    config.training.log_metrics_jsonl = args.log_metrics_jsonl
    config.training.metrics_jsonl_path = args.metrics_path

    # Initial benchmark configuration
    config.training.skip_initial_benchmark = args.no_initial_benchmark
    if (
        args.max_steps is not None
        and args.max_steps <= 5
        and not config.training.skip_initial_benchmark
    ):
        logger.info(
            "Auto-skipping initial benchmark for short verification run (max_steps=%d).",
            args.max_steps,
        )
        config.training.skip_initial_benchmark = True
    
    # Checkpoint configuration
    if args.no_checkpoints:
        config.training.checkpoint_dir = None

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

    # Create checkpoint directory only when checkpointing is enabled.
    if config.training.checkpoint_dir:
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Print configuration
    logger.info("Training Configuration:")
    logger.info("  Model: %s", config.model.model_id)
    logger.info("  LoRA Rank: %s", config.lora.rank)
    logger.info("  LoRA Alpha: %s", config.lora.alpha)
    logger.info("  Group Size: %s", config.grpo.group_size)
    logger.info("  Batch Size: %s", config.training.batch_size)
    logger.info("  Clip Epsilon: %s", config.grpo.clip_epsilon)
    logger.info("  Epsilon High: %s", config.grpo.epsilon_high)
    logger.info("  Delta (safety cap): %s", config.grpo.delta)
    logger.info("  Length Penalty Coef: %s", config.grpo.length_penalty_coef)
    logger.info("  Mask Truncated: %s", config.grpo.mask_truncated_completions)
    logger.info("  Learning Rate: %s", config.training.learning_rate)
    logger.info("  Epochs: %s", config.training.num_epochs)
    logger.info(
        "  Gradient Accumulation: %s", config.training.gradient_accumulation_steps
    )
    logger.info("  Entropy Mask: %s", config.entropy.use_entropy_mask)
    logger.info("  Entropy Percentile: %s", config.entropy.percentile)
    logger.info("  Entropy Min Tokens: %s", config.entropy.min_tokens)
    logger.info("  SENT Enabled: %s", config.sent.enabled)
    logger.info("  Triton Kernels: %s", config.training.use_triton_kernels)
    logger.info("  Triton Generation: %s", config.training.use_triton_generation)
    logger.info(
        "  Triton Generation Mode: %s", config.training.triton_generation_mode
    )
    logger.info("  Triton GRPO Loss: %s", config.training.use_triton_grpo_loss)
    logger.info("  Triton Entropy Mask: %s", config.training.use_triton_entropy_mask)
    logger.info("  Triton LoRA: %s", config.training.use_triton_lora)
    logger.info("  Max Prompt Length: %s", config.training.max_prompt_length)
    logger.info("  Max Response Length: %s", config.training.max_response_length)
    logger.info("  Generation Temperature: %s", config.training.generation_temperature)
    logger.info("  Generation Top-p: %s", config.training.generation_top_p)
    logger.info("  Generation Do Sample: %s", config.training.generation_do_sample)
    logger.info("  Output Directory: %s", config.training.output_dir)
    logger.info("  WandB Enabled: %s", config.wandb.enabled)
    logger.info("  Profiler Enabled: %s", config.training.profile_enabled)
    logger.info("  Metrics JSONL: %s", config.training.log_metrics_jsonl)
    if config.training.log_metrics_jsonl:
        logger.info(
            "  Metrics Path: %s",
            config.training.metrics_jsonl_path or os.path.join(args.output_dir, "metrics.jsonl"),
        )
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

    if config.training.profile_enabled:
        from tools.vram_profiler.profiler_server import start_server

        start_server(port=8550)

    try:
        trainer.setup()

        if args.resume or args.resume_checkpoint:
            checkpoint_manager = trainer.checkpoint_manager
            if checkpoint_manager is None:
                raise RuntimeError("Checkpoint manager not initialized after setup.")
            checkpoint_path = args.resume_checkpoint
            if checkpoint_path is None:
                checkpoint_path = checkpoint_manager.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "Resume requested but no checkpoint found in output directory."
                )
            if args.resume_checkpoint is not None:
                logger.info(
                    "Resume requested: using explicit checkpoint path: %s",
                    checkpoint_path,
                )
            else:
                logger.info(
                    "Resume requested: using latest checkpoint from %s: %s",
                    config.training.checkpoint_dir,
                    checkpoint_path,
                )
            resume_info = trainer.load_checkpoint(checkpoint_path)
            logger.info(
                "Resume loaded: step=%s epoch=%s",
                resume_info.get("step"),
                resume_info.get("epoch"),
            )

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
