"""
Checkpoint utilities for saving and loading model states.
"""
import os
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .logging_utils import get_logger

logger = get_logger("utils.checkpoint")


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer
            scheduler: Learning rate scheduler (optional)
            step: Current training step
            epoch: Current epoch
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
            checkpoint_name: Custom checkpoint name (optional)
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{step}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved to %s", checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best model to %s", best_path)
        
        # Save latest checkpoint info
        latest_info = {
            'latest_checkpoint': str(checkpoint_path),
            'step': step,
            'epoch': epoch,
        }
        with open(self.checkpoint_dir / "latest.json", 'w') as f:
            json.dump(latest_info, f, indent=2)
        
        return str(checkpoint_path)
    
    def save_lora_weights(
        self,
        model: torch.nn.Module,
        save_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save only LoRA weights (much smaller than full model).
        
        Args:
            model: Model with LoRA layers
            save_path: Path to save weights
            metadata: Additional metadata to save
            
        Returns:
            Path to saved weights
        """
        # Extract only LoRA parameters
        lora_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                lora_state_dict[name] = param.cpu()
        
        save_dict = {
            'lora_weights': lora_state_dict,
            'metadata': metadata or {},
        }
        
        torch.save(save_dict, save_path)
        logger.info("Saved LoRA weights to %s", save_path)
        logger.info("  - Number of LoRA tensors: %d", len(lora_state_dict))
        
        return save_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state (optional)
            scheduler: Scheduler to load state (optional)
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Dictionary with checkpoint info (step, epoch, metrics)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)
        
        logger.info("Loading from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        info = {
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
        }
        
        logger.info("Loaded step %d, epoch %d", info['step'], info['epoch'])
        
        return info
    
    def load_lora_weights(
        self,
        weights_path: str,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Load LoRA weights into model.
        
        Args:
            weights_path: Path to LoRA weights file
            model: Model with LoRA layers
            
        Returns:
            Metadata dictionary
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError("LoRA weights not found: %s" % weights_path)
        
        logger.info("Loading LoRA weights from %s", weights_path)
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        lora_weights = checkpoint['lora_weights']
        
        # Load only LoRA parameters
        model_state = model.state_dict()
        for name, param in lora_weights.items():
            if name in model_state:
                model_state[name].copy_(param)
            else:
                logger.warning("Layer %s not found in model", name)
        
        logger.info("Loaded %d LoRA tensors", len(lora_weights))
        
        return checkpoint.get('metadata', {})
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        latest_file = self.checkpoint_dir / "latest.json"
        
        if not latest_file.exists():
            return None
        
        with open(latest_file, 'r') as f:
            info = json.load(f)
        
        checkpoint_path = info.get('latest_checkpoint')
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path
        
        return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoints = []
        for file in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            checkpoints.append(str(file))
        
        return sorted(checkpoints)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Sort by modification time
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda x: os.path.getmtime(x),
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint in checkpoints_sorted[keep_last_n:]:
            os.remove(checkpoint)
            logger.info("Removed old checkpoint: %s", checkpoint)


def save_training_config(config, save_path: str):
    """
    Save training configuration to JSON.
    
    Args:
        config: Configuration object
        save_path: Path to save config
    """
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("Saved config to %s", save_path)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from JSON.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config
