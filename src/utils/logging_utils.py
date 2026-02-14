"""
Logging infrastructure for GRPO training engine.

Provides:
- Custom TRACE level (5) below DEBUG (10)
- TqdmLoggingHandler to prevent progress bar corruption
- setup_logging() for verbosity-based configuration
- get_logger() convenience function
- log_tensor_meta() for safe tensor metadata logging
"""
import logging
import os
from typing import Optional

# Custom levels below DEBUG (10)
VERBOSE = 8
TRACE = 5

logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(TRACE, "TRACE")


def _verbose(self, message, *args, **kwargs):
    """Log at VERBOSE level (between DEBUG and TRACE)."""
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


def _trace(self, message, *args, **kwargs):
    """Log at TRACE level (below VERBOSE)."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


logging.Logger.verbose = _verbose
logging.Logger.trace = _trace


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes output through tqdm.write().

    Prevents log messages from corrupting tqdm progress bars.
    """

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(verbosity: int = 0, log_dir: Optional[str] = None) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: 0=INFO, 1=DEBUG, 2=VERBOSE, 3=TRACE
        log_dir: If provided and verbosity >= 2, write logs to {log_dir}/training.log
    """
    # Map verbosity to logging level
    level_map = {
        0: logging.INFO,
        1: logging.DEBUG,
        2: VERBOSE,
        3: TRACE,
    }
    level = level_map.get(verbosity, TRACE if verbosity >= 3 else logging.INFO)

    # Configure project root logger
    logger = logging.getLogger("grpo")
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates on repeated calls
    logger.handlers.clear()

    # Console handler using tqdm.write
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_fmt = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler for verbosity >= 2
    if verbosity >= 2 and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"), mode="a"
        )
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s][%(name)s] %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    # Silence third-party loggers
    for lib_name in [
        "transformers", "urllib3", "bitsandbytes",
        "wandb", "httpx", "filelock",
    ]:
        logging.getLogger(lib_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the grpo namespace.

    Args:
        name: Module name (e.g., 'trainer', 'core.lora')

    Returns:
        Logger named 'grpo.{name}'
    """
    return logging.getLogger(f"grpo.{name}")


def log_tensor_meta(
    logger: logging.Logger, msg: str, tensor, level: int = TRACE
) -> None:
    """Log tensor metadata without stringifying the full tensor.

    Logs shape, dtype, device, and scalar stats (min/max/mean) if floating-point.
    Guards with isEnabledFor() to avoid computation when level is disabled.

    Args:
        logger: Logger instance to use
        msg: Description message
        tensor: PyTorch tensor to inspect
        level: Logging level (default: TRACE)
    """
    if not logger.isEnabledFor(level):
        return
    meta = "shape=%s, dtype=%s, device=%s" % (
        tuple(tensor.shape), tensor.dtype, tensor.device
    )
    if tensor.numel() > 0 and tensor.is_floating_point():
        meta += ", min=%.4f, max=%.4f, mean=%.4f" % (
            tensor.min().item(), tensor.max().item(), tensor.mean().item()
        )
    logger.log(level, "%s: %s", msg, meta)
