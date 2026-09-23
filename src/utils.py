"""Seeding, device selection, logging and checkpoint I/O."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import Config

__all__ = [
    "set_seed",
    "get_device",
    "get_logger",
    "save_checkpoint",
    "load_checkpoint",
    "ensure_dir",
    "AverageMeter",
    "count_metrics",
]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, optionally forcing deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device: cuda > mps > cpu.

    ``prefer`` forces a specific device string when it is actually available.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(
    name: str = "crowd",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger writing to stdout and, optionally, to ``log_file``.

    Handlers are only attached once, so repeated calls are cheap and do not
    duplicate output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        existing = {
            getattr(h, "baseFilename", None) for h in logger.handlers
        }
        if str(log_file.resolve()) not in existing:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_mae: float = float("inf"),
    config: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Path:
    """Write a checkpoint containing everything needed to resume training."""
    path = Path(path)
    ensure_dir(path.parent)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_mae": best_mae,
        "config": dict(config.to_dict() if isinstance(config, Config) else config)
        if config is not None
        else None,
    }
    payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, restoring any objects that are passed in.

    Returns the raw checkpoint dict so callers can read ``epoch``/``best_mae``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # Tolerate raw state_dicts saved by other scripts.
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if model is not None:
        model.load_state_dict(state, strict=strict)
    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt if isinstance(ckpt, dict) else {"model": ckpt}


class AverageMeter:
    """Running mean of a scalar, used for per-epoch loss reporting."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def count_metrics(
    predictions: "np.ndarray | list[float]", targets: "np.ndarray | list[float]"
) -> tuple[float, float]:
    """Return ``(MAE, RMSE)`` between predicted and ground-truth counts."""
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(targets, dtype=np.float64)
    if pred.size == 0:
        return float("nan"), float("nan")
    errors = pred - gt
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return mae, rmse
