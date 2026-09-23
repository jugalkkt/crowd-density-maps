"""Training entry point.

    python -m src.train --config configs/default.yaml --set data.part=B
    python -m src.train --resume checkpoints/part_B/last.pth

Every epoch writes ``last.pth`` so a Colab/Kaggle disconnect costs at most one
epoch; ``best.pth`` is written whenever validation MAE improves. Point
``paths.checkpoint_dir`` at Google Drive or ``/kaggle/working`` to keep both
across sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, add_config_args, config_from_args
from .data.dataset import build_dataloaders
from .data.transforms import denormalize
from .models import MCNN
from .utils import (
    AverageMeter,
    count_metrics,
    ensure_dir,
    get_device,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from .visualize import save_comparison

__all__ = ["build_optimizer", "build_scheduler", "density_loss", "train_one_epoch",
           "validate", "train", "main"]

METRIC_FIELDS = ("epoch", "train_loss", "val_mae", "val_rmse", "lr", "seconds")


def density_loss(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "sum_over_batch"
) -> torch.Tensor:
    """Pixel-wise MSE between predicted and target density maps.

    ``sum_over_batch`` (the default) sums the squared error over every pixel and
    divides by the batch size — the original MCNN objective. Its gradient
    magnitude is independent of crop size, which matters because per-pixel
    densities are tiny and a ``mean`` reduction makes "predict zero everywhere"
    an attractive local minimum. Use ``mean`` together with a large
    ``train.density_scale`` if you prefer.
    """
    if reduction == "mean":
        return nn.functional.mse_loss(prediction, target, reduction="mean")
    if reduction == "sum_over_batch":
        return nn.functional.mse_loss(prediction, target, reduction="sum") / (
            prediction.shape[0]
        )
    raise ValueError(f"Unknown loss reduction: {reduction!r}")


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Adam (default) or SGD, configured from ``cfg.train``."""
    name = str(cfg.train.get("optimizer", "adam")).lower()
    lr = float(cfg.train.get("lr", 1e-5))
    weight_decay = float(cfg.train.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.train.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    """``ReduceLROnPlateau`` on validation MAE, or ``None``."""
    name = str(cfg.train.get("scheduler", "none")).lower()
    if name in ("none", "off", ""):
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.train.get("scheduler_factor", 0.5)),
            patience=int(cfg.train.get("scheduler_patience", 20)),
        )
    raise ValueError(f"Unknown scheduler: {name!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int,
    scaler: Any | None = None,
    logger=None,
    writer=None,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    meter = AverageMeter()
    reduction = str(cfg.train.get("loss_reduction", "sum_over_batch"))
    log_every = int(cfg.train.get("log_every", 20))
    use_amp = scaler is not None and scaler.is_enabled()

    progress = tqdm(loader, desc=f"epoch {epoch}", unit="batch", leave=False)
    for step, (images, targets, _) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast:
            predictions = model(images)
            loss = density_loss(predictions, targets, reduction)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), images.shape[0])
        progress.set_postfix(loss=f"{meter.avg:.4f}")

        if logger and log_every and step % log_every == 0:
            logger.info(
                "epoch %d | step %d/%d | loss %.4f",
                epoch, step, len(loader), meter.avg,
            )
        if writer is not None and log_every and step % log_every == 0:
            writer.add_scalar("train/loss_step", loss.item(), epoch * len(loader) + step)

    return meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    density_scale: float = 1.0,
    desc: str = "val",
) -> tuple[float, float, list[float], list[float]]:
    """Evaluate counting accuracy on full images.

    Returns ``(mae, rmse, predicted_counts, gt_counts)``. Predictions are
    divided by ``density_scale`` so counts are in real units.
    """
    model.eval()
    predicted: list[float] = []
    ground_truth: list[float] = []

    for images, targets, counts in tqdm(loader, desc=desc, unit="img", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        # Sum per sample over channel/height/width.
        batch_counts = outputs.sum(dim=(1, 2, 3)) / density_scale
        predicted.extend(batch_counts.detach().float().cpu().tolist())
        ground_truth.extend(counts.tolist())

    mae, rmse = count_metrics(predicted, ground_truth)
    return mae, rmse, predicted, ground_truth


@torch.no_grad()
def save_val_sample(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: Path,
    density_scale: float = 1.0,
) -> None:
    """Save an image / GT / prediction triptych for the first validation image."""
    model.eval()
    images, targets, counts = next(iter(loader))
    prediction = model(images.to(device)).cpu()
    image = (denormalize(images[0]).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    save_comparison(
        image,
        targets[0, 0].numpy() / density_scale,
        prediction[0, 0].numpy() / density_scale,
        path,
        title=f"val sample (GT {counts[0]:.1f})",
    )


def _append_metrics(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row to ``metrics.csv``, writing the header if it is new."""
    ensure_dir(csv_path.parent)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(METRIC_FIELDS))
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def _make_writer(log_dir: Path):
    """Create a TensorBoard writer, or ``None`` if TensorBoard is unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return SummaryWriter(log_dir=str(log_dir))


def train(cfg: Config, resume: str | Path | None = None) -> dict[str, Any]:
    """Run the full training loop; returns a summary dict."""
    part = str(cfg.data.part).upper()
    run_dir = ensure_dir(Path(cfg.paths.checkpoint_dir) / f"part_{part}")
    output_dir = ensure_dir(Path(cfg.paths.output_dir) / f"part_{part}")
    logger = get_logger("crowd.train", run_dir / "train.log")

    set_seed(int(cfg.train.get("seed", 42)))
    device = get_device()
    logger.info("Device: %s", device)
    logger.info("Config: %s", json.dumps(cfg.to_dict(), default=str))

    loaders = build_dataloaders(cfg, splits=("train", "val"))
    logger.info(
        "Data: %d train / %d val images",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
    )

    model = MCNN().to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    amp_enabled = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        logger.info("Mixed precision enabled")

    start_epoch = 1
    best_mae = float("inf")
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mae = float(ckpt.get("best_mae", float("inf")))
        if ckpt.get("scaler") and amp_enabled:
            scaler.load_state_dict(ckpt["scaler"])
        logger.info("Resumed from %s at epoch %d (best MAE %.2f)",
                    resume, start_epoch, best_mae)

    writer = _make_writer(run_dir / "tb")
    metrics_csv = run_dir / "metrics.csv"
    density_scale = float(cfg.train.get("density_scale", 1.0))
    epochs = int(cfg.train.get("epochs", 400))
    val_every = max(1, int(cfg.train.get("val_every", 1)))
    viz_every = int(cfg.train.get("viz_every", 0))

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, device, cfg, epoch,
            scaler=scaler, logger=logger, writer=writer,
        )

        val_mae = val_rmse = float("nan")
        if epoch % val_every == 0 or epoch == epochs:
            val_mae, val_rmse, _, _ = validate(
                model, loaders["val"], device, density_scale
            )
            if scheduler is not None:
                scheduler.step(val_mae)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - started
        logger.info(
            "epoch %d/%d | loss %.4f | val MAE %.2f | val RMSE %.2f | lr %.2e | %.1fs",
            epoch, epochs, train_loss, val_mae, val_rmse, lr, elapsed,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_mae": round(val_mae, 4) if np.isfinite(val_mae) else "",
            "val_rmse": round(val_rmse, 4) if np.isfinite(val_rmse) else "",
            "lr": lr,
            "seconds": round(elapsed, 2),
        }
        _append_metrics(metrics_csv, row)
        history.append(row)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            if np.isfinite(val_mae):
                writer.add_scalar("val/mae", val_mae, epoch)
                writer.add_scalar("val/rmse", val_rmse, epoch)

        save_checkpoint(
            run_dir / "last.pth", model, optimizer, scheduler,
            epoch=epoch, best_mae=best_mae, config=cfg,
            scaler=scaler.state_dict() if amp_enabled else None,
        )
        if np.isfinite(val_mae) and val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                run_dir / "best.pth", model, optimizer, scheduler,
                epoch=epoch, best_mae=best_mae, config=cfg,
                val_rmse=val_rmse,
            )
            logger.info("New best val MAE %.2f -> %s", best_mae, run_dir / "best.pth")

        if viz_every and epoch % viz_every == 0:
            save_val_sample(
                model, loaders["val"], device,
                output_dir / "val_samples" / f"epoch_{epoch:04d}.png",
                density_scale,
            )

    if writer is not None:
        writer.close()
    logger.info("Training finished. Best val MAE: %.2f", best_mae)
    return {"best_mae": best_mae, "epochs": epochs, "history": history,
            "checkpoint_dir": str(run_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(
        argparse.ArgumentParser(description="Train MCNN on ShanghaiTech.")
    )
    parser.add_argument(
        "--resume", default=None, help="Path to a last.pth to resume from."
    )
    args = parser.parse_args(argv)
    train(config_from_args(args), resume=args.resume)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
