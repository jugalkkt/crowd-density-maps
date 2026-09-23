"""Smoke tests for the training loop (Module 4)."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.config import apply_overrides
from src.data.dataset import build_dataloaders
from src.models import MCNN
from src.train import build_optimizer, density_loss, train, validate


def _short_run_config(processed_config, epochs: int = 2):
    """Copy of the fixture config tuned for a 2-epoch CPU run."""
    return apply_overrides(
        processed_config,
        [
            f"train.epochs={epochs}",
            "train.lr=1e-4",
            "train.num_workers=0",
            "train.amp=false",
            "train.log_every=1",
            "train.viz_every=1",
            "data.crop_size=64",
        ],
    )


def test_two_epochs_write_checkpoints_and_metrics(processed_config) -> None:
    cfg = _short_run_config(processed_config, epochs=2)
    summary = train(cfg)

    run_dir = Path(summary["checkpoint_dir"])
    assert (run_dir / "last.pth").is_file()
    assert (run_dir / "best.pth").is_file()

    metrics = run_dir / "metrics.csv"
    assert metrics.is_file()
    with metrics.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1" and rows[1]["epoch"] == "2"
    assert float(rows[-1]["val_mae"]) >= 0

    # The periodic validation visualisation was written.
    samples = Path(cfg.paths.output_dir) / f"part_{cfg.data.part}" / "val_samples"
    assert any(samples.glob("epoch_*.png"))


def test_resume_continues_from_last_epoch(processed_config) -> None:
    cfg = _short_run_config(processed_config, epochs=2)
    summary = train(cfg)
    last = Path(summary["checkpoint_dir"]) / "last.pth"

    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2

    resumed = train(apply_overrides(cfg, ["train.epochs=4"]), resume=last)
    assert resumed["history"][0]["epoch"] == 3
    assert resumed["history"][-1]["epoch"] == 4

    with (Path(summary["checkpoint_dir"]) / "metrics.csv").open() as fh:
        epochs = [int(r["epoch"]) for r in csv.DictReader(fh)]
    assert epochs == [1, 2, 3, 4]


def test_overfitting_a_single_image_reduces_loss(processed_config) -> None:
    cfg = apply_overrides(
        processed_config, ["train.density_scale=100", "data.crop_size=64"]
    )
    loader = build_dataloaders(cfg, splits=("train",))["train"]
    images, targets, _ = next(iter(loader))

    torch.manual_seed(0)
    model = MCNN(verbose=False)
    optimizer = build_optimizer(model, apply_overrides(cfg, ["train.lr=1e-3"]))

    first = last = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = density_loss(model(images), targets)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < 0.5 * first, f"loss only went {first:.4f} -> {last:.4f}"


def test_validate_returns_counts_in_real_units(processed_config) -> None:
    cfg = apply_overrides(processed_config, ["train.density_scale=100"])
    loader = build_dataloaders(cfg, splits=("val",))["val"]
    model = MCNN(verbose=False)

    mae, rmse, predicted, gt = validate(
        model, loader, torch.device("cpu"), density_scale=100.0
    )
    assert len(predicted) == len(gt) == len(loader.dataset)
    assert rmse >= mae >= 0
    assert all(p >= 0 for p in predicted)


def test_loss_reductions_agree_up_to_a_constant() -> None:
    prediction = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    summed = density_loss(prediction, target, "sum_over_batch")
    meaned = density_loss(prediction, target, "mean")
    assert torch.allclose(summed, meaned * 64)
