"""Evaluate a checkpoint on the official ShanghaiTech test split.

    python -m src.evaluate --checkpoint checkpoints/part_B/best.pth --part B

Writes ``outputs/eval_part_X.json`` with MAE/RMSE, optionally a per-image CSV,
a predicted-vs-ground-truth scatter plot, and density comparisons for the
worst-error images.

Reference numbers from the MCNN paper: Part A MAE 110.2 / RMSE 173.2,
Part B MAE 26.4 / RMSE 41.3.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import Config, add_config_args, config_from_args
from .data.dataset import CrowdDataset, build_dataloaders
from .data.transforms import denormalize
from .models import MCNN
from .utils import count_metrics, ensure_dir, get_device, get_logger, load_checkpoint
from .visualize import save_comparison, save_scatter

__all__ = ["evaluate", "main"]

PAPER_REFERENCE = {
    "A": {"mae": 110.2, "rmse": 173.2},
    "B": {"mae": 26.4, "rmse": 41.3},
}


@torch.no_grad()
def evaluate(
    cfg: Config,
    checkpoint: str | Path,
    part: str | None = None,
    split: str = "test",
    save_predictions: bool = False,
    n_worst: int = 10,
) -> dict[str, Any]:
    """Run counting evaluation and write the report files.

    Returns a summary dict that is also saved as ``eval_part_X.json``.
    """
    part = (part or cfg.data.part).upper()
    logger = get_logger("crowd.evaluate")
    device = get_device()
    density_scale = float(cfg.train.get("density_scale", 1.0))

    model = MCNN(verbose=False).to(device)
    ckpt = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    logger.info(
        "Loaded %s (epoch %s, val MAE %s)",
        checkpoint, ckpt.get("epoch", "?"), ckpt.get("best_mae", "?"),
    )

    loader = build_dataloaders(cfg, part=part, splits=(split,))[split]
    dataset: CrowdDataset = loader.dataset
    records: list[dict[str, Any]] = []

    for index, (images, _, counts) in enumerate(
        tqdm(loader, desc=f"eval part_{part}/{split}", unit="img")
    ):
        prediction = model(images.to(device))
        predicted = float(prediction.sum().item()) / density_scale
        ground_truth = float(counts[0])
        records.append(
            {
                "image": dataset.stems[index],
                "gt_count": round(ground_truth, 3),
                "pred_count": round(predicted, 3),
                "abs_error": round(abs(predicted - ground_truth), 3),
            }
        )

    mae, rmse = count_metrics(
        [r["pred_count"] for r in records], [r["gt_count"] for r in records]
    )
    reference = PAPER_REFERENCE.get(part, {})
    summary = {
        "part": part,
        "split": split,
        "checkpoint": str(checkpoint),
        "n_images": len(records),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "paper_reference": reference,
        "mean_gt_count": round(float(np.mean([r["gt_count"] for r in records])), 3),
        "mean_pred_count": round(float(np.mean([r["pred_count"] for r in records])), 3),
    }

    output_dir = ensure_dir(cfg.paths.output_dir)
    json_path = output_dir / f"eval_part_{part}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    save_scatter(
        [r["gt_count"] for r in records],
        [r["pred_count"] for r in records],
        output_dir / f"scatter_part_{part}.png",
        title=f"Part {part} {split}: predicted vs GT (MAE {mae:.1f})",
    )

    if save_predictions:
        csv_path = output_dir / f"predictions_part_{part}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["image", "gt_count", "pred_count", "abs_error"]
            )
            writer.writeheader()
            writer.writerows(records)
        summary["predictions_csv"] = str(csv_path)

        worst = sorted(records, key=lambda r: -r["abs_error"])[:n_worst]
        worst_dir = ensure_dir(output_dir / f"worst_part_{part}")
        for record in worst:
            index = dataset.stems.index(record["image"])
            image_t, target, _ = dataset[index]
            prediction = model(image_t.unsqueeze(0).to(device)).cpu()
            image = (
                denormalize(image_t).permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            save_comparison(
                image,
                target[0].numpy() / density_scale,
                prediction[0, 0].numpy() / density_scale,
                worst_dir / f"{record['image']}_err{record['abs_error']:.0f}.png",
                title=f"{record['image']} (GT {record['gt_count']:.0f})",
            )
        summary["worst_dir"] = str(worst_dir)

    _print_summary(summary, logger)
    return summary


def _print_summary(summary: dict[str, Any], logger) -> None:
    """Log a small results table next to the paper's numbers."""
    reference = summary.get("paper_reference") or {}
    lines = [
        "",
        f"  Part {summary['part']} / {summary['split']} "
        f"({summary['n_images']} images)",
        "  " + "-" * 42,
        f"  {'metric':<10}{'this model':>14}{'MCNN paper':>16}",
        f"  {'MAE':<10}{summary['mae']:>14.2f}"
        f"{reference.get('mae', float('nan')):>16.2f}",
        f"  {'RMSE':<10}{summary['rmse']:>14.2f}"
        f"{reference.get('rmse', float('nan')):>16.2f}",
        "  " + "-" * 42,
        f"  mean GT count {summary['mean_gt_count']:.1f}, "
        f"mean predicted {summary['mean_pred_count']:.1f}",
        "",
    ]
    logger.info("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(
        argparse.ArgumentParser(description="Evaluate an MCNN checkpoint.")
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth.")
    parser.add_argument("--part", default=None, help="A or B (default: config).")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write a per-image CSV and density maps for the worst errors.",
    )
    parser.add_argument(
        "--n-worst", type=int, default=10, help="How many worst-error images to save."
    )
    args = parser.parse_args(argv)
    evaluate(
        config_from_args(args),
        args.checkpoint,
        part=args.part,
        split=args.split,
        save_predictions=args.save_predictions,
        n_worst=args.n_worst,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
