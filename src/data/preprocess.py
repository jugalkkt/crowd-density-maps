"""Turn the raw ShanghaiTech download into image / density-map pairs.

Usage::

    python -m src.data.preprocess --part B
    python -m src.data.preprocess --part A --set density.mode=adaptive --visualize 5

For every raw image this writes ``processed/part_X/<split>/IMG_n.h5`` holding
the float32 density map and the ground-truth count, alongside a copy of the
image under ``processed/part_X/<split>/images/``. A seeded train/val split is
recorded in ``processed/part_X/splits.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import Config, add_config_args, config_from_args
from ..utils import ensure_dir, get_logger
from .density import generate_density_map
from .parse_annotations import annotation_path_for_image, load_points

__all__ = ["find_part_dir", "list_split_images", "preprocess_part", "main"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "test")

# Defaults the paper uses per part, applied unless the config says otherwise.
PART_DEFAULT_MODE = {"A": "adaptive", "B": "fixed"}


def find_part_dir(raw_root: str | Path, part: str) -> Path:
    """Locate the ``part_A`` / ``part_B`` folder under a raw dataset root.

    Kaggle mirrors differ in their top-level folder name and in the separator
    used (``part_A``, ``part-A``, ``partA``), so the search is recursive and
    case-insensitive.
    """
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_root}. Set paths.raw_data_dir "
            f"to the folder containing the ShanghaiTech download."
        )

    wanted = {f"part_{part.lower()}", f"part-{part.lower()}", f"part{part.lower()}"}
    candidates = [
        d
        for d in [raw_root, *raw_root.rglob("*")]
        if d.is_dir() and d.name.lower() in wanted
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a `part_{part}` directory under {raw_root}."
        )
    # Prefer the shallowest match, which is the dataset root rather than a
    # nested duplicate inside an archive folder.
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def list_split_images(part_dir: Path, split: str) -> list[Path]:
    """Return the sorted image paths of ``train``/``test`` inside a part dir."""
    split_dir = part_dir / f"{split}_data"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    images = [
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda p: (len(p.stem), p.stem))


def _density_settings(cfg: Config, part: str) -> dict:
    """Density kwargs, defaulting `mode` to the paper's per-part choice."""
    density_cfg = cfg.get("density", Config())
    mode = density_cfg.get("mode") or PART_DEFAULT_MODE[part.upper()]
    return {
        "mode": str(mode),
        "fixed_sigma": float(density_cfg.get("fixed_sigma", 4.0)),
        "beta": float(density_cfg.get("beta", 0.3)),
        "k": int(density_cfg.get("k_neighbors", 3)),
        "truncate": float(density_cfg.get("truncate", 3.0)),
    }


def _write_splits_json(
    out_dir: Path, train_stems: list[str], test_stems: list[str],
    val_fraction: float, seed: int,
) -> dict:
    """Record a reproducible train/val split (ShanghaiTech has no official one)."""
    shuffled = list(train_stems)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    n_val = min(max(n_val, 1 if shuffled else 0), max(len(shuffled) - 1, 0))
    splits = {
        "seed": seed,
        "val_fraction": val_fraction,
        "val": sorted(shuffled[:n_val]),
        "train": sorted(shuffled[n_val:]),
        "test": sorted(test_stems),
    }
    with (out_dir / "splits.json").open("w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    return splits


def _save_visualisation(
    image: Image.Image, density: np.ndarray, count: float, dest: Path
) -> None:
    """Save a side-by-side image / density-map PNG for eyeballing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"{dest.stem}  (GT count {count:.1f})")
    axes[1].imshow(density, cmap="jet")
    axes[1].set_title(f"density (sum {density.sum():.2f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    ensure_dir(dest.parent)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def preprocess_part(
    cfg: Config,
    part: str,
    overwrite: bool = False,
    visualize: int = 0,
    copy_images: bool = True,
) -> dict:
    """Generate density maps for one part and return the split index.

    Args:
        cfg: loaded configuration.
        part: ``"A"`` or ``"B"``.
        overwrite: regenerate ``.h5`` files that already exist.
        visualize: save this many sanity-check PNGs per split to the output dir.
        copy_images: copy images into the processed tree (symlink if possible).
    """
    part = part.upper()
    logger = get_logger("crowd.preprocess")
    raw_part_dir = find_part_dir(cfg.paths.raw_data_dir, part)
    out_dir = ensure_dir(Path(cfg.paths.processed_dir) / f"part_{part}")
    density_kwargs = _density_settings(cfg, part)
    logger.info("Raw part dir: %s", raw_part_dir)
    logger.info("Density settings: %s", density_kwargs)

    stems: dict[str, list[str]] = {}
    for split in SPLITS:
        images = list_split_images(raw_part_dir, split)
        split_dir = ensure_dir(out_dir / split)
        images_out = ensure_dir(split_dir / "images")
        stems[split] = [p.stem for p in images]
        n_viz = 0

        for image_path in tqdm(images, desc=f"part_{part}/{split}", unit="img"):
            h5_path = split_dir / f"{image_path.stem}.h5"
            image_dest = images_out / image_path.name

            if copy_images and not image_dest.exists():
                try:
                    image_dest.symlink_to(image_path.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(image_path, image_dest)

            if h5_path.exists() and not overwrite and n_viz >= visualize:
                continue

            # Grayscale images appear in Part A; everything becomes RGB.
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                width, height = image.size

                points = load_points(annotation_path_for_image(image_path))
                density = generate_density_map(
                    (height, width), points, **density_kwargs
                )

                if not h5_path.exists() or overwrite:
                    with h5py.File(h5_path, "w") as fh:
                        fh.create_dataset(
                            "density", data=density, compression="gzip"
                        )
                        fh.attrs["count"] = float(len(points))
                        fh.attrs["density_sum"] = float(density.sum())
                        fh.attrs["image"] = image_path.name
                        fh.attrs["height"] = height
                        fh.attrs["width"] = width

                if n_viz < visualize:
                    _save_visualisation(
                        image,
                        density,
                        float(len(points)),
                        Path(cfg.paths.output_dir)
                        / "density_samples"
                        / f"part_{part}_{split}_{image_path.stem}.png",
                    )
                    n_viz += 1

    splits = _write_splits_json(
        out_dir,
        stems["train"],
        stems["test"],
        float(cfg.data.get("val_fraction", 0.1)),
        int(cfg.train.get("seed", 42)),
    )
    logger.info(
        "part_%s: %d train / %d val / %d test",
        part,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


def verify_part(cfg: Config, part: str, tolerance: float = 1e-2) -> tuple[int, int]:
    """Check every processed file's density sum against its head count.

    Returns ``(n_checked, n_bad)`` and logs each mismatch.
    """
    logger = get_logger("crowd.preprocess")
    out_dir = Path(cfg.paths.processed_dir) / f"part_{part.upper()}"
    checked = bad = 0
    for h5_path in sorted(out_dir.rglob("*.h5")):
        with h5py.File(h5_path, "r") as fh:
            count = float(fh.attrs["count"])
            total = float(np.asarray(fh["density"]).sum())
        checked += 1
        if abs(total - count) > tolerance:
            bad += 1
            logger.warning(
                "%s: density sum %.4f != count %.1f", h5_path.name, total, count
            )
    logger.info("Verified %d files, %d mismatches (tol %.0e)", checked, bad, tolerance)
    return checked, bad


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(
        argparse.ArgumentParser(description="Preprocess the ShanghaiTech dataset.")
    )
    parser.add_argument(
        "--part",
        default=None,
        help="Part to process: A, B, or 'both'. Defaults to data.part from config.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate existing .h5 files."
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        metavar="N",
        help="Save N side-by-side image/density PNGs per split to outputs/.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After processing, check every density sum against its head count.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not link/copy images into the processed tree.",
    )
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    requested = (args.part or cfg.data.part).upper()
    parts = ["A", "B"] if requested == "BOTH" else [requested]

    for part in parts:
        preprocess_part(
            cfg,
            part,
            overwrite=args.overwrite,
            visualize=args.visualize,
            copy_images=not args.no_copy_images,
        )
        if args.verify:
            _, bad = verify_part(cfg, part)
            if bad:
                return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
