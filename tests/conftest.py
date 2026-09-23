"""Shared fixtures: a synthetic ShanghaiTech-shaped dataset on disk.

The real dataset is a multi-GB download, so the tests build a tiny raw tree
with the same folder layout and ``.mat`` annotation structure, then run the
real preprocessing code over it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from src.config import load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def write_annotation(mat_path: Path, points: np.ndarray) -> None:
    """Write head points in ShanghaiTech's ``image_info`` cell/struct layout."""
    inner = np.zeros((1, 1), dtype=[("location", "O"), ("number", "O")])
    inner[0, 0]["location"] = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inner[0, 0]["number"] = np.array([[len(points)]], dtype=np.float64)
    cell = np.zeros((1, 1), dtype=object)
    cell[0, 0] = inner
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(mat_path), {"image_info": cell})


def make_raw_dataset(
    root: Path,
    part: str = "B",
    n_train: int = 6,
    n_test: int = 3,
    size: tuple[int, int] = (120, 160),
    seed: int = 0,
) -> Path:
    """Create ``root/ShanghaiTech/part_X/{train,test}_data/{images,ground-truth}``."""
    rng = np.random.default_rng(seed)
    height, width = size
    part_dir = root / "ShanghaiTech" / f"part_{part}"

    for split, n_images in (("train", n_train), ("test", n_test)):
        images_dir = part_dir / f"{split}_data" / "images"
        gt_dir = part_dir / f"{split}_data" / "ground-truth"
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, n_images + 1):
            pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(images_dir / f"IMG_{i}.jpg", quality=90)
            n_points = int(rng.integers(3, 12))
            points = np.stack(
                [
                    rng.uniform(0, width - 1, n_points),
                    rng.uniform(0, height - 1, n_points),
                ],
                axis=1,
            )
            write_annotation(gt_dir / f"GT_IMG_{i}.mat", points)

    return part_dir


@pytest.fixture(scope="session")
def base_config():
    """The project's default config, loaded once."""
    return load_config(CONFIG_PATH)


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    """A synthetic raw dataset root (the parent of ``ShanghaiTech/``)."""
    raw_root = tmp_path / "raw"
    make_raw_dataset(raw_root, part="B")
    return raw_root


@pytest.fixture
def processed_config(tmp_path: Path, raw_dataset: Path):
    """A config pointed at the synthetic dataset, already preprocessed."""
    from src.data.preprocess import preprocess_part

    cfg = load_config(
        CONFIG_PATH,
        [
            f"paths.raw_data_dir={raw_dataset}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.checkpoint_dir={tmp_path / 'checkpoints'}",
            f"paths.output_dir={tmp_path / 'outputs'}",
            "data.part=B",
            "data.crop_size=64",
            "data.val_fraction=0.34",
            "train.num_workers=0",
        ],
    )
    preprocess_part(cfg, "B")
    return cfg
