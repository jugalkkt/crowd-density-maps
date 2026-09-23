"""PyTorch dataset and dataloaders over the preprocessed ShanghaiTech tree."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .transforms import EvalTransform, TrainTransform

__all__ = ["CrowdDataset", "build_dataloaders", "collate_single"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class CrowdDataset(Dataset):
    """Image / density-map pairs for one split of one ShanghaiTech part.

    Each item is ``(image [3, H, W], density [1, H/s, W/s], count)`` where ``s``
    is ``data.downsample``. ``count`` is the mass of the *returned* target
    before ``train.density_scale`` is applied, so it matches a random crop
    rather than the whole image during training; on full images it equals the
    annotated head count.
    """

    def __init__(
        self,
        cfg: Config,
        split: str,
        train: bool | None = None,
        part: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.part = (part or cfg.data.part).upper()
        self.train = bool(train) if train is not None else (split == "train")

        self.root = Path(cfg.paths.processed_dir) / f"part_{self.part}"
        splits_file = self.root / "splits.json"
        if not splits_file.is_file():
            raise FileNotFoundError(
                f"{splits_file} not found. Run: python -m src.data.preprocess "
                f"--part {self.part}"
            )
        with splits_file.open("r", encoding="utf-8") as fh:
            splits = json.load(fh)
        if split not in splits:
            raise KeyError(f"Unknown split {split!r}; have {sorted(splits)}")

        # train/val both come from the raw train_data folder; test from test_data.
        self.source_dir = self.root / ("test" if split == "test" else "train")
        self.stems: list[str] = list(splits[split])
        if not self.stems:
            raise ValueError(f"Split {split!r} of part_{self.part} is empty.")

        downsample = int(cfg.data.get("downsample", 4))
        density_scale = float(cfg.train.get("density_scale", 1.0))
        if self.train:
            aug = cfg.train.get("augment", Config())
            self.transform = TrainTransform(
                crop_size=int(cfg.data.get("crop_size", 256)),
                downsample=downsample,
                hflip=bool(aug.get("hflip", True)),
                color_jitter=float(aug.get("color_jitter", 0.0)),
                grayscale_prob=float(aug.get("grayscale_prob", 0.0)),
                density_scale=density_scale,
            )
        else:
            self.transform = EvalTransform(
                downsample=downsample, density_scale=density_scale
            )

    def __len__(self) -> int:
        return len(self.stems)

    def _image_path(self, stem: str) -> Path:
        images_dir = self.source_dir / "images"
        for suffix in IMAGE_SUFFIXES:
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image for {stem} under {images_dir}")

    def load_raw(self, index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Load the untransformed ``(image, density, annotated_count)``."""
        stem = self.stems[index]
        with Image.open(self._image_path(stem)) as img:
            # Part A contains grayscale images; force three channels.
            array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()

        with h5py.File(self.source_dir / f"{stem}.h5", "r") as fh:
            density = np.asarray(fh["density"], dtype=np.float32)
            annotated = float(fh.attrs.get("count", float(density.sum())))
        density_t = torch.from_numpy(density).unsqueeze(0)

        if image.shape[-2:] != density_t.shape[-2:]:
            raise ValueError(
                f"{stem}: image {tuple(image.shape[-2:])} and density "
                f"{tuple(density_t.shape[-2:])} disagree; re-run preprocessing."
            )
        return image, density_t, annotated

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        image, density, _ = self.load_raw(index)
        image, target, count = self.transform(image, density)
        return image, target, count


def collate_single(batch):
    """Collate for batch size 1 / variable-sized eval images."""
    images, targets, counts = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(targets),
        torch.tensor(counts, dtype=torch.float32),
    )


def build_dataloaders(
    cfg: Config, part: str | None = None, splits: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, DataLoader]:
    """Build the train / val / test loaders described by ``cfg``.

    Validation and test always use batch size 1, because full images have
    different sizes and cannot be stacked.
    """
    num_workers = int(cfg.train.get("num_workers", 0))
    loaders: dict[str, DataLoader] = {}

    for split in splits:
        is_train = split == "train"
        dataset = CrowdDataset(cfg, split, train=is_train, part=part)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(cfg.train.get("batch_size", 1)) if is_train else 1,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_single,
            persistent_workers=num_workers > 0,
        )
    return loaders
