"""Tests for the dataset, transforms and dataloaders (Module 2)."""

from __future__ import annotations

import torch

from src.data.dataset import CrowdDataset, build_dataloaders
from src.data.transforms import (
    EvalTransform,
    TrainTransform,
    paired_hflip,
    pad_to_multiple,
    sum_pool,
)


def test_train_item_shapes(processed_config) -> None:
    dataset = CrowdDataset(processed_config, "train", train=True)
    image, density, count = dataset[0]
    crop = int(processed_config.data.crop_size)
    factor = int(processed_config.data.downsample)

    assert image.shape == (3, crop, crop)
    assert density.shape == (1, crop // factor, crop // factor)
    assert image.dtype == torch.float32 and density.dtype == torch.float32
    assert isinstance(count, float)


def test_density_is_quarter_of_image(processed_config) -> None:
    dataset = CrowdDataset(processed_config, "test", train=False)
    image, density, _ = dataset[0]
    assert density.shape[-2] * 4 == image.shape[-2]
    assert density.shape[-1] * 4 == image.shape[-1]


def test_eval_count_matches_annotation(processed_config) -> None:
    dataset = CrowdDataset(processed_config, "test", train=False)
    for i in range(len(dataset)):
        _, _, annotated = dataset.load_raw(i)
        _, density, count = dataset[i]
        # Zero padding adds no mass, and sum-pooling preserves it.
        assert abs(count - annotated) < 1e-3
        assert abs(float(density.sum()) - annotated) < 1e-3


def test_cropped_count_does_not_exceed_full_count(processed_config) -> None:
    dataset = CrowdDataset(processed_config, "train", train=True)
    for i in range(len(dataset)):
        _, _, full = dataset.load_raw(i)
        _, _, cropped = dataset[i]
        assert cropped <= full + 1e-3


def test_hflip_preserves_count() -> None:
    image = torch.rand(3, 32, 48)
    density = torch.rand(1, 32, 48)
    flipped_image, flipped_density = paired_hflip(image, density)
    assert torch.allclose(flipped_density.sum(), density.sum())
    assert flipped_image.shape == image.shape
    # Flipping twice is the identity, so image and density stay aligned.
    assert torch.equal(paired_hflip(flipped_image, flipped_density)[1], density)


def test_sum_pool_preserves_mass() -> None:
    density = torch.rand(1, 64, 64)
    pooled = sum_pool(density, 4)
    assert pooled.shape == (1, 16, 16)
    assert torch.allclose(pooled.sum(), density.sum(), atol=1e-4)


def test_pad_to_multiple_adds_no_mass() -> None:
    image = torch.rand(3, 333, 517)
    density = torch.rand(1, 333, 517)
    padded_image, padded_density = pad_to_multiple(image, density, 4)
    assert padded_image.shape[-2] % 4 == 0 and padded_image.shape[-1] % 4 == 0
    assert padded_image.shape[-2:] == padded_density.shape[-2:]
    assert torch.allclose(padded_density.sum(), density.sum())


def test_small_image_is_padded_to_crop_size() -> None:
    transform = TrainTransform(crop_size=256, downsample=4, hflip=False)
    image, density, count = transform(torch.rand(3, 100, 90), torch.rand(1, 100, 90))
    assert image.shape == (3, 256, 256)
    assert density.shape == (1, 64, 64)
    assert count >= 0


def test_density_scale_is_applied() -> None:
    density = torch.rand(1, 64, 64)
    scaled = EvalTransform(downsample=4, density_scale=100.0)(
        torch.rand(3, 64, 64), density
    )[1]
    plain = EvalTransform(downsample=4, density_scale=1.0)(
        torch.rand(3, 64, 64), density
    )[1]
    assert torch.allclose(scaled, plain * 100.0, atol=1e-3)


def test_build_dataloaders_one_epoch(processed_config) -> None:
    loaders = build_dataloaders(processed_config)
    assert set(loaders) == {"train", "val", "test"}

    seen = 0
    for images, densities, counts in loaders["train"]:
        assert images.ndim == 4 and densities.ndim == 4
        assert images.shape[0] == densities.shape[0] == counts.shape[0]
        seen += images.shape[0]
    assert seen == len(loaders["train"].dataset)

    for images, densities, counts in loaders["val"]:
        assert images.shape[0] == 1
        assert images.shape[-2] % 4 == 0 and images.shape[-1] % 4 == 0


def test_splits_are_disjoint(processed_config) -> None:
    train = set(CrowdDataset(processed_config, "train").stems)
    val = set(CrowdDataset(processed_config, "val", train=False).stems)
    assert train and val and not (train & val)
