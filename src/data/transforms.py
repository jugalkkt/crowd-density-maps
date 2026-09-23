"""Paired image / density-map transforms.

Every geometric operation is applied to the image *and* its density map with
the same random parameters, and the density map is reduced to the model's
output stride by **sum**-pooling, which preserves the count exactly.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "normalize",
    "denormalize",
    "pad_to_multiple",
    "pad_to_at_least",
    "paired_random_crop",
    "paired_hflip",
    "jitter_image",
    "sum_pool",
    "TrainTransform",
    "EvalTransform",
]

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def normalize(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalisation to a ``[3, H, W]`` float tensor."""
    return TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Invert :func:`normalize`, returning values back in ``[0, 1]``."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def pad_to_multiple(
    image: torch.Tensor, density: torch.Tensor | None, multiple: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Zero-pad the bottom/right so both sides are divisible by ``multiple``.

    Padding with zeros adds no density, so the count is unchanged.
    """
    _, height, width = image.shape
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, density
    image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    if density is not None:
        density = F.pad(density, (0, pad_w, 0, pad_h), value=0.0)
    return image, density


def pad_to_at_least(
    image: torch.Tensor, density: torch.Tensor | None, size: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Zero-pad the bottom/right so both sides are at least ``size`` pixels."""
    _, height, width = image.shape
    pad_h = max(0, size - height)
    pad_w = max(0, size - width)
    if pad_h == 0 and pad_w == 0:
        return image, density
    image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    if density is not None:
        density = F.pad(density, (0, pad_w, 0, pad_h), value=0.0)
    return image, density


def paired_random_crop(
    image: torch.Tensor, density: torch.Tensor, size: int, rng: random.Random | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the same random ``size x size`` window from image and density."""
    rng = rng or random
    _, height, width = image.shape
    top = rng.randint(0, height - size) if height > size else 0
    left = rng.randint(0, width - size) if width > size else 0
    return (
        image[:, top : top + size, left : left + size],
        density[:, top : top + size, left : left + size],
    )


def paired_hflip(
    image: torch.Tensor, density: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror image and density together; the count is unchanged."""
    return torch.flip(image, dims=[-1]), torch.flip(density, dims=[-1])


def jitter_image(
    image: torch.Tensor,
    strength: float = 0.0,
    grayscale_prob: float = 0.0,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Photometric augmentation on the image only (density is unaffected)."""
    rng = rng or random
    if strength > 0:
        image = TF.adjust_brightness(image, 1.0 + rng.uniform(-strength, strength))
        image = TF.adjust_contrast(image, 1.0 + rng.uniform(-strength, strength))
        image = TF.adjust_saturation(image, 1.0 + rng.uniform(-strength, strength))
        image = image.clamp(0.0, 1.0)
    if grayscale_prob > 0 and rng.random() < grayscale_prob:
        image = TF.rgb_to_grayscale(image, num_output_channels=3)
    return image


def sum_pool(density: torch.Tensor, factor: int) -> torch.Tensor:
    """Reduce a ``[1, H, W]`` density map by ``factor`` using sum-pooling.

    ``avg_pool2d * factor**2`` is the sum over each window, so the total mass —
    the crowd count — is preserved exactly. A bilinear resize would not be.
    """
    if factor == 1:
        return density
    _, height, width = density.shape
    if height % factor or width % factor:
        raise ValueError(
            f"Density map {height}x{width} is not divisible by {factor}; "
            "pad before pooling."
        )
    pooled = F.avg_pool2d(density.unsqueeze(0), factor).squeeze(0)
    return pooled * (factor * factor)


class TrainTransform:
    """Random crop + flip + photometric jitter, then normalise and sum-pool."""

    def __init__(
        self,
        crop_size: int = 256,
        downsample: int = 4,
        hflip: bool = True,
        color_jitter: float = 0.0,
        grayscale_prob: float = 0.0,
        density_scale: float = 1.0,
    ) -> None:
        if crop_size % downsample:
            raise ValueError(
                f"crop_size ({crop_size}) must be a multiple of "
                f"downsample ({downsample})."
            )
        self.crop_size = crop_size
        self.downsample = downsample
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.grayscale_prob = grayscale_prob
        self.density_scale = density_scale

    def __call__(
        self, image: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        image, density = pad_to_at_least(image, density, self.crop_size)
        image, density = paired_random_crop(image, density, self.crop_size)
        if self.hflip and random.random() < 0.5:
            image, density = paired_hflip(image, density)
        image = jitter_image(image, self.color_jitter, self.grayscale_prob)
        count = float(density.sum())
        target = sum_pool(density, self.downsample) * self.density_scale
        return normalize(image), target, count


class EvalTransform:
    """Full image, zero-padded to a multiple of the model's output stride."""

    def __init__(self, downsample: int = 4, density_scale: float = 1.0) -> None:
        self.downsample = downsample
        self.density_scale = density_scale

    def __call__(
        self, image: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        image, density = pad_to_multiple(image, density, self.downsample)
        count = float(density.sum())
        target = sum_pool(density, self.downsample) * self.density_scale
        return normalize(image), target, count
