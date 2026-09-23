"""Density-map generation from head point annotations.

A density map has the same spatial size as its image and sums to the number of
annotated heads. Each head contributes a normalised 2-D Gaussian; the Gaussian
is built once at the required sigma and pasted locally rather than filtering the
whole image per point, which keeps dense Part A images fast.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial import KDTree

__all__ = [
    "generate_density_map",
    "adaptive_sigmas",
    "gaussian_kernel",
]

# Sigmas are rounded to this resolution before caching kernels, so that the
# thousands of slightly different adaptive sigmas in a dense image reuse a
# handful of kernels instead of rebuilding one each time.
_SIGMA_QUANTUM = 0.05
_MIN_SIGMA = 0.5
_MAX_SIGMA = 64.0


@lru_cache(maxsize=512)
def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Return a square, unit-sum Gaussian kernel of radius ``truncate*sigma``.

    Args:
        sigma: standard deviation in pixels.
        truncate: kernel radius expressed in sigmas.

    Returns:
        ``(2r+1, 2r+1)`` float32 array summing to 1.0.
    """
    radius = max(1, int(round(truncate * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    line = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel = np.outer(line, line)
    total = kernel.sum()
    if total <= 0:  # pragma: no cover - only for absurdly small sigma
        kernel = np.zeros_like(kernel)
        kernel[radius, radius] = 1.0
        return kernel
    return (kernel / total).astype(np.float32)


def _quantise(sigma: float) -> float:
    """Clamp and round a sigma so kernels can be cached and reused."""
    sigma = float(np.clip(sigma, _MIN_SIGMA, _MAX_SIGMA))
    return round(sigma / _SIGMA_QUANTUM) * _SIGMA_QUANTUM


def adaptive_sigmas(
    points: np.ndarray,
    beta: float = 0.3,
    k: int = 3,
    fallback_sigma: float = 4.0,
) -> np.ndarray:
    """Geometry-adaptive sigmas: ``beta * mean distance to the k nearest heads``.

    With fewer than two points there are no neighbours to measure, so every
    point falls back to ``fallback_sigma``.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.full(1, fallback_sigma, dtype=np.float32)

    k_eff = min(k, n - 1)
    tree = KDTree(points)
    # The first neighbour of a point is itself (distance 0), so ask for k+1.
    distances, _ = tree.query(points, k=k_eff + 1)
    distances = np.atleast_2d(distances)
    mean_dist = distances[:, 1:].mean(axis=1)
    sigmas = beta * mean_dist
    # Coincident annotations give a zero mean distance; keep them usable.
    sigmas[sigmas <= 0] = fallback_sigma
    return sigmas.astype(np.float32)


def generate_density_map(
    shape: tuple[int, int],
    points: np.ndarray,
    mode: str = "fixed",
    fixed_sigma: float = 4.0,
    beta: float = 0.3,
    k: int = 3,
    truncate: float = 3.0,
) -> np.ndarray:
    """Build a density map whose sum equals the number of in-bounds heads.

    Args:
        shape: ``(H, W)`` of the target map.
        points: ``(N, 2)`` array of ``(x, y)`` head coordinates.
        mode: ``"adaptive"`` (sigma from k-nearest-neighbour distance, the
            Part A setting) or ``"fixed"`` (constant sigma, the Part B setting).
        fixed_sigma: sigma for ``"fixed"`` mode and the single-point fallback.
        beta: scale applied to the mean neighbour distance in adaptive mode.
        k: number of neighbours used in adaptive mode.
        truncate: kernel radius in sigmas.

    Returns:
        ``(H, W)`` float32 density map. Head coordinates are clipped into the
        image bounds (real annotations occasionally place a head a fraction of
        a pixel past the last row/column, e.g. ShanghaiTech Part B test image
        126); every head contributes exactly 1.0, including heads on a border
        whose kernel is clipped and heads clipped onto the border itself.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid density map shape: {shape}")

    density = np.zeros((height, width), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return density

    # Round to the containing pixel, then clip onto the image so every
    # annotated head is kept (the density sum should equal the head count).
    xs = np.clip(np.round(points[:, 0]), 0, width - 1).astype(np.int64)
    ys = np.clip(np.round(points[:, 1]), 0, height - 1).astype(np.int64)

    mode = mode.lower()
    if mode == "adaptive":
        sigmas = adaptive_sigmas(
            points, beta=beta, k=k, fallback_sigma=fixed_sigma
        )
    elif mode == "fixed":
        sigmas = np.full(xs.size, fixed_sigma, dtype=np.float32)
    else:
        raise ValueError(f"Unknown density mode: {mode!r}")

    for x, y, sigma in zip(xs, ys, sigmas):
        kernel = gaussian_kernel(_quantise(float(sigma)), truncate)
        radius = kernel.shape[0] // 2

        # Region of the image the kernel lands on, clipped to the bounds ...
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        # ... and the matching slice of the kernel itself.
        ky0, ky1 = y0 - (y - radius), kernel.shape[0] - ((y + radius + 1) - y1)
        kx0, kx1 = x0 - (x - radius), kernel.shape[1] - ((x + radius + 1) - x1)

        patch = kernel[ky0:ky1, kx0:kx1]
        patch_sum = patch.sum()
        if patch_sum <= 0:  # pragma: no cover - kernel entirely outside
            continue
        # Renormalise so a clipped kernel still contributes exactly 1.0.
        density[y0:y1, x0:x1] += patch / patch_sum

    return density
