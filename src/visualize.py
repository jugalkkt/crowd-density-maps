"""Heatmap rendering, overlays and diagnostic plots.

Everything here is for display only: density maps are upsampled and normalised
for visualisation, so never read a count back out of a heatmap — sum the raw
density map instead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir

__all__ = [
    "density_to_heatmap",
    "overlay",
    "draw_count",
    "annotate_frame",
    "save_comparison",
    "save_scatter",
]


def density_to_heatmap(
    density: np.ndarray,
    target_size: tuple[int, int] | None = None,
    colormap: int = cv2.COLORMAP_JET,
    vmax: float | None = None,
) -> np.ndarray:
    """Render a density map as an RGB heatmap.

    Args:
        density: ``(h, w)`` non-negative density map.
        target_size: ``(width, height)`` to upsample to, or ``None`` to keep size.
        colormap: an OpenCV colormap constant.
        vmax: value mapped to the top of the colour range; defaults to the max.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    density = np.asarray(density, dtype=np.float32)
    if density.ndim != 2:
        density = density.squeeze()
    if density.ndim != 2:
        raise ValueError(f"Expected a 2-D density map, got shape {density.shape}")

    if target_size is not None:
        density = cv2.resize(
            density, (int(target_size[0]), int(target_size[1])),
            interpolation=cv2.INTER_CUBIC,
        )
        density = np.clip(density, 0.0, None)

    peak = float(vmax if vmax is not None else density.max())
    normalised = density / peak if peak > 0 else np.zeros_like(density)
    scaled = (np.clip(normalised, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(scaled, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend ``heatmap`` over ``image`` (both RGB uint8) with weight ``alpha``."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(
            heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(heatmap, alpha, image, 1.0 - alpha, 0.0)


def draw_count(
    image: np.ndarray,
    count: float,
    label: str = "Count",
    origin: tuple[int, int] = (12, 12),
) -> np.ndarray:
    """Draw a ``Count: N`` box in the top-left corner of a copy of ``image``."""
    canvas = np.ascontiguousarray(image.copy())
    text = f"{label}: {count:.1f}"
    scale = max(0.6, min(canvas.shape[1] / 900.0, 1.6))
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = origin
    pad = int(8 * scale)
    cv2.rectangle(
        canvas,
        (x, y),
        (x + text_w + 2 * pad, y + text_h + baseline + 2 * pad),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        canvas,
        text,
        (x + pad, y + text_h + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def annotate_frame(
    image: np.ndarray,
    density: np.ndarray,
    count: float,
    alpha: float = 0.5,
    vmax: float | None = None,
) -> np.ndarray:
    """Heatmap overlay plus a count box — one call for video/demo frames."""
    heatmap = density_to_heatmap(
        density, target_size=(image.shape[1], image.shape[0]), vmax=vmax
    )
    return draw_count(overlay(image, heatmap, alpha), count)


def save_comparison(
    image: np.ndarray,
    gt_density: np.ndarray,
    pred_density: np.ndarray,
    path: str | Path,
    title: str = "",
) -> Path:
    """Save an image / ground-truth / prediction triptych for inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_density = np.asarray(gt_density).squeeze()
    pred_density = np.asarray(pred_density).squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(np.asarray(image))
    axes[0].set_title(title or "image")
    axes[1].imshow(gt_density, cmap="jet")
    axes[1].set_title(f"ground truth: {gt_density.sum():.1f}")
    # Share the colour range so the two density maps are comparable by eye.
    axes[2].imshow(pred_density, cmap="jet", vmin=0, vmax=max(gt_density.max(), 1e-8))
    axes[2].set_title(f"prediction: {pred_density.sum():.1f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_scatter(
    gt_counts, pred_counts, path: str | Path, title: str = "Predicted vs GT counts"
) -> Path:
    """Scatter predicted against ground-truth counts with the ``y = x`` line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.asarray(gt_counts, dtype=np.float64)
    pred = np.asarray(pred_counts, dtype=np.float64)
    limit = float(max(gt.max(initial=1.0), pred.max(initial=1.0))) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, pred, s=18, alpha=0.65, edgecolors="none")
    ax.plot([0, limit], [0, limit], "r--", linewidth=1, label="y = x")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("ground-truth count")
    ax.set_ylabel("predicted count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
