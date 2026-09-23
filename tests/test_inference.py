"""Tests for the inference API, visualisation and evaluation (Modules 5-6)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.evaluate import evaluate
from src.inference import CrowdCounter, is_video
from src.models import MCNN
from src.utils import save_checkpoint
from src.visualize import density_to_heatmap, draw_count, overlay


@pytest.fixture
def counter(base_config) -> CrowdCounter:
    """An untrained counter — enough to exercise shapes and plumbing."""
    return CrowdCounter(checkpoint=None, cfg=base_config, device="cpu")


@pytest.mark.parametrize("size", [(333, 517), (64, 64), (101, 97)])
def test_predict_image_handles_odd_sizes(counter: CrowdCounter, size) -> None:
    height, width = size
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    count, density = counter.predict_image(image)

    assert isinstance(count, float)
    assert count >= 0
    assert density.ndim == 2
    assert (density >= 0).all()
    assert density.shape == (height // 4, width // 4)
    assert count == pytest.approx(float(density.sum()), rel=1e-5)


def test_predict_image_accepts_pil_grayscale_and_float(counter: CrowdCounter) -> None:
    pil = Image.fromarray(
        np.random.randint(0, 255, (80, 120), dtype=np.uint8)
    )  # single channel
    count, density = counter.predict_image(pil)
    assert density.shape == (20, 30) and count >= 0

    as_float = np.random.rand(80, 120, 3).astype(np.float32)
    count2, _ = counter.predict_image(as_float)
    assert count2 >= 0


def test_large_image_is_resized_to_max_side(base_config) -> None:
    counter = CrowdCounter(checkpoint=None, cfg=base_config, device="cpu", max_side=128)
    _, density = counter.predict_image(
        np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    )
    # Longest side capped at 128 -> density at 1/4 of that.
    assert max(density.shape) == 128 // 4


def test_render_returns_input_sized_images(counter: CrowdCounter) -> None:
    image = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
    count, heatmap, blended = counter.render(image, alpha=0.5)
    assert heatmap.shape == image.shape
    assert blended.shape == image.shape
    assert blended.dtype == np.uint8
    assert count >= 0


def test_heatmap_and_overlay_shapes() -> None:
    density = np.abs(np.random.randn(16, 24)).astype(np.float32)
    image = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
    heatmap = density_to_heatmap(density, target_size=(96, 64))
    assert heatmap.shape == (64, 96, 3) and heatmap.dtype == np.uint8
    assert overlay(image, heatmap, 0.5).shape == image.shape
    assert draw_count(image, 42.0).shape == image.shape


def test_zero_density_heatmap_is_safe() -> None:
    heatmap = density_to_heatmap(np.zeros((8, 8), dtype=np.float32), (32, 32))
    assert heatmap.shape == (32, 32, 3)


def _write_tiny_video(path: Path, frames: int = 10, size=(96, 64)) -> Path:
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()
    return path


def test_predict_video_writes_outputs(counter: CrowdCounter, tmp_path: Path) -> None:
    video = _write_tiny_video(tmp_path / "clip.mp4", frames=10)
    out_video = tmp_path / "out" / "clip_annotated.mp4"
    out_csv = tmp_path / "out" / "clip_counts.csv"

    results = counter.predict_video(
        video, frame_stride=2, output_video=out_video, output_csv=out_csv,
        progress=False,
    )

    assert len(results) == 5
    assert [i for i, _ in results] == [0, 2, 4, 6, 8]
    assert all(c >= 0 for _, c in results)
    assert out_video.is_file() and out_video.stat().st_size > 0
    assert out_csv.is_file()

    with out_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5
    assert set(rows[0]) == {"frame", "time_s", "count"}


def test_predict_video_respects_max_frames(counter: CrowdCounter, tmp_path: Path) -> None:
    video = _write_tiny_video(tmp_path / "clip.mp4", frames=10)
    results = counter.predict_video(
        video, frame_stride=1, max_frames=3, progress=False
    )
    assert len(results) == 3


def test_is_video() -> None:
    assert is_video("a/b/clip.MP4") and is_video("clip.mov")
    assert not is_video("photo.jpg")


def test_missing_checkpoint_raises_helpful_error(base_config, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        CrowdCounter(tmp_path / "nope.pth", cfg=base_config, device="cpu")


def test_counter_loads_a_checkpoint(base_config, tmp_path: Path) -> None:
    path = tmp_path / "best.pth"
    save_checkpoint(path, MCNN(verbose=False), epoch=1, best_mae=1.0, config=base_config)
    counter = CrowdCounter(path, cfg=base_config, device="cpu")
    count, _ = counter.predict_image(np.zeros((32, 32, 3), dtype=np.uint8))
    assert count >= 0


def test_evaluate_writes_report(processed_config, tmp_path: Path) -> None:
    checkpoint = tmp_path / "eval.pth"
    save_checkpoint(
        checkpoint, MCNN(verbose=False), epoch=1, best_mae=1.0, config=processed_config
    )

    summary = evaluate(
        processed_config, checkpoint, part="B", save_predictions=True, n_worst=2
    )
    output_dir = Path(processed_config.paths.output_dir)

    assert summary["n_images"] == 3
    assert summary["mae"] >= 0 and summary["rmse"] >= summary["mae"]
    assert (output_dir / "eval_part_B.json").is_file()
    assert (output_dir / "scatter_part_B.png").is_file()
    assert (output_dir / "predictions_part_B.csv").is_file()
    assert len(list((output_dir / "worst_part_B").glob("*.png"))) == 2

    with (output_dir / "eval_part_B.json").open(encoding="utf-8") as fh:
        assert json.load(fh)["part"] == "B"
