"""Image and video inference.

Importable API::

    counter = CrowdCounter("checkpoints/part_B/best.pth")
    count, density = counter.predict_image(image)

CLI::

    python -m src.inference --checkpoint checkpoints/part_B/best.pth \
        --input path/to/img_or_video --output outputs/

Resizing an image changes its area but not the number of people in it, and the
density map is *summed* rather than resized, so no area correction is applied.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import Config, add_config_args, config_from_args, load_config
from .data.transforms import normalize
from .models import MCNN
from .utils import ensure_dir, get_device, get_logger, load_checkpoint
from .visualize import annotate_frame, density_to_heatmap, overlay

__all__ = ["CrowdCounter", "is_video", "main"]

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_video(path: str | Path) -> bool:
    """True when the path looks like a video file."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def _to_rgb_array(image: "np.ndarray | Image.Image") -> np.ndarray:
    """Normalise any supported input into an ``(H, W, 3)`` uint8 RGB array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype != np.uint8:
        # Accept float images in [0, 1] as well as [0, 255].
        peak = float(array.max()) if array.size else 0.0
        array = array * 255.0 if peak <= 1.0 else array
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class CrowdCounter:
    """Loads a checkpoint once and counts people in images or videos."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        cfg: Config | None = None,
        device: torch.device | str | None = None,
        max_side: int | None = None,
        density_scale: float | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.device = torch.device(device) if device else get_device()
        self.logger = get_logger("crowd.inference")

        self.model = MCNN(verbose=False).to(self.device)
        self.checkpoint = str(checkpoint) if checkpoint else None
        if checkpoint is not None:
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {path}. Train a model first "
                    f"(python -m src.train) or download one into "
                    f"{self.cfg.paths.checkpoint_dir}/."
                )
            ckpt = load_checkpoint(path, self.model, map_location=self.device)
            # Prefer the density scale the checkpoint was trained with.
            saved = (ckpt.get("config") or {}).get("train", {})
            if density_scale is None and "density_scale" in saved:
                density_scale = float(saved["density_scale"])
        else:
            self.logger.warning(
                "No checkpoint given: using randomly initialised weights."
            )
        self.model.eval()

        self.downsample = int(self.model.downsample)
        self.max_side = int(
            max_side if max_side is not None
            else self.cfg.inference.get("max_side", 1024)
        )
        self.density_scale = float(
            density_scale if density_scale is not None
            else self.cfg.train.get("density_scale", 1.0)
        )

    # -- preprocessing --------------------------------------------------
    def _prepare(self, rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize, normalise and pad; returns the tensor and the resized size."""
        height, width = rgb.shape[:2]
        longest = max(height, width)
        if self.max_side and longest > self.max_side:
            scale = self.max_side / longest
            width = max(self.downsample, int(round(width * scale)))
            height = max(self.downsample, int(round(height * scale)))
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = normalize(tensor.permute(2, 0, 1).contiguous())

        pad_h = (-height) % self.downsample
        pad_w = (-width) % self.downsample
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
        return tensor.unsqueeze(0), (height, width)

    # -- images ---------------------------------------------------------
    @torch.no_grad()
    def predict_image(
        self, image: "np.ndarray | Image.Image"
    ) -> tuple[float, np.ndarray]:
        """Count people in one image.

        Returns:
            ``(count, density)`` where ``density`` is the 2-D map at 1/4 of the
            *resized* input resolution, with the padding cropped off. Its sum
            equals the returned count.
        """
        rgb = _to_rgb_array(image)
        tensor, (height, width) = self._prepare(rgb)
        prediction = self.model(tensor.to(self.device))

        # Crop the padding back off before summing.
        out_h = height // self.downsample
        out_w = width // self.downsample
        density = prediction[0, 0, :out_h, :out_w].float().cpu().numpy()
        density = np.clip(density / self.density_scale, 0.0, None)
        return float(density.sum()), density

    def predict_batch(
        self, images: Iterable["np.ndarray | Image.Image"]
    ) -> list[tuple[float, np.ndarray]]:
        """Convenience wrapper: :meth:`predict_image` over an iterable."""
        return [self.predict_image(image) for image in images]

    def render(
        self,
        image: "np.ndarray | Image.Image",
        alpha: float | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Predict and render: returns ``(count, heatmap, overlay)`` as RGB."""
        rgb = _to_rgb_array(image)
        count, density = self.predict_image(rgb)
        alpha = float(
            alpha if alpha is not None
            else self.cfg.inference.get("overlay_alpha", 0.5)
        )
        heatmap = density_to_heatmap(density, (rgb.shape[1], rgb.shape[0]))
        return count, heatmap, overlay(rgb, heatmap, alpha)

    # -- video ----------------------------------------------------------
    @torch.no_grad()
    def predict_video(
        self,
        path: str | Path,
        frame_stride: int | None = None,
        output_video: str | Path | None = None,
        output_csv: str | Path | None = None,
        max_frames: int | None = None,
        alpha: float | None = None,
        progress: bool = True,
    ) -> list[tuple[int, float]]:
        """Count people every ``frame_stride`` frames of a video.

        Args:
            path: input video.
            frame_stride: process one frame in every ``frame_stride``.
            output_video: write an annotated MP4 here (skipped when ``None``).
            output_csv: write ``frame,time_s,count`` rows here.
            max_frames: stop after this many *processed* frames.
            alpha: heatmap opacity for the annotated video.

        Returns:
            ``[(frame_index, count), ...]`` for the processed frames.
        """
        path = Path(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        stride = int(
            frame_stride if frame_stride is not None
            else self.cfg.inference.get("video_frame_stride", 5)
        )
        stride = max(1, stride)
        max_frames = int(
            max_frames if max_frames is not None
            else self.cfg.inference.get("video_max_frames", 0) or 0
        )
        alpha = float(
            alpha if alpha is not None
            else self.cfg.inference.get("overlay_alpha", 0.5)
        )

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        writer = None
        results: list[tuple[int, float]] = []

        bar = tqdm(
            total=(total // stride) if total else None,
            desc=path.name,
            unit="frame",
            disable=not progress,
        )
        try:
            frame_index = 0
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    count, density = self.predict_image(rgb)
                    results.append((frame_index, count))

                    if output_video is not None:
                        annotated = annotate_frame(rgb, density, count, alpha)
                        if writer is None:
                            output_video = Path(output_video)
                            ensure_dir(output_video.parent)
                            writer = cv2.VideoWriter(
                                str(output_video),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                max(1.0, fps / stride),
                                (annotated.shape[1], annotated.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(
                                    f"Could not open video writer for {output_video}"
                                )
                        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                    bar.update(1)
                    if max_frames and len(results) >= max_frames:
                        break
                frame_index += 1
        finally:
            bar.close()
            capture.release()
            if writer is not None:
                writer.release()

        if output_csv is not None:
            output_csv = Path(output_csv)
            ensure_dir(output_csv.parent)
            with output_csv.open("w", newline="", encoding="utf-8") as fh:
                csv_writer = csv.writer(fh)
                csv_writer.writerow(["frame", "time_s", "count"])
                for index, count in results:
                    csv_writer.writerow([index, round(index / fps, 3), round(count, 2)])

        return results


def _iter_inputs(target: Path) -> list[Path]:
    """Expand a file or directory into the media files to process."""
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
        )
    return [target]


def run_cli(
    cfg: Config,
    checkpoint: str | Path,
    inputs: Sequence[Path],
    output_dir: Path,
    frame_stride: int | None = None,
    alpha: float | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """Process files, writing overlays / annotated videos into ``output_dir``."""
    logger = get_logger("crowd.inference")
    counter = CrowdCounter(checkpoint, cfg=cfg)
    ensure_dir(output_dir)
    summaries: list[dict] = []

    for item in inputs:
        if is_video(item):
            results = counter.predict_video(
                item,
                frame_stride=frame_stride,
                output_video=output_dir / f"{item.stem}_annotated.mp4",
                output_csv=output_dir / f"{item.stem}_counts.csv",
                max_frames=max_frames,
                alpha=alpha,
            )
            counts = [c for _, c in results]
            mean_count = float(np.mean(counts)) if counts else 0.0
            logger.info(
                "%s: %d frames, mean count %.1f", item.name, len(results), mean_count
            )
            summaries.append(
                {"input": str(item), "frames": len(results), "mean_count": mean_count}
            )
        else:
            with Image.open(item) as img:
                count, _, blended = counter.render(img, alpha=alpha)
            destination = output_dir / f"{item.stem}_overlay.png"
            cv2.imwrite(str(destination), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            logger.info("%s: count %.1f -> %s", item.name, count, destination)
            summaries.append(
                {"input": str(item), "count": count, "output": str(destination)}
            )

    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(
        argparse.ArgumentParser(description="Run MCNN inference on images or video.")
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth file.")
    parser.add_argument(
        "--input", required=True, help="Image, video, or directory of either."
    )
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--frame-stride", type=int, default=None, help="Video: process 1 in N frames."
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Video: cap processed frames."
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Heatmap overlay opacity."
    )
    args = parser.parse_args(argv)

    target = Path(args.input)
    if not target.exists():
        raise FileNotFoundError(f"Input not found: {target}")

    run_cli(
        config_from_args(args),
        args.checkpoint,
        _iter_inputs(target),
        Path(args.output),
        frame_stride=args.frame_stride,
        alpha=args.alpha,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
