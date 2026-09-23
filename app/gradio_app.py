"""Gradio web demo for MCNN crowd counting.

    python app/gradio_app.py                       # local, CPU
    python app/gradio_app.py --share               # public link (Colab)
    python app/gradio_app.py --enable-video        # adds the video tab

Models are loaded lazily and cached, so switching between the dense (Part A)
and sparse (Part B) checkpoints costs nothing after the first use.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

# Allow `python app/gradio_app.py` from a clean checkout, without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.inference import CrowdCounter  # noqa: E402
from src.visualize import density_to_heatmap  # noqa: E402

MODEL_CHOICES = {
    "Dense crowds (Part A)": "A",
    "Sparse crowds (Part B)": "B",
}

COUNT_TEMPLATE = """
<div style="text-align:center;padding:18px 8px;">
  <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;
              opacity:.65;">Estimated people</div>
  <div style="font-size:56px;font-weight:700;line-height:1.1;">{count}</div>
  <div style="font-size:12px;opacity:.65;">{note}</div>
</div>
"""


class ModelRegistry:
    """Lazily loads and caches one :class:`CrowdCounter` per part."""

    def __init__(self, cfg, checkpoints: dict[str, Path], device: str = "cpu") -> None:
        self.cfg = cfg
        self.checkpoints = checkpoints
        self.device = device
        self._cache: dict[str, CrowdCounter] = {}

    def available(self) -> list[str]:
        """Labels whose checkpoint file actually exists on disk."""
        return [
            label
            for label, part in MODEL_CHOICES.items()
            if self.checkpoints[part].is_file()
        ]

    def missing_message(self) -> str:
        """Instructions shown when no checkpoint could be found."""
        lines = "\n".join(
            f"- **Part {part}** → `{path}`" for part, path in self.checkpoints.items()
        )
        return (
            "### No trained model found\n\n"
            "The demo needs at least one checkpoint. Expected locations:\n\n"
            f"{lines}\n\n"
            "Train one with `python -m src.train --set data.part=B`, or pass "
            "`--checkpoint-a` / `--checkpoint-b` to point at a file you already "
            "have."
        )

    def get(self, label: str) -> CrowdCounter:
        part = MODEL_CHOICES[label]
        if part not in self._cache:
            path = self.checkpoints[part]
            if not path.is_file():
                raise gr.Error(
                    f"No checkpoint for Part {part} at {path}. Train one with "
                    f"`python -m src.train --set data.part={part}`."
                )
            self._cache[part] = CrowdCounter(path, cfg=self.cfg, device=self.device)
        return self._cache[part]


def build_interface(
    registry: ModelRegistry, enable_video: bool, examples_dir: Path
) -> gr.Blocks:
    """Assemble the Gradio UI."""
    available = registry.available()
    default_choice = available[0] if available else list(MODEL_CHOICES)[0]

    def count_image(image, model_label: str, alpha: float):
        if image is None:
            raise gr.Error("Upload an image first.")
        counter = registry.get(model_label)
        rgb = np.asarray(image)
        count, _, blended = counter.render(rgb, alpha=alpha)
        _, density = counter.predict_image(rgb)
        heatmap = density_to_heatmap(density)
        note = f"model: {model_label} · density sums to {density.sum():.2f}"
        return (
            COUNT_TEMPLATE.format(count=f"{count:.0f}", note=note),
            blended,
            heatmap,
        )

    def count_video(video_path, model_label: str, alpha: float, stride: int):
        if not video_path:
            raise gr.Error("Upload a video first.")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counter = registry.get(model_label)
        out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
        results = counter.predict_video(
            video_path,
            frame_stride=int(stride),
            output_video=out_path,
            max_frames=int(registry.cfg.inference.get("video_max_frames", 300)),
            alpha=alpha,
            progress=False,
        )
        if not results:
            raise gr.Error("No frames could be read from that video.")

        frames = [f for f, _ in results]
        counts = [c for _, c in results]
        figure, ax = plt.subplots(figsize=(7, 3))
        ax.plot(frames, counts, linewidth=1.6)
        ax.set_xlabel("frame")
        ax.set_ylabel("people")
        ax.set_title(f"mean {np.mean(counts):.1f} · peak {max(counts):.1f}")
        ax.grid(alpha=0.25)
        figure.tight_layout()

        summary = COUNT_TEMPLATE.format(
            count=f"{np.mean(counts):.0f}",
            note=f"mean over {len(results)} processed frames",
        )
        return summary, str(out_path), figure

    # Gradio 6 moved `theme` from the Blocks constructor to launch(); pass it
    # wherever the installed version accepts it so both 4/5 and 6 work.
    blocks_kwargs = {"title": "Crowd Counting (MCNN)"}
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kwargs["theme"] = gr.themes.Soft()

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# Crowd counting with density maps\n"
            "A Multi-Column CNN predicts a density map whose pixels sum to the "
            "number of people in the scene. Bright regions are where the model "
            "thinks the heads are."
        )
        if not available:
            gr.Markdown(registry.missing_message())

        with gr.Row():
            model_selector = gr.Radio(
                choices=list(MODEL_CHOICES),
                value=default_choice,
                label="Model",
                info="Part A is trained on dense crowds, Part B on street scenes.",
            )
            alpha_slider = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05, label="Heatmap opacity"
            )

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="numpy", label="Input image")
                        image_button = gr.Button("Count people", variant="primary")
                    with gr.Column():
                        count_output = gr.HTML(
                            COUNT_TEMPLATE.format(count="—", note="upload an image")
                        )
                        overlay_output = gr.Image(label="Heatmap overlay")
                        density_output = gr.Image(label="Raw density map")

                examples = sorted(
                    p for p in examples_dir.glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ) if examples_dir.is_dir() else []
                if examples:
                    gr.Examples(
                        examples=[[str(p)] for p in examples],
                        inputs=[image_input],
                        label="Examples",
                    )

                image_button.click(
                    count_image,
                    inputs=[image_input, model_selector, alpha_slider],
                    outputs=[count_output, overlay_output, density_output],
                )

            if enable_video:
                with gr.Tab("Video"):
                    gr.Markdown(
                        "Short clips only — every Nth frame is processed on CPU."
                    )
                    with gr.Row():
                        with gr.Column():
                            video_input = gr.Video(label="Input video")
                            stride_slider = gr.Slider(
                                1, 30, value=5, step=1, label="Frame stride"
                            )
                            video_button = gr.Button("Count people", variant="primary")
                        with gr.Column():
                            video_count = gr.HTML(
                                COUNT_TEMPLATE.format(
                                    count="—", note="upload a video"
                                )
                            )
                            video_output = gr.Video(label="Annotated video")
                            plot_output = gr.Plot(label="Count over time")

                    video_button.click(
                        count_video,
                        inputs=[
                            video_input, model_selector, alpha_slider, stride_slider
                        ],
                        outputs=[video_count, video_output, plot_output],
                    )

        gr.Markdown(
            "Counts are the sum of the predicted density map, so they are "
            "continuous rather than integer — the model estimates crowd mass, "
            "it does not detect individual people."
        )

    return demo


def build_registry(args: argparse.Namespace) -> ModelRegistry:
    """Resolve checkpoint paths from CLI flags, falling back to the config."""
    cfg = load_config(args.config)
    demo_cfg = cfg.get("demo", {})
    checkpoints = {
        "A": Path(args.checkpoint_a or demo_cfg.get(
            "checkpoint_a", "checkpoints/part_A/best.pth")),
        "B": Path(args.checkpoint_b or demo_cfg.get(
            "checkpoint_b", "checkpoints/part_B/best.pth")),
    }
    return ModelRegistry(cfg, checkpoints, device=args.device)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the crowd counting demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint-a", default=None, help="Part A checkpoint.")
    parser.add_argument("--checkpoint-b", default=None, help="Part B checkpoint.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--share", action="store_true", help="Create a public link (use on Colab)."
    )
    parser.add_argument(
        "--device", default="cpu", help="Inference device; CPU is plenty for MCNN."
    )
    parser.add_argument(
        "--enable-video", action="store_true", help="Show the video tab."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = build_registry(args)
    enable_video = args.enable_video or bool(
        registry.cfg.get("demo", {}).get("enable_video", False)
    )
    examples_dir = Path(
        registry.cfg.get("demo", {}).get("examples_dir", "app/examples")
    )

    if not registry.available():
        print(registry.missing_message(), file=sys.stderr)

    demo = build_interface(registry, enable_video, examples_dir)
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
    }
    if "theme" in inspect.signature(demo.launch).parameters:
        launch_kwargs["theme"] = gr.themes.Soft()
    demo.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
