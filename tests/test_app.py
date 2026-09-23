"""Tests for the Gradio demo wiring (Module 7) — the UI is built, not launched."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.models import MCNN
from src.utils import save_checkpoint

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "gradio_app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("gradio_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load_app_module()


def test_registry_reports_missing_checkpoints(app, base_config, tmp_path: Path) -> None:
    registry = app.ModelRegistry(
        base_config,
        {"A": tmp_path / "a.pth", "B": tmp_path / "b.pth"},
    )
    assert registry.available() == []
    message = registry.missing_message()
    assert "No trained model found" in message
    assert str(tmp_path / "a.pth") in message


def test_registry_loads_and_caches(app, base_config, tmp_path: Path) -> None:
    checkpoint = tmp_path / "b.pth"
    save_checkpoint(checkpoint, MCNN(verbose=False), config=base_config)
    registry = app.ModelRegistry(
        base_config, {"A": tmp_path / "missing.pth", "B": checkpoint}
    )

    assert registry.available() == ["Sparse crowds (Part B)"]
    counter = registry.get("Sparse crowds (Part B)")
    assert registry.get("Sparse crowds (Part B)") is counter  # cached

    count, _ = counter.predict_image(
        np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
    )
    assert count >= 0


def test_missing_part_raises_gradio_error(app, base_config, tmp_path: Path) -> None:
    import gradio as gr

    registry = app.ModelRegistry(
        base_config, {"A": tmp_path / "missing.pth", "B": tmp_path / "missing.pth"}
    )
    with pytest.raises(gr.Error):
        registry.get("Dense crowds (Part A)")


@pytest.mark.parametrize("enable_video", [False, True])
def test_interface_builds(app, base_config, tmp_path: Path, enable_video: bool) -> None:
    checkpoint = tmp_path / "b.pth"
    save_checkpoint(checkpoint, MCNN(verbose=False), config=base_config)
    registry = app.ModelRegistry(
        base_config, {"A": tmp_path / "missing.pth", "B": checkpoint}
    )
    demo = registry and app.build_interface(registry, enable_video, tmp_path)
    assert demo is not None
    assert demo.title == "Crowd Counting (MCNN)"


def test_cli_flags(app) -> None:
    args = app.parse_args(
        ["--share", "--port", "1234", "--checkpoint-b", "x.pth", "--enable-video"]
    )
    assert args.share and args.port == 1234
    assert args.checkpoint_b == "x.pth" and args.enable_video
