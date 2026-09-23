"""Reading ShanghaiTech head annotations out of the MATLAB ``.mat`` files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

__all__ = ["load_points", "annotation_path_for_image"]


def load_points(mat_path: str | Path) -> np.ndarray:
    """Load head coordinates from a ShanghaiTech ``GT_IMG_n.mat`` file.

    The coordinates live at ``mat["image_info"][0, 0][0, 0][0]`` as an
    ``(N, 2)`` array of ``(x, y)`` pairs. A few mirrors of the dataset store
    them under ``annPoints`` instead, so that key is accepted as a fallback.

    Returns:
        ``(N, 2)`` float32 array of ``(x, y)``; shape ``(0, 2)`` when the image
        has no annotated heads.
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {mat_path}")

    mat = loadmat(str(mat_path))
    points = None

    if "image_info" in mat:
        try:
            points = mat["image_info"][0, 0][0, 0][0]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Unexpected `image_info` structure in {mat_path}"
            ) from exc
    elif "annPoints" in mat:
        points = mat["annPoints"]
    else:
        keys = [k for k in mat if not k.startswith("__")]
        raise ValueError(
            f"No head annotations found in {mat_path}; available keys: {keys}"
        )

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return points


def annotation_path_for_image(image_path: str | Path) -> Path:
    """Map ``.../images/IMG_7.jpg`` to ``.../ground-truth/GT_IMG_7.mat``.

    Both the ``ground-truth`` and ``ground_truth`` spellings found in the
    various dataset mirrors are tried.
    """
    image_path = Path(image_path)
    split_dir = image_path.parent.parent
    for folder in ("ground-truth", "ground_truth", "ground-truth-h5"):
        candidate = split_dir / folder / f"GT_{image_path.stem}.mat"
        if candidate.is_file():
            return candidate
    # Return the canonical spelling so the caller can report a useful error.
    return split_dir / "ground-truth" / f"GT_{image_path.stem}.mat"
