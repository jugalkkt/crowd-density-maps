"""Tests for density-map generation (Module 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.density import adaptive_sigmas, gaussian_kernel, generate_density_map

SHAPE = (128, 160)
POINTS = np.array(
    [[20.0, 30.0], [60.0, 40.0], [100.0, 90.0], [130.0, 20.0], [80.0, 110.0]],
    dtype=np.float32,
)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_sum_equals_point_count(mode: str) -> None:
    density = generate_density_map(SHAPE, POINTS, mode=mode, fixed_sigma=4.0)
    assert abs(density.sum() - len(POINTS)) < 1e-3


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize(
    "corner", [(0.0, 0.0), (159.0, 0.0), (0.0, 127.0), (159.0, 127.0)]
)
def test_corner_point_still_sums_to_one(mode: str, corner) -> None:
    density = generate_density_map(
        SHAPE, np.array([corner], dtype=np.float32), mode=mode, fixed_sigma=4.0
    )
    assert abs(density.sum() - 1.0) < 1e-3


def test_empty_points_gives_zero_map() -> None:
    density = generate_density_map(SHAPE, np.zeros((0, 2), dtype=np.float32))
    assert density.shape == SHAPE
    assert density.dtype == np.float32
    assert np.count_nonzero(density) == 0


def test_dtype_and_shape() -> None:
    density = generate_density_map(SHAPE, POINTS, mode="fixed")
    assert density.dtype == np.float32
    assert density.shape == SHAPE
    assert (density >= 0).all()


def test_out_of_bounds_points_are_clipped_not_dropped() -> None:
    # Real ShanghaiTech annotations occasionally place a head a fraction of a
    # pixel past the image edge (e.g. Part B test image 126: x=1023.57 in a
    # 1024-wide image rounds to column 1024). Every head must still count.
    points = np.array([[10.0, 10.0], [-5.0, 10.0], [500.0, 2.0], [10.0, 999.0]])
    density = generate_density_map(SHAPE, points, mode="fixed")
    assert abs(density.sum() - len(points)) < 1e-2


def test_far_out_of_bounds_point_lands_on_the_border() -> None:
    density = generate_density_map(
        SHAPE, np.array([[-500.0, -500.0]], dtype=np.float32), mode="fixed"
    )
    assert abs(density.sum() - 1.0) < 1e-3
    peak = np.unravel_index(np.argmax(density), density.shape)
    assert peak == (0, 0)


def test_mass_sits_on_the_head() -> None:
    density = generate_density_map(
        SHAPE, np.array([[80.0, 64.0]], dtype=np.float32), mode="fixed", fixed_sigma=4.0
    )
    peak = np.unravel_index(np.argmax(density), density.shape)
    assert peak == (64, 80)


def test_kernel_is_unit_sum_and_odd_sized() -> None:
    kernel = gaussian_kernel(4.0, truncate=3.0)
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1
    assert abs(kernel.sum() - 1.0) < 1e-5


def test_adaptive_sigma_shrinks_in_dense_regions() -> None:
    dense = np.stack(
        np.meshgrid(np.arange(0, 20, 2.0), np.arange(0, 20, 2.0)), -1
    ).reshape(-1, 2)
    sparse = np.stack(
        np.meshgrid(np.arange(0, 200, 20.0), np.arange(0, 200, 20.0)), -1
    ).reshape(-1, 2)
    assert adaptive_sigmas(dense).mean() < adaptive_sigmas(sparse).mean()


def test_single_point_adaptive_uses_fallback_sigma() -> None:
    sigmas = adaptive_sigmas(
        np.array([[10.0, 10.0]], dtype=np.float32), fallback_sigma=7.0
    )
    assert sigmas.shape == (1,)
    assert sigmas[0] == pytest.approx(7.0)


def test_many_points_sum_is_exact() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 127, size=(300, 2)).astype(np.float32)
    density = generate_density_map((128, 128), points, mode="adaptive")
    assert abs(density.sum() - 300) < 1e-2


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        generate_density_map(SHAPE, POINTS, mode="bogus")
