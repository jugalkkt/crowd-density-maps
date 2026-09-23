"""Tests for the MCNN architecture (Module 3)."""

from __future__ import annotations

import pytest
import torch

from src.models import MCNN, count_parameters


@pytest.fixture(scope="module")
def model() -> MCNN:
    return MCNN(verbose=False)


def test_square_input_gives_quarter_resolution(model: MCNN) -> None:
    out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.parametrize("size", [(768, 1024), (128, 192), (100, 68)])
def test_non_square_input(model: MCNN, size) -> None:
    height, width = size
    out = model(torch.randn(1, 3, height, width))
    assert out.shape == (1, 1, height // 4, width // 4)


def test_output_is_non_negative(model: MCNN) -> None:
    out = model(torch.randn(2, 3, 64, 64))
    assert torch.all(out >= 0)


def test_forward_backward_on_cpu() -> None:
    model = MCNN(verbose=False)
    out = model(torch.randn(1, 3, 64, 64))
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_column_output_channels(model: MCNN) -> None:
    x = torch.randn(1, 3, 64, 64)
    assert model.column_large(x).shape[1] == 8
    assert model.column_medium(x).shape[1] == 10
    assert model.column_small(x).shape[1] == 12
    assert model.fuse.in_channels == 30


def test_weight_initialisation() -> None:
    model = MCNN(verbose=False)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
            assert module.weight.std().item() < 0.05


def test_parameter_count_is_reasonable(model: MCNN) -> None:
    # The paper reports roughly 130k parameters.
    n = count_parameters(model)
    assert 100_000 < n < 200_000
