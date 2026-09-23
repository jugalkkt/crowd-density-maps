"""Multi-Column CNN (Zhang et al., CVPR 2016).

Three parallel columns with different receptive-field sizes each see the whole
image; their feature maps are concatenated and fused by a 1x1 convolution into
a single-channel density map at 1/4 of the input resolution (two 2x max-pools).
Large kernels model heads that appear large (close to the camera), small
kernels model distant heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MCNN", "count_parameters"]

# (kernel size, output channels) per layer, per column, exactly as in Table 1
# of the paper.
COLUMN_SPECS: dict[str, tuple[tuple[int, int], ...]] = {
    "large": ((9, 16), (7, 32), (7, 16), (7, 8)),
    "medium": ((7, 20), (5, 40), (5, 20), (5, 10)),
    "small": ((5, 24), (3, 48), (3, 24), (3, 12)),
}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters in a module."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _column(in_channels: int, spec: tuple[tuple[int, int], ...]) -> nn.Sequential:
    """Build one column: conv-pool-conv-pool-conv-conv, all with 'same' padding."""
    (k1, c1), (k2, c2), (k3, c3), (k4, c4) = spec
    return nn.Sequential(
        nn.Conv2d(in_channels, c1, k1, padding=k1 // 2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, k2, padding=k2 // 2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(c2, c3, k3, padding=k3 // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(c3, c4, k4, padding=k4 // 2),
        nn.ReLU(inplace=True),
    )


class MCNN(nn.Module):
    """MCNN density estimator.

    Args:
        in_channels: 3 for RGB input.
        verbose: print the parameter count when the model is created.

    Shape:
        input ``[B, C, H, W]`` -> output ``[B, 1, H/4, W/4]``, non-negative.
    """

    downsample: int = 4

    def __init__(self, in_channels: int = 3, verbose: bool = True) -> None:
        super().__init__()
        self.column_large = _column(in_channels, COLUMN_SPECS["large"])
        self.column_medium = _column(in_channels, COLUMN_SPECS["medium"])
        self.column_small = _column(in_channels, COLUMN_SPECS["small"])

        fused_channels = sum(spec[-1][1] for spec in COLUMN_SPECS.values())  # 30
        self.fuse = nn.Conv2d(fused_channels, 1, kernel_size=1)
        # Density is a non-negative quantity; clamping here also stops the
        # count from being reduced by cancelling positive and negative pixels.
        self.output_activation = nn.ReLU(inplace=True)

        self._initialise_weights()
        if verbose:
            print(f"MCNN: {count_parameters(self):,} trainable parameters")

    def _initialise_weights(self) -> None:
        """Normal(0, 0.01) convolution weights and zero biases, as in the paper."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.column_large(x), self.column_medium(x), self.column_small(x)],
            dim=1,
        )
        return self.output_activation(self.fuse(features))


if __name__ == "__main__":  # pragma: no cover - quick manual check
    model = MCNN()
    dummy = torch.zeros(1, 3, 256, 256)
    print("output shape:", tuple(model(dummy).shape))
