"""Multi-scale local feature branch.

NEU-CLS defects live at very different spatial scales: `scratches` are long
thin high-contrast lines, `crazing` is a dense fine-grained crack network, and
`patches` are large low-frequency blobs. A single receptive field is a poor fit
for all three, so each block mixes four parallel receptive fields and fuses
them with a 1x1 convolution.

Dilated 3x3 convolutions are used instead of larger dense kernels for the wide
receptive fields: they reach the same extent at a fraction of the parameters
and FLOPs, which matters because this branch runs alongside EfficientNet-B0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """Parallel 1x1 / 3x3 / dilated-3x3 / 5x5 paths, concatenated and fused.

    Receptive fields per path: 1, 3, 5 (3x3 d=2), 5. The four paths each emit
    ``out_channels // 4`` channels so the concatenation width is exactly
    ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4:
            raise ValueError(f"out_channels must be divisible by 4, got {out_channels}")
        b = out_channels // 4

        def path(k: int, d: int) -> nn.Sequential:
            pad = d * (k - 1) // 2
            return nn.Sequential(
                nn.Conv2d(in_channels, b, kernel_size=k, padding=pad, dilation=d, bias=False),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
            )

        self.p1 = path(1, 1)   # pointwise - pure channel mixing
        self.p3 = path(3, 1)   # fine detail: cracks, thin scratches
        self.p3d = path(3, 2)  # wider context at 3x3 cost
        self.p5 = path(5, 1)   # blob-scale structure: patches, pitted regions

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Projection shortcut keeps gradients healthy when stacking blocks.
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                               nn.BatchNorm2d(out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.p1(x), self.p3(x), self.p3d(x), self.p5(x)], dim=1)
        return self.fuse(y) + self.shortcut(x)


class MultiScaleCNN(nn.Module):
    """Lightweight local branch returning a spatial feature map.

    Three stages at /2, /4, /8 of the input resolution. The map is returned
    (not pooled) because adaptive chunk selection needs spatial structure.
    """

    def __init__(self, in_channels: int = 1, widths: tuple[int, ...] = (32, 64, 128)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        c_in = widths[0]
        for c_out in widths:
            stages.append(nn.Sequential(MultiScaleBlock(c_in, c_out), nn.MaxPool2d(2)))
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.out_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))
