"""Reproduction of the Jia et al. (2023) chunk-based CNN baseline.

Reproduced from the published description (Journal of Cloud Computing 12:175,
DOI 10.1186/s13677-023-00537-0). The paper's prose is internally inconsistent
in places, so every assumption we had to make is recorded here and in
reports/baseline_reproduction.md.

What the paper states explicitly
--------------------------------
* Images resized to 112x112, grayscale, histogram-equalised (section "Data
  preprocessing"; Table 1 and Table 2 both use 112x112).
* Augmentation: rotation +/-15 deg, horizontal flip, translation +/-10%.
* "a convolutional layer with 7x7 convolution kernels, followed by an
  activation function layer and a fully connected layer with 256 input neurons
  and 128 output neurons. Finally ... an output layer with 128 input neurons
  and 6 output neurons."
* ReLU in the plaintext model; a **square** activation in the CKKS-compatible
  model (Table 3 lists the activation function as "Square").
* Adam optimiser, cross-entropy loss.
* Chunking: 28x28 chunks over a 112x112 image -> a 4x4 grid of 16 chunks
  (Table 2 states 16 chunks at 28x28; the prose's "28x26" is inconsistent with
  both the chunk count and the image size, and is treated as a typo).

What the paper does NOT state (our documented assumptions)
----------------------------------------------------------
* Number of convolution filters -> we use 32. The paper never gives a channel
  count. 32 is chosen so that the post-pool feature map can be projected to
  the stated 256-dimensional fully-connected input.
* How the conv output reaches "256 input neurons" -> the paper gives no
  pooling or flatten arithmetic that produces 256 from a 7x7 conv over
  112x112. We insert an adaptive average pool to a fixed spatial size, which
  is the minimal, standard construction that satisfies the stated 256 figure.
* Stride/padding of the 7x7 convolution -> stride 1, padding 3 ("same").
* Train/val/test split sizes, epoch count, learning rate, batch size -> not
  reported. We use this project's common protocol so that the baseline and
  SAGE-HE are compared on identical data (see configs/).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Square(nn.Module):
    """x -> x^2, the CKKS-friendly activation used by the base paper.

    Unlike ReLU this is a degree-2 polynomial, so it can be evaluated directly
    on CKKS ciphertexts at the cost of one multiplicative level.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class JiaChunkCNN(nn.Module):
    """The base paper's classifier.

    Architecture (as reproduced):
        (B,1,112,112)
          -> Conv2d(1, 32, k=7, s=1, p=3)      -> (B,32,112,112)
          -> activation
          -> MaxPool2d(2)                      -> (B,32,56,56)
          -> AdaptiveAvgPool2d(target)         -> (B,32,h,w) with 32*h*w == 256
          -> flatten                           -> (B,256)
          -> Linear(256,128) -> activation
          -> Linear(128,6)
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        conv_channels: int = 32,
        activation: str = "relu",
        fc_in: int = 256,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "square"}:
            raise ValueError(f"activation must be 'relu' or 'square', got {activation!r}")
        act = (lambda: nn.ReLU(inplace=True)) if activation == "relu" else Square

        # Pool to whatever spatial size makes channels * h * w equal the stated
        # 256-unit fully-connected input. Capping the width at 4 reproduces the
        # original shapes exactly — 16->(4,4), 32->(2,4), 64->(1,4) — while
        # also covering widths where 256/channels < 4, which an earlier
        # fixed `(side // 4, 4)` form silently floored to a zero-height pool.
        if fc_in % conv_channels:
            raise ValueError(
                f"conv_channels={conv_channels} does not divide fc_in={fc_in}"
            )
        q = fc_in // conv_channels
        w = min(4, q)
        self.pool_hw = (q // w, w)
        if conv_channels * self.pool_hw[0] * self.pool_hw[1] != fc_in:
            raise ValueError(
                f"conv_channels={conv_channels} cannot be pooled to fc_in={fc_in}"
            )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=7, stride=1, padding=3),
            act(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(self.pool_hw)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            act(),
            nn.Linear(128, num_classes),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
