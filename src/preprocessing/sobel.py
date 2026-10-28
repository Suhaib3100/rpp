"""Sobel gradient analysis and chunk-level gradient energy.

This module implements both the base paper's selection rule and the gradient
statistic that conditions SAGE-HE's adaptive chunk selection.

Jia et al. partition the image into a grid of chunks, compute a Sobel gradient
per chunk, and encrypt only those chunks whose gradient exceeds a threshold.
Their published description fixes the image size (112x112) and chunk size
(28x28 -> a 4x4 grid of 16 chunks) but never states the threshold value, so
``select_chunks_threshold`` exposes it as an explicit, documented parameter and
``otsu_threshold`` provides a data-driven default.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Fixed 3x3 Sobel kernels, registered as buffers by callers that need them on
# an accelerator.
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]])
SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0]])


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """|grad| of a uint8 grayscale image, as float32."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def chunk_gradient_energy(img: np.ndarray, grid: int) -> np.ndarray:
    """Mean Sobel magnitude per chunk on a ``grid`` x ``grid`` partition.

    Returns a (grid*grid,) float32 array in row-major chunk order.
    """
    h, w = img.shape[:2]
    if h % grid or w % grid:
        raise ValueError(f"image {h}x{w} is not divisible into a {grid}x{grid} grid")
    mag = sobel_magnitude(img)
    ch, cw = h // grid, w // grid
    return (mag.reshape(grid, ch, grid, cw)
               .mean(axis=(1, 3))
               .astype(np.float32)
               .reshape(-1))


def select_chunks_threshold(energy: np.ndarray, threshold: float) -> np.ndarray:
    """The base paper's rule: keep chunks whose gradient exceeds a threshold.

    Returns a boolean mask. At least one chunk is always kept, otherwise a
    uniform image would encrypt nothing and the server would receive no data.
    """
    mask = energy > threshold
    if not mask.any():
        mask[int(np.argmax(energy))] = True
    return mask


def otsu_threshold(energy: np.ndarray) -> float:
    """Otsu's method on the chunk-energy histogram.

    Used as the documented default for the baseline, since Jia et al. do not
    report the threshold they used. Otsu picks the value minimising intra-class
    variance of the two resulting chunk groups, which is the natural
    parameter-free reading of "chunks with / without important features".
    """
    e = energy.astype(np.float64)
    lo, hi = e.min(), e.max()
    if hi <= lo:
        return float(lo)
    norm = ((e - lo) / (hi - lo) * 255).astype(np.uint8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + (t / 255.0) * (hi - lo))


def batch_chunk_energy(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Differentiable-friendly chunk gradient energy for a batch of tensors.

    Args:
        x: (B, 1, H, W) float tensor.
        grid: partition grid size.
    Returns:
        (B, grid*grid) tensor of mean |grad| per chunk.
    """
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"expected (B,1,H,W), got {tuple(x.shape)}")
    kx = SOBEL_X.to(x.device, x.dtype).view(1, 1, 3, 3)
    ky = SOBEL_Y.to(x.device, x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    # Average-pool each chunk down to a single value.
    b, _, h, w = mag.shape
    pooled = F.adaptive_avg_pool2d(mag, output_size=(grid, grid))
    return pooled.view(b, grid * grid)
