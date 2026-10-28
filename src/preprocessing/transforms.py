"""Preprocessing and augmentation pipelines.

Two pipelines are provided:

``jia_baseline`` reproduces the preprocessing described by Jia et al. (2023):
resize to 112x112, grayscale, global histogram equalisation, and augmentation
by rotation (+/-15 deg), horizontal flip and translation (+/-10%).

``sage`` is this project's pipeline. It differs in two ways, both of which are
measured in the design study (`run_suite.py design`) rather than assumed:

* CLAHE instead of global histogram equalisation. The hypothesis was that global
  equalisation applies one intensity mapping to the whole frame and crushes the
  low-contrast `crazing` texture, while CLAHE equalises locally and preserves
  it. **The measurement does not support the hypothesis, but it does not refute
  it either**: across three seeds in the reduced-data regime, no contrast
  enhancement at all has the higher mean (+1.8 points over CLAHE), yet the
  seed-to-seed spread (~4.7 points) is larger than the gap, so the three options
  are not distinguishable at this sample size. The default is kept so that every
  reported run stays reproducible; see the design-study table in
  reports/technical_report.md.
* No vertical flip. NEU-CLS defect classes are not vertically symmetric in
  appearance (`scratches` and `rolled-in_scale` carry a strong horizontal
  orientation prior, see the v/h gradient ratios in
  results/metrics/dataset_verification.json), so a vertical flip would blur a
  genuinely discriminative cue. This one is an argument from the data's own
  statistics, not an ablated choice.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

# NEU-CLS is single-channel. These are the ImageNet grayscale-equivalent stats
# (the mean of the RGB means/stds), used because the global branch starts from
# ImageNet-pretrained weights.
GRAY_MEAN = 0.449
GRAY_STD = 0.226


def _to_float(img: np.ndarray) -> np.ndarray:
    return ((img.astype(np.float32) / 255.0) - GRAY_MEAN) / GRAY_STD


def histogram_equalise(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation - the base paper's contrast step."""
    return cv2.equalizeHist(img)


def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def build_transform(
    pipeline: str,
    size: int,
    train: bool,
    contrast: str = "clahe",
) -> A.Compose:
    """Build an albumentations pipeline.

    Args:
        pipeline: "jia_baseline" or "sage".
        size: output square edge length in pixels.
        train: include stochastic augmentation.
        contrast: "clahe" | "histeq" | "none" (ablatable).
    """
    ops: list = [A.Resize(size, size, interpolation=cv2.INTER_AREA)]

    if train:
        # Note on padding: Rotate takes `border_mode` and uses reflect padding,
        # but albumentations' Affine takes `mode`, not `border_mode`, and
        # defaults to BORDER_CONSTANT with cval=0. Affine therefore pads with
        # black. That is stated explicitly below rather than left to a silently
        # ignored keyword, and it is applied identically to both models, so the
        # comparison is unaffected. It is kept as-is because every run reported
        # in results/ was trained under it; changing it would make the seeds
        # non-comparable.
        if pipeline == "jia_baseline":
            ops += [
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=(-0.10, 0.10),
                         mode=cv2.BORDER_CONSTANT, cval=0, p=0.5),
            ]
        else:
            ops += [
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=(-0.10, 0.10),
                         scale=(0.92, 1.08),
                         mode=cv2.BORDER_CONSTANT, cval=0, p=0.5),
                # Mild photometric jitter: rolling-mill lighting varies, and the
                # per-class intensity means differ by >75 grey levels, so we do
                # not want the model keying on absolute brightness alone.
                A.RandomBrightnessContrast(brightness_limit=0.15,
                                           contrast_limit=0.15, p=0.3),
            ]
    return A.Compose(ops)


def apply_contrast(img: np.ndarray, contrast: str) -> np.ndarray:
    if contrast == "clahe":
        return clahe(img)
    if contrast == "histeq":
        return histogram_equalise(img)
    if contrast == "none":
        return img
    raise ValueError(f"unknown contrast mode: {contrast!r}")
