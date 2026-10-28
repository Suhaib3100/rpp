"""NEU-CLS torch dataset."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.transforms import GRAY_MEAN, GRAY_STD, apply_contrast, build_transform

CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_ABBREV = ["Cr", "In", "Pa", "PS", "RS", "Sc"]


class NEUCLSDataset(Dataset):
    """Grayscale NEU-CLS images with a configurable preprocessing pipeline.

    Yields ``(image, label)`` where image is a (1, size, size) float tensor,
    contrast-normalised and standardised. The single channel is expanded to
    three inside the model's global branch, so that the dataset stays
    backbone-agnostic and the Sobel chunk statistics are computed on the true
    single-channel intensity.
    """

    def __init__(
        self,
        records: list[tuple[str, int]],
        images_dir: Path,
        size: int = 160,
        train: bool = False,
        pipeline: str = "sage",
        contrast: str = "clahe",
    ) -> None:
        self.records = list(records)
        self.images_dir = Path(images_dir)
        self.size = size
        self.train = train
        self.pipeline = pipeline
        self.contrast = contrast
        self.tf = build_transform(pipeline, size, train, contrast)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fn, label = self.records[idx]
        path = self.images_dir / fn
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")

        # Contrast enhancement runs on the raw uint8 image, before geometry,
        # so that reflect-padding introduced by rotation cannot bias the
        # histogram the equaliser sees.
        img = apply_contrast(img, self.contrast)
        img = self.tf(image=img)["image"]

        x = ((img.astype(np.float32) / 255.0) - GRAY_MEAN) / GRAY_STD
        return torch.from_numpy(x).unsqueeze(0), int(label)


def load_label_map(path: Path) -> dict[int, str]:
    data = json.loads(Path(path).read_text())
    return {int(k): v for k, v in data["index_to_name"].items()}
