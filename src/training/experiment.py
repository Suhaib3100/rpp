"""Experiment definition and runner.

One function, ``run_experiment``, builds the data, model and trainer from a
single declarative config and returns a result record. Every experiment in the
project (baseline reproduction, ablations, final model, seed repeats) goes
through this path, so they are guaranteed to share the same splits, the same
evaluation code and the same selection protocol.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.neu_cls import CLASS_NAMES, NEUCLSDataset
from src.datasets.splits import load_splits
from src.evaluation.metrics import compute_metrics, wilson_interval
from src.models.baseline_jia import JiaChunkCNN
from src.models.sage_he import SageHE, SageHEConfig
from src.training.trainer import TrainConfig, Trainer
from src.utils.device import get_device
from src.utils.seed import seed_everything, worker_init_fn

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExperimentConfig:
    name: str
    model: str = "sage_he"            # "sage_he" | "jia_baseline"
    image_size: int = 160
    pipeline: str = "sage"            # preprocessing pipeline
    contrast: str = "clahe"           # "clahe" | "histeq" | "none"
    seed: int = 42
    # --- sage_he knobs ---
    fusion: str = "gated"
    use_local: bool = True
    use_global: bool = True
    use_attention: bool = True
    use_sobel: bool = True
    top_k: int | None = None
    compressed_dim: int = 64
    head_hidden: int = 32
    chunk_grid: int = 4
    pretrained: bool = True
    # --- jia_baseline knobs ---
    activation: str = "square"
    # Convolution width of the baseline. The paper never states it, so our
    # reproduction had to pick one; this makes the assumption a measurable
    # parameter rather than a buried constant.
    conv_channels: int = 32
    # --- training ---
    epochs: int = 30
    lr: float = 3e-4
    batch_size: int = 32
    weight_decay: float = 1e-4
    patience: int = 8
    # 0 = load in the training process. NEU-CLS is 1223 small images and the
    # transform is a few milliseconds each, so worker processes buy almost no
    # throughput here — but each one carries a ~2.5 GB copy-on-write footprint,
    # which on an 8 GB box with no swap is enough to lose a run to the OOM
    # killer mid-sweep. Inline loading is both lighter and, under CPU
    # contention with the compute threads, no slower.
    num_workers: int = 0
    # Fraction of the training split to actually train on (stratified).
    # On full data a pretrained backbone saturates NEU-CLS at ~100% test
    # accuracy, which puts every architectural variant at ceiling and makes the
    # ablation uninformative. Training on a stratified subset restores the
    # dynamic range needed to measure each component's contribution. Validation
    # and test splits are never subsampled.
    train_fraction: float = 1.0
    notes: str = ""


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    if cfg.model == "jia_baseline":
        return JiaChunkCNN(num_classes=6, activation=cfg.activation,
                           conv_channels=cfg.conv_channels)
    if cfg.model == "sage_he":
        return SageHE(SageHEConfig(
            num_classes=6,
            pretrained=cfg.pretrained,
            compressed_dim=cfg.compressed_dim,
            head_hidden=cfg.head_hidden,
            chunk_grid=cfg.chunk_grid,
            top_k=cfg.top_k,
            fusion=cfg.fusion,
            use_sobel=cfg.use_sobel,
            use_attention=cfg.use_attention,
            use_local=cfg.use_local,
            use_global=cfg.use_global,
        ))
    raise ValueError(f"unknown model {cfg.model!r}")


def build_loaders(cfg: ExperimentConfig) -> dict[str, DataLoader]:
    splits = load_splits(ROOT / "data/splits.json")
    images = ROOT / "data/raw/images"

    if cfg.train_fraction < 1.0:
        from sklearn.model_selection import train_test_split as _tts
        recs = splits["train"]
        labels = [y for _, y in recs]
        kept, _ = _tts(recs, train_size=cfg.train_fraction,
                       random_state=cfg.seed, stratify=labels)
        splits["train"] = sorted(kept)

    loaders = {}
    for split in ("train", "val", "test"):
        ds = NEUCLSDataset(
            splits[split], images,
            size=cfg.image_size,
            train=(split == "train"),
            pipeline=cfg.pipeline,
            contrast=cfg.contrast,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
            drop_last=False,
            persistent_workers=cfg.num_workers > 0,
        )
    return loaders


def run_experiment(cfg: ExperimentConfig, out_root: Path | None = None,
                   evaluate_test: bool = True, resume: bool = True) -> dict:
    out_root = Path(out_root or ROOT / "experiments")
    out_dir = out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    dev = get_device()
    print(f"[{cfg.name}] device={dev} seed={cfg.seed}")

    loaders = build_loaders(cfg)
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    tcfg = TrainConfig(
        epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay, patience=cfg.patience, seed=cfg.seed,
    )
    trainer = Trainer(model, tcfg, dev, out_dir)

    t0 = time.time()
    fit = trainer.fit(loaders["train"], loaders["val"], resume=resume)
    train_seconds = time.time() - t0

    # Restore the selected checkpoint before any test-set contact.
    trainer.load_best()
    val_loss, val_metrics, _ = trainer.evaluate(loaders["val"])

    result = {
        "config": asdict(cfg),
        "n_parameters": n_params,
        "n_train": len(loaders["train"].dataset),
        "n_val": len(loaders["val"].dataset),
        "n_test": len(loaders["test"].dataset),
        "device": str(dev),
        "train_seconds": train_seconds,
        "fit": {k: v for k, v in fit.items() if k != "history"},
        "history": fit["history"],
        "val_metrics": val_metrics.to_dict(),
        "class_names": CLASS_NAMES,
    }

    if evaluate_test:
        test_loss, test_metrics, logits = trainer.evaluate(loaders["test"])
        correct = int(round(test_metrics.accuracy * test_metrics.n_samples))
        lo, hi = wilson_interval(correct, test_metrics.n_samples)
        result["test_metrics"] = test_metrics.to_dict()
        result["test_loss"] = test_loss
        result["test_accuracy_ci95"] = [lo, hi]
        np.save(out_dir / "test_logits.npy", logits)
        print(f"[{cfg.name}] TEST {test_metrics.summary()} "
              f"(95% CI on acc: {lo:.4f}-{hi:.4f})")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
