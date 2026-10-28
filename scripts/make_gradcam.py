#!/usr/bin/env python
"""Generate Grad-CAM and chunk-attention visualisations.

Produces two figures:

  gradcam_per_class.png   one correctly-classified example per defect class,
                          with Grad-CAM and the model's own chunk attention
                          side by side.
  gradcam_errors.png      every misclassified test image (if any), so that
                          failure cases are shown rather than only successes.

The two columns answer different questions. Grad-CAM is a post-hoc gradient
attribution over the backbone; chunk attention is the model's own selection
weight. Agreement between them is weak evidence that the model is using the
defect region; disagreement is informative about which branch is driving the
decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.neu_cls import CLASS_NAMES, NEUCLSDataset  # noqa: E402
from src.datasets.splits import load_splits  # noqa: E402
from src.explainability.gradcam import GradCAM, attention_map  # noqa: E402
from src.viz import theme as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _overlay(ax, img: np.ndarray, cam: np.ndarray | None, title: str,
             title_color: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if cam is not None:
        ax.imshow(cam, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9, color=title_color, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(T.BORDER)


def denorm(x: torch.Tensor) -> np.ndarray:
    from src.preprocessing.transforms import GRAY_MEAN, GRAY_STD
    a = x.squeeze().cpu().numpy() * GRAY_STD + GRAY_MEAN
    return np.clip(a, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main/sage_he_seed42")
    args = ap.parse_args()

    from scripts.run_encrypted_eval import load_model  # reuse the loader
    T.apply_theme()

    model, cfg = load_model(args.exp)
    grid = cfg.chunk_grid
    splits = load_splits(ROOT / "data/splits.json")
    ds = NEUCLSDataset(splits["test"], ROOT / "data/raw/images", size=cfg.image_size,
                       train=False, pipeline=cfg.pipeline, contrast=cfg.contrast)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    # ---- collect predictions
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x).argmax(1).numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    wrong = np.flatnonzero(preds != trues)
    print(f"test accuracy {np.mean(preds == trues):.4f}  ({len(wrong)} errors)")

    out_dir = ROOT / "results" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- one correct example per class
    picks = []
    for c in range(6):
        idx = np.flatnonzero((trues == c) & (preds == c))
        if len(idx):
            picks.append(int(idx[0]))

    fig, axes = plt.subplots(3, len(picks), figsize=(2.3 * len(picks), 7.4))
    fig.patch.set_facecolor(T.PAGE)
    cam_engine = GradCAM(model, model.gradcam_target_layer())
    for col, i in enumerate(picks):
        x, y = ds[i]
        xb = x.unsqueeze(0)
        cam, logits = cam_engine(xb)
        attn = attention_map(model, xb, grid)
        img = denorm(x)
        _overlay(axes[0, col], img, None, CLASS_NAMES[y], T.INK)
        _overlay(axes[1, col], img, cam[0], "Grad-CAM", T.INK_MUTED)
        _overlay(axes[2, col], img, None if attn is None else attn[0],
                 "chunk attention", T.INK_MUTED)
    cam_engine.remove()
    for row, lab in enumerate(["input", "Grad-CAM", "attention"]):
        axes[row, 0].set_ylabel(lab, color=T.INK_SECONDARY, fontsize=10)
    fig.suptitle("Grad-CAM and chunk attention  ·  correctly classified samples",
                 color=T.INK, fontsize=13, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_per_class.png", dpi=200, facecolor=T.PAGE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out_dir/'gradcam_per_class.png'}")

    # ---- misclassified samples
    if len(wrong):
        n = min(len(wrong), 6)
        fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2), squeeze=False)
        fig.patch.set_facecolor(T.PAGE)
        cam_engine = GradCAM(model, model.gradcam_target_layer())
        for col, i in enumerate(wrong[:n]):
            x, y = ds[int(i)]
            xb = x.unsqueeze(0)
            cam, _ = cam_engine(xb)
            img = denorm(x)
            _overlay(axes[0, col], img, None,
                     f"true: {CLASS_NAMES[y]}", T.INK)
            _overlay(axes[1, col], img, cam[0],
                     f"pred: {CLASS_NAMES[preds[i]]}", T.CRITICAL)
        cam_engine.remove()
        fig.suptitle(f"Misclassified test images  ·  {len(wrong)} of {len(trues)}",
                     color=T.INK, fontsize=13, fontweight="semibold", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out_dir / "gradcam_errors.png", dpi=200, facecolor=T.PAGE,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"written: {out_dir/'gradcam_errors.png'}")
    else:
        print("no misclassified test images for this model")
        # The brief asks for explanations on misclassified samples. This model
        # has none, so the next most informative set is the images the
        # *reproduced baseline* got wrong: the genuinely hard cases, shown with
        # this model's attribution on them. That is a real comparison rather
        # than an empty figure.
        base_logits = ROOT / "experiments" / "main/jia_baseline_seed42" / "test_logits.npy"
        if base_logits.exists():
            base_pred = np.load(base_logits).argmax(1)
            hard = np.flatnonzero(base_pred != trues)
            if len(hard):
                n = min(len(hard), 6)
                fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2), squeeze=False)
                fig.patch.set_facecolor(T.PAGE)
                cam_engine = GradCAM(model, model.gradcam_target_layer())
                for col, i in enumerate(hard[:n]):
                    x, y = ds[int(i)]
                    cam, _ = cam_engine(x.unsqueeze(0))
                    img = denorm(x)
                    _overlay(axes[0, col], img, None, f"true: {CLASS_NAMES[y]}", T.INK)
                    _overlay(axes[1, col], img, cam[0],
                             f"baseline said: {CLASS_NAMES[base_pred[i]]}", T.CRITICAL)
                cam_engine.remove()
                fig.suptitle(
                    f"Hard cases — the {len(hard)} images the reproduced baseline "
                    f"misclassified (SAGE-HE classifies all of them correctly)",
                    color=T.INK, fontsize=13, fontweight="semibold", x=0.01, ha="left")
                fig.tight_layout()
                fig.savefig(out_dir / "gradcam_hard_cases.png", dpi=200,
                            facecolor=T.PAGE, bbox_inches="tight")
                plt.close(fig)
                print(f"written: {out_dir/'gradcam_hard_cases.png'} ({len(hard)} hard cases)")
                meta_extra = {"n_baseline_errors": int(len(hard))}
            else:
                meta_extra = {}
        else:
            meta_extra = {}

    # ---- record the learned Sobel trust parameter
    meta = {"experiment": args.exp, "n_errors": int(len(wrong)),
            "test_accuracy": float(np.mean(preds == trues))}
    meta.update(locals().get("meta_extra", {}) or {})
    if getattr(model, "chunk_attn", None) is not None:
        meta["learned_sobel_weight_lambda"] = float(model.chunk_attn.sobel_weight.item())
    (out_dir / "explainability_meta.json").write_text(json.dumps(meta, indent=2))
    print("lambda (Sobel trust):", meta.get("learned_sobel_weight_lambda"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
