#!/usr/bin/env python
"""Before/after comparisons — statistical and visual.

Produces the side-by-side evidence that the deck and report need:

  statistical
    results/metrics/before_after.json    paired McNemar test + bootstrap CI on
                                         the accuracy delta, plus a per-class
                                         before/after table

  visual
    figures/ba_preprocessing.png         raw -> histeq (base paper) -> CLAHE (ours)
    figures/ba_chunk_selection.png       fixed Sobel threshold vs learned attention
    figures/ba_encryption.png            plaintext vs decrypted logits + error
    figures/ba_per_class.png             per-class F1, baseline vs SAGE-HE
    figures/ba_confusion.png             both confusion matrices, one scale

Usage: python scripts/make_comparisons.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.neu_cls import CLASS_NAMES, NEUCLSDataset  # noqa: E402
from src.datasets.splits import load_splits  # noqa: E402
from src.evaluation.aggregate import confusion_for, per_class_table  # noqa: E402
from src.evaluation.significance import bootstrap_delta_ci, mcnemar  # noqa: E402
from src.preprocessing.sobel import (  # noqa: E402
    chunk_gradient_energy, otsu_threshold, select_chunks_threshold,
)
from src.preprocessing.transforms import clahe, histogram_equalise  # noqa: E402
from src.viz import theme as T  # noqa: E402
from src.viz.diagram import save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
BASE = "main/jia_baseline_seed42"
SAGE = "main/sage_he_seed42"


def _axes_off(ax, title="", color=None):
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9.5, color=color or T.INK_SECONDARY, pad=4)
    for s in ax.spines.values():
        s.set_color(T.BORDER)


# ---------------------------------------------------------------- statistics
def statistical_comparison() -> dict | None:
    """Paired McNemar test between the reproduced baseline and SAGE-HE."""
    pa = ROOT / "experiments" / BASE / "test_logits.npy"
    pb = ROOT / "experiments" / SAGE / "test_logits.npy"
    if not (pa.exists() and pb.exists()):
        print("  [skip] statistical comparison: logits missing")
        return None

    splits = load_splits(ROOT / "data/splits.json")
    y_true = np.array([y for _, y in splits["test"]])
    pred_a = np.load(pa).argmax(1)
    pred_b = np.load(pb).argmax(1)

    res = mcnemar(y_true, pred_a, pred_b)
    lo, hi = bootstrap_delta_ci(y_true, pred_a, pred_b)

    # per-class before/after
    per_class = []
    ta = per_class_table(BASE, CLASS_NAMES)
    tb = per_class_table(SAGE, CLASS_NAMES)
    if ta is not None and tb is not None:
        for i, name in enumerate(CLASS_NAMES):
            per_class.append({
                "class": name,
                "before_f1": float(ta["f1"][i]),
                "after_f1": float(tb["f1"][i]),
                "delta_f1": float(tb["f1"][i] - ta["f1"][i]),
                "before_recall": float(ta["recall"][i]),
                "after_recall": float(tb["recall"][i]),
            })

    out = {
        "before": {"name": BASE, "label": "Jia et al. (2023), reproduced"},
        "after": {"name": SAGE, "label": "SAGE-HE (proposed)"},
        "mcnemar": res.to_dict(),
        "accuracy_delta_ci95": [lo, hi],
        "per_class": per_class,
        "note": (
            "Both models were evaluated on the identical 360-image held-out test "
            "split, so the comparison is paired and McNemar's test is the correct "
            "choice. The bootstrap CI resamples images, keeping each image's pair "
            "of predictions together."
        ),
    }
    p = ROOT / "results" / "metrics" / "before_after.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"  statistical: {res.summary()}")
    print(f"    accuracy delta 95% CI: [{lo*100:+.2f}, {hi*100:+.2f}] pts")
    print(f"  -> {p.name}")
    return out


# ------------------------------------------------------- preprocessing B/A
def fig_preprocessing() -> None:
    splits = load_splits(ROOT / "data/splits.json")
    by: dict[int, str] = {}
    for fn, y in splits["train"]:
        by.setdefault(y, fn)

    rows = [("raw 200×200", lambda a: a),
            ("histogram equalisation\n(base paper)", histogram_equalise),
            ("CLAHE\n(SAGE-HE)", clahe)]
    fig, axes = plt.subplots(3, 6, figsize=(13.2, 7.0))
    fig.patch.set_facecolor(T.PAGE)
    for c in range(6):
        img = cv2.imread(str(ROOT / "data/raw/images" / by[c]), cv2.IMREAD_GRAYSCALE)
        for r, (_, fn) in enumerate(rows):
            axes[r, c].imshow(fn(img), cmap="gray", vmin=0, vmax=255)
            _axes_off(axes[r, c], CLASS_NAMES[c] if r == 0 else "",
                      T.INK if r == 0 else None)
    for r, (label, _) in enumerate(rows):
        axes[r, 0].set_ylabel(label, color=T.ACCENT_BRIGHT, fontsize=10.5)
    fig.suptitle("Preprocessing · before and after",
                 color=T.INK, fontsize=14, fontweight="semibold", x=0.01, ha="left")
    fig.text(0.01, 0.015,
             "Global equalisation lifts contrast everywhere; CLAHE equalises locally. Measured "
             "over 3 seeds against no enhancement at all, the three are indistinguishable — the "
             "seed-to-seed spread (~4.7 pts) is wider than any gap between them.",
             color=T.INK_MUTED, fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save(fig, FIG / "ba_preprocessing", formats=("png",))
    print("  -> ba_preprocessing.png")


# ------------------------------------------------------ chunk selection B/A
def fig_chunk_selection(model=None, cfg=None) -> None:
    """Fixed Sobel threshold (base paper) vs learned attention (ours)."""
    splits = load_splits(ROOT / "data/splits.json")
    by: dict[int, str] = {}
    for fn, y in splits["test"]:
        by.setdefault(y, fn)
    grid = 4

    ds = None
    if model is not None and cfg is not None:
        ds = NEUCLSDataset(splits["test"], ROOT / "data/raw/images", size=cfg.image_size,
                           train=False, pipeline=cfg.pipeline, contrast=cfg.contrast)
        index_of = {fn: i for i, (fn, _) in enumerate(splits["test"])}

    conc: list[dict] = []
    nrows = 3 if ds is not None else 2
    fig, axes = plt.subplots(nrows, 6, figsize=(13.2, 2.45 * nrows + 0.9))
    fig.patch.set_facecolor(T.PAGE)

    for c in range(6):
        fn = by[c]
        img = cv2.imread(str(ROOT / "data/raw/images" / fn), cv2.IMREAD_GRAYSCALE)
        img112 = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

        axes[0, c].imshow(img112, cmap="gray")
        _axes_off(axes[0, c], CLASS_NAMES[c], T.INK)

        # --- base paper: Otsu threshold on chunk gradient energy.
        # Discarded chunks are dimmed and kept chunks left clear, so the
        # overlay reads the way the rule works. (A diverging colormap here
        # would imply an ordering the binary rule does not have.)
        e = chunk_gradient_energy(img112, grid)
        mask = select_chunks_threshold(e, otsu_threshold(e))
        axes[1, c].imshow(img112, cmap="gray")
        dim = np.zeros((grid, grid, 4))
        dim[..., :3] = 0.03               # near-black veil over discarded chunks
        dim[..., 3] = (~mask).reshape(grid, grid) * 0.72
        axes[1, c].imshow(dim, extent=(0, 112, 112, 0), interpolation="nearest")
        for k in np.flatnonzero(mask):    # outline the kept chunks
            r, cc = divmod(int(k), grid)
            axes[1, c].add_patch(plt.Rectangle(
                (cc * 112 / grid, r * 112 / grid), 112 / grid, 112 / grid,
                fill=False, edgecolor=T.SERIES[1], lw=1.8))
        _axes_off(axes[1, c], f"{int(mask.sum())}/16 chunks kept")

        # --- ours: learned attention weights
        if ds is not None:
            x, _ = ds[index_of[fn]]
            with torch.no_grad():
                model(x.unsqueeze(0))
            w = model.last_attention[0].numpy().reshape(grid, grid)
            axes[2, c].imshow(img112, cmap="gray")
            axes[2, c].imshow(w, cmap="viridis", alpha=0.52,
                              extent=(0, 112, 112, 0), interpolation="nearest")
            # Concentration: 1.0 = all mass on one chunk, 0.0 = uniform.
            flat = w.ravel()
            ent = -(flat * np.log(flat + 1e-12)).sum() / np.log(len(flat))
            conc.append({"class": CLASS_NAMES[c], "max_weight": float(flat.max()),
                         "concentration": float(1 - ent)})
            _axes_off(axes[2, c], f"max w = {w.max():.2f}  ·  conc {1-ent:.2f}")

    axes[0, 0].set_ylabel("input", color=T.ACCENT_BRIGHT, fontsize=10.5)
    axes[1, 0].set_ylabel("BEFORE\nfixed Sobel threshold", color=T.SERIES[1], fontsize=10.5)
    if ds is not None:
        axes[2, 0].set_ylabel("AFTER\nlearned attention", color=T.SERIES[0], fontsize=10.5)

    fig.suptitle("Chunk selection · before and after",
                 color=T.INK, fontsize=14, fontweight="semibold", x=0.01, ha="left")
    fig.text(0.01, 0.012,
             "The threshold rule is binary and photometric — a chunk is in or out. The learned "
             "attention is continuous and task-conditioned, and every chunk still receives gradient.",
             color=T.INK_MUTED, fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    save(fig, FIG / "ba_chunk_selection", formats=("png",))
    print("  -> ba_chunk_selection.png")

    if conc:
        out = ROOT / "results" / "metrics" / "attention_concentration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(conc, indent=2))
        print("     attention concentration (1.0 = one chunk, 0.0 = uniform):")
        for r in conc:
            print(f"       {r['class']:<17} max_w={r['max_weight']:.3f}  conc={r['concentration']:.3f}")


# ----------------------------------------------------------- encryption B/A
def fig_encryption() -> None:
    p = ROOT / "results/encryption_benchmarks/encrypted_inference.json"
    if not p.exists():
        print("  [skip] ba_encryption: benchmark missing")
        return
    b = json.loads(p.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    fig.patch.set_facecolor(T.PAGE)

    ax = axes[0]
    stages = ["plaintext\ninference", "encrypted\ninference"]
    vals = [b["plaintext_inference_ms"]["mean"], b["he_inference_ms"]["mean"]]
    bars = ax.bar(stages, vals, color=[T.SERIES[1], T.SERIES[0]], width=0.5)
    ax.set_yscale("log")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15,
                f"{v:.3f} ms" if v < 1 else f"{v:.0f} ms",
                ha="center", fontsize=11, color=T.INK, fontweight="semibold")
    T.style_axes(ax, ylabel="Milliseconds per sample (log)",
                 title=f"Cost of encryption: {vals[1]/max(vals[0],1e-9):,.0f}× slower")

    ax = axes[1]
    acc_b = b.get("accuracy_plaintext") or 0
    acc_a = b.get("accuracy_encrypted") or 0
    bars = ax.bar(["plaintext", "encrypted"], [acc_b * 100, acc_a * 100],
                  color=[T.SERIES[1], T.SERIES[0]], width=0.5)
    for bar, v in zip(bars, [acc_b, acc_a]):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 1,
                f"{v*100:.2f}%", ha="center", fontsize=12,
                color=T.INK, fontweight="semibold")
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    T.style_axes(ax, ylabel="Test accuracy",
                 title=f"Accuracy cost of encryption: {(acc_a-acc_b)*100:+.2f} points")
    ax.text(0.5, 0.06,
            f"agreement {b['agreement_rate']*100:.2f}% · "
            f"max logit error {b['max_abs_logit_error']:.1e} · "
            f"{b.get('error_to_margin_ratio', float('nan'))*100:.2f}% of the decision budget",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=T.INK_MUTED, style="italic")

    fig.suptitle("Encryption · before and after",
                 color=T.INK, fontsize=14, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, FIG / "ba_encryption", formats=("png",))
    print("  -> ba_encryption.png")


# ------------------------------------------------------------ per-class B/A
def fig_per_class(stats: dict | None) -> None:
    if not stats or not stats.get("per_class"):
        print("  [skip] ba_per_class: statistics missing")
        return
    pc = stats["per_class"]
    names = [r["class"] for r in pc]
    before = np.array([r["before_f1"] for r in pc]) * 100
    after = np.array([r["after_f1"] for r in pc]) * 100

    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = np.arange(len(names))
    ax.hlines(y, before, after, color=T.BORDER, lw=2.5, zorder=1)
    # Rings on both markers so a zero-delta class (identical values, coincident
    # points) still reads as two overlapping measurements rather than as one
    # missing series.
    ax.scatter(after, y, s=150, color=T.SERIES[0], zorder=2,
               edgecolors=T.PAGE, linewidths=1.6, label="AFTER · SAGE-HE")
    ax.scatter(before, y, s=90, color=T.SERIES[1], zorder=3,
               edgecolors=T.PAGE, linewidths=1.6,
               label="BEFORE · Jia et al. (reproduced)")
    for i, (b, a) in enumerate(zip(before, after)):
        delta = a - b
        colour = T.GOOD if delta > 0.05 else (T.CRITICAL if delta < -0.05 else T.INK_MUTED)
        xpos = (b + a) / 2 if abs(delta) > 0.05 else a
        ax.text(xpos, i - 0.3, f"{delta:+.1f}", ha="center",
                fontsize=9.5, color=colour, fontweight="semibold")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlim(min(before.min(), after.min()) - 5, 103)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(len(names) - 0.4, -0.75)   # headroom for the top delta label
    T.style_axes(ax, xlabel="Per-class F1", title="Per-class performance · before and after")
    # Legend below the plot so it cannot sit on top of the bottom row.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    save(fig, FIG / "ba_per_class", formats=("png",))
    print("  -> ba_per_class.png")


# ------------------------------------------------------------ confusion B/A
def fig_confusion() -> None:
    ca, cb = confusion_for(BASE), confusion_for(SAGE)
    if ca is None or cb is None:
        print("  [skip] ba_confusion: confusion matrices missing")
        return
    vmax = max(ca.max(), cb.max())
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    fig.patch.set_facecolor(T.PAGE)
    for ax, cm, title, color in [
        (axes[0], ca, "BEFORE · Jia et al. (reproduced)", T.SERIES[1]),
        (axes[1], cb, "AFTER · SAGE-HE", T.SERIES[0]),
    ]:
        im = ax.imshow(cm, cmap=T.sequential_cmap(), vmin=0, vmax=vmax)
        n = len(CLASS_NAMES)
        ax.set_xticks(range(n), [c[:9] for c in CLASS_NAMES], rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(n), [c[:9] for c in CLASS_NAMES], fontsize=9)
        ax.set_title(title, color=color, fontsize=12, loc="left", pad=10, fontweight="semibold")
        ax.grid(False)
        ax.set_xlabel("predicted", color=T.INK_MUTED, fontsize=10)
        for i in range(n):
            for j in range(n):
                v = int(cm[i, j])
                if v:
                    ax.text(j, i, v, ha="center", va="center", fontsize=10,
                            fontweight="bold" if i == j else "normal",
                            color=T.SURFACE if v > vmax * 0.55 else T.INK_SECONDARY)
    axes[0].set_ylabel("true", color=T.INK_MUTED, fontsize=10)
    fig.suptitle("Confusion matrices · before and after (identical test images)",
                 color=T.INK, fontsize=14, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, FIG / "ba_confusion", formats=("png",))
    print("  -> ba_confusion.png")


def main() -> int:
    T.apply_theme()
    FIG.mkdir(parents=True, exist_ok=True)
    print("before/after comparisons")

    stats = statistical_comparison()
    fig_preprocessing()
    fig_per_class(stats)
    fig_confusion()
    fig_encryption()

    model = cfg = None
    try:
        from scripts.run_encrypted_eval import load_model
        model, cfg = load_model(SAGE)
    except Exception as exc:      # trained model not available yet
        print(f"  [note] chunk-selection overlay without model: {exc}")
    fig_chunk_selection(model, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
