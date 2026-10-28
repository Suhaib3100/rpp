"""Experimental result charts.

Design rules followed throughout (see src/viz/theme.py for the validated
palette): categorical hues assigned in fixed slot order and never cycled; one
y-axis per chart, never two; sequential magnitude uses a single hue ramp;
status colours reserved. Charts with more than two series always carry direct
value labels or a legend, so colour never carries identity alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import theme as T
from src.viz.diagram import save

PUBLISHED_JIA = 97.1   # Jia et al. (2023), Table 3, NEU-CLS, square activation


def _pct(ax) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def _bar_labels(ax, bars, fmt="{:.1f}", dy=0.6, color=None) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                    ha="center", va="bottom", fontsize=9.5,
                    color=color or T.INK, fontweight="semibold")


# ------------------------------------------------------------------ headline
def accuracy_comparison(summary: pd.DataFrame, out: Path) -> list:
    """Published figure vs reproduced baseline vs proposed model."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    rows = []
    for arch, label, color in [
        ("jia_baseline", "Jia et al. (2023)\nreproduced", T.SERIES[1]),
        ("sage_he", "SAGE-HE\n(proposed)", T.SERIES[0]),
    ]:
        r = summary[summary["arch"] == arch]
        if len(r):
            rows.append((label, float(r["test_accuracy_mean"].iloc[0]) * 100,
                         float(r["test_accuracy_std"].fillna(0).iloc[0]) * 100, color))

    labels = ["Jia et al. (2023)\npublished (Table 3)"] + [r[0] for r in rows]
    vals = [PUBLISHED_JIA] + [r[1] for r in rows]
    errs = [0.0] + [r[2] for r in rows]
    colors = [T.INK_MUTED] + [r[3] for r in rows]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.58, color=colors,
                  yerr=errs, capsize=5,
                  error_kw=dict(ecolor=T.INK_SECONDARY, lw=1.4))
    # hatch the published bar: it is a literature value, not something we ran
    bars[0].set_hatch("///")
    bars[0].set_edgecolor(T.SURFACE)
    bars[0].set_linewidth(0)

    for b, v, e in zip(bars, vals, errs):
        txt = f"{v:.2f}" + (f" ± {e:.2f}" if e > 0 else "")
        # clear the error-bar cap, otherwise the ± sits on top of it
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.5, txt,
                ha="center", va="bottom", fontsize=10.5,
                color=T.INK, fontweight="semibold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(80, 103)
    _pct(ax)
    T.style_axes(ax, ylabel="Test accuracy", title="Accuracy on NEU-CLS")
    ax.axhline(PUBLISHED_JIA, color=T.INK_MUTED, lw=1, ls=(0, (5, 4)), zorder=0)
    ax.text(len(labels) - 0.4, PUBLISHED_JIA + 0.25, "published 97.1%",
            color=T.INK_MUTED, fontsize=9, ha="right")
    # The caveat belongs outside the bar: inside it, the hatch swallows it.
    fig.text(0.5, -0.02,
             "Hatched bar is a literature value on an unknown split, not a run of ours — "
             "it is shown for reference and is not a like-for-like comparison.",
             ha="center", fontsize=9, color=T.INK_MUTED, style="italic")
    fig.tight_layout()
    return save(fig, out)


def metric_comparison(summary: pd.DataFrame, out: Path) -> list:
    """Macro precision / recall / F1, reproduced baseline vs proposed."""
    metrics = [("test_macro_precision", "Macro precision"),
               ("test_macro_recall", "Macro recall"),
               ("test_macro_f1", "Macro F1")]
    archs = [("jia_baseline", "Jia et al. (reproduced)", T.SERIES[1]),
             ("sage_he", "SAGE-HE (proposed)", T.SERIES[0])]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(metrics))
    w = 0.34
    for i, (arch, label, color) in enumerate(archs):
        r = summary[summary["arch"] == arch]
        if not len(r):
            continue
        vals = [float(r[f"{m}_mean"].iloc[0]) * 100 for m, _ in metrics]
        errs = [float(r[f"{m}_std"].fillna(0).iloc[0]) * 100 for m, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.9, label=label, color=color,
                      yerr=errs, capsize=4,
                      error_kw=dict(ecolor=T.INK_SECONDARY, lw=1.2))
        _bar_labels(ax, bars, "{:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(80, 103)
    _pct(ax)
    T.style_axes(ax, ylabel="Score", title="Macro-averaged metrics on the held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out)


# --------------------------------------------------------------- confusion
def confusion_matrix_fig(cm: np.ndarray, class_names: list[str], out: Path,
                         title: str = "Confusion matrix") -> list:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = T.sequential_cmap()
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())

    n = len(class_names)
    ax.set_xticks(range(n), class_names, rotation=35, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted", color=T.INK_SECONDARY)
    ax.set_ylabel("True", color=T.INK_SECONDARY)
    ax.set_title(title, color=T.INK, loc="left", pad=12, fontweight="semibold")
    ax.grid(False)

    thresh = cm.max() * 0.55
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    fontsize=11, fontweight="bold" if i == j else "normal",
                    color=T.SURFACE if v > thresh else T.INK_SECONDARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_edgecolor(T.BORDER)
    cb.ax.tick_params(colors=T.INK_MUTED)
    fig.tight_layout()
    return save(fig, out)


# ----------------------------------------------------------- training curves
def training_curves(hist: pd.DataFrame, out: Path, title: str) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    a, b = axes

    a.plot(hist["epoch"], hist["train_loss"], color=T.SERIES[0], label="train")
    a.plot(hist["epoch"], hist["val_loss"], color=T.SERIES[1], label="validation")
    T.style_axes(a, xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    a.legend()

    b.plot(hist["epoch"], hist["val_accuracy"] * 100, color=T.SERIES[0], label="accuracy")
    b.plot(hist["epoch"], hist["val_macro_f1"] * 100, color=T.SERIES[2], label="macro F1")
    T.style_axes(b, xlabel="Epoch", ylabel="Validation score", title="Validation performance")
    _pct(b)
    b.legend(loc="lower right")

    fig.suptitle(title, color=T.INK, fontsize=13, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out)


# ------------------------------------------------------------------ ablation
def aggregate_ablation(df: pd.DataFrame, metric: str = "test_macro_f1") -> pd.DataFrame:
    """Collapse per-seed ablation runs into mean / std / n per stage."""
    d = df.copy()
    d["stage"] = d["short"].str.replace(r"_seed\d+$", "", regex=True)
    g = d.groupby("stage")[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values("stage").reset_index(drop=True)


# Below this, a run failed to train rather than performed badly (chance on six
# balanced classes is ~16.7% macro F1). Kept consistent with
# scripts/analyse_components.py.
COLLAPSE_F1 = 40.0


def ablation_chart(df: pd.DataFrame, out: Path, metric: str = "test_macro_f1",
                   title: str = "") -> list:
    """Bars are the median across seeds; every individual run is plotted on top.

    A mean ± sd summary is actively misleading here: with three seeds, one
    training collapse drags the mean down ~25 points and inflates the error bar
    into a symmetric band that implies a spread the data does not have. Plotting
    each run makes a bimodal "two good, one collapsed" outcome legible as what
    it is, and the median bar stays robust to it.
    """
    g = aggregate_ablation(df, metric)
    d = df.copy()
    d["stage"] = d["short"].str.replace(r"_seed\d+$", "", regex=True)

    stages = list(g["stage"])
    labels = [s.split("_", 1)[1].replace("_", " ") for s in stages]
    med = np.array([d.loc[d["stage"] == s, metric].median() * 100 for s in stages])
    n_seeds = int(g["count"].max()) if len(g) else 0

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    colors = [T.SERIES[0] if ("chunk attention" in l and "no sobel" not in l)
              else T.SERIES[1] if "baseline" in l else T.ACCENT_DIM for l in labels]
    ax.barh(range(len(labels)), med, color=colors, height=0.6)

    any_collapse = False
    for i, s in enumerate(stages):
        runs = d.loc[d["stage"] == s, metric].values * 100
        ok = runs >= COLLAPSE_F1
        if ok.any():
            ax.scatter(runs[ok], np.full(ok.sum(), i), s=34, zorder=3,
                       color=T.INK, edgecolors=T.PAGE, linewidths=1.1)
        if (~ok).any():
            any_collapse = True
            ax.scatter(runs[~ok], np.full((~ok).sum(), i), s=56, zorder=3,
                       marker="X", color=T.CRITICAL, edgecolors=T.PAGE, linewidths=1.1)
        # The median label goes *inside* the bar, left-aligned. Outside the bar
        # it collides with the individual-run dots, which cluster just past the
        # bar end; anchoring it to the furthest run instead parks it at the far
        # right of a bimodal row where it reads as the wrong value.
        if med[i] > 12:
            ax.text(1.6, i, f"{med[i]:.1f}", va="center", ha="left",
                    fontsize=9.5, color=T.INK, fontweight="semibold")
        else:
            ax.text(max(runs.max(), med[i]) + 1.4, i, f"{med[i]:.1f}",
                    va="center", fontsize=9.5, color=T.INK, fontweight="semibold")

    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

    handles = [plt.Line2D([], [], marker="o", ls="", color=T.INK,
                          markeredgecolor=T.PAGE, label=f"individual run (n={n_seeds} seeds)")]
    if any_collapse:
        handles.append(plt.Line2D([], [], marker="X", ls="", color=T.CRITICAL,
                                  markeredgecolor=T.PAGE,
                                  label=f"collapsed run (< {COLLAPSE_F1:.0f}% — failed to train)"))
    # Legend below the axes: at the bottom-right it sits on the last row's marks.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    T.style_axes(ax, xlabel="Test macro F1 (bar = median across seeds)",
                 title=title or f"Component ablation ({n_seeds} seeds)")
    fig.tight_layout()
    return save(fig, out)


# ------------------------------------------------------------ learning curve
def learning_curve(df: pd.DataFrame, out: Path) -> list:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model, label, color in [("sage", "SAGE-HE", T.SERIES[0]),
                                ("jia", "Jia et al. (reproduced)", T.SERIES[1])]:
        sub = df[df["short"].str.startswith(model)].sort_values("n_train")
        if sub.empty:
            continue
        ax.plot(sub["n_train"], sub["test_macro_f1"] * 100, marker="o",
                color=color, label=label)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['test_macro_f1']*100:.1f}",
                        (r["n_train"], r["test_macro_f1"] * 100),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5, color=T.INK_SECONDARY)
    ax.set_xscale("log")
    # Ticks come from the data, not a hard-coded list: the fractions resolve to
    # whatever stratified subsampling actually produced.
    sizes = sorted({int(v) for v in df["n_train"].dropna().unique()})
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    _pct(ax)
    T.style_axes(ax, xlabel="Training images (log scale)", ylabel="Test macro F1",
                 title="Learning curve: where the architectures separate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out)


# --------------------------------------------------------------- encryption
def encryption_breakdown(bench: dict, out: Path) -> list:
    """Latency split across the three protocol stages (one axis, stacked)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    stages = [("encrypt_ms", "Client: encrypt", T.SERIES[2]),
              ("he_inference_ms", "Server: homomorphic evaluation", T.SERIES[0]),
              ("decrypt_ms", "Client: decrypt", T.SERIES[3])]
    left = 0.0
    for key, label, color in stages:
        v = bench[key]["mean"]
        ax.barh([0], [v], left=left, color=color, label=f"{label}  ({v:.1f} ms)",
                height=0.42, edgecolor=T.SURFACE, linewidth=2)
        left += v
    ax.set_yticks([])
    ax.set_xlim(0, left * 1.02)
    ax.grid(axis="x")
    T.style_axes(ax, xlabel="Milliseconds per sample",
                 title=f"Encrypted inference latency  ·  total {left:.0f} ms/sample")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.tight_layout()
    return save(fig, out)


def compression_tradeoff(rows: list[dict], out: Path) -> list:
    """Accuracy and HE latency against compressed dimension - two panels, never two y-axes."""
    rows = sorted(rows, key=lambda r: r["dim"])
    dims = [r["dim"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    a, b = axes

    f1 = [r["macro_f1"] * 100 for r in rows]
    bars = a.bar([str(d) for d in dims], f1, color=T.SERIES[0], width=0.55)
    _bar_labels(a, bars, "{:.2f}", dy=0.15)
    a.set_ylim(min(f1) - 3, 101)
    _pct(a)
    T.style_axes(a, xlabel="Compressed dimension d", ylabel="Test macro F1",
                 title="Accuracy vs. compression")

    lat = [r["he_ms"] for r in rows]
    bars = b.bar([str(d) for d in dims], lat, color=T.SERIES[3], width=0.55)
    _bar_labels(b, bars, "{:.0f}", dy=max(lat) * 0.01)
    T.style_axes(b, xlabel="Compressed dimension d", ylabel="Homomorphic evaluation (ms)",
                 title="Encrypted latency vs. compression")
    # The two panels are measured in different regimes and must not be read as
    # one curve: accuracy comes from the reduced-data design study (where the
    # dimensions are separable at all), latency is a property of d alone.
    fig.text(0.5, -0.03,
             "Accuracy is measured in the reduced-data design regime, where the dimensions "
             "are separable; at full training size all of them sit at ceiling. "
             "Latency depends only on d and is independent of the training regime.",
             ha="center", fontsize=9, color=T.INK_MUTED, style="italic")
    fig.tight_layout()
    return save(fig, out)


# ---------------------------------------------------------------- per class
def per_class_chart(tbl: pd.DataFrame, out: Path, title: str) -> list:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(tbl))
    w = 0.26
    for i, (col, label) in enumerate([("precision", "Precision"),
                                      ("recall", "Recall"), ("f1", "F1")]):
        bars = ax.bar(x + (i - 1) * w, tbl[col] * 100, w * 0.92,
                      label=label, color=T.SERIES[i])
        _bar_labels(ax, bars, "{:.0f}", dy=0.5)
    ax.set_xticks(x, tbl["class"], rotation=20, ha="right")
    ax.set_ylim(0, 108)
    _pct(ax)
    T.style_axes(ax, ylabel="Score", title=title)
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    return save(fig, out)
