#!/usr/bin/env python
"""Generate every experimental figure from stored results.

Figures are written to results/figures/. Each one is skipped with a notice if
the experiment it depends on has not run yet, so this is safe to call at any
point during the experiment sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.neu_cls import CLASS_NAMES  # noqa: E402
from src.datasets.splits import load_splits  # noqa: E402
from src.evaluation.aggregate import (  # noqa: E402
    confusion_for, history_for, per_class_table, results_table, seed_summary,
)
from src.viz import charts as C  # noqa: E402
from src.viz import theme as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def dataset_samples(out: Path) -> None:
    """One row per defect class, four examples each."""
    import cv2
    sp = load_splits(ROOT / "data/splits.json")
    by = {}
    for fn, y in sp["train"]:
        by.setdefault(y, []).append(fn)
    # Each cell holds n images side by side, so the cell content is n:1. The
    # figure height is chosen to match that aspect, otherwise matplotlib pads
    # the rows with large bands of empty background.
    n = 4
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 3.35))
    fig.patch.set_facecolor(T.PAGE)
    for ci, ax in enumerate(axes.ravel()):
        strip = [cv2.imread(str(ROOT / "data/raw/images" / f), cv2.IMREAD_GRAYSCALE)
                 for f in sorted(by[ci])[:n]]
        ax.imshow(np.hstack(strip), cmap="gray")
        ax.set_title(CLASS_NAMES[ci], color=T.INK, fontsize=12,
                     loc="left", pad=6, fontweight="semibold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
    fig.tight_layout()
    fig.savefig(out, dpi=190, facecolor=T.PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"  dataset_samples -> {out.name}")


def model_parameters(out: Path) -> None:
    from src.models.sage_he import SageHE, SageHEConfig
    m = SageHE(SageHEConfig(pretrained=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m.count_parameters(), indent=2))
    print(f"  model_parameters -> {out.name}")


def main() -> int:
    T.apply_theme()
    FIG.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []

    dataset_samples(FIG / "dataset_samples.png")
    model_parameters(ROOT / "results/metrics/model_parameters.json")

    # ---- headline comparisons
    summ = seed_summary("main")
    if summ is not None and not summ.empty and "test_accuracy_mean" in summ:
        C.accuracy_comparison(summ, FIG / "accuracy_comparison")
        C.metric_comparison(summ, FIG / "metric_comparison")
        made += ["accuracy_comparison", "metric_comparison"]
    else:
        skipped.append("accuracy/metric comparison (needs main suite)")

    # ---- confusion matrices, per-class, curves
    for key, label, fname in [("main/sage_he_seed42", "SAGE-HE", "sage"),
                              ("main/jia_baseline_seed42", "Jia et al. (reproduced)", "jia")]:
        cm = confusion_for(key)
        if cm is not None:
            C.confusion_matrix_fig(cm, CLASS_NAMES, FIG / f"confusion_{fname}",
                                   f"Confusion matrix · {label}")
            made.append(f"confusion_{fname}")
            tbl = per_class_table(key, CLASS_NAMES)
            if tbl is not None:
                C.per_class_chart(tbl, FIG / f"per_class_{fname}",
                                  f"Per-class performance · {label}")
                made.append(f"per_class_{fname}")
            h = history_for(key)
            if h is not None and len(h):
                C.training_curves(h, FIG / f"curves_{fname}",
                                  f"Training dynamics · {label}")
                made.append(f"curves_{fname}")
        else:
            skipped.append(f"confusion/{fname}")

    # ---- ablation
    abl = results_table("abl")
    if abl is not None and not abl.empty and abl["test_macro_f1"].notna().any():
        frac = abl["train_fraction"].iloc[0]
        n_tr = int(abl["n_train"].iloc[0])
        C.ablation_chart(abl, FIG / "ablation",
                         title=f"Component ablation · {int(frac*100)}% of training data "
                               f"({n_tr} images)")
        made.append("ablation")
    else:
        skipped.append("ablation (needs ablation suite)")

    # ---- learning curve: capacity runs + the 100% point from the main suite
    cap = results_table("capacity")
    main = results_table("main")
    if cap is not None and not cap.empty:
        rows = cap.copy()
        if main is not None and not main.empty:
            extra = main[main["seed"] == 42].copy()
            extra["short"] = extra["short"].str.replace("jia_baseline_seed42", "jia_100")
            extra["short"] = extra["short"].str.replace("sage_he_seed42", "sage_100")
            rows = __import__("pandas").concat([rows, extra], ignore_index=True)
        C.learning_curve(rows, FIG / "learning_curve")
        made.append("learning_curve")
    else:
        skipped.append("learning curve (needs capacity suite)")

    # ---- encryption
    encp = ROOT / "results/encryption_benchmarks/encrypted_inference.json"
    if encp.exists():
        C.encryption_breakdown(json.loads(encp.read_text()), FIG / "encryption_breakdown")
        made.append("encryption_breakdown")
    else:
        skipped.append("encryption breakdown (needs run_encrypted_eval.py)")

    sweepp = ROOT / "results/encryption_benchmarks/compression_sweep.json"
    if sweepp.exists():
        sweep = json.loads(sweepp.read_text())
        if sweep:
            C.compression_tradeoff(sweep, FIG / "compression_tradeoff")
            made.append("compression_tradeoff")
    else:
        skipped.append("compression tradeoff (needs --sweep)")

    print(f"\n{len(made)} figures written to {FIG}")
    for s in skipped:
        print(f"  [skipped] {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
