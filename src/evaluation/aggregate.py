"""Collect experiment results into tidy tables.

Every figure and every number in the report and the deck is read from here, so
there is exactly one path from a training run to a reported value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"


def load_results(group: str | None = None) -> list[dict]:
    """Load every result.json, optionally filtered to one group prefix."""
    out = []
    for p in sorted(EXPERIMENTS.glob("*/*/result.json")):
        r = json.loads(p.read_text())
        name = r["config"]["name"]
        if group and not name.startswith(f"{group}/"):
            continue
        r["_path"] = str(p)
        out.append(r)
    return out


def results_table(group: str | None = None) -> pd.DataFrame:
    rows = []
    for r in load_results(group):
        cfg = r["config"]
        t = r.get("test_metrics") or {}
        v = r.get("val_metrics") or {}
        ci = r.get("test_accuracy_ci95") or [np.nan, np.nan]
        rows.append({
            "name": cfg["name"],
            "group": cfg["name"].split("/")[0],
            "short": cfg["name"].split("/", 1)[1],
            "model": cfg["model"],
            "seed": cfg["seed"],
            "train_fraction": cfg.get("train_fraction", 1.0),
            "n_train": r.get("n_train"),
            "image_size": cfg["image_size"],
            "contrast": cfg["contrast"],
            "fusion": cfg.get("fusion"),
            "use_local": cfg.get("use_local"),
            "use_global": cfg.get("use_global"),
            "use_attention": cfg.get("use_attention"),
            "use_sobel": cfg.get("use_sobel"),
            "top_k": cfg.get("top_k"),
            "compressed_dim": cfg.get("compressed_dim"),
            "n_parameters": r.get("n_parameters"),
            "epochs_run": r["fit"]["epochs_run"],
            "best_epoch": r["fit"]["best_epoch"],
            "train_seconds": r.get("train_seconds"),
            "val_accuracy": v.get("accuracy"),
            "val_macro_f1": v.get("macro_f1"),
            "test_accuracy": t.get("accuracy"),
            "test_macro_precision": t.get("macro_precision"),
            "test_macro_recall": t.get("macro_recall"),
            "test_macro_f1": t.get("macro_f1"),
            "test_acc_lo": ci[0],
            "test_acc_hi": ci[1],
            "notes": cfg.get("notes", ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("name").reset_index(drop=True) if len(df) else df


def seed_summary(group: str = "main") -> pd.DataFrame:
    """Mean +/- std across seeds for each architecture in a group."""
    df = results_table(group)
    if df.empty:
        return df
    df["arch"] = df["short"].str.replace(r"_seed\d+$", "", regex=True)
    metrics = ["test_accuracy", "test_macro_precision",
               "test_macro_recall", "test_macro_f1"]
    g = df.groupby("arch")[metrics]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def confusion_for(name: str) -> np.ndarray | None:
    p = EXPERIMENTS / name / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    cm = (r.get("test_metrics") or {}).get("confusion")
    return np.array(cm) if cm else None


def history_for(name: str) -> pd.DataFrame | None:
    p = EXPERIMENTS / name / "result.json"
    if not p.exists():
        return None
    return pd.DataFrame(json.loads(p.read_text())["history"])


def per_class_table(name: str, class_names: list[str]) -> pd.DataFrame | None:
    p = EXPERIMENTS / name / "result.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text()).get("test_metrics")
    if not t:
        return None
    return pd.DataFrame({
        "class": class_names,
        "precision": t["per_class_precision"],
        "recall": t["per_class_recall"],
        "f1": t["per_class_f1"],
        "support": t["per_class_support"],
    })
