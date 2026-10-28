"""Reproducible, leakage-free train/val/test partitioning of NEU-CLS.

Two properties matter here and are enforced rather than assumed:

1. **No duplicate leakage.** The source mirror contains one exact-duplicate
   image pair (verified by SHA-256 over raw pixels). If the two copies landed
   on opposite sides of a split, the test set would contain an image the model
   had already memorised. We deduplicate by pixel hash *before* splitting, so
   every evaluation image is unique.

2. **The test set is touched exactly once.** Splits are written to a manifest
   and re-read; model selection only ever sees train/val.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 1337
    test_size: float = 0.20
    val_size: float = 0.15  # fraction of the post-test remainder


def build_splits(
    manifest_path: Path,
    out_path: Path,
    cfg: SplitConfig = SplitConfig(),
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest["records"]

    # ---- deduplicate by pixel content -------------------------------------
    seen: dict[str, str] = {}
    unique, dropped = [], []
    for r in records:
        sha = r["sha256_pixels"]
        if sha in seen:
            dropped.append({"dropped": r["filename"], "duplicate_of": seen[sha],
                            "label_index": r["label_index"]})
            continue
        seen[sha] = r["filename"]
        unique.append(r)

    files = [r["filename"] for r in unique]
    labels = [r["label_index"] for r in unique]

    # ---- stratified train / val / test ------------------------------------
    f_trval, f_test, y_trval, y_test = train_test_split(
        files, labels, test_size=cfg.test_size, random_state=cfg.seed, stratify=labels
    )
    f_train, f_val, y_train, y_val = train_test_split(
        f_trval, y_trval, test_size=cfg.val_size, random_state=cfg.seed, stratify=y_trval
    )

    splits = {
        "train": sorted(zip(f_train, y_train)),
        "val": sorted(zip(f_val, y_val)),
        "test": sorted(zip(f_test, y_test)),
    }

    # ---- assert the guarantees we just claimed ----------------------------
    sets = {k: {f for f, _ in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"
    assert sum(len(v) for v in splits.values()) == len(unique), "split lost samples"

    out = {
        "config": asdict(cfg),
        "source_manifest": str(manifest_path),
        "n_source_records": len(records),
        "n_unique_after_dedup": len(unique),
        "duplicates_removed": dropped,
        "counts": {k: len(v) for k, v in splits.items()},
        "per_class_counts": {
            k: {str(c): n for c, n in sorted(Counter(y for _, y in v).items())}
            for k, v in splits.items()
        },
        "splits": {k: [{"filename": f, "label_index": y} for f, y in v] for k, v in splits.items()},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


def load_splits(path: Path) -> dict[str, list[tuple[str, int]]]:
    data = json.loads(Path(path).read_text())
    return {
        k: [(r["filename"], r["label_index"]) for r in v]
        for k, v in data["splits"].items()
    }
