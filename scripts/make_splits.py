#!/usr/bin/env python
"""Build the reproducible train/val/test manifest."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.splits import build_splits, SplitConfig

ROOT = Path(__file__).resolve().parents[1]
out = build_splits(ROOT/"data/raw/manifest.json", ROOT/"data/splits.json", SplitConfig())
print(json.dumps({k: out[k] for k in
      ["n_source_records","n_unique_after_dedup","duplicates_removed","counts","per_class_counts"]}, indent=2))
