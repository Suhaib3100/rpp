#!/usr/bin/env python
"""Render every architecture diagram to SVG + PNG."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz import theme as T
T.apply_theme()
from src.viz.architecture_diagrams import ALL_DIAGRAMS

OUT = Path(__file__).resolve().parents[1] / "diagrams"
for name, fn in ALL_DIAGRAMS:
    paths = fn(OUT / name)
    print(f"  {name:<32} -> {', '.join(p.suffix[1:] for p in paths)}")
print(f"\n{len(ALL_DIAGRAMS)} diagrams written to {OUT}")
