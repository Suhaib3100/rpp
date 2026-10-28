"""The architecture diagrams must agree with the implementation.

A diagram that disagrees with the code is worse than no diagram, so the tensor
shapes and CKKS parameters annotated on the figures are asserted against the
real model rather than trusted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_diagram_annotations_match_implementation():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_diagram_shapes.py")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=900,
    )
    assert r.returncode == 0, f"diagram annotations disagree with the code:\n{r.stdout}"
