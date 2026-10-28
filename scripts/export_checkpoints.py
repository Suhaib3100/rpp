#!/usr/bin/env python
"""Export inference-ready weights for the headline models.

Training checkpoints in `experiments/` carry optimiser moments, scheduler and
scaler state and the full epoch history — roughly three times the size of the
weights themselves, and useless to anyone who only wants to run the model. This
writes a slim, self-describing checkpoint per headline model into
`checkpoints/`, which is what gets committed.

Each file contains the state dict, the exact config needed to rebuild the
architecture, and the test metrics it achieved, so a reader can verify the
numbers without rerunning anything.

Usage:  python scripts/export_checkpoints.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkpoints"

# The two models every reported comparison rests on.
HEADLINE = {
    "sage_he": "main/sage_he_seed42",
    "jia_baseline": "main/jia_baseline_seed42",
}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    index = []
    for name, exp in HEADLINE.items():
        src = ROOT / "experiments" / exp
        ck = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
        res = json.loads((src / "result.json").read_text())

        slim = {
            "model_state_dict": ck["model"],
            "config": res["config"],
            "test_metrics": res.get("test_metrics"),
            "test_accuracy_ci95": res.get("test_accuracy_ci95"),
            "n_parameters": res.get("n_parameters"),
            "source_experiment": exp,
            "note": (
                "Inference-ready weights. Rebuild the architecture from "
                "`config` via src.training.experiment.build_model, then "
                "load_state_dict(model_state_dict)."
            ),
        }
        dest = OUT / f"{name}.pt"
        torch.save(slim, dest)
        mb = dest.stat().st_size / 1048576
        full_mb = (src / "best.pt").stat().st_size / 1048576
        acc = (res.get("test_metrics") or {}).get("accuracy")
        print(f"  {name:<14} {mb:6.1f} MB  (training ckpt was {full_mb:.1f} MB)"
              f"   test acc {acc*100:.2f}%" if acc else f"  {name}: {mb:.1f} MB")
        index.append({"name": name, "file": f"checkpoints/{name}.pt",
                      "size_mb": round(mb, 2), "source_experiment": exp,
                      "test_accuracy": acc})

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwritten: {OUT}/  ({len(index)} checkpoints + index.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
