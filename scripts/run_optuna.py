#!/usr/bin/env python
"""Run the Optuna hyperparameter study."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimization.optuna_search import run_study

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=15)
ap.add_argument("--fraction", type=float, default=0.10)
ap.add_argument("--epochs", type=int, default=8)
a = ap.parse_args()
out = run_study(n_trials=a.trials, fraction=a.fraction, epochs=a.epochs)
print(json.dumps({k: out[k] for k in ["n_trials","best_value","best_params","best_trial_number"]}, indent=2))
