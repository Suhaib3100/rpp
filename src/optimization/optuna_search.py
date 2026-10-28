"""Optuna hyperparameter optimisation for SAGE-HE.

The search is deliberately run in the reduced-data regime. On full NEU-CLS the
validation macro-F1 saturates at 1.0 within two epochs, so every trial would
score identically and the study would rank noise. At a small training fraction
the objective has real dynamic range, and the hyperparameters that matter there
are the ones that matter when data is scarce - which is the regime an
industrial deployment with limited labelled defects actually faces.

Objective: validation macro-F1 (maximise), with validation loss as the
tiebreaker, matching the model-selection rule used everywhere else.
"""
from __future__ import annotations

import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.training.experiment import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parents[2]


def build_trial_config(trial: optuna.Trial, fraction: float, epochs: int,
                       seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"optuna/trial_{trial.number:03d}",
        model="sage_he",
        image_size=160,
        contrast=trial.suggest_categorical("contrast", ["clahe", "histeq"]),
        lr=trial.suggest_float("lr", 5e-5, 2e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32]),
        compressed_dim=trial.suggest_categorical("compressed_dim", [32, 64, 128]),
        head_hidden=trial.suggest_categorical("head_hidden", [16, 32, 64]),
        fusion=trial.suggest_categorical("fusion", ["gated", "attention", "concat"]),
        use_sobel=trial.suggest_categorical("use_sobel", [True, False]),
        epochs=epochs,
        patience=epochs,          # no early stop inside a short study
        seed=seed,
        train_fraction=fraction,
        notes="Optuna trial",
    )


def objective_factory(fraction: float, epochs: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        cfg = build_trial_config(trial, fraction, epochs, seed)
        # Trials never touch the test set.
        result = run_experiment(cfg, out_root=ROOT / "experiments",
                                evaluate_test=False, resume=False)
        trial.set_user_attr("val_accuracy", result["val_metrics"]["accuracy"])
        trial.set_user_attr("best_epoch", result["fit"]["best_epoch"])
        trial.set_user_attr("n_parameters", result["n_parameters"])
        return float(result["val_metrics"]["macro_f1"])
    return objective


def run_study(n_trials: int = 15, fraction: float = 0.10, epochs: int = 8,
              seed: int = 42) -> dict:
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        pruner=MedianPruner(n_startup_trials=4),
        study_name="sage_he_lowdata",
    )
    study.optimize(objective_factory(fraction, epochs, seed), n_trials=n_trials)

    rows = [{
        "number": t.number,
        "value": t.value,
        "state": str(t.state),
        "params": t.params,
        "val_accuracy": t.user_attrs.get("val_accuracy"),
        "n_parameters": t.user_attrs.get("n_parameters"),
    } for t in study.trials]

    out = {
        "n_trials": len(study.trials),
        "train_fraction": fraction,
        "epochs_per_trial": epochs,
        "objective": "validation macro-F1",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "trials": rows,
        "param_importances": _importances(study),
    }
    path = ROOT / "results" / "metrics" / "optuna_study.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def _importances(study: optuna.Study) -> dict:
    try:
        return {k: float(v) for k, v in
                optuna.importance.get_param_importances(study).items()}
    except (ValueError, RuntimeError):
        # needs several completed trials with differing params
        return {}
