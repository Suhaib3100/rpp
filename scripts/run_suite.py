#!/usr/bin/env python
"""Run a named group of experiments.

Groups
------
main       Jia baseline reproduction + SAGE-HE, 3 seeds each, full data.
ablation   The nine-stage component ablation from the project brief, run in
           the low-data regime where the components are separable.
design     Supporting design studies: resolution, contrast, fusion, Sobel,
           top-k, compressed dimension.
capacity   SAGE-HE vs baseline across training-set fractions (the learning
           curve that justifies the low-data ablation protocol).

Usage:
    python scripts/run_suite.py main
    python scripts/run_suite.py ablation --fraction 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.experiment import ExperimentConfig, run_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Low-data fraction used for the ablation, chosen from the measured learning
# curve rather than guessed. The capacity suite shows SAGE-HE reaching 99.72%
# test accuracy at 15% of the training split and 99.17% at 10% — both still at
# ceiling, where an ablation would again be ranking noise. At 5% (61 images,
# ~10 per class) it sits at 88.33%, leaving ~12 points of headroom for a
# component to actually move. Runs at this size are also cheap enough to repeat
# over seeds, which matters because 61 images is a noisy regime: the ablation
# is therefore run over ABL_SEEDS and reported as mean ± spread, not as single
# point estimates.
ABL_FRAC = 0.05
ABL_SEEDS = (42, 1337, 2024)

# Shared training budget. SAGE-HE converges on NEU-CLS within a handful of
# epochs; the baseline trains from scratch and needs more.
SAGE_EPOCHS = 20
BASE_EPOCHS = 60


def main_group(seeds=(42, 1337, 2024)) -> list[ExperimentConfig]:
    cfgs = []
    for s in seeds:
        cfgs.append(ExperimentConfig(
            name=f"main/jia_baseline_seed{s}", model="jia_baseline",
            image_size=112, pipeline="jia_baseline", contrast="histeq",
            activation="square", epochs=BASE_EPOCHS, lr=1e-3, patience=15, seed=s,
            notes="Faithful reproduction of Jia et al. (2023) preprocessing + architecture.",
        ))
        cfgs.append(ExperimentConfig(
            name=f"main/sage_he_seed{s}", model="sage_he",
            image_size=160, pipeline="sage", contrast="clahe",
            epochs=SAGE_EPOCHS, lr=3e-4, patience=8, seed=s,
            notes="Proposed SAGE-HE: EfficientNet-B0 + multi-scale local branch + "
                  "Sobel-gated chunk attention + gated fusion + 64-d compression.",
        ))
    return cfgs


def ablation_group(frac: float = ABL_FRAC,
                   seeds: tuple[int, ...] = ABL_SEEDS) -> list[ExperimentConfig]:
    """The ablation stages named in the brief, repeated over seeds.

    At 61 training images a single run is noisy, so each stage is trained with
    every seed in ``seeds`` and reported as a mean with spread. Runs at this
    size take a couple of minutes each, so the repetition is affordable.
    """
    cfgs: list[ExperimentConfig] = []
    for seed in seeds:
        base = dict(image_size=160, pipeline="sage", contrast="clahe",
                    epochs=SAGE_EPOCHS, lr=3e-4, patience=8, seed=seed,
                    train_fraction=frac)
        s = f"_seed{seed}"
        cfgs += [
            # 1. baseline CNN (the base paper's architecture)
            ExperimentConfig(name=f"abl/1_baseline_cnn{s}", model="jia_baseline",
                             image_size=112, pipeline="jia_baseline", contrast="histeq",
                             activation="square", epochs=BASE_EPOCHS, lr=1e-3,
                             patience=15, seed=seed, train_fraction=frac,
                             notes="Stage 1: base paper CNN."),
            # 2. EfficientNet alone
            ExperimentConfig(name=f"abl/2_global_only{s}", use_local=False,
                             notes="Stage 2: EfficientNet-B0 global branch alone.", **base),
            # 3. multi-scale CNN alone
            ExperimentConfig(name=f"abl/3_local_only{s}", use_global=False,
                             use_attention=False,
                             notes="Stage 3: multi-scale local CNN alone.", **base),
            # 4. dual branch, plain concatenation
            ExperimentConfig(name=f"abl/4_dual_concat{s}", fusion="concat",
                             use_attention=False,
                             notes="Stage 4: dual branch, concatenation fusion.", **base),
            # 5. dual branch + scalar attention fusion
            ExperimentConfig(name=f"abl/5_dual_attention{s}", fusion="attention",
                             use_attention=False,
                             notes="Stage 5: dual branch, scalar attention fusion.", **base),
            # 6. dual branch + channel-wise gated fusion
            ExperimentConfig(name=f"abl/6_dual_gated{s}", fusion="gated",
                             use_attention=False,
                             notes="Stage 6: dual branch, channel-wise gated fusion.", **base),
            # 7. + adaptive chunk selection = full SAGE-HE
            ExperimentConfig(name=f"abl/7_chunk_attention{s}", fusion="gated",
                             use_attention=True, use_sobel=True,
                             notes="Stage 7: + Sobel-gated adaptive chunk selection = SAGE-HE.",
                             **base),
            # 8. control isolating the Sobel prior
            ExperimentConfig(name=f"abl/8_chunk_attention_no_sobel{s}", fusion="gated",
                             use_attention=True, use_sobel=False,
                             notes="Stage 7 control: learned chunk attention WITHOUT the "
                                   "Sobel prior, isolating the prior's contribution.", **base),
        ]
    return cfgs


def design_group(frac: float = ABL_FRAC,
                 seeds: tuple[int, ...] = ABL_SEEDS) -> list[ExperimentConfig]:
    """Design studies, at the same reduced size and over the same seeds.

    Run at ABL_FRAC for the same reason as the ablation: at 10% and above the
    model is already at ceiling, so a resolution or contrast change would show
    no difference whatever its real effect. Seeded because 61 images is noisy
    enough that a single run could reverse any of these comparisons.
    """
    cfgs: list[ExperimentConfig] = []
    for seed in seeds:
        base = dict(image_size=160, pipeline="sage", contrast="clahe",
                    epochs=SAGE_EPOCHS, lr=3e-4, patience=8, seed=seed,
                    train_fraction=frac)
        s = f"_seed{seed}"
        for size in (112, 200):
            d = dict(base); d["image_size"] = size
            cfgs.append(ExperimentConfig(name=f"design/res_{size}{s}",
                                         notes=f"Input resolution study: {size}px.", **d))
        for contrast in ("none", "histeq"):
            d = dict(base); d["contrast"] = contrast
            cfgs.append(ExperimentConfig(name=f"design/contrast_{contrast}{s}",
                                         notes=f"Contrast study: {contrast}.", **d))
        for dim in (32, 128, 256):
            cfgs.append(ExperimentConfig(name=f"design/compress_{dim}{s}",
                                         compressed_dim=dim,
                                         notes=f"Compressed dimension {dim}.", **base))
        for k in (4, 8):
            cfgs.append(ExperimentConfig(name=f"design/topk_{k}{s}", top_k=k,
                                         notes=f"Hard top-{k} chunk selection.", **base))
    return cfgs


def capacity_group(seed: int = 42) -> list[ExperimentConfig]:
    """Learning curve. The 100% point is deliberately absent: it is exactly the
    `main/*_seed42` configuration, so the curve reuses those results instead of
    retraining the same two models for an hour."""
    cfgs = []
    for frac in (0.05, 0.10, 0.15, 0.25, 0.50):
        pct = int(frac * 100)
        cfgs.append(ExperimentConfig(
            name=f"capacity/sage_{pct:03d}", image_size=160, contrast="clahe",
            epochs=SAGE_EPOCHS, lr=3e-4, patience=8, seed=seed, train_fraction=frac,
            notes=f"SAGE-HE learning curve at {pct}% of training data."))
        cfgs.append(ExperimentConfig(
            name=f"capacity/jia_{pct:03d}", model="jia_baseline", image_size=112,
            pipeline="jia_baseline", contrast="histeq", activation="square",
            epochs=BASE_EPOCHS, lr=1e-3, patience=15, seed=seed, train_fraction=frac,
            notes=f"Jia baseline learning curve at {pct}% of training data."))
    return cfgs


def component_group(seeds: tuple[int, ...] = ABL_SEEDS) -> list[ExperimentConfig]:
    """Does the proposed architecture beat its own backbone, at any data size?

    The 5% ablation says no: plain EfficientNet-B0 outscores every variant that
    adds the local branch and the chunk attention. That could be an artefact of
    the very-low-data regime, where the from-scratch local branch has no chance
    to learn anything useful. This study settles it by running the two endpoints
    - backbone alone vs the full model - head to head at larger fractions, with
    matched seeds so the comparison is paired.

    Deliberately only two configurations at one fraction: this is a targeted
    question, not another sweep.
    """
    # Only 10%. The measured learning curve puts SAGE-HE at 99.17% macro F1 on
    # 10% of the data and 98.34% on 25%, so by 25% both arms are at ceiling and
    # the contrast cannot discriminate. 10% is the first point above the
    # degenerate 5% regime, which is exactly what this study needs to test.
    cfgs: list[ExperimentConfig] = []
    for frac in (0.10,):
        pct = int(frac * 100)
        for seed in seeds:
            base = dict(image_size=160, pipeline="sage", contrast="clahe",
                        epochs=SAGE_EPOCHS, lr=3e-4, patience=8, seed=seed,
                        train_fraction=frac)
            cfgs.append(ExperimentConfig(
                name=f"component/global_only_{pct:03d}_seed{seed}", use_local=False,
                notes=f"Backbone alone at {pct}% of training data.", **base))
            cfgs.append(ExperimentConfig(
                name=f"component/full_sage_{pct:03d}_seed{seed}",
                fusion="gated", use_attention=True, use_sobel=True,
                notes=f"Full SAGE-HE at {pct}% of training data.", **base))
    return cfgs


def baseline_sensitivity_group(seed: int = 42) -> list[ExperimentConfig]:
    """Is the failed reproduction our assumption's fault?

    We report that the published 97.1% does not reproduce. That claim leans on
    choices the paper forced us to make, the largest being the convolution
    width, which it never states. If a different width recovers 97.1%, the
    honest conclusion is "the paper under-specifies its architecture", not "the
    result does not reproduce" — so the assumption is tested rather than
    defended. Also varies the activation, since the paper uses ReLU in
    plaintext and square under encryption.
    """
    cfgs: list[ExperimentConfig] = []
    base = dict(model="jia_baseline", image_size=112, pipeline="jia_baseline",
                contrast="histeq", epochs=BASE_EPOCHS, lr=1e-3, patience=15,
                seed=seed)
    for ch in (16, 64, 128):
        cfgs.append(ExperimentConfig(
            name=f"sensitivity/jia_ch{ch}", conv_channels=ch, activation="square",
            notes=f"Baseline reproduction with {ch} convolution channels.", **base))
    cfgs.append(ExperimentConfig(
        name="sensitivity/jia_relu", conv_channels=32, activation="relu",
        notes="Baseline reproduction with ReLU (the paper's plaintext activation).",
        **base))
    return cfgs


GROUPS = {
    "main": main_group,
    "ablation": ablation_group,
    "design": design_group,
    "capacity": capacity_group,
    "component": component_group,
    "sensitivity": baseline_sensitivity_group,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=sorted(GROUPS))
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--only", type=str, default=None, help="substring filter on name")
    args = ap.parse_args()

    fn = GROUPS[args.group]
    cfgs = fn(args.fraction) if (args.fraction and args.group in {"ablation", "design"}) else fn()
    if args.only:
        cfgs = [c for c in cfgs if args.only in c.name]

    print(f"=== group '{args.group}': {len(cfgs)} experiments ===", flush=True)
    results, failures = [], []
    for i, cfg in enumerate(cfgs, 1):
        done = ROOT / "experiments" / cfg.name / "result.json"
        if done.exists():
            print(f"[{i}/{len(cfgs)}] {cfg.name}: already complete, skipping", flush=True)
            results.append(json.loads(done.read_text()))
            continue
        print(f"\n[{i}/{len(cfgs)}] {cfg.name}", flush=True)
        try:
            results.append(run_experiment(cfg))
        except Exception:
            print(f"!! {cfg.name} FAILED:\n{traceback.format_exc()}", flush=True)
            failures.append(cfg.name)

    print(f"\n=== group '{args.group}' done: {len(results)} ok, {len(failures)} failed ===")
    for f in failures:
        print("   FAILED:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
