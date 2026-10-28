#!/usr/bin/env python
"""Run encrypted inference over the held-out test set.

Loads a trained SAGE-HE checkpoint, runs the client-side feature extractor in
plaintext, then encrypts each compressed vector and evaluates the polynomial
head homomorphically. Reports agreement with the plaintext model, the numerical
error CKKS introduces, and the full latency/size breakdown.

Usage:
    python scripts/run_encrypted_eval.py                       # default model
    python scripts/run_encrypted_eval.py --limit 60            # quick pass
    python scripts/run_encrypted_eval.py --sweep               # dimension sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.neu_cls import NEUCLSDataset  # noqa: E402
from src.datasets.splits import load_splits  # noqa: E402
from src.encryption.benchmark import run_encrypted_benchmark  # noqa: E402
from src.encryption.ckks import CKKSParams  # noqa: E402
from src.models.sage_he import SageHE, SageHEConfig  # noqa: E402
from src.training.experiment import ExperimentConfig  # noqa: E402
from src.utils.device import get_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_model(exp_name: str) -> tuple[SageHE, ExperimentConfig]:
    exp_dir = ROOT / "experiments" / exp_name
    cfg = ExperimentConfig(**json.loads((exp_dir / "result.json").read_text())["config"])
    model = SageHE(SageHEConfig(
        pretrained=False,
        compressed_dim=cfg.compressed_dim,
        head_hidden=cfg.head_hidden,
        chunk_grid=cfg.chunk_grid,
        top_k=cfg.top_k,
        fusion=cfg.fusion,
        use_sobel=cfg.use_sobel,
        use_attention=cfg.use_attention,
        use_local=cfg.use_local,
        use_global=cfg.use_global,
    ))
    ck = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_test_features(model: SageHE, cfg: ExperimentConfig,
                          limit: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Client-side plaintext pass: image -> compressed vector v."""
    splits = load_splits(ROOT / "data/splits.json")
    recs = splits["test"][:limit] if limit else splits["test"]
    ds = NEUCLSDataset(recs, ROOT / "data/raw/images", size=cfg.image_size,
                       train=False, pipeline=cfg.pipeline, contrast=cfg.contrast)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    feats, labels = [], []
    t0 = time.perf_counter()
    for x, y in loader:
        feats.append(model.extract_features(x).cpu().numpy())
        labels.append(y.numpy())
    elapsed = time.perf_counter() - t0
    f = np.concatenate(feats)
    return f, np.concatenate(labels), elapsed / len(f) * 1000  # ms/sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main/sage_he_seed42")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test images")
    ap.add_argument("--sweep", action="store_true",
                    help="also benchmark the compressed-dimension variants")
    ap.add_argument("--sweep-limit", type=int, default=40)
    args = ap.parse_args()

    dev = get_device()
    out_dir = ROOT / "results" / "encryption_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== encrypted inference: {args.exp} on {dev} ===")
    model, cfg = load_model(args.exp)
    feats, labels, client_ms = extract_test_features(model, cfg, args.limit)
    print(f"client-side feature extraction: {client_ms:.1f} ms/sample, "
          f"features {feats.shape}")

    bench = run_encrypted_benchmark(
        feats, model.head.export_weights(), CKKSParams(), labels=labels
    )
    d = bench.to_dict()
    d["experiment"] = args.exp
    d["client_feature_extraction_ms"] = client_ms

    print(f"\n  agreement with plaintext : {bench.agreement_rate*100:.2f}% "
          f"({bench.n_disagreements} disagreements of {bench.n_samples})")
    print(f"  max |logit error|        : {bench.max_abs_logit_error:.3e}")
    print(f"  mean |logit error|       : {bench.mean_abs_logit_error:.3e}")
    print(f"  encrypt                  : {bench.encrypt_ms['mean']:.1f} ms")
    print(f"  homomorphic evaluation   : {bench.he_inference_ms['mean']:.0f} ms")
    print(f"  decrypt                  : {bench.decrypt_ms['mean']:.1f} ms")
    print(f"  ciphertext               : {bench.ciphertext_bytes/1024:.0f} KB "
          f"({bench.expansion_factor:.0f}x the plaintext vector)")
    print(f"  accuracy plain/encrypted : {bench.accuracy_plaintext:.4f} / "
          f"{bench.accuracy_encrypted:.4f}")
    print(f"  {bench.security_note}")

    path = out_dir / "encrypted_inference.json"
    path.write_text(json.dumps(d, indent=2))
    print(f"\nwritten: {path}")

    # ------------------------------------------------ compressed-dim sweep
    if args.sweep:
        sweep = []
        # The design sweep is seeded; latency and ciphertext size are
        # seed-independent, so one representative checkpoint is timed, while
        # accuracy is averaged over every seed that trained at that dimension.
        candidates = [("design/compress_32_seed42", 32),
                      (f"design/compress_{cfg.compressed_dim}_seed42", cfg.compressed_dim),
                      ("design/compress_128_seed42", 128),
                      ("design/compress_256_seed42", 256)]
        if not (ROOT / "experiments" / candidates[1][0] / "result.json").exists():
            # 64-d is the main model's dimension and has no design run of its own
            candidates[1] = (args.exp, cfg.compressed_dim)

        def _mean_metric(dim: int, metric: str, fallback: str) -> float:
            """Accuracy for a compressed width, measured in ONE regime.

            The design sweep trains 32/128/256 in the reduced-data regime. The
            default width (64) has no design run of its own, but the ablation's
            stage 7 *is* that architecture at that width in that same regime, so
            it is the correct same-regime source. Falling through to the
            full-data model instead would put a 100% number beside 5%-data
            numbers and make the compression curve meaningless.
            """
            paths = sorted((ROOT / "experiments" / "design").glob(
                f"compress_{dim}_seed*/result.json"))
            if not paths:
                paths = sorted((ROOT / "experiments" / "abl").glob(
                    "7_chunk_attention_seed*/result.json"))
            if not paths:
                paths = [ROOT / "experiments" / fallback / "result.json"]
            vals = [json.loads(p.read_text())["test_metrics"][metric]
                    for p in paths if p.exists()]
            return float(np.mean(vals)) if vals else float("nan")

        for name, dim in candidates:
            if not (ROOT / "experiments" / name / "result.json").exists():
                print(f"  [skip] {name} not trained")
                continue
            m, c = load_model(name)
            f, lab, _ = extract_test_features(m, c, args.sweep_limit)
            b = run_encrypted_benchmark(f, m.head.export_weights(),
                                        CKKSParams(), labels=lab, verbose=False)
            sweep.append({
                "name": name,
                "dim": dim,
                "macro_f1": _mean_metric(dim, "macro_f1", name),
                "accuracy": _mean_metric(dim, "accuracy", name),
                "n_seeds_averaged": len(sorted((ROOT / "experiments" / "design").glob(
                    f"compress_{dim}_seed*/result.json"))) or 1,
                "he_ms": b.he_inference_ms["mean"],
                "encrypt_ms": b.encrypt_ms["mean"],
                "ciphertext_bytes": b.ciphertext_bytes,
                "agreement_rate": b.agreement_rate,
            })
            print(f"  d={dim:<4} F1={sweep[-1]['macro_f1']:.4f} "
                  f"(mean of {sweep[-1]['n_seeds_averaged']} seed(s))  "
                  f"HE={b.he_inference_ms['mean']:.0f}ms  "
                  f"ct={b.ciphertext_bytes/1024:.0f}KB")
        p = out_dir / "compression_sweep.json"
        p.write_text(json.dumps(sweep, indent=2))
        print(f"written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
