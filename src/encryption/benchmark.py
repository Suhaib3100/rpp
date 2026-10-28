"""End-to-end encrypted inference benchmark.

Measures the quantities the project brief asks for: encryption time, encrypted
inference time, decryption time, ciphertext size, memory, prediction
consistency against the plaintext model, and the numerical error CKKS
introduces. Timings are repeated per sample and summarised with mean/std/median
plus p95, because CKKS latency has a long right tail on a shared CPU.
"""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass, asdict, field

import numpy as np

from src.encryption.ckks import CKKSParams, CKKSSession, EncryptedPolynomialHead


def _peak_rss_mb() -> float:
    # ru_maxrss is kB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class EncryptedBenchmark:
    params: dict
    security_note: str
    n_samples: int
    compressed_dim: int
    keygen_seconds: float
    key_sizes_bytes: dict
    ciphertext_bytes: int
    plaintext_vector_bytes: int
    expansion_factor: float
    encrypt_ms: dict
    he_inference_ms: dict
    decrypt_ms: dict
    total_ms: dict
    plaintext_inference_ms: dict
    agreement_rate: float
    n_disagreements: int
    max_abs_logit_error: float
    mean_abs_logit_error: float
    max_rel_logit_error: float
    # How much headroom the argmax actually has. The decision margin is the
    # plaintext gap between the top-1 and top-2 logits; a prediction can only
    # flip if CKKS error exceeds half that gap. Reporting the ratio turns
    # "encryption did not change the prediction" from an observation on this
    # test set into a quantified safety margin.
    min_decision_margin: float
    mean_decision_margin: float
    error_to_margin_ratio: float
    peak_rss_mb: float
    accuracy_plaintext: float | None = None
    accuracy_encrypted: float | None = None
    disagreement_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def run_encrypted_benchmark(
    features: np.ndarray,
    head_weights: dict[str, list],
    params: CKKSParams = CKKSParams(),
    labels: np.ndarray | None = None,
    verbose: bool = True,
) -> EncryptedBenchmark:
    """
    Args:
        features: (N, d) client-side compressed feature vectors, plaintext.
        head_weights: output of ``PolynomialHead.export_weights``.
        params: CKKS parameters.
        labels: optional ground truth, to report plaintext vs encrypted accuracy.
    """
    session = CKKSSession(params)
    head = EncryptedPolynomialHead(head_weights)

    t_enc, t_inf, t_dec, t_plain = [], [], [], []
    enc_logits, plain_logits = [], []
    ct_bytes = 0

    for i, v in enumerate(features.astype(np.float64)):
        t0 = time.perf_counter()
        ct = session.encrypt(v)
        t_enc.append((time.perf_counter() - t0) * 1000)

        if i == 0:
            ct_bytes = len(ct.serialize())

        t0 = time.perf_counter()
        out = head(ct)
        t_inf.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        dec = session.decrypt(out)
        t_dec.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ref = head.plaintext(v)
        t_plain.append((time.perf_counter() - t0) * 1000)

        enc_logits.append(np.asarray(dec[: head.out_dim]))
        plain_logits.append(ref)

        if verbose and (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(features)}] "
                  f"enc {np.mean(t_enc):.1f}ms  he {np.mean(t_inf):.0f}ms", flush=True)

    enc_logits = np.stack(enc_logits)
    plain_logits = np.stack(plain_logits)

    enc_pred = enc_logits.argmax(1)
    plain_pred = plain_logits.argmax(1)
    disagree = np.flatnonzero(enc_pred != plain_pred)

    abs_err = np.abs(enc_logits - plain_logits)
    denom = np.maximum(np.abs(plain_logits), 1e-9)

    # Decision margin: top-1 minus top-2 plaintext logit, per sample.
    srt = np.sort(plain_logits, axis=1)
    margins = srt[:, -1] - srt[:, -2]
    min_margin = float(margins.min())
    # A flip needs error > margin/2, so this ratio is the fraction of the
    # tightest decision's budget that CKKS error actually consumes.
    ratio = float(abs_err.max() / (min_margin / 2)) if min_margin > 0 else float("inf")

    total = [a + b + c for a, b, c in zip(t_enc, t_inf, t_dec)]
    d = features.shape[1]

    bench = EncryptedBenchmark(
        params=asdict(params),
        security_note=params.security_note(),
        n_samples=int(features.shape[0]),
        compressed_dim=int(d),
        keygen_seconds=session.keygen_seconds,
        key_sizes_bytes=session.key_sizes(),
        ciphertext_bytes=ct_bytes,
        plaintext_vector_bytes=d * 4,            # float32
        expansion_factor=ct_bytes / (d * 4),
        encrypt_ms=_stats(t_enc),
        he_inference_ms=_stats(t_inf),
        decrypt_ms=_stats(t_dec),
        total_ms=_stats(total),
        plaintext_inference_ms=_stats(t_plain),
        agreement_rate=float((enc_pred == plain_pred).mean()),
        n_disagreements=int(disagree.size),
        disagreement_indices=[int(i) for i in disagree],
        max_abs_logit_error=float(abs_err.max()),
        mean_abs_logit_error=float(abs_err.mean()),
        max_rel_logit_error=float((abs_err / denom).max()),
        min_decision_margin=min_margin,
        mean_decision_margin=float(margins.mean()),
        error_to_margin_ratio=ratio,
        peak_rss_mb=_peak_rss_mb(),
    )
    if labels is not None:
        bench.accuracy_plaintext = float((plain_pred == labels).mean())
        bench.accuracy_encrypted = float((enc_pred == labels).mean())
    return bench
