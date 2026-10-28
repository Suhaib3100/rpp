#!/usr/bin/env python
"""Check that the tensor shapes written on the architecture diagrams are real.

A diagram that disagrees with the code is worse than no diagram, so the shapes
annotated in src/viz/architecture_diagrams.py are asserted here against the
actual implementation by running a tensor through it and reading the real
sizes. Run as part of the test suite and before regenerating the diagrams.

Usage:  python scripts/verify_diagram_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encryption.ckks import CKKSParams  # noqa: E402
from src.models.sage_he import SageHE, SageHEConfig  # noqa: E402

IMG = 160
BATCH = 2

# (label on the diagram, expected shape) - exactly as annotated.
EXPECTED = {
    "input": (BATCH, 1, IMG, IMG),
    "global pooled": (BATCH, 1280),
    "local feature map": (BATCH, 128, 10, 10),
    "chunk descriptor": (BATCH, 128),
    "projected branches": (BATCH, 256),
    "fused": (BATCH, 256),
    "compressed v": (BATCH, 64),
    "logits": (BATCH, 6),
    "chunk attention": (BATCH, 16),
}


def main() -> int:
    cfg = SageHEConfig(pretrained=False, compressed_dim=64, head_hidden=32, chunk_grid=4)
    model = SageHE(cfg).eval()
    x = torch.randn(*EXPECTED["input"])

    actual: dict[str, tuple] = {"input": tuple(x.shape)}
    with torch.no_grad():
        g_pool = model.backbone(x)
        actual["global pooled"] = tuple(g_pool.shape)
        actual["projected branches"] = tuple(model.global_proj(g_pool).shape)

        fmap = model.local(x)
        actual["local feature map"] = tuple(fmap.shape)

        v = model.extract_features(x)
        actual["compressed v"] = tuple(v.shape)
        actual["chunk attention"] = tuple(model.last_attention.shape)
        actual["logits"] = tuple(model(x).shape)

        # descriptor + fused are internal; recompute them the same way forward does
        from src.preprocessing.sobel import batch_chunk_energy
        desc, _ = model.chunk_attn(fmap, batch_chunk_energy(x, cfg.chunk_grid))
        actual["chunk descriptor"] = tuple(desc.shape)
        actual["fused"] = tuple(model.fusion(
            model.global_proj(g_pool), model.local_proj(desc)).shape)

    ok = True
    print(f"{'annotation':<22}{'diagram says':<22}{'code produces':<22}")
    for k, exp in EXPECTED.items():
        got = actual[k]
        match = got == exp
        ok &= match
        print(f"{k:<22}{str(exp):<22}{str(got):<22}{'OK' if match else 'MISMATCH'}")

    # Encryption annotations on diagrams 01 and 09.
    p = CKKSParams()
    checks = [
        ("poly modulus degree 16384", p.poly_modulus_degree == 16384),
        ("coeff chain [60,40,40,40,60]", tuple(p.coeff_mod_bit_sizes) == (60, 40, 40, 40, 60)),
        ("scale 2^40", p.global_scale_bits == 40),
        ("depth available == 3", p.multiplicative_depth == 3),
        ("240 total modulus bits", p.total_modulus_bits == 240),
    ]
    print()
    for label, good in checks:
        ok &= good
        print(f"{label:<34}{'OK' if good else 'MISMATCH'}")

    # Head dimensions quoted on the diagram: 64 -> 32 -> 6
    w = model.head.export_weights()
    head_ok = (len(w["W1"]), len(w["b1"]), len(w["b2"])) == (64, 32, 6)
    ok &= head_ok
    print(f"{'encrypted head 64 -> 32 -> 6':<34}{'OK' if head_ok else 'MISMATCH'}")

    print("\n" + ("All diagram annotations match the implementation."
                  if ok else "DIAGRAM ANNOTATIONS DISAGREE WITH THE CODE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
