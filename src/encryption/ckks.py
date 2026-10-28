"""CKKS homomorphic encryption for SAGE-HE's classification head.

Threat model
------------
Two parties, honest-but-curious server:

* **Client** owns the image and the secret key. It runs the plaintext feature
  extractor (EfficientNet-B0 + local branch + attention + fusion + compression)
  and encrypts only the resulting ``d``-dimensional vector ``v``.
* **Server** owns the classifier weights and holds only the *public* context
  (encryption, relinearisation and Galois keys). It evaluates the polynomial
  head on ciphertext and returns encrypted logits. It never holds the secret
  key, so it learns neither the image nor the prediction.

What this protects, and what it does not
----------------------------------------
Protected: the feature vector in transit and at the server, and therefore the
prediction. The server cannot decrypt either.

Not protected: this is *not* the base paper's setting. Jia et al. encrypt
image chunks and claim to run convolutions on them, so in their model the
client does almost no work. Here the client runs the entire feature extractor,
so the client must be trusted with the model weights of the extractor, and the
scheme protects the *server* side of the boundary only. The honest comparison
is in reports/technical_report.md; we do not claim equivalence.

Also out of scope: CKKS is not circuit-private. The returned ciphertext can
leak information about the server's weights to a client that chooses inputs
adaptively; defending that needs noise flooding, which we do not implement.

Parameter choice
----------------
The head computes ``W2 @ square(W1 @ v + b1) + b2`` -> multiplicative depth 3
(matmul, square, matmul). A CKKS coefficient-modulus chain of L primes
supports L-2 multiplications after the initial scale, so we need 5 primes:

    poly_modulus_degree = 16384
    coeff_mod_bit_sizes = [60, 40, 40, 40, 60]   (240 bits total)
    global_scale        = 2**40

Security: for degree 16384 the HomomorphicEncryption.org standard permits up to
438 coefficient-modulus bits at the 128-bit classical security level. We use
240, comfortably inside that bound, so the instantiation meets >=128-bit
security. (Using *fewer* modulus bits at a fixed degree increases the security
margin.) The base paper reports degree 16384 for its 28x28 chunks, so our
parameters are directly comparable on that axis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import tenseal as ts


@dataclass(frozen=True)
class CKKSParams:
    poly_modulus_degree: int = 16384
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 40, 60)
    global_scale_bits: int = 40

    @property
    def total_modulus_bits(self) -> int:
        return sum(self.coeff_mod_bit_sizes)

    @property
    def multiplicative_depth(self) -> int:
        return len(self.coeff_mod_bit_sizes) - 2

    @property
    def max_slots(self) -> int:
        return self.poly_modulus_degree // 2

    def security_note(self) -> str:
        limits = {8192: 218, 16384: 438, 32768: 881}
        cap = limits.get(self.poly_modulus_degree)
        if cap is None:
            return "security level not tabulated for this degree"
        ok = self.total_modulus_bits <= cap
        return (f"n={self.poly_modulus_degree}, {self.total_modulus_bits} modulus bits "
                f"(<= {cap} permitted at 128-bit security): "
                f"{'meets' if ok else 'DOES NOT MEET'} 128-bit classical security")


class CKKSSession:
    """Holds the client (secret) context and derives the server's public one."""

    def __init__(self, params: CKKSParams = CKKSParams()) -> None:
        self.params = params
        t0 = time.perf_counter()
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.poly_modulus_degree,
            coeff_mod_bit_sizes=list(params.coeff_mod_bit_sizes),
        )
        ctx.global_scale = 2 ** params.global_scale_bits
        ctx.generate_galois_keys()      # needed for the rotations inside mm()
        ctx.generate_relin_keys()       # needed after every ciphertext multiply
        self.keygen_seconds = time.perf_counter() - t0
        self._client_ctx = ctx

        # The server copy has the secret key stripped. Any attempt to decrypt
        # with it raises, which is asserted in tests/test_encryption.py.
        pub = ctx.copy()
        pub.make_context_public()
        self._server_ctx = pub

    @property
    def client_context(self) -> ts.Context:
        return self._client_ctx

    @property
    def server_context(self) -> ts.Context:
        return self._server_ctx

    def key_sizes(self) -> dict[str, int]:
        """Serialised sizes in bytes of the context the server receives."""
        return {
            "server_context_bytes": len(self._server_ctx.serialize(
                save_public_key=True, save_secret_key=False,
                save_galois_keys=True, save_relin_keys=True)),
            "client_context_bytes": len(self._client_ctx.serialize(
                save_public_key=True, save_secret_key=True,
                save_galois_keys=True, save_relin_keys=True)),
        }

    # -------------------------------------------------------------- client
    def encrypt(self, v: Sequence[float]) -> ts.CKKSVector:
        return ts.ckks_vector(self._client_ctx, list(v))

    def decrypt(self, ct: ts.CKKSVector) -> list[float]:
        return ct.decrypt(self._client_ctx.secret_key())


class EncryptedPolynomialHead:
    """Server-side homomorphic evaluation of the SAGE-HE classification head.

    The weights are plaintext (the server owns the model); only the activations
    are encrypted. ``W1``/``W2`` must already be transposed for row-vector
    multiplication - see ``PolynomialHead.export_weights``.
    """

    def __init__(self, weights: dict[str, list]) -> None:
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]
        self.in_dim = len(self.W1)
        self.hidden = len(self.b1)
        self.out_dim = len(self.b2)

    def __call__(self, ct: ts.CKKSVector) -> ts.CKKSVector:
        # TenSEAL contexts default to auto_relin / auto_rescale / auto_mod_switch,
        # so relinearisation after the squaring and the rescale that follows each
        # multiplication are applied for us. The depth budget is still ours to
        # respect: three multiplications against a five-prime chain.
        x = ct.mm(self.W1) + self.b1    # depth 1
        x = x.square()                  # depth 2
        return x.mm(self.W2) + self.b2  # depth 3

    def plaintext(self, v: np.ndarray) -> np.ndarray:
        """Identical arithmetic in the clear, for approximation-error analysis."""
        h = v @ np.asarray(self.W1) + np.asarray(self.b1)
        return (h * h) @ np.asarray(self.W2) + np.asarray(self.b2)
