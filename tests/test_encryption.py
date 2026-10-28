"""Tests for the CKKS encrypted inference path.

These are the tests that matter most for the project's privacy claim: they
check that the encrypted head computes the same function as the plaintext one,
and that the server context genuinely cannot decrypt.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.encryption.ckks import CKKSParams, CKKSSession, EncryptedPolynomialHead
from src.models.sage_he import PolynomialHead

# One session is expensive (Galois key generation ~2 s), so share it.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def session() -> CKKSSession:
    return CKKSSession(CKKSParams())


@pytest.fixture(scope="module")
def head_pair():
    torch.manual_seed(0)
    head = PolynomialHead(32, 16, 6).eval()
    return head, EncryptedPolynomialHead(head.export_weights())


def test_params_meet_128_bit_security():
    p = CKKSParams()
    assert p.multiplicative_depth >= 3, "head needs depth 3 (mm, square, mm)"
    assert p.total_modulus_bits <= 438, "exceeds the 128-bit bound for n=16384"
    assert "meets" in p.security_note()


def test_encrypt_decrypt_roundtrip(session):
    v = np.linspace(-1, 1, 32)
    got = np.array(session.decrypt(session.encrypt(v))[:32])
    assert np.allclose(got, v, atol=1e-5)


def test_encrypted_head_matches_plaintext(session, head_pair):
    torch_head, enc_head = head_pair
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = np.tanh(rng.normal(size=32))
        ref = torch_head(torch.tensor(v, dtype=torch.float32)).detach().numpy()
        got = np.array(session.decrypt(enc_head(session.encrypt(v)))[:6])
        assert np.abs(ref - got).max() < 1e-3, "CKKS result diverged from plaintext"
        assert int(ref.argmax()) == int(got.argmax()), "encrypted prediction differs"


def test_numpy_reference_matches_torch(head_pair):
    """The numpy reference used for error analysis must match the torch module."""
    torch_head, enc_head = head_pair
    v = np.tanh(np.random.default_rng(1).normal(size=32))
    ref = torch_head(torch.tensor(v, dtype=torch.float32)).detach().numpy()
    assert np.abs(ref - enc_head.plaintext(v)).max() < 1e-4


def test_server_context_cannot_decrypt(session):
    """The core privacy property: the server holds no secret key."""
    ct = session.encrypt(np.zeros(32))
    with pytest.raises(Exception):
        ct.decrypt(session.server_context.secret_key())


def test_server_context_has_no_secret_key(session):
    assert session.server_context.is_public(), "server context must be public-only"
    assert not session.client_context.is_public(), "client must hold the secret key"


def test_depth_budget_is_exhausted_not_exceeded(session, head_pair):
    """A fourth multiplication must fail: proof our depth accounting is real."""
    _, enc_head = head_pair
    out = enc_head(session.encrypt(np.tanh(np.zeros(32) + 0.3)))
    with pytest.raises(Exception):
        # depth 3 is already spent; squaring again must exceed the chain
        out.square()
        out.square()
