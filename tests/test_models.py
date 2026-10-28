"""Architecture tests: shapes, ablation switches, gradient flow, determinism."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.attention.chunk_attention import SobelGatedChunkAttention
from src.fusion.gated_fusion import build_fusion, GatedFusion
from src.models.baseline_jia import JiaChunkCNN, Square
from src.models.multiscale import MultiScaleCNN
from src.models.sage_he import SageHE, SageHEConfig, PolynomialHead
from src.preprocessing.sobel import (
    chunk_gradient_energy, otsu_threshold, select_chunks_threshold, batch_chunk_energy,
)


def cfg(**kw) -> SageHEConfig:
    return SageHEConfig(pretrained=False, **kw)


def test_sage_forward_shape():
    m = SageHE(cfg())
    assert m(torch.randn(2, 1, 160, 160)).shape == (2, 6)


def test_compressed_vector_is_bounded():
    """The encrypted head's input must stay in [-1,1] for CKKS scale safety."""
    m = SageHE(cfg(compressed_dim=64)).eval()
    v = m.extract_features(torch.randn(8, 1, 160, 160) * 5)
    assert v.shape == (8, 64)
    assert v.abs().max() <= 1.0 + 1e-6, "tanh bound violated"


@pytest.mark.parametrize("fusion", ["global_only", "local_only", "concat", "attention", "gated"])
def test_fusion_variants(fusion):
    m = SageHE(cfg(fusion=fusion))
    assert m(torch.randn(2, 1, 160, 160)).shape == (2, 6)


@pytest.mark.parametrize("kw", [
    {"use_local": False}, {"use_global": False},
    {"use_attention": False}, {"use_sobel": False}, {"top_k": 4},
])
def test_ablation_switches(kw):
    m = SageHE(cfg(**kw))
    assert m(torch.randn(2, 1, 160, 160)).shape == (2, 6)


def test_both_branches_disabled_is_rejected():
    with pytest.raises(ValueError):
        SageHE(cfg(use_local=False, use_global=False))


def test_gradients_reach_every_branch():
    m = SageHE(cfg())
    m(torch.randn(2, 1, 160, 160)).sum().backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no gradient reached {name}"


def test_sobel_weight_is_learnable():
    m = SageHE(cfg(use_sobel=True))
    m(torch.randn(2, 1, 160, 160)).sum().backward()
    assert m.chunk_attn.sobel_weight.grad is not None


def test_attention_weights_sum_to_one():
    attn = SobelGatedChunkAttention(channels=128, grid=4)
    feat = torch.randn(3, 128, 20, 20)
    energy = torch.rand(3, 16)
    desc, w = attn(feat, energy)
    assert desc.shape == (3, 128)
    assert torch.allclose(w.sum(dim=1), torch.ones(3), atol=1e-5)


def test_topk_zeroes_all_but_k_chunks():
    attn = SobelGatedChunkAttention(channels=64, grid=4, top_k=4)
    _, w = attn(torch.randn(2, 64, 16, 16), torch.rand(2, 16))
    assert ((w > 1e-6).sum(dim=1) <= 4).all()


def test_gated_fusion_is_channelwise():
    """The gate must be a vector, not a scalar - that is the design claim."""
    f = GatedFusion(dim=32)
    f(torch.randn(4, 32), torch.randn(4, 32))
    assert f.last_gate.shape == (4, 32)
    assert f.last_gate.std(dim=1).max() > 0, "gate is constant across channels"


def test_square_activation():
    x = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.allclose(Square()(x), torch.tensor([4.0, 0.0, 9.0]))


def test_polynomial_head_export_roundtrip():
    """Exported weights must reproduce the module in numpy, transposed correctly."""
    head = PolynomialHead(16, 8, 6).eval()
    w = head.export_weights()
    v = np.tanh(np.random.default_rng(0).normal(size=16))
    ref = head(torch.tensor(v, dtype=torch.float32)).detach().numpy()
    got = (np.maximum(v @ np.array(w["W1"]) + np.array(w["b1"]), -np.inf) ** 2) \
        @ np.array(w["W2"]) + np.array(w["b2"])
    assert np.abs(ref - got).max() < 1e-4


def test_jia_baseline_shapes_and_fc_width():
    for act in ("relu", "square"):
        m = JiaChunkCNN(activation=act)
        assert m(torch.randn(2, 1, 112, 112)).shape == (2, 6)
    # the paper states a 256-wide fully connected input
    assert JiaChunkCNN().classifier[1].in_features == 256


def test_jia_baseline_rejects_bad_activation():
    with pytest.raises(ValueError):
        JiaChunkCNN(activation="gelu")


def test_multiscale_downsamples_by_16():
    m = MultiScaleCNN()
    out = m(torch.randn(2, 1, 160, 160))
    assert out.shape == (2, 128, 10, 10)


def test_chunk_energy_matches_grid():
    img = (np.random.default_rng(0).random((112, 112)) * 255).astype(np.uint8)
    e = chunk_gradient_energy(img, grid=4)
    assert e.shape == (16,)
    assert (e >= 0).all()


def test_chunk_energy_rejects_indivisible_grid():
    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        chunk_gradient_energy(img, grid=7)


def test_threshold_selection_always_keeps_one_chunk():
    e = np.zeros(16, dtype=np.float32)
    assert select_chunks_threshold(e, threshold=1.0).sum() == 1


def test_otsu_threshold_on_constant_input():
    assert otsu_threshold(np.full(16, 5.0, dtype=np.float32)) == pytest.approx(5.0)


def test_batch_chunk_energy_rejects_multichannel():
    with pytest.raises(ValueError):
        batch_chunk_energy(torch.randn(2, 3, 64, 64), grid=4)


def test_batch_energy_tracks_image_structure():
    """A high-frequency chunk must score above a flat one."""
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, :32, :32] = torch.randn(32, 32) * 3   # noisy top-left quadrant
    e = batch_chunk_energy(x, grid=2)
    assert e[0, 0] > e[0, 3]


def test_build_fusion_rejects_unknown():
    with pytest.raises(ValueError):
        build_fusion("telepathy", 32)


def test_forward_is_deterministic_in_eval():
    m = SageHE(cfg()).eval()
    x = torch.randn(2, 1, 160, 160)
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))


@pytest.mark.parametrize("channels,expected_pool", [
    (16, (4, 4)), (32, (2, 4)), (64, (1, 4)), (128, (1, 2)), (256, (1, 1)),
])
def test_jia_baseline_pools_to_stated_fc_width(channels, expected_pool):
    """Any channel width dividing 256 must reach the paper's 256-unit FC input.

    Regression: an earlier `(256 // channels) // 4` form floored to a
    zero-height pool for channels > 64, so the sensitivity study crashed on
    128 channels. The expected shapes below also pin the widths used by runs
    already reported in results/, so this test fails if the pooling is ever
    changed in a way that would invalidate them.
    """
    m = JiaChunkCNN(conv_channels=channels)
    assert m.pool_hw == expected_pool
    assert channels * m.pool_hw[0] * m.pool_hw[1] == 256
    assert m(torch.randn(2, 1, 112, 112)).shape == (2, 6)


def test_jia_baseline_rejects_indivisible_channel_count():
    with pytest.raises(ValueError, match="does not divide"):
        JiaChunkCNN(conv_channels=24)
