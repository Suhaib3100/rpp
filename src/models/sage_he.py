"""SAGE-HE: Secure Adaptive Gated Ensemble with Homomorphic Encryption.

The network is split at an explicit **encryption boundary**:

    ---------------------------- CLIENT (plaintext, holds secret key) --------
      image
        -> global branch   : EfficientNet-B0           -> g  (d_f)
        -> local branch    : multi-scale CNN           -> F  (C,H,W)
        -> chunk attention : Sobel-informed, learned   -> l  (d_f)
        -> gated fusion    : z*g + (1-z)*l             -> f  (d_f)
        -> compression     : Linear + tanh             -> v  (d_c), bounded
        -> CKKS encryption                             -> Enc(v)
    ---------------------------- SERVER (ciphertext only, no key) ------------
        Enc(v) -> Linear -> Square -> Linear           -> Enc(logits)  (6)
    ---------------------------- CLIENT --------------------------------------
      decrypt -> softmax -> label

Everything up to and including compression runs client-side in plaintext, so
it may use any operation at all (BatchNorm, ReLU, softmax attention). Only
``PolynomialHead`` is evaluated homomorphically, and it is therefore restricted
to operations CKKS supports natively: plaintext-matrix multiplication and
squaring. The head is *trained* with exactly those operations (not with ReLU
and then swapped), so the plaintext and encrypted models are the same function
up to CKKS approximation error.

The compression layer ends in ``tanh`` for a concrete numerical reason: the
square activation amplifies its input, and CKKS has finite scale budget.
Bounding v to [-1,1] keeps the intermediate magnitudes inside the range where
the chosen coefficient moduli retain precision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import timm
import torch
import torch.nn as nn

from src.attention.chunk_attention import SobelGatedChunkAttention
from src.fusion.gated_fusion import build_fusion
from src.models.baseline_jia import Square
from src.models.multiscale import MultiScaleCNN
from src.preprocessing.sobel import batch_chunk_energy


@dataclass
class SageHEConfig:
    num_classes: int = 6
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    feature_dim: int = 256          # d_f, common width of both branches
    compressed_dim: int = 64        # d_c, the width that gets encrypted
    head_hidden: int = 32           # width inside the polynomial head
    chunk_grid: int = 4             # 4x4 = 16 chunks, matching the base paper
    top_k: int | None = None
    fusion: str = "gated"
    use_sobel: bool = True
    use_attention: bool = True
    use_local: bool = True
    use_global: bool = True
    local_widths: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.2


class PolynomialHead(nn.Module):
    """The only part of the network evaluated under encryption.

        logits = W2 @ square(W1 @ v + b1) + b2

    Multiplicative depth is 3 (matmul, square, matmul), which fixes the CKKS
    parameter choice in src/encryption/ckks.py.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = Square()
        self.fc2 = nn.Linear(hidden, num_classes)
        # Small init: square() doubles the exponent of its input, so a standard
        # init makes early logits explode and the loss diverge.
        nn.init.normal_(self.fc1.weight, std=0.10)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.10)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(v)))

    @torch.no_grad()
    def export_weights(self) -> dict[str, list]:
        """Plain weights for homomorphic evaluation (server side).

        Matrices are exported transposed because TenSEAL's ``CKKSVector.mm``
        computes ``v @ M`` for a row vector v, whereas ``nn.Linear`` stores
        ``W`` for ``v @ W.T``.
        """
        return {
            "W1": self.fc1.weight.detach().cpu().t().contiguous().tolist(),
            "b1": self.fc1.bias.detach().cpu().tolist(),
            "W2": self.fc2.weight.detach().cpu().t().contiguous().tolist(),
            "b2": self.fc2.bias.detach().cpu().tolist(),
        }


class SageHE(nn.Module):
    """The full proposed model."""

    def __init__(self, cfg: SageHEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SageHEConfig()
        c = self.cfg
        if not (c.use_global or c.use_local):
            raise ValueError("at least one of use_global / use_local must be True")

        # ---- global branch ------------------------------------------------
        if c.use_global:
            # in_chans=1 makes timm adapt the pretrained RGB stem by summing
            # its three input filters - the standard documented conversion for
            # grayscale inputs, and better than replicating the channel because
            # it preserves the filter response magnitude.
            self.backbone = timm.create_model(
                c.backbone, pretrained=c.pretrained, num_classes=0, in_chans=1
            )
            gdim = self.backbone.num_features
            self.global_proj = nn.Sequential(
                nn.Linear(gdim, c.feature_dim), nn.LayerNorm(c.feature_dim), nn.ReLU(inplace=True)
            )
        else:
            self.backbone = None
            self.global_proj = None

        # ---- local branch -------------------------------------------------
        if c.use_local:
            self.local = MultiScaleCNN(in_channels=1, widths=tuple(c.local_widths))
            ldim = self.local.out_channels
            if c.use_attention:
                self.chunk_attn = SobelGatedChunkAttention(
                    channels=ldim, grid=c.chunk_grid, top_k=c.top_k, use_sobel=c.use_sobel
                )
            else:
                self.chunk_attn = None
            self.local_proj = nn.Sequential(
                nn.Linear(ldim, c.feature_dim), nn.LayerNorm(c.feature_dim), nn.ReLU(inplace=True)
            )
        else:
            self.local = None
            self.chunk_attn = None
            self.local_proj = None

        # ---- fusion -------------------------------------------------------
        if c.use_global and c.use_local:
            fusion_name = c.fusion
        elif c.use_global:
            fusion_name = "global_only"
        else:
            fusion_name = "local_only"
        self.fusion = build_fusion(fusion_name, c.feature_dim)
        self.fusion_name = fusion_name

        # ---- compression (client-side, bounded) ---------------------------
        self.compress = nn.Sequential(
            nn.Dropout(c.dropout),
            nn.Linear(self.fusion.out_dim, c.compressed_dim),
            nn.Tanh(),
        )

        # ---- encrypted head ------------------------------------------------
        self.head = PolynomialHead(c.compressed_dim, c.head_hidden, c.num_classes)

        self._last_attn: torch.Tensor | None = None

    # ------------------------------------------------------------------ API
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Everything the client computes in plaintext: image -> v (d_c).

        This is the vector that gets encrypted. Separated from ``forward`` so
        that the encrypted-inference benchmark can reuse it verbatim.
        """
        g = l = None
        if self.backbone is not None:
            g = self.global_proj(self.backbone(x))
        if self.local is not None:
            fmap = self.local(x)
            if self.chunk_attn is not None:
                energy = batch_chunk_energy(x, self.cfg.chunk_grid)
                desc, attn = self.chunk_attn(fmap, energy)
                self._last_attn = attn.detach()
            else:
                desc = fmap.mean(dim=(2, 3))
                self._last_attn = None
            l = self.local_proj(desc)

        if g is None:
            g = l
        if l is None:
            l = g
        fused = self.fusion(g, l)
        return self.compress(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    @property
    def last_attention(self) -> torch.Tensor | None:
        return self._last_attn

    def gradcam_target_layer(self) -> nn.Module:
        """Deepest spatial feature map, for Grad-CAM."""
        if self.backbone is not None:
            return self.backbone.conv_head
        return self.local.stages[-1][0]

    def count_parameters(self) -> dict[str, int]:
        def n(m: nn.Module | None) -> int:
            return 0 if m is None else sum(p.numel() for p in m.parameters())
        return {
            "global_branch": n(self.backbone) + n(self.global_proj),
            "local_branch": n(self.local) + n(self.local_proj),
            "chunk_attention": n(self.chunk_attn),
            "fusion": n(self.fusion),
            "compression": n(self.compress),
            "encrypted_head": n(self.head),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
