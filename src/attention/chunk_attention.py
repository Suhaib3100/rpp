"""Sobel-informed adaptive chunk selection.

The base paper selects chunks with a *fixed hard threshold* on Sobel gradient
energy: a chunk is kept iff |grad| > tau. Two consequences follow. The rule is
non-differentiable, so it cannot adapt to the task; and it is purely
photometric, so a high-contrast but uninformative region (a lighting seam, a
mill-roll edge) scores exactly as highly as a genuine defect.

This module keeps the Sobel prior - it is a genuinely useful, free signal -
but makes the selection *learned and differentiable*:

    a_i = MLP(f_i) + lambda * e_hat_i          (chunk logit)
    w   = softmax(a / T)                       (soft selection)

where f_i is the pooled feature descriptor of chunk i, e_hat_i is its
standardised Sobel energy, and lambda is a single learnable scalar that lets
the network decide how much to trust the gradient prior. lambda is reported in
the results: it is the direct evidence for whether the Sobel prior earns its
place (see reports/technical_report.md).

Optional hard top-k keeps only the k highest-scoring chunks, matching the
base paper's "select a subset" behaviour while remaining differentiable
through the retained weights (straight-through on the mask).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGatedChunkAttention(nn.Module):
    """Attention over a grid of chunks, conditioned on a Sobel energy prior.

    Args:
        channels: channel width of the incoming feature map.
        grid: chunk grid is ``grid`` x ``grid``.
        hidden: width of the scoring MLP.
        temperature: softmax temperature.
        top_k: if set, keep only the k highest-scoring chunks.
        use_sobel: if False the prior is removed entirely (ablation).
    """

    def __init__(
        self,
        channels: int,
        grid: int = 4,
        hidden: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_sobel: bool = True,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.n_chunks = grid * grid
        self.temperature = temperature
        self.top_k = top_k
        self.use_sobel = use_sobel

        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Learnable trust in the gradient prior. Initialised at 1.0 so the
        # prior starts out meaningful but can be driven to ~0 if unhelpful.
        self.sobel_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        feat: torch.Tensor,
        sobel_energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W) local feature map.
            sobel_energy: (B, grid*grid) per-chunk Sobel energy, or None.
        Returns:
            descriptor: (B, C) attention-weighted chunk descriptor.
            weights: (B, grid*grid) attention weights, for visualisation.
        """
        b, c, _, _ = feat.shape
        # (B, C, g, g) -> (B, g*g, C): one descriptor per chunk.
        chunks = F.adaptive_avg_pool2d(feat, (self.grid, self.grid))
        chunks = chunks.flatten(2).transpose(1, 2)

        logits = self.score(chunks).squeeze(-1)             # (B, n_chunks)

        if self.use_sobel:
            if sobel_energy is None:
                raise ValueError("use_sobel=True requires sobel_energy")
            # Standardise per sample so the prior is scale-free across images.
            e = sobel_energy
            e_hat = (e - e.mean(dim=1, keepdim=True)) / (e.std(dim=1, keepdim=True) + 1e-6)
            logits = logits + self.sobel_weight * e_hat

        if self.top_k is not None and self.top_k < self.n_chunks:
            kth = logits.topk(self.top_k, dim=1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        weights = F.softmax(logits / self.temperature, dim=1)
        descriptor = torch.einsum("bn,bnc->bc", weights, chunks)
        return descriptor, weights
