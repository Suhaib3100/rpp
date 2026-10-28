"""Feature fusion strategies.

All five strategies named in the project brief are implemented behind one
interface so the ablation can swap them with a config string and nothing else
changes. ``GatedFusion`` is the proposed component; the others exist to test
whether it is actually worth its parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FusionBase(nn.Module):
    out_dim: int

    def forward(self, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:  # noqa: E741
        raise NotImplementedError


class GlobalOnly(FusionBase):
    """Discard the local branch (ablation)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_dim = dim

    def forward(self, g, l):  # noqa: E741
        return g


class LocalOnly(FusionBase):
    """Discard the global branch (ablation)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_dim = dim

    def forward(self, g, l):  # noqa: E741
        return l


class ConcatFusion(FusionBase):
    """Plain concatenation followed by a projection - the obvious baseline."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.ReLU(inplace=True)
        )
        self.out_dim = dim

    def forward(self, g, l):  # noqa: E741
        return self.proj(torch.cat([g, l], dim=1))


class AttentionFusion(FusionBase):
    """Scalar attention: one softmax weight per branch, shared across channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(2 * dim, 2)
        self.out_dim = dim

    def forward(self, g, l):  # noqa: E741
        w = torch.softmax(self.score(torch.cat([g, l], dim=1)), dim=1)
        return w[:, 0:1] * g + w[:, 1:2] * l


class GatedFusion(FusionBase):
    """Channel-wise gated fusion (proposed).

        z     = sigmoid(W [g ; l] + b)        in (0,1)^dim
        fused = z * g + (1 - z) * l

    The distinction from ``AttentionFusion`` is that the gate is a *vector*,
    not a scalar: the model can take some feature dimensions from the global
    branch and others from the local branch for the same image. That matters
    on NEU-CLS because the two branches are good at different classes - the
    global branch separates the large-structure classes (`patches`), the local
    branch the fine-texture ones (`crazing`, `scratches`) - and the per-class
    gate statistics are reported in the results as direct evidence.
    """

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or dim
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.out_dim = dim
        self._last_gate: torch.Tensor | None = None

    def forward(self, g, l):  # noqa: E741
        z = self.gate(torch.cat([g, l], dim=1))
        self._last_gate = z.detach()
        return z * g + (1.0 - z) * l

    @property
    def last_gate(self) -> torch.Tensor | None:
        """Most recent gate activations, for the gate-statistics analysis."""
        return self._last_gate


FUSIONS: dict[str, type[FusionBase]] = {
    "global_only": GlobalOnly,
    "local_only": LocalOnly,
    "concat": ConcatFusion,
    "attention": AttentionFusion,
    "gated": GatedFusion,
}


def build_fusion(name: str, dim: int) -> FusionBase:
    if name not in FUSIONS:
        raise ValueError(f"unknown fusion {name!r}; options: {sorted(FUSIONS)}")
    return FUSIONS[name](dim)
