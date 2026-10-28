"""Grad-CAM for SAGE-HE.

Standard Grad-CAM (Selvaraju et al.): weight each channel of a convolutional
feature map by the mean gradient of the target logit with respect to that
channel, sum, and keep the positive part.

A caveat specific to this model is worth stating, because it affects how the
heatmaps should be read: SAGE-HE's logits come from a *squared* activation, so
the gradient of a logit with respect to the features is not sign-stable the way
it is after a ReLU stack. Two consequences follow. First, gradients are taken
with respect to the target class logit *before* softmax, as usual, but the
square means a feature can increase the logit by being strongly negative.
Second, Grad-CAM here explains the client-side feature extractor, which is the
part with spatial structure - it does not and cannot explain the encrypted
head, which sees only a 64-dimensional vector with no spatial layout left.

The chunk-attention weights are a separate, complementary explanation: they are
the model's *own* spatial selection rather than a post-hoc gradient attribution,
and ``attention_map`` returns them for side-by-side comparison.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM over an arbitrary target layer.

    Hooks are registered on construction and released by ``remove()`` or by
    using the object as a context manager.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _m, _inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, _m, _gin, gout) -> None:
        self.gradients = gout[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: int | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (B, 1, H, W) input batch.
            class_idx: target class per sample; defaults to the prediction.
        Returns:
            cams: (B, H, W) float array in [0, 1], upsampled to input size.
            logits: the model output, so callers need not re-run the forward.
        """
        was_training = self.model.training
        self.model.eval()
        # Grad-CAM needs gradients w.r.t. activations, so no_grad is not an option.
        x = x.clone().requires_grad_(False)
        logits = self.model(x)

        if class_idx is None:
            target = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            target = torch.full((x.size(0),), class_idx, dtype=torch.long, device=x.device)
        else:
            target = class_idx.to(x.device)

        score = logits.gather(1, target.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer produced no activations/gradients")

        # Channel weights = global-average-pooled gradients.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Normalise each map to [0,1] independently.
        b = cam.size(0)
        flat = cam.view(b, -1)
        lo = flat.min(dim=1).values.view(b, 1, 1)
        hi = flat.max(dim=1).values.view(b, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)

        if was_training:
            self.model.train()
        return cam.cpu().numpy(), logits.detach()


@torch.no_grad()
def attention_map(model: nn.Module, x: torch.Tensor, grid: int) -> np.ndarray | None:
    """The model's own chunk-attention weights, upsampled to image size.

    Returns None if the model has no chunk attention (e.g. ablations that
    disable it).
    """
    model.eval()
    model(x)
    attn = getattr(model, "last_attention", None)
    if attn is None:
        return None
    b = attn.size(0)
    a = attn.view(b, 1, grid, grid)
    a = F.interpolate(a, size=x.shape[-2:], mode="nearest").squeeze(1)
    lo = a.view(b, -1).min(dim=1).values.view(b, 1, 1)
    hi = a.view(b, -1).max(dim=1).values.view(b, 1, 1)
    return ((a - lo) / (hi - lo + 1e-8)).cpu().numpy()
