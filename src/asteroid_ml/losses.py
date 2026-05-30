"""Classification losses used for the hierarchical asteroid CNN.

The main entry point is :class:`FocalSmoothedCE`, a small wrapper that combines
class-weighted cross-entropy with focal-loss focusing (Lin et al. 2017) and
label smoothing (Szegedy et al. 2016).  Both heads (fine + coarse) use this
loss with their own class weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalSmoothedCE(nn.Module):
    """Focal cross-entropy with label smoothing and per-class weights.

    Parameters
    ----------
    weight:
        Optional 1D tensor of per-class weights.
    gamma:
        Focal-loss focusing parameter.  ``gamma=0`` disables focusing and the
        loss reduces to weighted CE with label smoothing.
    label_smoothing:
        Smoothing parameter in ``[0, 1)``.  ``0`` disables smoothing.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)

        if target.dtype == torch.long:
            with torch.no_grad():
                eps = self.label_smoothing
                if eps > 0:
                    true_dist = torch.full_like(log_probs, eps / max(1, n_classes - 1))
                    true_dist.scatter_(1, target.unsqueeze(1), 1.0 - eps)
                else:
                    true_dist = F.one_hot(target, n_classes).to(log_probs.dtype)
        else:
            true_dist = target.to(log_probs.dtype)
            if self.label_smoothing > 0:
                eps = self.label_smoothing
                true_dist = true_dist * (1.0 - eps) + eps / max(1, n_classes - 1)

        if self.weight is not None:
            wt = self.weight.to(log_probs.device).unsqueeze(0)
        else:
            wt = torch.ones(1, n_classes, device=log_probs.device)

        if self.gamma > 0:
            probs = log_probs.exp().clamp(min=1e-8, max=1 - 1e-8)
            focal_factor = (1.0 - probs).pow(self.gamma)
            per_term = -(focal_factor * log_probs * true_dist * wt)
        else:
            per_term = -(log_probs * true_dist * wt)
        return per_term.sum(dim=1).mean()
