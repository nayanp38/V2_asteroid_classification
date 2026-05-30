"""Baseline 1D CNN for asteroid reflectance spectra.

Like :class:`SpectraNetLite`, the head is split into a shared bottleneck and
fine / optional coarse heads so the model works under the hierarchical loss.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class SpectrumCNN(nn.Module):
    """Two-channel 1D CNN (reflectance + mask) with dual classifier heads."""

    def __init__(
        self,
        n_outputs: int,
        in_channels: int = 2,
        n_coarse: int = 0,
        bottleneck: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=31, padding=15),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_channels = 64
        self.head_shared = nn.Sequential(
            nn.Linear(self.encoder_channels, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_fine = nn.Linear(bottleneck, n_outputs)
        self.head_coarse: Optional[nn.Linear] = (
            nn.Linear(bottleneck, n_coarse) if n_coarse > 0 else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.encode(x)
        h = self.pool(h).flatten(1)
        h = self.head_shared(h)
        logits_fine = self.head_fine(h)
        logits_coarse = self.head_coarse(h) if self.head_coarse is not None else None
        return logits_fine, logits_coarse
