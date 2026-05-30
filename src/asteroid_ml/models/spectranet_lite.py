"""SpectraNet-lite: multi-scale 1D CNN inspired by AppleCiDEr SpectraNet.

The classifier head is split into a shared bottleneck plus a fine head and an
optional coarse head, so the same encoder can drive the hierarchical loss and
constrained inference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MultiScaleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: tuple[int, ...] = (5, 15, 31)) -> None:
        super().__init__()
        n = len(kernels)
        per = out_ch // n
        remainder = out_ch - per * n
        channels = [per + (1 if i < remainder else 0) for i in range(n)]
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2)
                for k, ch in zip(kernels, channels)
            ]
        )
        self.out_channels = sum(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1)


class SpectraBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.ms = MultiScaleConv(in_ch, out_ch)
        self.norm = nn.BatchNorm1d(self.ms.out_channels)
        self.act = nn.GELU()
        self.pool_max = nn.MaxPool1d(2)
        self.pool_avg = nn.AvgPool1d(2)
        self.mid_channels = self.ms.out_channels
        self.out_channels = self.mid_channels * 2
        self.gate = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj = (
            nn.Conv1d(in_ch, self.mid_channels, kernel_size=1)
            if in_ch != self.mid_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.ms(x)))
        pooled = torch.cat([self.pool_max(h), self.pool_avg(h)], dim=1)
        gated = pooled * self.gate(pooled)
        res = self.proj(x)
        res_pooled = torch.cat([self.pool_max(res), self.pool_avg(res)], dim=1)
        return gated + res_pooled


class SpectraNetLite(nn.Module):
    """Multi-scale 1D CNN with optional dual (coarse + fine) heads."""

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        n_coarse: int = 0,
        bottleneck: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = SpectraBlock(in_channels, 32)
        self.block2 = SpectraBlock(self.block1.out_channels, 64)
        self.block3 = SpectraBlock(self.block2.out_channels, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_channels = self.block3.out_channels
        self.head_shared = nn.Sequential(
            nn.Linear(self.encoder_channels, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_fine = nn.Linear(bottleneck, n_classes)
        self.head_coarse: Optional[nn.Linear] = (
            nn.Linear(bottleneck, n_coarse) if n_coarse > 0 else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position encoder features (shape (B, C, L)) before GAP."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.encode(x)
        h = self.pool(h).flatten(1)
        h = self.head_shared(h)
        logits_fine = self.head_fine(h)
        logits_coarse = self.head_coarse(h) if self.head_coarse is not None else None
        return logits_fine, logits_coarse
