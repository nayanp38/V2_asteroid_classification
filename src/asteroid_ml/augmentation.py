"""Training-time augmentation for 1D reflectance spectra.

Operates on the ``(2, L)`` array produced by
:func:`asteroid_ml.spectrum_io.preprocess_spectrum` (channel 0 = normalized
reflectance, channel 1 = validity mask).

Two transforms only, both mask-aware so the augmentation never invents signal
in artifact regions:

  * **Additive Gaussian noise** on the reflectance channel
    (only where the mask is 1).
  * **Gaussian smoothing** of the reflectance channel
    (only where the mask is 1).

The previous v0.2 ``circular wavelength shift`` was removed because it broke
the 0.55 µm normalization anchor.  No other augmentations are applied to keep
the methodology simple and easy to describe in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    noise_std: float = 0.015
    smooth_sigma: float = 0.8
    p_noise: float = 0.8
    p_smooth: float = 0.3

    @classmethod
    def from_dict(cls, d: dict | None) -> "AugmentConfig":
        d = d or {}
        return cls(
            noise_std=float(d.get("noise_std", 0.015)),
            smooth_sigma=float(d.get("smooth_sigma", 0.8)),
            p_noise=float(d.get("p_noise", 0.8)),
            p_smooth=float(d.get("p_smooth", 0.3)),
        )


def _gaussian_kernel1d(sigma: float, size: int = 7) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return (k / k.sum()).astype(np.float32)


def augment_spectrum(
    x: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Apply Gaussian noise and Gaussian smoothing to a copy of ``x``.

    Both transforms are mask-aware: noise only adds to valid points (mask=1)
    and smoothing only replaces values where the mask is set, preserving the
    artifact-fill convention in masked regions.
    """
    out = x.copy()
    refl = out[0].copy()
    mask = out[1]
    valid = mask > 0.5

    if cfg.smooth_sigma > 0 and rng.random() < cfg.p_smooth:
        k = _gaussian_kernel1d(cfg.smooth_sigma)
        pad = len(k) // 2
        padded = np.pad(refl, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, k, mode="valid")
        refl = np.where(valid, smoothed, refl)

    if cfg.noise_std > 0 and rng.random() < cfg.p_noise:
        noise = rng.normal(0.0, cfg.noise_std, size=refl.shape).astype(np.float32)
        refl = refl + noise * mask

    out[0] = refl.astype(np.float32)
    return out
