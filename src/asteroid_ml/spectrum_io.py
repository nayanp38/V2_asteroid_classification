"""Load GP spectra and build CNN input tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """How `preprocess_spectrum` builds the mask and fills artifact regions."""

    std_mask_enabled: bool = True
    std_mask_floor: float = 0.02
    std_mask_k: float = 2.0
    # Detect "frozen" runs from sparse-GP extrapolation: a run of >=
    # ``frozen_run_min`` adjacent points whose reflectance changes by less than
    # ``frozen_eps`` is masked out.
    frozen_run_enabled: bool = True
    frozen_eps: float = 1e-4
    frozen_run_min: int = 4
    artifact_fill_value: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "PreprocessConfig":
        d = d or {}
        return cls(
            std_mask_enabled=bool(d.get("std_mask_enabled", True)),
            std_mask_floor=float(d.get("std_mask_floor", 0.02)),
            std_mask_k=float(d.get("std_mask_k", 2.0)),
            frozen_run_enabled=bool(d.get("frozen_run_enabled", True)),
            frozen_eps=float(d.get("frozen_eps", 1e-4)),
            frozen_run_min=int(d.get("frozen_run_min", 4)),
            artifact_fill_value=float(d.get("artifact_fill_value", 1.0)),
        )


def load_gp_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load wavelength, reflectance, gp_std from a GP-interpolated file."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    wl = data[:, 0]
    refl = data[:, 1].astype(np.float32)
    if data.shape[1] >= 3:
        std = data[:, 2].astype(np.float32)
    else:
        std = np.zeros_like(refl)
    return wl, refl, std


def reflectance_at_wavelength(
    wl: np.ndarray, refl: np.ndarray, target: float = 0.55
) -> float:
    """Linear interpolation of reflectance at target wavelength."""
    valid = np.isfinite(refl) & np.isfinite(wl)
    if not np.any(valid):
        return 1.0
    w = wl[valid]
    r = refl[valid]
    if target <= w.min():
        return float(r[0])
    if target >= w.max():
        return float(r[-1])
    return float(np.interp(target, w, r))


def _frozen_run_mask(refl: np.ndarray, eps: float, min_run: int) -> np.ndarray:
    """Mark points that belong to a run of >= ``min_run`` adjacent reflectance
    values identical within ``eps`` (signature of GP collapse to the mean).
    Returns a boolean array, True = part of a frozen run.
    """
    n = refl.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    diffs = np.abs(np.diff(refl))
    same = diffs < eps  # length n-1, same[i] => refl[i] ~= refl[i+1]
    frozen = np.zeros(n, dtype=bool)
    i = 0
    while i < n - 1:
        if not same[i]:
            i += 1
            continue
        j = i
        while j < n - 1 and same[j]:
            j += 1
        run_len = j - i + 1
        if run_len >= min_run:
            frozen[i : j + 1] = True
        i = j + 1
    return frozen


def _build_mask(
    refl: np.ndarray, std: np.ndarray, cfg: PreprocessConfig
) -> np.ndarray:
    """Boolean validity mask, True = trustworthy point."""
    valid = np.isfinite(refl)
    if cfg.std_mask_enabled and np.isfinite(std).any():
        finite_std = std[np.isfinite(std)]
        median_std = float(np.median(finite_std))
        threshold = max(cfg.std_mask_floor, cfg.std_mask_k * median_std)
        valid = valid & np.isfinite(std) & (std <= threshold)
    if cfg.frozen_run_enabled and refl.size > 0:
        frozen = _frozen_run_mask(
            np.where(np.isfinite(refl), refl, 0.0),
            cfg.frozen_eps,
            cfg.frozen_run_min,
        )
        valid = valid & ~frozen
    return valid


def preprocess_spectrum(
    path: Path,
    normalize_wavelength: float = 0.55,
    cfg: PreprocessConfig | None = None,
) -> np.ndarray:
    """
    Return tensor shaped (2, L): channel 0 = normalized reflectance with
    artifact regions filled at the anchor value, channel 1 = validity mask.

    The mask combines `isfinite(refl)` with a `gp_std`-based artifact filter
    so that GP extrapolation plateaus (constant reflectance + high std) are
    marked invalid even though they look numerically fine.
    """
    cfg = cfg or PreprocessConfig()
    wl, refl, std = load_gp_spectrum(path)
    mask_bool = _build_mask(refl, std, cfg)
    mask = mask_bool.astype(np.float32)

    refl_clean = refl.copy()
    nan_idx = ~np.isfinite(refl_clean)
    if nan_idx.any():
        if mask_bool.any():
            refl_clean[nan_idx] = float(np.nanmean(refl_clean[mask_bool]))
        else:
            refl_clean[nan_idx] = float(np.nanmean(refl_clean[np.isfinite(refl_clean)])) \
                if np.any(np.isfinite(refl_clean)) else 1.0

    norm_pool_wl = wl[mask_bool] if mask_bool.any() else wl[np.isfinite(refl_clean)]
    norm_pool_refl = refl_clean[mask_bool] if mask_bool.any() else refl_clean[np.isfinite(refl_clean)]
    if norm_pool_wl.size == 0:
        norm = 1.0
    else:
        norm = reflectance_at_wavelength(norm_pool_wl, norm_pool_refl, normalize_wavelength)
    if norm <= 0 or not np.isfinite(norm):
        norm = 1.0
    refl_norm = (refl_clean / norm).astype(np.float32)
    refl_norm = np.where(mask_bool, refl_norm, np.float32(cfg.artifact_fill_value))

    x = np.stack([refl_norm, mask], axis=0)
    return x


def expected_length(wl_min: float, wl_max: float, step: float) -> int:
    return int(round((wl_max - wl_min) / step)) + 1


def valid_fraction(
    path: Path,
    cfg: PreprocessConfig | None = None,
) -> float:
    """Fraction of points that survive the mask (artifact-free)."""
    cfg = cfg or PreprocessConfig()
    _wl, refl, std = load_gp_spectrum(path)
    mask = _build_mask(refl, std, cfg)
    if mask.size == 0:
        return 0.0
    return float(mask.sum() / mask.size)
