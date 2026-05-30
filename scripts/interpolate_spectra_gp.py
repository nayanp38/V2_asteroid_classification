#!/usr/bin/env python3
"""
Regenerate GP-interpolated spectra (401 points, step 0.005 µm).
Copied from the parent project; run from v2/ root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

DATA_ROOT = Path("data")
BINZEL_DIR = DATA_ROOT / "Binzel2019data"
DEMEO_DIR = DATA_ROOT / "DeMeo2009data"
MARSSET_DIR = DATA_ROOT / "MITHNEOS_spectra_Marsset2022"
BINZEL_OUT = DATA_ROOT / "Binzel2019data_gp"
DEMEO_OUT = DATA_ROOT / "DeMeo2009data_gp"
MARSSET_OUT = DATA_ROOT / "MITHNEOS_spectra_Marsset2022_gp"
X_MIN, X_MAX, X_STEP = 0.45, 2.45, 0.005


def load_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, dtype=float, usecols=(0, 1, 2), invalid_raise=False)
    if data.ndim == 1:
        data = data[None, :]
    data = data[~np.isnan(data).any(axis=1)]
    x, y = data[:, 0].astype(float), data[:, 1].astype(float)
    dy = data[:, 2].astype(float) if data.shape[1] >= 3 else np.full_like(y, 0.01)
    dy = np.where(dy <= 0, np.nanmedian(dy[dy > 0]) if np.any(dy > 0) else 1e-3, dy)
    order = np.argsort(x)
    return x[order], y[order], dy[order]


def load_mithneos_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MITHNEOS/Marsset 2022: wavelength, reflectance, uncertainty; -1 = missing."""
    data = np.genfromtxt(path, dtype=float, usecols=(0, 1, 2), invalid_raise=False)
    if data.ndim == 1:
        data = data[None, :]
    valid = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    valid &= data[:, 1] > 0
    data = data[valid]
    x, y = data[:, 0].astype(float), data[:, 1].astype(float)
    dy = data[:, 2].astype(float) if data.shape[1] >= 3 else np.full_like(y, 0.01)
    dy = np.where(
        (dy <= 0) | ~np.isfinite(dy),
        np.nanmedian(dy[(dy > 0) & np.isfinite(dy)]) if np.any((dy > 0) & np.isfinite(dy)) else 1e-3,
        dy,
    )
    order = np.argsort(x)
    return x[order], y[order], dy[order]


def make_standard_grid() -> np.ndarray:
    n = int(round((X_MAX - X_MIN) / X_STEP))
    return X_MIN + X_STEP * np.arange(n + 1, dtype=float)


def gp_interpolate(x, y, dy, x_grid):
    kernel = 1.0 * RBF(length_scale=0.05) + WhiteKernel(noise_level=1e-4)
    alpha = (np.clip(dy, 1e-6, np.inf)) ** 2
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
    gp.fit(x[:, None], y)
    y_mean = np.full_like(x_grid, np.nan)
    y_std = np.full_like(x_grid, np.nan)
    inside = (x_grid >= x.min()) & (x_grid <= x.max())
    if np.any(inside):
        y_pred, y_sigma = gp.predict(x_grid[inside, None], return_std=True)
        y_mean[inside], y_std[inside] = y_pred, y_sigma
    return y_mean, y_std


def interpolate_directory(
    in_dir: Path,
    out_dir: Path,
    x_grid: np.ndarray,
    loader=load_spectrum,
    patterns: Tuple[str, ...] = ("a*.txt", "au*.txt"),
) -> None:
    seen: set[str] = set()
    for pattern in patterns:
        for f in sorted(in_dir.glob(pattern)):
            if f.name in seen:
                continue
            seen.add(f.name)
            out = out_dir / f.name
            if out.exists():
                continue
            x, y, dy = loader(f)
            if x.size < 3:
                continue
            y_mean, y_std = gp_interpolate(x, y, dy, x_grid)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out, np.column_stack([x_grid, y_mean, y_std]), fmt="%.6f\t%.6f\t%.6f")


def main() -> None:
    grid = make_standard_grid()
    print(f"Grid: {len(grid)} points")
    interpolate_directory(BINZEL_DIR, BINZEL_OUT, grid)
    interpolate_directory(DEMEO_DIR, DEMEO_OUT, grid)
    if MARSSET_DIR.is_dir():
        print(f"Marsset 2022 MITHNEOS: {MARSSET_DIR}")
        interpolate_directory(
            MARSSET_DIR, MARSSET_OUT, grid, loader=load_mithneos_spectrum
        )


if __name__ == "__main__":
    main()
