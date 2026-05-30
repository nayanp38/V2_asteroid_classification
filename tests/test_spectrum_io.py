import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.spectrum_io import (
    PreprocessConfig,
    expected_length,
    load_gp_spectrum,
    preprocess_spectrum,
    valid_fraction,
)


def test_preprocess_shape():
    files = list((ROOT / "data" / "DeMeo2009data_gp").glob("a*.txt"))
    assert files
    x = preprocess_spectrum(files[0])
    assert x.shape[0] == 2
    assert x.shape[1] == expected_length(0.45, 2.45, 0.005)


def test_normalize_near_one_at_055():
    path = ROOT / "data" / "DeMeo2009data_gp" / "a000004.sp02.txt"
    wl, refl, _ = load_gp_spectrum(path)
    x = preprocess_spectrum(path)
    idx = int(np.argmin(np.abs(wl - 0.55)))
    assert abs(float(x[0, idx]) - 1.0) < 0.05


def test_frozen_run_is_masked():
    """A Binzel spectrum with a known frozen-GP region should be partly masked."""
    path = ROOT / "data" / "Binzel2019data_gp" / "a144411.visnir.txt"
    if not path.is_file():
        return
    cfg = PreprocessConfig()
    x = preprocess_spectrum(path, cfg=cfg)
    mask = x[1]
    assert mask.min() == 0.0
    assert mask.mean() < 0.95
    refl = x[0]
    assert np.allclose(refl[mask < 0.5], cfg.artifact_fill_value)


def test_valid_fraction_runs():
    path = ROOT / "data" / "DeMeo2009data_gp" / "a000004.sp02.txt"
    vf = valid_fraction(path)
    assert 0.0 <= vf <= 1.0
