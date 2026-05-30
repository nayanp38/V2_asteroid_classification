import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.augmentation import AugmentConfig, augment_spectrum


def test_augment_preserves_shape():
    rng = np.random.default_rng(0)
    x = np.stack([np.linspace(0.8, 1.2, 401), np.ones(401)], axis=0).astype(np.float32)
    cfg = AugmentConfig(noise_std=0.01, smooth_sigma=0.5)
    y = augment_spectrum(x, rng, cfg)
    assert y.shape == x.shape


def test_augment_respects_mask():
    """Augmentation must not write into masked-out regions of the reflectance."""
    rng = np.random.default_rng(0)
    refl = np.ones(401, dtype=np.float32) * 1.05
    mask = np.ones(401, dtype=np.float32)
    mask[:50] = 0.0  # masked region we expect to stay frozen at 1.05
    x = np.stack([refl, mask], axis=0)
    cfg = AugmentConfig(noise_std=0.05, smooth_sigma=0.0, p_noise=1.0, p_smooth=0.0)
    y = augment_spectrum(x, rng, cfg)
    assert np.allclose(y[0, :50], refl[:50])
    assert not np.allclose(y[0, 50:], refl[50:])