from asteroid_ml.models.spectranet_lite import SpectraNetLite
from asteroid_ml.models.spectrum_cnn import SpectrumCNN

__all__ = ["SpectrumCNN", "SpectraNetLite", "build_model"]


def build_model(
    name: str,
    n_classes: int,
    in_channels: int = 2,
    n_coarse: int = 0,
):
    """Construct a model by name.

    ``n_coarse`` controls whether the auxiliary coarse-class head is created.
    """
    name = name.lower().replace("-", "_")
    if name in ("spectrum_cnn", "spectrumcnn", "baseline"):
        return SpectrumCNN(
            n_outputs=n_classes,
            in_channels=in_channels,
            n_coarse=n_coarse,
        )
    if name in ("spectranet_lite", "spectranetlite", "spectranet"):
        return SpectraNetLite(
            n_classes=n_classes,
            in_channels=in_channels,
            n_coarse=n_coarse,
        )
    raise ValueError(f"Unknown model: {name}")
