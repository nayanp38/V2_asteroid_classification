"""1D Grad-CAM for spectral CNN models.

The CNNs now return ``(logits_fine, logits_coarse)`` so we hook the relevant
encoder layer (last conv in SpectrumCNN, ``block3.ms`` in SpectraNetLite) and
back-propagate through the fine-head argmax. The dataset emits a 5-tuple
``(x, y_fine, y_coarse, extra, path)`` which we unpack here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from asteroid_ml.spectrum_io import load_gp_spectrum


def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    name = model_name.lower()
    if "spectrum" in name:
        for i in range(len(model.features) - 1, -1, -1):
            if isinstance(model.features[i], nn.Conv1d):
                return model.features[i]
        raise ValueError("No Conv1d in SpectrumCNN.features")
    if hasattr(model, "block3"):
        return model.block3.ms
    raise ValueError(f"Unknown model for Grad-CAM: {model_name}")


class GradCAM1D:
    def __init__(self, model: nn.Module, model_name: str) -> None:
        self.model = model
        self.model_name = model_name
        self.target_layer = _get_target_layer(model, model_name)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def _save_activation(self, _module, _inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        self.model.eval()
        x = x.requires_grad_(True)
        out = self.model(x)
        logits_fine = out[0] if isinstance(out, tuple) else out
        if class_idx is None:
            class_idx = int(logits_fine.argmax(dim=1).item())
        score = logits_fine[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()

        L = x.shape[-1]
        if cam_np.shape[-1] != L:
            cam_np = np.interp(
                np.linspace(0, 1, L),
                np.linspace(0, 1, cam_np.shape[-1]),
                cam_np,
            )
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        return cam_np.astype(np.float32), class_idx


def plot_gradcam_sample(
    spectrum_path: Path,
    x: np.ndarray,
    cam: np.ndarray,
    class_name: str,
    true_class: Optional[str],
    out_path: Path,
    normalize_wavelength: float = 0.55,
) -> None:
    wl, refl, _ = load_gp_spectrum(spectrum_path)
    mask = x[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(wl, refl, color="0.3", lw=1, label="reflectance")
    axes[0].axvline(normalize_wavelength, color="red", ls="--", alpha=0.5, label="0.55 µm")
    title = f"Predicted: {class_name}"
    if true_class:
        title += f"  |  True: {true_class}"
    axes[0].set_title(title)
    axes[0].set_ylabel("Reflectance")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(wl, x[0], color="steelblue", lw=1, label="normalized input")
    masked = mask < 0.5
    if masked.any():
        axes[1].fill_between(wl, x[0].min(), x[0].max(), where=masked, color="0.85", step="mid",
                              alpha=0.7, label="masked (artifact)")
    ax2 = axes[1].twinx()
    ax2.fill_between(wl, 0, cam, color="orange", alpha=0.45, label="Grad-CAM")
    axes[1].set_xlabel("Wavelength (µm)")
    axes[1].set_ylabel("Norm. refl.")
    ax2.set_ylabel("CAM weight")
    axes[1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_report(
    model: nn.Module,
    model_name: str,
    loader,
    index_to_class: dict,
    data_root: Path,
    out_dir: Path,
    max_samples: int = 12,
    normalize_wavelength: float = 0.55,
) -> List[str]:
    """Generate Grad-CAM plots for up to ``max_samples`` test items.

    The loader is expected to emit ``(x, y_fine, y_coarse, path)``.
    """
    out_dir = out_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM1D(model, model_name)
    saved: List[str] = []
    try:
        count = 0
        for x, y, _y_coarse, paths in loader:
            for i in range(x.size(0)):
                if count >= max_samples:
                    break
                xi = x[i : i + 1]
                yi = int(y[i])
                path = paths[i]
                spectrum_path = data_root / path
                cam_map, pred_idx = cam(xi, class_idx=None)
                pred_name = index_to_class[pred_idx]
                true_name = index_to_class[yi]
                stem = Path(path).stem
                out_file = out_dir / f"{stem}_pred{pred_name}_true{true_name}.png"
                plot_gradcam_sample(
                    spectrum_path,
                    xi[0].detach().cpu().numpy(),
                    cam_map,
                    pred_name,
                    true_name,
                    out_file,
                    normalize_wavelength,
                )
                saved.append(str(out_file.relative_to(out_dir.parent)))
                count += 1
            if count >= max_samples:
                break
    finally:
        cam.close()
    return saved
