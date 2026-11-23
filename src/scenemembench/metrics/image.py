# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Deterministic image metrics for rotation-only continuation evaluation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class RotationMetricConfig:
    """Frozen preprocessing and model settings for one evaluation batch."""

    width: int = 448
    height: int = 252
    dino_model: str = "facebook/dinov2-base"
    dino_revision: str | None = None
    lpips_net: str = "alex"
    psnr_epsilon: float = 1e-10
    schema_version: str = "world-memory-rotation-metrics-v1"

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("metric dimensions must be positive")
        if self.width % 14 or self.height % 14:
            raise ValueError(
                "DINOv2 metric dimensions must be divisible by patch size 14"
            )
        if self.lpips_net not in {"alex", "vgg", "squeeze"}:
            raise ValueError(f"unsupported LPIPS network: {self.lpips_net}")
        if self.psnr_epsilon <= 0:
            raise ValueError("psnr_epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - selected environment controls deps
        raise RuntimeError("rotation metrics require numpy and pillow") from exc
    return np, Image


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_uint8(image: Any) -> Any:
    """Coerce a path, PIL image, tensor, or array to contiguous RGB uint8 HWC."""

    np, Image = _dependencies()
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return np.ascontiguousarray(
                np.asarray(source.convert("RGB"), dtype=np.uint8)
            )
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "convert") and hasattr(image, "size"):
        image = np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if (
        array.ndim == 3
        and array.shape[0] in {1, 3, 4}
        and array.shape[-1] not in {1, 3, 4}
    ):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"expected an HWC RGB image, got {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array *= 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def center_crop_to_aspect(image: Any, aspect: float) -> Any:
    """Return the centered maximum crop with the requested aspect ratio."""

    if aspect <= 0:
        raise ValueError("aspect must be positive")
    array = rgb_uint8(image)
    height, width = array.shape[:2]
    current = width / height
    if current > aspect:
        target_width = max(1, round(height * aspect))
        left = (width - target_width) // 2
        return array[:, left : left + target_width]
    if current < aspect:
        target_height = max(1, round(width / aspect))
        top = (height - target_height) // 2
        return array[top : top + target_height]
    return array


def resize_rgb(image: Any, width: int, height: int, *, cover: bool = False) -> Any:
    """Resize RGB with Lanczos, optionally preserving aspect via center cover crop."""

    np, Image = _dependencies()
    array = rgb_uint8(image)
    if cover:
        array = center_crop_to_aspect(array, width / height)
    resized = Image.fromarray(array).resize(
        (int(width), int(height)), Image.Resampling.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def native_gt_frame(image: Any, model_id: str) -> Any:
    """Apply the exact GT spatial transform used by each current resident prefill."""

    if model_id == "hy-worldplay":
        return resize_rgb(image, 848, 480, cover=True)
    if model_id == "matrix-game-3.5":
        return resize_rgb(image, 1280, 704, cover=False)
    if model_id == "matrix-game-3":
        return resize_rgb(image, 1280, 704, cover=False)
    if model_id == "lingbot-world-v2":
        # C3 LingBot v2 resident emits a 832x464 (16:9 latent) canvas.
        return resize_rgb(image, 832, 464, cover=False)
    if model_id == "sana-wm-streaming":
        # C3 SANA-WM resident emits the 1280x704 canvas.
        return resize_rgb(image, 1280, 704, cover=False)
    if model_id == "zing-world-0.5":
        # Zing-0.5 resizes every seed/context frame directly to its native
        # 1280x704 canvas before VAE encoding (no aspect-preserving crop).
        return resize_rgb(image, 1280, 704, cover=False)
    raise ValueError(f"rotation evaluator has no native GT transform for {model_id!r}")


def metric_frame(image: Any, config: RotationMetricConfig) -> Any:
    """Normalize either side of a metric pair to the canonical 16:9 canvas."""

    return resize_rgb(image, config.width, config.height, cover=True)


def compute_pixel_metrics(
    ground_truth: Sequence[Any],
    predicted: Sequence[Any],
    config: RotationMetricConfig = RotationMetricConfig(),
) -> dict[str, list[float]]:
    """Compute RGB PSNR and Gaussian-window SSIM for equally sized frame pairs."""

    if len(ground_truth) != len(predicted) or not ground_truth:
        raise ValueError("metric inputs must be non-empty and have equal lengths")
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pixel metrics require numpy and scikit-image") from exc
    psnr: list[float] = []
    ssim: list[float] = []
    for gt_raw, pred_raw in zip(ground_truth, predicted):
        gt = metric_frame(gt_raw, config).astype(np.float32) / 255.0
        pred = metric_frame(pred_raw, config).astype(np.float32) / 255.0
        mse = float(np.mean((gt - pred) ** 2))
        psnr.append(float(-10.0 * np.log10(max(mse, config.psnr_epsilon))))
        ssim.append(
            float(
                structural_similarity(
                    gt,
                    pred,
                    channel_axis=2,
                    data_range=1.0,
                    gaussian_weights=True,
                    win_size=11,
                )
            )
        )
    return {"psnr": psnr, "ssim": ssim}


class DINOv2Scorer:
    """Batch-reused DINOv2 CLS cosine scorer with frozen preprocessing."""

    def __init__(
        self,
        config: RotationMetricConfig,
        *,
        device: str = "cuda:0",
        local_files_only: bool = False,
        batch_size: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DINOv2 scoring requires torch and transformers"
            ) from exc
        self.config = config
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.model = AutoModel.from_pretrained(
            config.dino_model,
            revision=config.dino_revision,
            local_files_only=local_files_only,
        ).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None) or config.dino_revision
        )
        self.weights_sha256 = self._weight_sha256(local_files_only=local_files_only)

    def _weight_sha256(self, *, local_files_only: bool) -> str | None:
        try:
            from transformers.utils.hub import cached_file
        except ImportError:  # pragma: no cover
            return None
        for name in ("model.safetensors", "pytorch_model.bin"):
            path = cached_file(
                self.config.dino_model,
                name,
                revision=self.resolved_revision,
                local_files_only=local_files_only,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if path:
                return _sha256_path(Path(path))
        return None

    def identity(self) -> dict[str, Any]:
        return {
            "model_id": self.config.dino_model,
            "source_model_id": getattr(self.model.config, "_name_or_path", None),
            "requested_revision": self.config.dino_revision,
            "resolved_revision": self.resolved_revision,
            "weights_sha256": self.weights_sha256,
            "feature": "last_hidden_state_cls_l2_cosine",
            "dtype": "float32",
        }

    def _features(self, frames: Sequence[Any]):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        features = []
        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                arrays = [
                    metric_frame(frame, self.config)
                    for frame in frames[start : start + self.batch_size]
                ]
                pixels = torch.from_numpy(np.stack(arrays)).to(
                    self.device, dtype=torch.float32
                )
                pixels = pixels.permute(0, 3, 1, 2) / 255.0
                pixels = (pixels - mean) / std
                outputs = self.model(pixel_values=pixels, interpolate_pos_encoding=True)
                cls = functional.normalize(
                    outputs.last_hidden_state[:, 0].float(), dim=1
                )
                features.append(cls.cpu())
        return torch.cat(features, dim=0)

    def score(
        self, ground_truth: Sequence[Any], predicted: Sequence[Any]
    ) -> list[float]:
        if len(ground_truth) != len(predicted) or not ground_truth:
            raise ValueError("DINO inputs must be non-empty and have equal lengths")
        gt_features = self._features(ground_truth)
        pred_features = self._features(predicted)
        return (gt_features * pred_features).sum(dim=1).tolist()


class LPIPSScorer:
    """Batch-reused LPIPS scorer."""

    def __init__(
        self,
        config: RotationMetricConfig,
        *,
        device: str = "cuda:0",
        batch_size: int = 8,
        torch_home: str | Path | None = None,
    ) -> None:
        if torch_home is not None:
            os.environ["TORCH_HOME"] = str(Path(torch_home).expanduser().resolve())
        try:
            import lpips
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LPIPS scoring requires lpips and torch") from exc
        self.config = config
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.model = lpips.LPIPS(net=config.lpips_net).to(self.device).eval()
        self.weights_sha256 = self._trunk_sha256()
        calibration = (
            Path(lpips.__file__).resolve().parent
            / "weights"
            / "v0.1"
            / f"{config.lpips_net}.pth"
        )
        self.calibration_sha256 = (
            _sha256_path(calibration) if calibration.is_file() else None
        )

    def _trunk_sha256(self) -> str | None:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return None
        names = {
            "alex": "alexnet-owt-7be5be79.pth",
            "vgg": "vgg16-397923af.pth",
            "squeeze": "squeezenet1_1-b8a52dc0.pth",
        }
        path = Path(torch.hub.get_dir()) / "checkpoints" / names[self.config.lpips_net]
        return _sha256_path(path) if path.is_file() else None

    def identity(self) -> dict[str, Any]:
        try:
            import importlib.metadata

            version = importlib.metadata.version("lpips")
        except Exception:  # pragma: no cover
            version = "unknown"
        return {
            "network": self.config.lpips_net,
            "package_version": version,
            "trunk_weights_sha256": self.weights_sha256,
            "calibration_weights_sha256": self.calibration_sha256,
            "dtype": "float32",
        }

    def score(
        self, ground_truth: Sequence[Any], predicted: Sequence[Any]
    ) -> list[float]:
        if len(ground_truth) != len(predicted) or not ground_truth:
            raise ValueError("LPIPS inputs must be non-empty and have equal lengths")
        import numpy as np
        import torch

        scores = []
        with torch.inference_mode():
            for start in range(0, len(ground_truth), self.batch_size):
                gt_arrays = [
                    metric_frame(frame, self.config)
                    for frame in ground_truth[start : start + self.batch_size]
                ]
                pred_arrays = [
                    metric_frame(frame, self.config)
                    for frame in predicted[start : start + self.batch_size]
                ]
                gt = torch.from_numpy(np.stack(gt_arrays)).to(
                    self.device, dtype=torch.float32
                )
                pred = torch.from_numpy(np.stack(pred_arrays)).to(
                    self.device, dtype=torch.float32
                )
                gt = gt.permute(0, 3, 1, 2) / 127.5 - 1.0
                pred = pred.permute(0, 3, 1, 2) / 127.5 - 1.0
                values = self.model(gt, pred).reshape(-1)
                scores.extend(values.cpu().tolist())
        return [float(value) for value in scores]


def summarize_metric(values: Sequence[float]) -> dict[str, float | int]:
    """Return stable count/mean/std/min/max statistics for one metric."""

    if not values:
        raise ValueError("cannot summarize an empty metric")
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


__all__ = [
    "DINOv2Scorer",
    "LPIPSScorer",
    "RotationMetricConfig",
    "center_crop_to_aspect",
    "compute_pixel_metrics",
    "metric_frame",
    "native_gt_frame",
    "resize_rgb",
    "rgb_uint8",
    "summarize_metric",
]
