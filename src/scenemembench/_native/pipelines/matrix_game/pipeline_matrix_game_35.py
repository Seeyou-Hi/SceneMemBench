# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldFoundry pipeline facade for Matrix-Game 3.5 resident drivers."""

from __future__ import annotations

import importlib
import inspect
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ...synthesis.visual_generation.matrix_game.matrix_game_35_runtime import (
    MatrixGame35RealtimeSession,
    MatrixGame35RuntimeError,
)
from ..pipeline_utils import PipelineABC


def _load_driver(spec: str, **factory_kwargs: Any) -> Any:
    """Load ``module:factory`` without importing MG3.5 at WorldFoundry import time."""

    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"MG3.5 driver must use MODULE:FACTORY syntax, got {spec!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"MG3.5 driver factory is not callable: {spec}")
    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return factory(**factory_kwargs)
    accepted = {
        name: value
        for name, value in factory_kwargs.items()
        if name in signature.parameters
    }
    return factory(**accepted)


class MatrixGame35Pipeline(PipelineABC):
    """Pipeline facade with a stable resident API and injectable native driver."""

    MODEL_ID = "matrix-game-3.5"

    def __init__(
        self, *, backend: Any = None, device: str = "cuda", **kwargs: Any
    ) -> None:
        super().__init__(model_id=self.MODEL_ID, device=device, **kwargs)
        self.backend = backend
        self.device = device
        self._realtime_session: MatrixGame35RealtimeSession | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | os.PathLike[str] | None = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "MatrixGame35Pipeline":
        options = dict(required_components or {})
        options.update(kwargs)
        driver_spec = options.pop("driver", None) or os.environ.get(
            "SCENEMEMBENCH_MATRIX_GAME35_DRIVER", ""
        )
        if not driver_spec:
            root = Path(model_path).expanduser() if model_path else None
            raise MatrixGame35RuntimeError(
                "MG3.5 resident integration requires a native driver. Set "
                "SCENEMEMBENCH_MATRIX_GAME35_DRIVER=module:factory (or pass "
                "required_components={'driver': 'module:factory'}). The released "
                f"batch repository was found at {root or '<unspecified>'}, but does "
                "not expose a resident session API yet."
            )
        backend = _load_driver(
            driver_spec,
            model_path=str(model_path) if model_path is not None else "",
            device=device,
            options=dict(options),
        )
        return cls(backend=backend, device=device, **options)

    def _ensure_realtime_session(self) -> MatrixGame35RealtimeSession:
        if self._realtime_session is None:
            if self.backend is None:
                raise MatrixGame35RuntimeError(
                    "MatrixGame35Pipeline has no resident backend"
                )
            self._realtime_session = MatrixGame35RealtimeSession(self.backend)
        return self._realtime_session

    def prepare_realtime(self) -> Mapping[str, Any]:
        return {
            "realtime_spec": self._ensure_realtime_session()
            .realtime_spec()
            .to_payload()
        }

    def configure_realtime(
        self, images: Any, prompt: str = "", seed: int = 42, **kwargs: Any
    ) -> Mapping[str, Any]:
        if isinstance(images, (str, os.PathLike)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise TypeError("MG3.5 realtime requires a PIL image or image path")
        session = self._ensure_realtime_session()
        configured = session.configure(
            images, prompt=str(prompt or ""), seed=int(seed), **kwargs
        )
        if os.getenv(
            "SCENEMEMBENCH_MATRIX_GAME35_INITIAL_IMAGE_PREFILL", ""
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return configured

        # The distilled resident driver requires at least ten Context frames,
        # while single-initial-image rollouts start from one uploaded image. For
        # this explicit bootstrap mode, repeat that image at an identity
        # pose to initialize the causal cache without inventing camera motion.
        context_frames = 10
        with tempfile.TemporaryDirectory(prefix="worldfoundry_mg35_seed_") as temp_dir:
            seed_path = Path(temp_dir) / "initial.png"
            images.save(seed_path)
            identity = np.eye(4, dtype=np.float32).tolist()
            prefill = session.prefill_context(
                rgb_paths=[str(seed_path)] * context_frames,
                camera_to_world=[identity] * context_frames,
                frame_ids=list(range(context_frames)),
                fps=17.0,
                position_units="m",
                horizontal_fov_deg=float(kwargs.get("horizontal_fov_deg", 90.0)),
            )
        return {**dict(configured), "initial_image_prefill": prefill}

    def prefill_context_realtime(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._ensure_realtime_session().prefill_context(**kwargs)

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        action_motion_scale: float = 1.0,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        return self._ensure_realtime_session().generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
            motion_scale=action_motion_scale,
            **kwargs,
        )

    def realtime_next_output_frames(self) -> int:
        return self._ensure_realtime_session().next_output_frames()

    def reset_realtime(self) -> None:
        if self._realtime_session is not None:
            self._realtime_session.reset()


__all__ = ["MatrixGame35Pipeline"]
