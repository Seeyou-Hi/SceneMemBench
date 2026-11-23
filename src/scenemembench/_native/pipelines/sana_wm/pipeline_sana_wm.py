# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldFoundry pipeline for resident SANA-WM interaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..pipeline_utils import PipelineABC


class SanaWMPipeline(PipelineABC):
    """Quality-first image-and-camera world generation with a resident core."""

    MODEL_ID = "sana-wm"
    MODEL_PATH_OPTION = "checkpoint_source"

    def __init__(
        self,
        *,
        checkpoint_source: str | Path | None = None,
        device: str = "cuda",
        model_id: str = MODEL_ID,
        **options: Any,
    ) -> None:
        super().__init__(model_id=model_id, device=device, **options)
        self.checkpoint_source = checkpoint_source
        self._realtime_session: Any = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "SanaWMPipeline":
        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        checkpoint_source = options.pop(
            "checkpoint_source",
            options.pop("checkpoint_dir", options.pop("model_path", None)),
        )
        cls._strip_framework_loading_options(options)
        return cls(
            checkpoint_source=checkpoint_source,
            device=device,
            model_id=resolved_model_id,
            **options,
        )

    def _ensure_realtime_session(self) -> Any:
        if self._realtime_session is None:
            from ...synthesis.visual_generation.sana_wm.realtime import (
                SanaWMRealtimeSession,
            )

            self._realtime_session = SanaWMRealtimeSession(self.checkpoint_source)
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        session = self._ensure_realtime_session()
        return {
            "realtime_spec": session.realtime_spec().to_payload(),
            "runtime_info": session.runtime_info(),
        }

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        window_frames: int | None = None,
        step: int = 4,
        cfg_scale: float = 1.0,
        horizontal_fov_deg: float = 55.0,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(images, (str, Path)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("SANA-WM realtime requires a PIL image or image path.")
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("SANA-WM realtime requires a user-provided text prompt.")
        return self._ensure_realtime_session().configure(
            image=images,
            prompt=prompt,
            seed=seed,
            fps=fps,
            num_frames=window_frames,
            step=step,
            cfg_scale=cfg_scale,
            horizontal_fov_deg=horizontal_fov_deg,
        )

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        camera_trajectory_ue: Sequence[Sequence[Sequence[float]]] | None = None,
        trajectory_mode: str | None = None,
        position_units: str = "m",
        **_: Any,
    ) -> dict[str, Any]:
        result = self._ensure_realtime_session().generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
            seed=seed,
            prompt=prompt,
            camera_trajectory_ue=camera_trajectory_ue,
            trajectory_mode=trajectory_mode,
            position_units=position_units,
        )
        # The common resident adapter consumes ``video`` while the public SANA
        # route historically returned ``frames``. Preserve both spellings.
        result.setdefault("video", result.get("frames"))
        return result

    def prefill_context_realtime(self, **kwargs: Any) -> dict[str, Any]:
        """Teacher-force GT video through the native Self-Forcing history path."""

        return self._ensure_realtime_session().prefill_context(**kwargs)

    def realtime_next_output_frames(self) -> int:
        return int(self._ensure_realtime_session().output_frames)

    def reset_realtime(self) -> None:
        if self._realtime_session is not None:
            self._realtime_session.reset()

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[str] | None = None,
        fps: int = 16,
        num_frames: int = 25,
        step: int = 4,
        cfg_scale: float = 1.0,
        seed: int = 42,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        window_frames = int(kwargs.pop("window_frames", num_frames))
        self.configure_realtime(
            images=images,
            prompt=prompt,
            seed=seed,
            fps=fps,
            window_frames=window_frames,
            step=step,
            cfg_scale=cfg_scale,
            **kwargs,
        )
        result = self.stream_realtime(interactions=interactions, seed=seed, **kwargs)
        return result if return_dict else result["frames"]


SanaWMStreamingPipeline = SanaWMPipeline

__all__ = ["SanaWMPipeline", "SanaWMStreamingPipeline"]
