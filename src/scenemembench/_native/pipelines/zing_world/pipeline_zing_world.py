# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldFoundry wrapper for Zing-0.5 native and resident inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..pipeline_utils import PipelineABC


class ZingWorldPipeline(PipelineABC):
    """Keep the official Zing path available while exposing resident controls."""

    MODEL_ID = "zing-world-0.5"
    MODEL_PATH_OPTION = "checkpoint_source"

    def __init__(
        self,
        *,
        checkpoint_source=None,
        pretrained_dir=None,
        source_root=None,
        device="cuda",
        **options,
    ):
        super().__init__(model_id=self.MODEL_ID, device=device, **options)
        self.checkpoint_source = checkpoint_source
        self.pretrained_dir = pretrained_dir
        self.source_root = source_root
        self._realtime_session = None

    @classmethod
    def from_pretrained(
        cls, model_path=None, required_components=None, device="cuda", **kwargs
    ):
        options = cls._runtime_options(model_path, required_components, kwargs)
        checkpoint_source = options.pop(
            "checkpoint_source",
            options.pop("checkpoint_dir", options.pop("model_path", None)),
        )
        pretrained_dir = options.pop("pretrained_dir", None)
        source_root = options.pop("source_root", None)
        cls._strip_framework_loading_options(options)
        return cls(
            checkpoint_source=checkpoint_source,
            pretrained_dir=pretrained_dir,
            source_root=source_root,
            device=device,
            **options,
        )

    def _ensure_realtime_session(self):
        if self._realtime_session is None:
            from ...synthesis.visual_generation.zing_world import (
                ZingWorldRealtimeSession,
            )

            self._realtime_session = ZingWorldRealtimeSession(
                self.checkpoint_source,
                pretrained_dir=self.pretrained_dir,
                source_root=self.source_root,
                device=self.device,
            )
        return self._realtime_session

    def prepare_realtime(self):
        return {"runtime_info": self._ensure_realtime_session().runtime_info()}

    def configure_realtime(
        self, images: Any, prompt: str = "", seed: int = 42, fps: int = 24, **kwargs
    ):
        return self._ensure_realtime_session().configure(
            images, prompt=prompt, seed=seed, fps=fps, **kwargs
        )

    def prefill_context_realtime(self, **kwargs):
        return self._ensure_realtime_session().prefill_context(**kwargs)

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        return self._ensure_realtime_session().generate(
            interactions=interactions,
            control_segments=realtime_segments,
            prompt=prompt,
            seed=seed,
            **kwargs,
        )

    def realtime_next_output_frames(self):
        return int(
            self._ensure_realtime_session().runtime_info()["native_chunk_frames"]
        )

    def reset_realtime(self):
        if self._realtime_session is not None:
            self._realtime_session.reset()

    def finalize_realtime(self):
        """Decode the accumulated resident Eval latents as one video."""

        return self._ensure_realtime_session().finalize_generated_video()

    def set_realtime_defer_decode(self, enabled: bool = True):
        self._ensure_realtime_session().set_defer_decode(enabled)

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[str] | None = None,
        fps: int = 24,
        num_frames: int = 16,
        seed: int = 42,
        return_dict: bool = True,
        **kwargs,
    ):
        """Run one native-shaped image/action call through the official modules."""

        session = self._ensure_realtime_session()
        session.configure(images, prompt=prompt, seed=seed, fps=fps, **kwargs)
        # Native single-image calls have no GT prefill; the reference image is
        # encoded once and then one resident block is sampled.  The standalone
        # upstream ``run.sh`` remains untouched and is the exact official CLI.
        result = session.generate(interactions=interactions, seed=seed, **kwargs)
        return result if return_dict else result["frames"]


__all__ = ["ZingWorldPipeline"]
