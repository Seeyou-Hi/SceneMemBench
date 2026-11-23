# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldFoundry resident lifecycle for Matrix-Game 3.5.

MG3.5's public sampler is batch-oriented.  This module deliberately keeps the
WorldFoundry-facing state machine small and model-agnostic: a native driver
owns the DiT, VAE, rolling KV cache and Patch/Mosaic memory, while this session
enforces one configure/prefill/step/reset lifecycle and monotonic cursors.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from PIL import Image

from scenemembench._native.core.realtime import RealtimeSpec


class MatrixGame35RuntimeError(RuntimeError):
    """Raised when a MG3.5 resident driver violates the runtime contract."""


@runtime_checkable
class MatrixGame35Backend(Protocol):
    """Minimal API implemented by the MG3.5 native driver."""

    def configure(
        self, image: Image.Image, *, prompt: str, seed: int, **kwargs: Any
    ) -> Mapping[str, Any] | None: ...

    def prefill_context(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        motion_scale: float = 1.0,
        **kwargs: Any,
    ) -> Mapping[str, Any] | None: ...

    def reset(self) -> None: ...


@dataclass
class _SessionState:
    configured: bool = False
    prefilled: bool = False
    prompt: str = ""
    seed: int = 42
    step_id: int = 0
    history_cursor: int = 0
    context_frame_ids: tuple[int, ...] = ()
    accepted_frame_ids: tuple[int, ...] = ()
    state_digest: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MatrixGame35RealtimeSession:
    """One resident MG3.5 session backed by a native driver."""

    fps = 17
    first_output_frames = 12
    steady_output_frames = 12

    def __init__(self, backend: MatrixGame35Backend) -> None:
        if not isinstance(backend, MatrixGame35Backend):
            raise TypeError("MG3.5 backend does not implement the resident protocol")
        self.backend = backend
        self.state = _SessionState()

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
            controls=(
                "forward",
                "backward",
                "left",
                "right",
                "camera_up",
                "camera_down",
                "camera_l",
                "camera_r",
            ),
        )

    def next_output_frames(self) -> int:
        return (
            self.first_output_frames
            if self.state.step_id == 0
            else self.steady_output_frames
        )

    @staticmethod
    def _digest(*values: Any) -> str:
        payload = repr(values).encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()[:16]

    def configure(
        self,
        image: Image.Image,
        *,
        prompt: str = "",
        seed: int = 42,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("MG3.5 realtime configuration requires a PIL image")
        self.reset()
        self.state.prompt = str(prompt or "")
        self.state.seed = int(seed)
        started = time.perf_counter()
        result = (
            self.backend.configure(
                image.convert("RGB"),
                prompt=self.state.prompt,
                seed=self.state.seed,
                **kwargs,
            )
            or {}
        )
        if not isinstance(result, Mapping):
            raise MatrixGame35RuntimeError(
                "MG3.5 backend.configure must return a mapping"
            )
        self.state.configured = True
        self.state.state_digest = str(result.get("state_digest", "")) or self._digest(
            "configured", self.state.seed, self.state.prompt
        )
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                **dict(result.get("realtime_metrics", {})),
                "total_ms": float(
                    result.get("total_ms", (time.perf_counter() - started) * 1000.0)
                ),
            },
        }

    def prefill_context(self, **kwargs: Any) -> dict[str, Any]:
        if not self.state.configured:
            raise MatrixGame35RuntimeError(
                "configure must run before MG3.5 Context prefill"
            )
        frame_ids = tuple(int(value) for value in kwargs.get("frame_ids", ()))
        if not frame_ids:
            raise ValueError("MG3.5 Context prefill requires frame_ids")
        if tuple(sorted(frame_ids)) != frame_ids or len(set(frame_ids)) != len(
            frame_ids
        ):
            raise ValueError("MG3.5 Context frame_ids must be strictly increasing")
        camera_to_world = kwargs.get("camera_to_world")
        if camera_to_world is None or len(camera_to_world) != len(frame_ids):
            raise ValueError("MG3.5 Context requires one camera pose per frame")
        result = self.backend.prefill_context(**kwargs)
        if not isinstance(result, Mapping):
            raise MatrixGame35RuntimeError(
                "MG3.5 backend.prefill_context must return a mapping"
            )
        accepted = tuple(
            int(value) for value in result.get("accepted_frame_ids", frame_ids)
        )
        if not set(accepted).issubset(set(frame_ids)):
            raise MatrixGame35RuntimeError(
                "MG3.5 backend accepted a frame outside the Context"
            )
        before = int(result.get("history_cursor_before", self.state.history_cursor))
        after = int(result.get("history_cursor_after", before + len(accepted)))
        if after <= before:
            raise MatrixGame35RuntimeError(
                "MG3.5 Context prefill did not advance history cursor"
            )
        self.state.prefilled = True
        self.state.context_frame_ids = frame_ids
        self.state.accepted_frame_ids = accepted
        self.state.history_cursor = after
        self.state.state_digest = str(
            result.get("state_digest_after", "")
        ) or self._digest("prefill", frame_ids, after)
        return {
            "status": str(result.get("status", "accepted")),
            "accepted_frame_ids": accepted,
            "rejected_frame_ids": tuple(
                int(value) for value in result.get("rejected_frame_ids", ())
            ),
            "native_latent_count": int(result.get("native_latent_count", 0)),
            "native_chunk_count": int(result.get("native_chunk_count", 0)),
            "history_cursor_before": before,
            "history_cursor_after": after,
            "last_context_pose": result.get("last_context_pose", {}),
            "boundary_mapping": dict(result.get("boundary_mapping", {})),
            "state_digest_before": str(result.get("state_digest_before", "")),
            "state_digest_after": self.state.state_digest,
            "warnings": tuple(str(value) for value in result.get("warnings", ())),
        }

    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        motion_scale: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.state.prefilled:
            raise MatrixGame35RuntimeError("MG3.5 step requires Context prefill")
        if not 0.0 < float(motion_scale) <= 1.0:
            raise ValueError("MG3.5 motion_scale must be in (0, 1]")
        step_id = self.state.step_id
        cursor_before = self.state.history_cursor
        started = time.perf_counter()
        result = (
            self.backend.generate(
                interactions=tuple(str(value) for value in interactions),
                control_segments=control_segments,
                motion_scale=float(motion_scale),
                step_id=step_id,
                **kwargs,
            )
            or {}
        )
        if not isinstance(result, Mapping):
            raise MatrixGame35RuntimeError(
                "MG3.5 backend.generate must return a mapping"
            )
        video = result.get("video")
        generated_count = int(result.get("generated_frame_count", 0))
        if generated_count <= 0 and video is not None:
            generated_count = int(getattr(video, "shape", [0])[0])
        if generated_count <= 0:
            raise MatrixGame35RuntimeError("MG3.5 backend generated no frames")
        cursor_after = int(
            result.get("history_cursor_after", cursor_before + generated_count)
        )
        if cursor_after <= cursor_before:
            raise MatrixGame35RuntimeError(
                "MG3.5 generation did not advance history cursor"
            )
        self.state.step_id += 1
        self.state.history_cursor = cursor_after
        self.state.state_digest = str(result.get("state_digest", "")) or self._digest(
            "step", self.state.step_id, cursor_after
        )
        metrics = dict(result.get("realtime_metrics", {}))
        metrics.setdefault("history_cursor_before", cursor_before)
        metrics.setdefault("history_cursor_after", cursor_after)
        metrics.setdefault("state_digest", self.state.state_digest)
        metrics.setdefault("total_ms", (time.perf_counter() - started) * 1000.0)
        return {
            "video": video,
            "generated_frame_count": generated_count,
            "realtime_metrics": metrics,
            "memory_selection": dict(result.get("memory_selection", {})),
            "warnings": tuple(str(value) for value in result.get("warnings", ())),
        }

    def reset(self) -> None:
        self.backend.reset()
        self.state = _SessionState()


__all__ = [
    "MatrixGame35Backend",
    "MatrixGame35RealtimeSession",
    "MatrixGame35RuntimeError",
]
