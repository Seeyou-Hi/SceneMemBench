# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Stable contracts for GT Context prefill and resident continuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable


PrefillMode = Literal[
    "native_video_prefill",
    "partial_native_video_prefill",
    "last_frame_only",
    "unsupported",
]


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What a model adapter can prove about its resident continuation path."""

    model_id: str
    prefill_mode: PrefillMode
    native_chunk_frames: int
    native_fps: int
    native_pose_format: str
    supports_gt_video_prefill: bool
    supports_persistent_step: bool
    supports_intra_step_action_segments: bool = False
    supports_action_motion_scale: bool = False
    state_observables: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.native_chunk_frames < 1 or self.native_fps < 1:
            raise ValueError("native chunk and fps must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state_observables"] = list(self.state_observables)
        value["limitations"] = list(self.limitations)
        return value


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Validated GT Context and its camera trajectory."""

    case_id: str
    frame_ids: tuple[int, ...]
    rgb_paths: tuple[Path, ...]
    timestamps_sec: tuple[float, ...]
    ue_poses: tuple[tuple[float, ...], ...]
    camera_to_world: tuple[tuple[tuple[float, ...], ...], ...]
    context_range: tuple[int, int]
    eval_range: tuple[int, int]
    fps: float
    width: int
    height: int
    horizontal_fov_deg: float
    prompt_id: str = "world_memory_default"
    context_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {
            len(self.frame_ids),
            len(self.rgb_paths),
            len(self.timestamps_sec),
            len(self.ue_poses),
            len(self.camera_to_world),
        }
        if len(lengths) != 1:
            raise ValueError(
                "ContextBundle frame, RGB, timestamp, and pose lengths must match"
            )
        if not self.frame_ids:
            raise ValueError("ContextBundle must contain at least one frame")
        if tuple(sorted(self.frame_ids)) != self.frame_ids:
            raise ValueError("ContextBundle frame_ids must be sorted")
        if any(
            right <= left
            for left, right in zip(self.timestamps_sec, self.timestamps_sec[1:])
        ):
            raise ValueError("ContextBundle timestamps must be strictly increasing")
        if self.context_range[0] < 0 or self.context_range[1] <= self.context_range[0]:
            raise ValueError("invalid context_range")
        if (
            self.eval_range[0] < self.context_range[1]
            or self.eval_range[1] <= self.eval_range[0]
        ):
            raise ValueError("eval_range must start at or after context_range end")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("ContextBundle media metadata must be positive")

    @property
    def context_frame_count(self) -> int:
        return len(self.frame_ids)

    @property
    def last_context_pose(self) -> tuple[float, ...]:
        return self.ue_poses[-1]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rgb_paths"] = [str(path) for path in self.rgb_paths]
        value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """Canonical action passed to a model-specific resident session."""

    step_id: int
    canonical_action: str
    duration_frames: int = 1
    translation: tuple[float, float, float] | None = None
    rotation_deg: tuple[float, float, float] | None = None
    target_pose_ue: tuple[float, ...] | None = None
    action_segments: tuple[tuple[str, float], ...] = ()
    camera_trajectory_ue: tuple[tuple[tuple[float, ...], ...], ...] = ()
    trajectory_source_frame_ids: tuple[int, ...] = ()
    phase_names: tuple[str, ...] = ()
    conditioning_only: tuple[bool, ...] = ()
    trajectory_mode: str = "canonical_action"
    motion_scale: float = 1.0
    source: Literal[
        "oracle",
        "fsm",
        "vlm",
        "route_replay_v0",
        "gt_eval_visual_oracle_v0",
        "frozen_action_replay",
        "open_loop_native_action_v0",
    ] = "oracle"

    def __post_init__(self) -> None:
        if self.step_id < 0 or self.duration_frames < 1:
            raise ValueError(
                "step_id must be non-negative and duration_frames must be positive"
            )
        if not self.canonical_action:
            raise ValueError("canonical_action must not be empty")
        if any(not name or duration <= 0 for name, duration in self.action_segments):
            raise ValueError("action segments require a name and positive duration")
        if not 0.0 < self.motion_scale <= 1.0:
            raise ValueError("motion_scale must be in (0, 1]")
        if self.camera_trajectory_ue and self.trajectory_source_frame_ids:
            if len(self.camera_trajectory_ue) != len(self.trajectory_source_frame_ids):
                raise ValueError(
                    "trajectory poses and source frame ids must have equal lengths"
                )
        if (
            self.conditioning_only
            and len(self.conditioning_only) != self.duration_frames
        ):
            raise ValueError("conditioning_only must match duration_frames")
        if self.trajectory_mode not in {"canonical_action", "prescribed_pose"}:
            raise ValueError(f"unsupported trajectory_mode: {self.trajectory_mode}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrefillResult:
    """Auditable result of writing GT Context into resident model state."""

    status: Literal["accepted", "partial", "rejected"]
    accepted_frame_ids: tuple[int, ...]
    rejected_frame_ids: tuple[int, ...] = ()
    native_latent_count: int = 0
    native_chunk_count: int = 0
    history_cursor_before: int = 0
    history_cursor_after: int = 0
    last_context_pose: Mapping[str, Any] = field(default_factory=dict)
    boundary_mapping: Mapping[str, Any] = field(default_factory=dict)
    state_digest_before: str = ""
    state_digest_after: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["accepted_frame_ids"] = list(self.accepted_frame_ids)
        value["rejected_frame_ids"] = list(self.rejected_frame_ids)
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True, slots=True)
class StepResult:
    """One resident continuation chunk."""

    step_id: int
    generated_frame_count: int
    canonical_action: str
    native_action: Mapping[str, Any]
    history_cursor_before: int
    history_cursor_after: int
    last_generated_pose: Mapping[str, Any] | None = None
    state_digest: str = ""
    duration_ms: float = 0.0
    memory_selection: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    video: Any = None

    def to_dict(self, *, include_video: bool = False) -> dict[str, Any]:
        value = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if include_video or f.name != "video"
        }
        value["warnings"] = list(self.warnings)
        return value


@runtime_checkable
class ContextualWorldModelAdapter(Protocol):
    """Lifecycle shared by all four model adapters."""

    def capabilities(self) -> ModelCapabilities: ...

    def begin_case(
        self, context: ContextBundle, *, prompt: str, seed: int = 42
    ) -> None: ...

    def prefill_context(self, context: ContextBundle) -> PrefillResult: ...

    def step(self, action: ActionCommand) -> StepResult: ...

    def end_case(self) -> None: ...


def pose_mapping(pose: Sequence[float]) -> dict[str, Any]:
    """Convert the manifest UE pose vector into an audit-friendly mapping."""

    values = tuple(float(item) for item in pose)
    if len(values) != 6:
        raise ValueError(f"UE pose must contain six values, got {len(values)}")
    return {
        "x_cm": values[0],
        "y_cm": values[1],
        "z_cm": values[2],
        "pitch_deg": values[3],
        "yaw_deg": values[4],
        "roll_deg": values[5],
    }


__all__ = [
    "ActionCommand",
    "ContextBundle",
    "ContextualWorldModelAdapter",
    "ModelCapabilities",
    "PrefillResult",
    "StepResult",
    "pose_mapping",
]
