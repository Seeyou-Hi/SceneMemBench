# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Shared lifecycle glue around model-owned resident pipelines."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from ..contracts import (
    ActionCommand,
    ContextBundle,
    ModelCapabilities,
    PrefillResult,
    StepResult,
    pose_mapping,
)


class ResidentMemoryAdapter(ABC):
    """Keep benchmark lifecycle uniform while leaving state ownership in runtime."""

    _CANONICAL_TO_NATIVE = {
        "forward": "forward",
        "forward_short": "forward",
        "backward": "backward",
        "turn_left": "camera_l",
        "turn_left_short": "camera_l",
        "turn_right": "camera_r",
        "turn_right_short": "camera_r",
        "camera_left": "camera_l",
        "camera_right": "camera_r",
        "look_up": "camera_up",
        "look_down": "camera_down",
        "left": "left",
        "right": "right",
        "stop": "nomove",
        "idle": "nomove",
    }
    _CANONICAL_TO_KEY = {
        "forward": "w",
        "forward_short": "w",
        "backward": "s",
        "turn_left": "j",
        "turn_left_short": "j",
        "turn_right": "l",
        "turn_right_short": "l",
        "camera_left": "j",
        "camera_right": "l",
        "look_up": "i",
        "look_down": "k",
        "left": "a",
        "right": "d",
        "stop": "q",
        "idle": "q",
    }

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.context: ContextBundle | None = None
        self._step_id = 0
        self._prompt = ""
        self._seed = 42

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    def begin_case(
        self, context: ContextBundle, *, prompt: str, seed: int = 42
    ) -> None:
        from PIL import Image

        self.context = context
        self._step_id = 0
        self._prompt = str(prompt or "")
        self._seed = int(seed)
        with Image.open(context.rgb_paths[0]) as image:
            first_frame = image.convert("RGB")
        retrieval_config = context.metadata.get("point_cloud_retrieval")
        if retrieval_config is None:
            retrieval_config = context.metadata.get("worldplay_retrieval_config")
        retrieval_runtime = context.metadata.get("point_cloud_runtime")
        if retrieval_runtime is None:
            retrieval_runtime = context.metadata.get("worldplay_point_cloud_runtime")
        # The point-cloud binding owns the UCM geometry runtime.  Context
        # manifests from older runs do not carry this flag, so enable it only
        # for the dedicated point-cloud adapter and keep native HY unchanged.
        if self.capabilities().model_id == "hy-worldplay-pointcloud":
            retrieval_config = dict(retrieval_config or {})
            retrieval_config.setdefault("retrieval_unit", "latent")
            retrieval_config.setdefault("latents_per_query", 4)
            retrieval_config.setdefault("ablation", "fused_coverage")
            retrieval_config.setdefault("initial_chunks", 1)
            retrieval_config.setdefault("temporal_chunks", 3)
            retrieval_config.setdefault("ucm_warp_rope_enabled", True)
        self.pipeline.configure_realtime(
            images=first_frame,
            prompt=self._prompt,
            seed=self._seed,
            fps=self.capabilities().native_fps,
            horizontal_fov_deg=context.horizontal_fov_deg,
            point_cloud_config=retrieval_config,
            point_cloud_runtime=retrieval_runtime,
        )

    def prefill_context(self, context: ContextBundle) -> PrefillResult:
        if self.context is None or self.context.case_id != context.case_id:
            raise RuntimeError("begin_case must be called before prefill_context")
        prefill = getattr(self.pipeline, "prefill_context_realtime", None)
        if not callable(prefill):
            raise RuntimeError(
                f"{self.capabilities().model_id} pipeline does not expose prefill_context_realtime"
            )
        return PrefillResult(
            **prefill_context_kwargs(
                prefill, context, native_fps=self.capabilities().native_fps
            )
        )

    def step(self, action: ActionCommand) -> StepResult:
        if self.context is None:
            raise RuntimeError("begin_case must be called before step")
        started = time.perf_counter()
        native_action = self._CANONICAL_TO_NATIVE.get(
            action.canonical_action.strip().lower(),
            action.canonical_action.strip().lower(),
        )
        canonical_segments = action.action_segments or (
            (action.canonical_action, float(action.duration_frames)),
        )
        realtime_segments = []
        for segment_action, segment_duration in canonical_segments:
            segment_native = self._CANONICAL_TO_NATIVE.get(
                segment_action.strip().lower(), segment_action.strip().lower()
            )
            segment_key = self._CANONICAL_TO_KEY.get(
                segment_action.strip().lower(), segment_native
            )
            realtime_segments.append(
                {
                    "duration": float(segment_duration)
                    / float(self.capabilities().native_fps),
                    "keys": [segment_key],
                }
            )
        stream_kwargs = {
            "interactions": [native_action],
            "realtime_segments": realtime_segments,
            "action_motion_scale": action.motion_scale,
            "rotation_deg": action.rotation_deg,
            "seed": self._seed + action.step_id,
        }
        # Prescribed trajectories are optional.  Existing realtime pipelines
        # accept and ignore these kwargs until their model-specific pose branch
        # is enabled; discrete actions therefore retain their old behavior.
        if action.camera_trajectory_ue:
            stream_kwargs.update(
                camera_trajectory_ue=action.camera_trajectory_ue,
                trajectory_source_frame_ids=action.trajectory_source_frame_ids,
                phase_names=action.phase_names,
                trajectory_mode=action.trajectory_mode,
                position_units=str(
                    self.context.metadata.get("pose_format", {}).get(
                        "position_units", "m"
                    )
                ),
            )
        result = self.pipeline.stream_realtime(**stream_kwargs)
        if result is None:
            try:
                import torch.distributed as dist

                distributed_nonzero_rank = (
                    dist.is_available()
                    and dist.is_initialized()
                    and dist.get_rank() != 0
                )
            except ImportError:
                distributed_nonzero_rank = False
            if not distributed_nonzero_rank:
                raise RuntimeError(
                    f"{self.capabilities().model_id} returned no step result"
                )
            result = {
                "video": None,
                "realtime_metrics": {
                    "history_cursor_before": action.step_id,
                    "history_cursor_after": action.step_id + 1,
                    "warnings": ["rank_zero_only_result"],
                },
            }
        metrics = (
            result.get("realtime_metrics", {}) if isinstance(result, Mapping) else {}
        )
        generated = result.get("video") if isinstance(result, Mapping) else None
        if generated is None:
            generated_count = int(metrics.get("generated_frame_count", 0))
        else:
            shape = getattr(generated, "shape", None)
            generated_count = (
                int(shape[0]) if shape is not None else int(len(generated))
            )
        cursor_before = int(
            metrics.get(
                "history_cursor_before", metrics.get("ar_index", action.step_id)
            )
        )
        cursor_after = int(
            metrics.get("history_cursor_after", cursor_before + generated_count)
        )
        memory_keys = (
            "selected_history_latent_indices",
            "anchor_memory_latent_indices",
            "retrieved_memory_latent_indices",
            "temporal_context_latent_indices",
            "selected_context_latent_indices",
            "selected_context_memory_bank_frame_ids",
            "selected_reference_frame_ids",
            "selected_generated_latent_indices",
            "selected_context_pose_source_frame_ids",
            "memory_bank_context_latent_indices",
            "memory_bank_context_pose_source_frame_ids",
            "retrieved_context_memory_latent_indices",
            "retrieved_context_memory_pose_source_frame_ids",
            "memory_bank_read",
            "retrieved_context_memory_bank_hit",
            "context_latent_count",
            "memory_query_frame_indices",
            "selected_history_frame_indices",
            "memory_fov_confidences",
            "memory_candidate_frame_range",
            "fov_retrieval",
            "context_slot_roles",
            "context_slot_attention_weights",
            "camera_condition",
            "previous_vision_cache_released",
            "cuda_allocated_mb_before_vision_release",
            "cuda_allocated_mb_after_vision_release",
            "cuda_allocated_mb_after_vision_rebuild",
            "point_cloud_retrieval",
        )
        memory_selection = (
            dict(result.get("memory_selection", {}))
            if isinstance(result, Mapping)
            else {}
        )
        memory_selection.update(
            {key: metrics[key] for key in memory_keys if key in metrics}
        )
        self._step_id = max(self._step_id, action.step_id + 1)
        return StepResult(
            step_id=action.step_id,
            generated_frame_count=generated_count,
            canonical_action=action.canonical_action,
            native_action={
                "interactions": [native_action],
                "segments": realtime_segments,
                "trajectory_mode": action.trajectory_mode,
                "trajectory_source_frame_ids": list(action.trajectory_source_frame_ids),
                "phase_names": list(action.phase_names),
                "conditioning_only": list(action.conditioning_only),
                "motion_scale": action.motion_scale,
                "source": action.source,
            },
            history_cursor_before=cursor_before,
            history_cursor_after=cursor_after,
            last_generated_pose=(
                pose_mapping(action.target_pose_ue) if action.target_pose_ue else None
            ),
            state_digest=str(metrics.get("state_digest", "")),
            duration_ms=float(
                metrics.get("total_ms", (time.perf_counter() - started) * 1000.0)
            ),
            memory_selection=memory_selection,
            warnings=tuple(str(item) for item in metrics.get("warnings", ())),
            video=generated,
        )

    def end_case(self) -> None:
        reset = getattr(self.pipeline, "reset_realtime", None)
        if callable(reset):
            reset()
        self.context = None


def prefill_context_kwargs(
    prefill: Any,
    context: ContextBundle,
    *,
    native_fps: int,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a model-owned method and normalize its result into PrefillResult kwargs."""

    native_indices = _native_sample_indices(context, native_fps)
    call_kwargs = {
        "rgb_paths": [context.rgb_paths[index] for index in native_indices],
        "camera_to_world": [context.camera_to_world[index] for index in native_indices],
        "frame_ids": [context.frame_ids[index] for index in native_indices],
        "fps": native_fps,
        "horizontal_fov_deg": context.horizontal_fov_deg,
        "position_units": str(
            context.metadata.get("pose_format", {}).get("position_units", "m")
        ),
        "coordinate_system": str(
            context.metadata.get("pose_format", {}).get("coordinate_system", "")
        ),
        "handedness": str(
            context.metadata.get("pose_format", {}).get("handedness", "")
        ),
    }
    call_kwargs.update(dict(extra_kwargs or {}))
    result = prefill(**call_kwargs)
    if not isinstance(result, Mapping):
        raise TypeError("prefill_context_realtime must return a mapping")
    accepted = tuple(
        int(item) for item in result.get("accepted_frame_ids", context.frame_ids)
    )
    rejected = tuple(int(item) for item in result.get("rejected_frame_ids", ()))
    warnings = tuple(str(item) for item in result.get("warnings", ()))
    last_pose = result.get("last_context_pose") or pose_mapping(
        context.last_context_pose
    )
    boundary_mapping = dict(result.get("boundary_mapping", {}))
    boundary_mapping["source_context_fps"] = context.fps
    boundary_mapping["native_context_fps"] = native_fps
    boundary_mapping["source_frame_count"] = len(context.frame_ids)
    boundary_mapping["native_sampled_frame_count"] = len(native_indices)
    if len(native_indices) != len(context.frame_ids):
        boundary_mapping["native_sampled_source_indices"] = native_indices
        warnings = (*warnings, f"resampled_context_fps={context.fps:g}->{native_fps}")
    status = str(result.get("status", "accepted"))
    if len(native_indices) != len(context.frame_ids) and status == "accepted":
        status = "partial"
    return {
        "status": status,
        "accepted_frame_ids": accepted,
        "rejected_frame_ids": rejected,
        "native_latent_count": int(result.get("native_latent_count", 0)),
        "native_chunk_count": int(result.get("native_chunk_count", 0)),
        "history_cursor_before": int(result.get("history_cursor_before", 0)),
        "history_cursor_after": int(result.get("history_cursor_after", 0)),
        "last_context_pose": last_pose,
        "boundary_mapping": boundary_mapping,
        "state_digest_before": str(result.get("state_digest_before", "")),
        "state_digest_after": str(result.get("state_digest_after", "")),
        "warnings": warnings,
    }


def _native_sample_indices(context: ContextBundle, native_fps: int) -> list[int]:
    """Choose deterministic nearest source frames for a model-native cadence."""

    if abs(context.fps - native_fps) < 1e-6:
        return list(range(len(context.frame_ids)))
    start = context.timestamps_sec[0]
    end = context.timestamps_sec[-1]
    native_count = max(1, int(round((end - start) * native_fps)) + 1)
    indices: list[int] = []
    for native_index in range(native_count):
        target = start + native_index / float(native_fps)
        source_index = min(
            range(len(context.timestamps_sec)),
            key=lambda index: abs(context.timestamps_sec[index] - target),
        )
        if not indices or source_index != indices[-1]:
            indices.append(source_index)
    if indices[-1] != len(context.frame_ids) - 1:
        indices.append(len(context.frame_ids) - 1)
    return indices


__all__ = ["ResidentMemoryAdapter"]
