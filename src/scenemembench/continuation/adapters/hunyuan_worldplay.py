# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""HY-WorldPlay Context adapter."""

from __future__ import annotations

from ..contracts import ModelCapabilities
from .base import ResidentMemoryAdapter


class HunyuanWorldPlayMemoryAdapter(ResidentMemoryAdapter):
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id="hy-worldplay",
            prefill_mode="native_video_prefill",
            native_chunk_frames=4,
            native_fps=24,
            native_pose_format="ue_pose_to_hy_relative_camera_action",
            supports_gt_video_prefill=True,
            supports_persistent_step=True,
            supports_intra_step_action_segments=True,
            supports_action_motion_scale=True,
            state_observables=(
                "latent_history",
                "cond_latent_history",
                "motion_history",
                "vae_state",
                "point_cloud_memory_bank",
                "retrieval_selection",
            ),
        )


class HunyuanWorldPlayPointCloudMemoryAdapter(HunyuanWorldPlayMemoryAdapter):
    """Resident adapter for the separate point-cloud HY model binding."""

    def capabilities(self) -> ModelCapabilities:
        value = super().capabilities()
        return ModelCapabilities(
            model_id="hy-worldplay-pointcloud",
            prefill_mode=value.prefill_mode,
            native_chunk_frames=value.native_chunk_frames,
            native_fps=value.native_fps,
            native_pose_format=value.native_pose_format,
            supports_gt_video_prefill=value.supports_gt_video_prefill,
            supports_persistent_step=value.supports_persistent_step,
            supports_intra_step_action_segments=value.supports_intra_step_action_segments,
            supports_action_motion_scale=value.supports_action_motion_scale,
            state_observables=value.state_observables,
            limitations=value.limitations,
        )


__all__ = [
    "HunyuanWorldPlayMemoryAdapter",
    "HunyuanWorldPlayPointCloudMemoryAdapter",
]
