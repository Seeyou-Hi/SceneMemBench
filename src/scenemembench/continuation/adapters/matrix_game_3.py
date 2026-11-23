# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Matrix-Game-3 Context adapter."""

from __future__ import annotations

from ..contracts import ModelCapabilities
from .base import ResidentMemoryAdapter


class MatrixGame3MemoryAdapter(ResidentMemoryAdapter):
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id="matrix-game-3",
            prefill_mode="native_video_prefill",
            native_chunk_frames=4,
            native_fps=17,
            native_pose_format="ue_pose_to_matrix_plucker_and_latent_memory",
            supports_gt_video_prefill=True,
            supports_persistent_step=True,
            supports_intra_step_action_segments=True,
            supports_action_motion_scale=True,
            state_observables=(
                "latent_history",
                "camera_history",
                "action_history",
                "streaming_vae_state",
            ),
            limitations=(
                "Matrix-Game-3 has explicit latent memory rather than transformer KV prefill.",
            ),
        )


__all__ = ["MatrixGame3MemoryAdapter"]
