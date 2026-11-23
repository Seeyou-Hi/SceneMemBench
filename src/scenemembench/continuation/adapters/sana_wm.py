# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""SANA-WM Streaming Context adapter."""

from __future__ import annotations

from ..contracts import ModelCapabilities
from .base import ResidentMemoryAdapter


class SanaWMMemoryAdapter(ResidentMemoryAdapter):
    """Adapter for SANA-WM's native Self-Forcing continuation state."""

    def __init__(self, pipeline, *, model_id: str = "sana-wm-streaming"):
        super().__init__(pipeline)
        self._model_id = model_id
        # The shared adapter key table predates SANA-WM's unified DSL.  Keep
        # SANA's native mapping explicit: a/d yaw, j/l strafe.
        self._CANONICAL_TO_KEY = dict(self._CANONICAL_TO_KEY)
        self._CANONICAL_TO_KEY.update(
            {
                "turn_left": "a",
                "turn_left_short": "a",
                "turn_right": "d",
                "turn_right_short": "d",
                "camera_left": "a",
                "camera_right": "d",
                "left": "j",
                "right": "l",
            }
        )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            prefill_mode="native_video_prefill",
            native_chunk_frames=24,
            native_fps=16,
            native_pose_format="ue_pose_to_sana_c2w_intrinsics",
            supports_gt_video_prefill=True,
            supports_persistent_step=True,
            supports_intra_step_action_segments=True,
            supports_action_motion_scale=False,
            state_observables=(
                "stage1_context_latents",
                "gdn_recurrent_state",
                "softmax_sink_recent_kv",
                "refiner_sliding_kv",
                "causal_vae_state",
                "generated_history_cursor",
            ),
            limitations=(
                "History follows native cache policy; there is no pose- or similarity-based frame retrieval.",
                "WorldFoundry replays retained clean stage-1 latents to reconstruct native caches at each interactive step.",
            ),
        )


SanaWMStreamingMemoryAdapter = SanaWMMemoryAdapter

__all__ = ["SanaWMMemoryAdapter", "SanaWMStreamingMemoryAdapter"]
