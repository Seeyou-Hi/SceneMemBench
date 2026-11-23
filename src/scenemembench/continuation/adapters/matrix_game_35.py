# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Matrix-Game-3.5 Context adapter."""

from __future__ import annotations

from ..contracts import ModelCapabilities
from .base import ResidentMemoryAdapter


class MatrixGame35MemoryAdapter(ResidentMemoryAdapter):
    """WorldFoundry lifecycle adapter for the MG3.5 resident pipeline."""

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id="matrix-game-3.5",
            prefill_mode="native_video_prefill",
            native_chunk_frames=12,
            native_fps=17,
            native_pose_format="extrinsics_c2w_metric_plus_patch_memory",
            supports_gt_video_prefill=True,
            supports_persistent_step=True,
            supports_intra_step_action_segments=True,
            supports_action_motion_scale=True,
            state_observables=(
                "rolling_kv_cache",
                "rolling_latent_history",
                "patch_memory_bank",
                "dynamic_context_pool",
                "streaming_vae_state",
            ),
            limitations=(
                "The checked-in native driver registers all GT Context frames into the "
                "Patch/Mosaic candidate bank; rolling KV still uses the native "
                "budgeted per-chunk context selector.",
                "Initial integration targets the distilled first-person checkpoint.",
                "Base and third-person checkpoints are not declared causal residents.",
            ),
        )


__all__ = ["MatrixGame35MemoryAdapter"]
