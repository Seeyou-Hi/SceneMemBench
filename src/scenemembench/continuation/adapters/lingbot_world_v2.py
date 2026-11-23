# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""LingBot-World-V2 Context adapter."""

from __future__ import annotations

from ..contracts import ModelCapabilities
from .base import ResidentMemoryAdapter


class LingBotWorldV2MemoryAdapter(ResidentMemoryAdapter):
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id="lingbot-world-v2",
            prefill_mode="native_video_prefill",
            native_chunk_frames=4,
            native_fps=16,
            native_pose_format="ue_pose_to_lingbot_c2w_plucker",
            supports_gt_video_prefill=True,
            supports_persistent_step=True,
            state_observables=(
                "causal_latent_history",
                "self_kv_cache",
                "cross_attention_cache",
                "vae_state",
            ),
        )


__all__ = ["LingBotWorldV2MemoryAdapter"]
