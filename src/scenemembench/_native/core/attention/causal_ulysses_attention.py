# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from scenemembench._native.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_attention import (
    flash_attention,
)
from scenemembench._native.core.attention.ulysses_attention import (
    distributed_attention as _distributed_attention,
)


def distributed_attention(
    q,
    k,
    v,
    seq_lens,
    window_size=(-1, -1),
):
    """Run Ulysses attention with the causal LingBot attention kernel."""
    return _distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=window_size,
        attention_fn=flash_attention,
    )
