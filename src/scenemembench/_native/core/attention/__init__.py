# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from importlib import import_module

_EXPORTS = {
    "flash_attention": "scenemembench._native.core.attention.varlen",
    "apply_nd_rotary_embedding": "scenemembench._native.core.attention.rope_nd",
    "apply_rotary_embedding": "scenemembench._native.core.attention.rope",
    "get_1d_rotary_pos_embed": "scenemembench._native.core.attention.rope_nd",
    "get_nd_rotary_pos_embed": "scenemembench._native.core.attention.rope_nd",
    "attention": "scenemembench._native.core.attention.varlen",
    "scaled_dot_product_attention": "scenemembench._native.core.attention.native",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name, module)
    globals()[name] = value
    return value
