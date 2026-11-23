# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from importlib import import_module

_EXPORTS = {
    "load_tensor_state_dict": "scenemembench._native.core.checkpoint.safe_loading",
    "load_weights_only": "scenemembench._native.core.checkpoint.safe_loading",
    "require_mapping": "scenemembench._native.core.checkpoint.safe_loading",
    "require_tensor": "scenemembench._native.core.checkpoint.safe_loading",
    "tensor_state_dict": "scenemembench._native.core.checkpoint.safe_loading",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name, module)
    globals()[name] = value
    return value
