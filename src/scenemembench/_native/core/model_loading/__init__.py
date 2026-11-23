# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from importlib import import_module

_EXPORTS = {
    "load_model": "scenemembench._native.core.model_loading.model",
    "load_state_dict": "scenemembench._native.core.model_loading.file",
    "load_torch_checkpoint": "scenemembench._native.core.model_loading.file",
    "load_torch_state_dict": "scenemembench._native.core.model_loading.file",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name, module)
    globals()[name] = value
    return value
