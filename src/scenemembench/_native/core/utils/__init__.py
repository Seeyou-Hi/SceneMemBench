# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from importlib import import_module

_EXPORTS = {"load_pil_image": "scenemembench._native.core.utils.image_utils"}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name, module)
    globals()[name] = value
    return value
