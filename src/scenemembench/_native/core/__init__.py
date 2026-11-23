# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from importlib import import_module

_EXPORTS = {
    "autocast_context": "scenemembench._native.core.inference",
    "cuda_visible_devices_from_device": "scenemembench._native.core.device",
    "load_video_frames": "scenemembench._native.core.io",
    "load_pil_image": "scenemembench._native.core.utils",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name, module)
    globals()[name] = value
    return value
