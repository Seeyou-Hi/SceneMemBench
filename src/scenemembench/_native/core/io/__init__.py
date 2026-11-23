# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from importlib import import_module

_EXPORTS = {
    "load_video_frames": "scenemembench._native.core.io.video",
    "hf_download_or_fpath": "scenemembench._native.core.io.hf",
    "resolve_hf_path": "scenemembench._native.core.io.hf",
    "write_video": "scenemembench._native.core.io.video",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name, module)
    globals()[name] = value
    return value
