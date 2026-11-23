# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Minimal autocast helper extracted from WorldFoundry; platform registries omitted."""

from __future__ import annotations
from typing import Any
from contextlib import nullcontext


def autocast_context(
    device: Any,
    *,
    dtype: Any | None = None,
    enabled: bool = True,
) -> Any:
    """Return a CUDA autocast context and a no-op context for non-CUDA devices."""

    try:
        import torch
    except Exception:
        return nullcontext()

    if _device_type(device) != "cuda":
        return nullcontext()

    kwargs: dict[str, Any] = {"device_type": "cuda", "enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.amp.autocast(**kwargs)


def _device_type(device: Any) -> str:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        if isinstance(device, torch.Tensor):
            return device.device.type
        if isinstance(device, torch.device):
            return device.type
    return str(device).split(":", maxsplit=1)[0]
