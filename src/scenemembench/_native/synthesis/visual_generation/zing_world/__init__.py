# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Zing-0.5 official runtime bridge and resident continuation session."""

from .realtime import ZingWorldRealtimeSession

__all__ = ["ZingWorldRealtimeSession"]
