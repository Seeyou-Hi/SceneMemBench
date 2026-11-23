# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compatibility exports for WorldFoundry's shared attention implementation."""

from scenemembench._native.core.attention.varlen import attention, flash_attention

__all__ = ["attention", "flash_attention"]
