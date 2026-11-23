# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""Dataset registry stub retained for mmcv ``build_from_cfg`` compatibility."""

try:
    from mmcv import Registry
except ImportError:  # mmcv-lite >= 2 exposes registries through mmengine.
    from mmengine import Registry

DATASETS = Registry("datasets")
