# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldPlay memory selection backends used by the resident runtime.

The point-cloud selector is vendored from the HY-WorldPlay inference path so
WorldFoundry can own the lifecycle and still preserve the upstream retrieval
semantics.  Imports stay lazy at call sites because the regular FOV-only path
must not require any optional point-cloud assets.
"""

from .point_cloud_retrieval import PointCloudRetrievalConfig

__all__ = ["PointCloudRetrievalConfig"]
