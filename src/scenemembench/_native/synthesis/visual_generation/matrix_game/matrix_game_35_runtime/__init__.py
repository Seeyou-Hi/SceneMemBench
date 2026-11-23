# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Resident runtime boundary for Matrix-Game 3.5.

The released MG3.5 repository currently exposes a batch validation entrypoint.
WorldFoundry talks to it through the small backend protocol in ``realtime`` so
the model-owned KV/memory implementation can evolve independently from the
benchmark lifecycle.
"""

from .realtime import (
    MatrixGame35Backend,
    MatrixGame35RealtimeSession,
    MatrixGame35RuntimeError,
)

__all__ = [
    "MatrixGame35Backend",
    "MatrixGame35RealtimeSession",
    "MatrixGame35RuntimeError",
]
