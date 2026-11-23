# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Independent endpoint depth and normal metrics extracted from WorldFoundry."""

from __future__ import annotations
from typing import Any
import numpy as np


def robust_depth_scale(
    reference: np.ndarray, estimate: np.ndarray
) -> tuple[float, int]:
    """Return the robust scalar s for reference ~= s * estimate.

    DA3's relative/metric branches can differ by a global scale after pose
    alignment.  A median ratio is deliberately used here instead of fitting a
    free affine transform, so scene structure is not removed by the metric.
    """
    valid = (
        np.isfinite(reference)
        & np.isfinite(estimate)
        & (reference > 1e-4)
        & (estimate > 1e-4)
    )
    if not valid.any():
        return 1.0, 0
    ratio = reference[valid].astype(np.float64) / estimate[valid].astype(np.float64)
    ratio = ratio[np.isfinite(ratio) & (ratio > 1e-4)]
    if not len(ratio):
        return 1.0, 0
    # Trim extreme sky/occlusion ratios before taking the median.
    lo, hi = (
        np.percentile(ratio, [10, 90])
        if len(ratio) >= 10
        else (ratio.min(), ratio.max())
    )
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    return float(np.median(trimmed if len(trimmed) else ratio)), int(valid.sum())


def camera_aware_normal_map(
    depth: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals from a depth map after pinhole back-projection."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.asarray(depth, dtype=np.float32)
    depth_valid = np.isfinite(d) & (d > 1e-4)
    x = (xx.astype(np.float32) - float(k[0, 2])) / max(float(k[0, 0]), 1e-6) * d
    y = (yy.astype(np.float32) - float(k[1, 2])) / max(float(k[1, 1]), 1e-6) * d
    points = np.stack([x, y, d], axis=-1)
    normals = np.full((h, w, 3), np.nan, dtype=np.float32)
    if h < 3 or w < 3:
        return normals, np.zeros((h, w), dtype=bool)
    center_valid = (
        depth_valid[1:-1, 1:-1]
        & depth_valid[1:-1, :-2]
        & depth_valid[1:-1, 2:]
        & depth_valid[:-2, 1:-1]
        & depth_valid[2:, 1:-1]
    )
    tangent_x = points[1:-1, 2:] - points[1:-1, :-2]
    tangent_y = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(tangent_x, tangent_y)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-8)
    center_points = points[1:-1, 1:-1]
    flip = np.sum(n * (-center_points), axis=-1) < 0
    n[flip] *= -1.0
    normal_valid_inner = (
        center_valid & np.isfinite(n).all(axis=-1) & (n_norm[..., 0] > 1e-8)
    )
    normal_valid = np.zeros((h, w), dtype=bool)
    normal_valid[1:-1, 1:-1] = normal_valid_inner
    normals[1:-1, 1:-1] = n
    return normals, normal_valid


def endpoint_scale_invariant_depth_metrics(
    gt_depth: np.ndarray, gen_depth: np.ndarray
) -> dict[str, Any]:
    """Fit one scalar on an endpoint pair and compute coverage-aware SI-AbsRel."""
    gt_valid = np.isfinite(gt_depth) & (gt_depth > 1e-4)
    gen_valid = np.isfinite(gen_depth) & (gen_depth > 1e-4)
    joint = gt_valid & gen_valid
    gt_count = int(gt_valid.sum())
    joint_count = int(joint.sum())
    invalid_count = gt_count - joint_count
    scale, scale_count = robust_depth_scale(gt_depth, gen_depth)
    output: dict[str, Any] = {
        "endpoint_depth_scale_gt_over_gen": float(scale),
        "scale_fit_valid_pixels": int(scale_count),
        "gt_valid_pixels": gt_count,
        "joint_valid_pixels": joint_count,
        "valid_coverage": float(joint_count / gt_count) if gt_count else 0.0,
        "invalid_gen_fraction": float(invalid_count / gt_count) if gt_count else 1.0,
        "endpoint_si_absrel": None,
        "endpoint_si_absrel_valid_only": None,
        "raw_absrel_diagnostic": None,
    }
    if not gt_count or not joint_count:
        return output
    gt = gt_depth[joint].astype(np.float64)
    generated = gen_depth[joint].astype(np.float64)
    aligned_error = np.abs(gt - float(scale) * generated) / np.maximum(gt, 1e-6)
    raw_error = np.abs(gt - generated) / np.maximum(gt, 1e-6)
    # Invalid generated pixels contribute a fixed unit error and therefore
    # cannot improve a model by disappearing from the valid intersection.
    output["endpoint_si_absrel"] = float(
        (aligned_error.sum() + invalid_count) / gt_count
    )
    output["endpoint_si_absrel_valid_only"] = float(aligned_error.mean())
    output["raw_absrel_diagnostic"] = float(
        (raw_error.sum() + invalid_count) / gt_count
    )
    return output
