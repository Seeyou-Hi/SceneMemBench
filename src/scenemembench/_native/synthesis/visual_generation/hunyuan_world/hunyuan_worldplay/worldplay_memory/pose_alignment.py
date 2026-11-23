# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Small, service-independent pose alignment primitives used during inference."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _poses(value: Sequence[np.ndarray], name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N,4,4], got {poses.shape}")
    return poses


def apply_sim3_to_c2w(
    c2w_poses: Sequence[np.ndarray],
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> List[np.ndarray]:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    result: List[np.ndarray] = []
    for pose in c2w_poses:
        pose = np.asarray(pose, dtype=np.float64)
        aligned = np.eye(4, dtype=np.float64)
        aligned[:3, :3] = rotation @ pose[:3, :3]
        aligned[:3, 3] = float(scale) * (rotation @ pose[:3, 3]) + translation
        result.append(aligned)
    return result


def fit_sim3_c2w_to_target(
    source_c2w: Sequence[np.ndarray],
    target_c2w: Sequence[np.ndarray],
    *,
    orientation_weight: float = 1.0,
) -> Tuple[Dict[str, object], List[np.ndarray]]:
    """Fit a similarity transform while using camera orientation in degenerate paths."""
    source = _poses(source_c2w, "source_c2w")
    target = _poses(target_c2w, "target_c2w")
    count = min(len(source), len(target))
    if count == 0:
        raise ValueError("Sim3 alignment requires at least one pose pair")
    source_fit = source[:count]
    target_fit = target[:count]
    source_points = source_fit[:, :3, 3]
    target_points = target_fit[:, :3, 3]
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source0 = source_points - source_mean
    target0 = target_points - target_mean

    correlation = target0.T @ source0
    position_norm = float(np.linalg.norm(correlation))
    if position_norm > 1e-10:
        correlation /= position_norm
    else:
        correlation[:] = 0.0
    orientation = np.einsum("nij,nkj->ik", target_fit[:, :3, :3], source_fit[:, :3, :3])
    orientation_norm = float(np.linalg.norm(orientation))
    if orientation_norm > 1e-10 and orientation_weight > 0:
        correlation += float(orientation_weight) * orientation / orientation_norm
    if float(np.linalg.norm(correlation)) <= 1e-10:
        rotation = np.eye(3, dtype=np.float64)
    else:
        u, _, vt = np.linalg.svd(correlation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt

    denom = float((source0 * source0).sum())
    position_degenerate = denom < 1e-10
    scale = (
        1.0
        if position_degenerate
        else max(float(((source0 @ rotation.T) * target0).sum() / denom), 1e-8)
    )
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = apply_sim3_to_c2w(
        source, scale=scale, rotation=rotation, translation=translation
    )
    aligned_centers = np.asarray([pose[:3, 3] for pose in aligned[:count]])
    rmse = float(
        np.sqrt(np.mean(np.sum((aligned_centers - target_points) ** 2, axis=1)))
    )
    debug: Dict[str, object] = {
        "enabled": True,
        "fit_count": int(count),
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "center_rmse": rmse,
        "position_degenerate": bool(position_degenerate),
    }
    return debug, aligned


def anchor_aligned_c2w_to_observed_source(
    aligned_c2w: Sequence[np.ndarray],
    *,
    aligned_source_c2w: np.ndarray,
    observed_source_c2w: np.ndarray,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Anchor a Sim3-aligned block to the latest observed LingBot pose."""
    aligned_source = np.asarray(aligned_source_c2w, dtype=np.float64)
    observed_source = np.asarray(observed_source_c2w, dtype=np.float64)
    source_residual = observed_source @ np.linalg.inv(aligned_source)
    anchored = [
        source_residual @ np.asarray(pose, dtype=np.float64) for pose in aligned_c2w
    ]
    residual_rotation = source_residual[:3, :3]
    residual_cos = float(np.clip((np.trace(residual_rotation) - 1.0) * 0.5, -1.0, 1.0))
    return anchored, {
        "source_anchor_translation": float(np.linalg.norm(source_residual[:3, 3])),
        "source_anchor_rotation_deg": float(np.degrees(np.arccos(residual_cos))),
        "source_residual_c2w": source_residual.tolist(),
    }


def preserve_translation_magnitude(
    command_delta: np.ndarray, aligned_delta: np.ndarray
) -> np.ndarray:
    """Use the aligned direction/rotation without inheriting its Sim3 scale."""
    command_delta = np.asarray(command_delta, dtype=np.float64)
    aligned_delta = np.asarray(aligned_delta, dtype=np.float64)
    result = aligned_delta.copy()
    wanted = float(np.linalg.norm(command_delta[:3, 3]))
    direction = aligned_delta[:3, 3]
    norm = float(np.linalg.norm(direction))
    if wanted <= 1e-12:
        result[:3, 3] = 0.0
    elif norm > 1e-12:
        result[:3, 3] = direction * (wanted / norm)
    else:
        result[:3, 3] = command_delta[:3, 3]
    return result


def remap_lingbot_anchor_to_generation(
    lingbot_history: Sequence[np.ndarray],
    generation_history: Sequence[np.ndarray],
    latest_lingbot_c2w: np.ndarray,
    generation_base_c2w: np.ndarray,
    *,
    max_rmse: float = 0.75,
    reject_correction: float = 2.0,
    max_correction: float = 0.32,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Map LingBot translation into generation space, then gate and cap correction."""
    debug, aligned = fit_sim3_c2w_to_target(lingbot_history, generation_history)
    latest = apply_sim3_to_c2w(
        [latest_lingbot_c2w],
        scale=float(debug["scale"]),
        rotation=np.asarray(debug["rotation"]),
        translation=np.asarray(debug["translation"]),
    )[0]
    base = np.asarray(generation_base_c2w, dtype=np.float64)
    correction = latest[:3, 3] - base[:3, 3]
    correction_norm = float(np.linalg.norm(correction))
    accepted = (
        float(debug["center_rmse"]) <= max_rmse and correction_norm <= reject_correction
    )
    result = base.copy()
    if accepted and correction_norm > 0:
        correction *= min(1.0, float(max_correction) / correction_norm)
        result[:3, 3] += correction
    debug.update(
        {
            "accepted": bool(accepted),
            "correction_norm": correction_norm,
            "applied_correction": correction.tolist() if accepted else [0.0, 0.0, 0.0],
        }
    )
    return result, debug
