# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Trajectory state shared by online retrieval and future interactive generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pose_alignment import (
    anchor_aligned_c2w_to_observed_source,
    apply_sim3_to_c2w,
    fit_sim3_c2w_to_target,
)


def _copy_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"pose must be a finite 4x4 matrix, got {pose.shape}")
    return pose.copy()


def _translation_scale(
    source: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    *,
    end: int,
    window: int,
    scale_min: float,
    scale_max: float,
    min_source_step: float,
) -> Tuple[float, Dict[str, object]]:
    end = min(int(end), len(source), len(target))
    begin = max(1, end - max(1, int(window)))
    ratios: List[float] = []
    rejected_small_source_steps = 0
    for idx in range(begin, end):
        source_step = float(np.linalg.norm(source[idx][:3, 3] - source[idx - 1][:3, 3]))
        target_step = float(np.linalg.norm(target[idx][:3, 3] - target[idx - 1][:3, 3]))
        if source_step < float(min_source_step):
            rejected_small_source_steps += 1
            continue
        if target_step > 1e-8:
            ratios.append(target_step / source_step)
    if ratios:
        raw_scale = float(np.median(ratios))
        scale = float(np.clip(raw_scale, float(scale_min), float(scale_max)))
    else:
        raw_scale = 1.0
        scale = 1.0
    return scale, {
        "local_scale": scale,
        "local_scale_raw": raw_scale,
        "local_scale_count": int(len(ratios)),
        "local_scale_window": int(window),
        "local_scale_begin": int(begin),
        "local_scale_end": int(end),
        "local_scale_min_source_step": float(min_source_step),
        "local_scale_rejected_small_source_steps": int(rejected_small_source_steps),
    }


def _scale_delta_translation(delta: np.ndarray, scale: float) -> np.ndarray:
    result = np.asarray(delta, dtype=np.float64).copy()
    result[:3, 3] *= float(scale)
    return result


@dataclass
class TrajectoryState:
    chunk_size: int = 4
    command_pose_history: List[np.ndarray] = field(default_factory=list)
    generation_pose_history: List[np.ndarray] = field(default_factory=list)
    lingbot_pose_history: List[np.ndarray] = field(default_factory=list)
    chunk_index: int = 0
    last_motion_key: Optional[str] = None
    last_alignment: Dict[str, object] = field(default_factory=dict)
    frozen_prefix_sim3_alignment: Dict[str, object] = field(default_factory=dict)
    trusted_prefix_sim3_anchor: Dict[str, object] = field(default_factory=dict)
    last_ucm_p2l_source: Optional[int] = None
    last_ucm_p2l_sources: List[Optional[int]] = field(default_factory=list)

    @classmethod
    def from_w2c(
        cls,
        command_w2c: Sequence[np.ndarray],
        generation_w2c: Optional[Sequence[np.ndarray]] = None,
    ) -> "TrajectoryState":
        command = [_copy_pose(np.linalg.inv(pose)) for pose in command_w2c]
        generation = (
            command
            if generation_w2c is None
            else [_copy_pose(np.linalg.inv(pose)) for pose in generation_w2c]
        )
        return cls(
            command_pose_history=[pose.copy() for pose in command],
            generation_pose_history=[pose.copy() for pose in generation],
        )

    def update_lingbot_poses(
        self,
        lingbot_poses: Sequence[np.ndarray],
        available: int,
        *,
        convention: str = "w2c",
    ) -> None:
        """Normalize decoded LingBot extrinsics to physical C2W trajectory poses."""
        convention = str(convention).strip().lower()
        if convention not in {"w2c", "c2w"}:
            raise ValueError(f"unsupported LingBot pose convention: {convention}")
        available = min(int(available), len(lingbot_poses))
        self.lingbot_pose_history = [
            _copy_pose(
                np.asarray(lingbot_poses[idx], dtype=np.float64)
                if convention == "c2w"
                else np.linalg.inv(np.asarray(lingbot_poses[idx], dtype=np.float64))
            )
            for idx in range(available)
        ]

    def update_lingbot_w2c(
        self, lingbot_w2c: Sequence[np.ndarray], available: int
    ) -> None:
        """Backward-compatible wrapper for decoded LingBot W2C matrices."""
        self.update_lingbot_poses(lingbot_w2c, available, convention="w2c")

    def query_c2w_in_lingbot_space(
        self,
        start: int,
        count: int,
        *,
        alignment_mode: str = "prefix_sim3",
        local_window: int = 64,
        scale_min: float = 0.05,
        scale_max: float = 4.0,
        min_command_step: float = 1e-4,
        orientation_weight: float = 1.0,
        min_pairs: int = 1,
        anchor_max_translation_step: float = 0.0,
        anchor_max_rotation_step_deg: float = 30.0,
    ) -> Tuple[List[np.ndarray], Dict[str, object]]:
        stop = min(start + count, len(self.command_pose_history))
        query = self.command_pose_history[start:stop]
        fit_count = min(start, len(self.lingbot_pose_history))
        if alignment_mode == "prefix_sim3_anchor" and fit_count >= int(min_pairs):
            calibration_created = not bool(self.frozen_prefix_sim3_alignment)
            if calibration_created:
                calibration, _ = fit_sim3_c2w_to_target(
                    self.command_pose_history[:fit_count],
                    self.lingbot_pose_history[:fit_count],
                    orientation_weight=orientation_weight,
                )
                raw_scale = float(calibration["scale"])
                frozen_scale = float(
                    np.clip(raw_scale, float(scale_min), float(scale_max))
                )
                rotation = np.asarray(calibration["rotation"], dtype=np.float64)
                source_centers = np.asarray(
                    [pose[:3, 3] for pose in self.command_pose_history[:fit_count]],
                    dtype=np.float64,
                )
                target_centers = np.asarray(
                    [pose[:3, 3] for pose in self.lingbot_pose_history[:fit_count]],
                    dtype=np.float64,
                )
                translation = target_centers.mean(axis=0) - frozen_scale * (
                    rotation @ source_centers.mean(axis=0)
                )
                aligned_centers = (
                    frozen_scale * (source_centers @ rotation.T) + translation
                )
                calibration.update(
                    {
                        "scale": frozen_scale,
                        "scale_raw": raw_scale,
                        "translation": translation.tolist(),
                        "center_rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (aligned_centers - target_centers) ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        ),
                        "calibration_fit_count": int(fit_count),
                        "calibration_start": 0,
                    }
                )
                self.frozen_prefix_sim3_alignment = dict(calibration)
            calibration = dict(self.frozen_prefix_sim3_alignment)
            base_idx = min(start - 1, fit_count - 1, len(self.command_pose_history) - 1)
            command_base = self.command_pose_history[base_idx]
            observed_base = self.lingbot_pose_history[base_idx]
            aligned_query = apply_sim3_to_c2w(
                query,
                scale=float(calibration["scale"]),
                rotation=np.asarray(calibration["rotation"]),
                translation=np.asarray(calibration["translation"]),
            )
            aligned_base = apply_sim3_to_c2w(
                [command_base],
                scale=float(calibration["scale"]),
                rotation=np.asarray(calibration["rotation"]),
                translation=np.asarray(calibration["translation"]),
            )[0]
            candidate_aligned, anchor_debug = anchor_aligned_c2w_to_observed_source(
                aligned_query,
                aligned_source_c2w=aligned_base,
                observed_source_c2w=observed_base,
            )
            candidate_residual = np.asarray(
                anchor_debug["source_residual_c2w"], dtype=np.float64
            )
            trusted_payload = self.trusted_prefix_sim3_anchor
            trusted_residual = (
                np.asarray(trusted_payload["source_residual_c2w"], dtype=np.float64)
                if trusted_payload.get("source_residual_c2w") is not None
                else None
            )
            if trusted_residual is None:
                anchor_step_translation = 0.0
                anchor_step_rotation_deg = 0.0
            else:
                anchor_step = candidate_residual @ np.linalg.inv(trusted_residual)
                anchor_step_translation = float(np.linalg.norm(anchor_step[:3, 3]))
                step_cos = float(
                    np.clip((np.trace(anchor_step[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
                )
                anchor_step_rotation_deg = float(np.degrees(np.arccos(step_cos)))
            translation_rejected = float(
                anchor_max_translation_step
            ) > 0 and anchor_step_translation > float(anchor_max_translation_step)
            rotation_rejected = float(
                anchor_max_rotation_step_deg
            ) > 0 and anchor_step_rotation_deg > float(anchor_max_rotation_step_deg)
            anchor_accepted = not (translation_rejected or rotation_rejected)
            rejection_reasons = []
            if translation_rejected:
                rejection_reasons.append("translation_step")
            if rotation_rejected:
                rejection_reasons.append("rotation_step")
            if anchor_accepted:
                applied_residual = candidate_residual
                aligned = candidate_aligned
                self.trusted_prefix_sim3_anchor = {
                    "source_residual_c2w": candidate_residual.tolist(),
                    "anchor_index": int(base_idx),
                }
            else:
                assert trusted_residual is not None
                applied_residual = trusted_residual
                aligned = [trusted_residual @ pose for pose in aligned_query]
            applied_rotation = applied_residual[:3, :3]
            applied_cos = float(
                np.clip((np.trace(applied_rotation) - 1.0) * 0.5, -1.0, 1.0)
            )
            debug = dict(calibration)
            debug.update(anchor_debug)
            debug.update(
                {
                    "fit_count": int(fit_count),
                    "calibration_frozen": True,
                    "calibration_created": bool(calibration_created),
                    "anchor_index": int(base_idx),
                    "orientation_weight": float(orientation_weight),
                    "min_pairs": int(min_pairs),
                    "query_pose_source": (
                        "command_frozen_context_sim3_to_lingbot_with_latest_rigid_anchor"
                        if anchor_accepted
                        else "command_frozen_context_sim3_to_lingbot_with_trusted_rigid_anchor"
                    ),
                    "alignment_mode": alignment_mode,
                    "anchor_candidate_residual_c2w": candidate_residual.tolist(),
                    "anchor_step_translation": anchor_step_translation,
                    "anchor_step_rotation_deg": anchor_step_rotation_deg,
                    "anchor_max_translation_step": float(anchor_max_translation_step),
                    "anchor_max_rotation_step_deg": float(anchor_max_rotation_step_deg),
                    "anchor_accepted": bool(anchor_accepted),
                    "anchor_rejection_reason": (
                        "+".join(rejection_reasons) if rejection_reasons else None
                    ),
                    "trusted_anchor_index": int(
                        self.trusted_prefix_sim3_anchor.get("anchor_index", base_idx)
                    ),
                    "source_residual_c2w": applied_residual.tolist(),
                    "source_anchor_translation": float(
                        np.linalg.norm(applied_residual[:3, 3])
                    ),
                    "source_anchor_rotation_deg": float(
                        np.degrees(np.arccos(applied_cos))
                    ),
                }
            )
        elif alignment_mode == "anchor_delta_scaled" and fit_count > 0:
            base_idx = min(start - 1, fit_count - 1, len(self.command_pose_history) - 1)
            command_base = self.command_pose_history[base_idx]
            observed_base = self.lingbot_pose_history[base_idx]
            scale, debug = _translation_scale(
                self.command_pose_history,
                self.lingbot_pose_history,
                end=base_idx + 1,
                window=local_window,
                scale_min=scale_min,
                scale_max=scale_max,
                min_source_step=min_command_step,
            )
            aligned = [
                observed_base
                @ _scale_delta_translation(
                    np.linalg.inv(command_base) @ pose,
                    scale,
                )
                for pose in query
            ]
            debug.update(
                {
                    "enabled": True,
                    "fit_count": int(fit_count),
                    "anchor_index": int(base_idx),
                    "query_pose_source": "latest_lingbot_anchor_plus_scaled_command_delta",
                    "alignment_mode": alignment_mode,
                }
            )
        elif alignment_mode == "local_sim3_anchor" and fit_count > 0:
            fit_begin = max(0, fit_count - max(1, int(local_window)))
            debug, _ = fit_sim3_c2w_to_target(
                self.command_pose_history[fit_begin:fit_count],
                self.lingbot_pose_history[fit_begin:fit_count],
            )
            aligned = apply_sim3_to_c2w(
                query,
                scale=float(debug["scale"]),
                rotation=np.asarray(debug["rotation"]),
                translation=np.asarray(debug["translation"]),
            )
            base_idx = min(start - 1, fit_count - 1, len(self.command_pose_history) - 1)
            aligned_base = apply_sim3_to_c2w(
                [self.command_pose_history[base_idx]],
                scale=float(debug["scale"]),
                rotation=np.asarray(debug["rotation"]),
                translation=np.asarray(debug["translation"]),
            )[0]
            anchor_offset = (
                self.lingbot_pose_history[base_idx][:3, 3] - aligned_base[:3, 3]
            )
            for pose in aligned:
                pose[:3, 3] += anchor_offset
            debug.update(
                {
                    "fit_begin": int(fit_begin),
                    "anchor_index": int(base_idx),
                    "anchor_offset": anchor_offset.tolist(),
                    "query_pose_source": "local_sim3_to_lingbot_with_latest_anchor",
                    "alignment_mode": alignment_mode,
                }
            )
        elif alignment_mode == "prefix_sim3" and fit_count > 0:
            debug, _ = fit_sim3_c2w_to_target(
                self.command_pose_history[:fit_count],
                self.lingbot_pose_history[:fit_count],
            )
            aligned = apply_sim3_to_c2w(
                query,
                scale=float(debug["scale"]),
                rotation=np.asarray(debug["rotation"]),
                translation=np.asarray(debug["translation"]),
            )
            debug["query_pose_source"] = "command_prefix_sim3_to_lingbot"
            debug["alignment_mode"] = alignment_mode
        elif self.lingbot_pose_history and start > 0:
            command_base = self.command_pose_history[start - 1]
            observed_base = self.lingbot_pose_history[-1]
            aligned = [
                observed_base @ np.linalg.inv(command_base) @ pose for pose in query
            ]
            debug = {
                "enabled": False,
                "fit_count": 0,
                "query_pose_source": "latest_lingbot_plus_command_delta",
                "alignment_mode": alignment_mode,
            }
        else:
            aligned = [pose.copy() for pose in query]
            debug = {
                "enabled": False,
                "fit_count": 0,
                "query_pose_source": (
                    "command_pose_fallback_insufficient_pairs"
                    if alignment_mode == "prefix_sim3_anchor"
                    else "command_pose_fallback"
                ),
                "alignment_mode": alignment_mode,
                "min_pairs": int(min_pairs),
            }
        self.last_alignment = dict(debug)
        return aligned, debug

    def mark_chunk_complete(self, chunk_index: int) -> None:
        start = int(chunk_index) * self.chunk_size
        stop = min(start + self.chunk_size, len(self.command_pose_history))
        if stop > start:
            base = np.eye(4) if start == 0 else self.command_pose_history[start - 1]
            delta = np.linalg.inv(base) @ self.command_pose_history[stop - 1]
            translation = delta[:3, 3]
            if (
                abs(float(translation[2])) > abs(float(translation[0]))
                and abs(float(translation[2])) > 1e-8
            ):
                self.last_motion_key = "forward" if translation[2] > 0 else "backward"
            elif abs(float(translation[0])) > 1e-8:
                self.last_motion_key = "right" if translation[0] > 0 else "left"
            else:
                rotation_cos = float(
                    np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
                )
                self.last_motion_key = (
                    "rotation" if np.arccos(rotation_cos) > 1e-8 else "idle"
                )
        self.chunk_index = max(self.chunk_index, int(chunk_index) + 1)

    def update_ucm_p2l_sources(self, sources: Sequence[Optional[int]]) -> None:
        self.last_ucm_p2l_sources = [
            None if source is None else int(source) for source in sources
        ]
        selected = [
            source for source in self.last_ucm_p2l_sources if source is not None
        ]
        if selected:
            self.last_ucm_p2l_source = int(selected[-1])

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_size": int(self.chunk_size),
            "chunk_index": int(self.chunk_index),
            "last_motion_key": self.last_motion_key,
            "command_pose_history": [
                pose.tolist() for pose in self.command_pose_history
            ],
            "generation_pose_history": [
                pose.tolist() for pose in self.generation_pose_history
            ],
            "lingbot_pose_history": [
                pose.tolist() for pose in self.lingbot_pose_history
            ],
            "last_alignment": self.last_alignment,
            "frozen_prefix_sim3_alignment": self.frozen_prefix_sim3_alignment,
            "trusted_prefix_sim3_anchor": self.trusted_prefix_sim3_anchor,
            "last_ucm_p2l_source": self.last_ucm_p2l_source,
            "last_ucm_p2l_sources": self.last_ucm_p2l_sources,
        }
