# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Point-to-chunk retrieval over the incremental LingBot reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .session_state import TrajectoryState


@dataclass(frozen=True)
class PointCloudRetrievalConfig:
    chunk_size: int = 4
    initial_chunks: int = 1
    temporal_chunks: int = 2
    spatial_chunks: int = 2
    retrieval_unit: str = "chunk"
    latents_per_query: int = 4
    ucm_sources_per_query: int = 4
    ucm_shortlist_size: int = 8
    ucm_switch_penalty: float = 0.05
    ucm_coverage_weight: float = 0.85
    ucm_valid_source_weight: float = 0.15
    ucm_warp_rope_enabled: bool = False
    ucm_warp_rope_min_target_cover: float = 0.10
    ucm_warp_rope_conf_threshold: float = 0.0
    ucm_warp_rope_min_depth: float = 1e-4
    ucm_warp_rope_depth_scale: float = 1.0
    # The online generated frames are not a reliable camera-pose sensor.  A
    # frozen prefix calibration keeps commanded rotation geometry stable while
    # still using LingBot poses for the historical point map.
    query_alignment_mode: str = "prefix_sim3"
    query_alignment_window: int = 64
    query_alignment_scale_min: float = 0.05
    query_alignment_scale_max: float = 4.0
    query_alignment_min_command_step: float = 1e-4
    query_alignment_orientation_weight: float = 1.0
    query_alignment_min_pairs: int = 1
    query_alignment_anchor_max_translation_step: float = 0.0
    query_alignment_anchor_max_rotation_step_deg: float = 30.0
    cache_policy: str = "rotation_following"
    rotation_action_conditioning_scale: float = 1.0
    rotation_query_pose_scale: float = 1.0
    anchor_attention_mass_budget: float = 1.0
    temporal_attention_mass_budget: float = 12.0
    temporal_attention_decay: float = 1.0
    temporal_attention_chunk_decay: float = 1.0
    spatial_min_attention_weight: float = 0.05
    spatial_attention_gamma: float = 1.0
    spatial_attention_mass_budget: float = 1.0
    ablation: str = "s2_full"
    sample_stride: int = 4
    max_points: int = 250_000
    min_visible_points: int = 32
    confidence_threshold: float = 0.0
    far: Optional[float] = None
    zbuffer_cell_size: int = 8
    depth_abs_tol: float = 0.05
    depth_rel_tol: float = 0.02
    greedy_cell_size: int = 16
    tile_cols: int = 8
    tile_rows: int = 6
    top_k: int = 10
    debug_projection_top_k: int = 0
    debug_projection_samples_per_chunk: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.initial_chunks < 0 or self.temporal_chunks < 0:
            raise ValueError("invalid point-cloud memory layout")
        if self.spatial_chunks <= 0:
            raise ValueError("spatial_chunks must be positive")
        if self.retrieval_unit not in {"chunk", "latent"}:
            raise ValueError(
                f"unsupported point-cloud retrieval unit: {self.retrieval_unit}"
            )
        if self.latents_per_query <= 0:
            raise ValueError("latents_per_query must be positive")
        if self.ucm_sources_per_query <= 0:
            raise ValueError("ucm_sources_per_query must be positive")
        if self.ucm_shortlist_size <= 0:
            raise ValueError("ucm_shortlist_size must be positive")
        if self.ucm_switch_penalty < 0:
            raise ValueError("ucm_switch_penalty must be non-negative")
        if self.ucm_coverage_weight < 0 or self.ucm_valid_source_weight < 0:
            raise ValueError("UCM P2L score weights must be non-negative")
        if self.ucm_coverage_weight + self.ucm_valid_source_weight <= 0:
            raise ValueError("at least one UCM P2L score weight must be positive")
        if not 0.0 <= self.ucm_warp_rope_min_target_cover <= 1.0:
            raise ValueError("ucm_warp_rope_min_target_cover must be in [0, 1]")
        if self.ucm_warp_rope_conf_threshold < 0 or self.ucm_warp_rope_min_depth <= 0:
            raise ValueError("invalid UCM warped RoPE depth/confidence thresholds")
        if self.query_alignment_mode not in {
            "prefix_sim3",
            "prefix_sim3_anchor",
            "anchor_delta_scaled",
            "local_sim3_anchor",
        }:
            raise ValueError(
                f"unsupported point-cloud query alignment mode: {self.query_alignment_mode}"
            )
        if self.query_alignment_window <= 0:
            raise ValueError("query_alignment_window must be positive")
        if self.query_alignment_scale_min <= 0 or self.query_alignment_scale_max <= 0:
            raise ValueError("query alignment scale bounds must be positive")
        if self.query_alignment_scale_min > self.query_alignment_scale_max:
            raise ValueError(
                "query_alignment_scale_min cannot exceed query_alignment_scale_max"
            )
        if self.query_alignment_min_command_step < 0:
            raise ValueError("query_alignment_min_command_step must be non-negative")
        if self.query_alignment_orientation_weight < 0:
            raise ValueError("query_alignment_orientation_weight must be non-negative")
        if self.query_alignment_min_pairs <= 0:
            raise ValueError("query_alignment_min_pairs must be positive")
        if self.query_alignment_anchor_max_translation_step < 0:
            raise ValueError(
                "query alignment anchor translation step must be non-negative"
            )
        if self.query_alignment_anchor_max_rotation_step_deg < 0:
            raise ValueError(
                "query alignment anchor rotation step must be non-negative"
            )
        if self.cache_policy not in {"official", "rotation_following"}:
            raise ValueError(
                f"unsupported point-cloud cache policy: {self.cache_policy}"
            )
        if self.rotation_action_conditioning_scale <= 0:
            raise ValueError("rotation_action_conditioning_scale must be positive")
        if self.rotation_query_pose_scale <= 0:
            raise ValueError("rotation_query_pose_scale must be positive")
        if self.anchor_attention_mass_budget <= 0:
            raise ValueError("anchor_attention_mass_budget must be positive")
        if self.temporal_attention_mass_budget <= 0:
            raise ValueError("temporal_attention_mass_budget must be positive")
        if not 0.0 < self.temporal_attention_decay <= 1.0:
            raise ValueError("temporal_attention_decay must be in (0, 1]")
        if not 0.0 < self.temporal_attention_chunk_decay <= 1.0:
            raise ValueError("temporal_attention_chunk_decay must be in (0, 1]")
        if not 0.0 < self.spatial_min_attention_weight <= 1.0:
            raise ValueError("spatial_min_attention_weight must be in (0, 1]")
        if self.spatial_attention_gamma <= 0:
            raise ValueError("spatial_attention_gamma must be positive")
        if self.spatial_attention_mass_budget <= 0:
            raise ValueError("spatial_attention_mass_budget must be positive")
        if self.ablation not in {
            "baseline",
            "zbuffer",
            "temporal_overlap",
            "tile_coverage",
            "greedy_submodular",
            "fused_coverage",
            "s2_full",
        }:
            raise ValueError(f"unsupported point-cloud ablation: {self.ablation}")
        if (
            self.debug_projection_top_k < 0
            or self.debug_projection_samples_per_chunk < 0
        ):
            raise ValueError("debug projection settings must be non-negative")


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def order_context_slots_by_source(
    source_indices: Sequence[int],
    target_indices: Optional[Sequence[int]] = None,
    warp_flags: Optional[Sequence[bool]] = None,
) -> Tuple[List[int], Optional[List[int]], Optional[List[bool]]]:
    """Keep ordinary AR memory chronological while preserving slot metadata."""
    sources = [int(source) for source in source_indices]
    if target_indices is not None and len(target_indices) != len(sources):
        raise ValueError("target_indices must match source_indices")
    if warp_flags is not None and len(warp_flags) != len(sources):
        raise ValueError("warp_flags must match source_indices")

    order = sorted(range(len(sources)), key=lambda index: (sources[index], index))
    ordered_sources = [sources[index] for index in order]
    ordered_targets = (
        [int(target_indices[index]) for index in order]
        if target_indices is not None
        else None
    )
    ordered_flags = (
        [bool(warp_flags[index]) for index in order] if warp_flags is not None else None
    )
    return ordered_sources, ordered_targets, ordered_flags


def _p2c_target_latent(
    source_latent: int, current_frame_idx: int, chunk_size: int
) -> int:
    """Match each source position in a chunk to the same query position."""

    return int(current_frame_idx + (int(source_latent) % int(chunk_size)))


def _pixel_intrinsic(K: np.ndarray, height: int, width: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).copy()
    if max(abs(float(K[0, 2])), abs(float(K[1, 2]))) <= 2.0:
        K[0, 0] *= width
        K[1, 1] *= height
        K[0, 2] *= width
        K[1, 2] *= height
    return K


def build_source_world_point_cloud(
    world_points: np.ndarray,
    confidence: Optional[np.ndarray],
    *,
    frame_count: int,
    config: PointCloudRetrievalConfig,
    source_chunk_min: Optional[int] = None,
    source_chunk_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten LingBot's official depth-unprojected world points by source latent."""
    world_points = _numpy(world_points)
    confidence = None if confidence is None else _numpy(confidence)
    frame_count = min(int(frame_count), len(world_points))
    stride = max(1, int(config.sample_stride))
    xyz_parts: List[np.ndarray] = []
    conf_parts: List[np.ndarray] = []
    chunk_parts: List[np.ndarray] = []
    for frame_idx in range(frame_count):
        source_chunk = frame_idx // config.chunk_size
        if source_chunk_min is not None and source_chunk < source_chunk_min:
            continue
        if source_chunk_max is not None and source_chunk > source_chunk_max:
            continue
        xyz = np.asarray(world_points[frame_idx], dtype=np.float32)[::stride, ::stride]
        conf = (
            np.ones(xyz.shape[:2], dtype=np.float32)
            if confidence is None
            else np.asarray(confidence[frame_idx], dtype=np.float32)[::stride, ::stride]
        )
        valid = np.all(np.isfinite(xyz), axis=-1) & np.isfinite(conf)
        valid &= conf >= float(config.confidence_threshold)
        if not valid.any():
            continue
        xyz_parts.append(xyz[valid])
        conf_parts.append(conf[valid])
        chunk_parts.append(np.full(int(valid.sum()), source_chunk, dtype=np.int32))
    if not xyz_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    xyz = np.concatenate(xyz_parts)
    conf = np.concatenate(conf_parts)
    chunks = np.concatenate(chunk_parts)
    if config.max_points > 0 and len(xyz) > config.max_points:
        indices = np.linspace(0, len(xyz) - 1, config.max_points, dtype=np.int64)
        xyz, conf, chunks = xyz[indices], conf[indices], chunks[indices]
    return xyz, conf, chunks


def build_source_world_point_latent_cloud(
    world_points: np.ndarray,
    confidence: Optional[np.ndarray],
    *,
    frame_count: int,
    config: PointCloudRetrievalConfig,
    source_latent_min: Optional[int] = None,
    source_latent_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten LingBot world points while preserving each point's source latent."""
    world_points = _numpy(world_points)
    confidence = None if confidence is None else _numpy(confidence)
    frame_count = min(int(frame_count), len(world_points))
    stride = max(1, int(config.sample_stride))
    xyz_parts: List[np.ndarray] = []
    conf_parts: List[np.ndarray] = []
    latent_parts: List[np.ndarray] = []
    for frame_idx in range(frame_count):
        if source_latent_min is not None and frame_idx < source_latent_min:
            continue
        if source_latent_max is not None and frame_idx > source_latent_max:
            continue
        xyz = np.asarray(world_points[frame_idx], dtype=np.float32)[::stride, ::stride]
        conf = (
            np.ones(xyz.shape[:2], dtype=np.float32)
            if confidence is None
            else np.asarray(confidence[frame_idx], dtype=np.float32)[::stride, ::stride]
        )
        valid = np.all(np.isfinite(xyz), axis=-1) & np.isfinite(conf)
        valid &= conf >= float(config.confidence_threshold)
        if not valid.any():
            continue
        xyz_parts.append(xyz[valid])
        conf_parts.append(conf[valid])
        latent_parts.append(np.full(int(valid.sum()), frame_idx, dtype=np.int32))
    if not xyz_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    xyz = np.concatenate(xyz_parts)
    conf = np.concatenate(conf_parts)
    latents = np.concatenate(latent_parts)
    if config.max_points > 0 and len(xyz) > config.max_points:
        indices = np.linspace(0, len(xyz) - 1, config.max_points, dtype=np.int64)
        xyz, conf, latents = xyz[indices], conf[indices], latents[indices]
    return xyz, conf, latents


def load_source_point_shards(
    points_dir: str | Path,
    *,
    frame_count: int,
    max_points: int,
    source_field: str,
    source_min: Optional[int] = None,
    source_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the same online point shards consumed by WMFactory P2C."""
    xyz_parts: List[np.ndarray] = []
    conf_parts: List[np.ndarray] = []
    source_parts: List[np.ndarray] = []
    for path in sorted(Path(points_dir).glob("frame_*.npz")):
        try:
            frame_id = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if frame_id >= int(frame_count):
            continue
        with np.load(path) as data:
            if not {"xyz", "conf", source_field}.issubset(data.files) or not len(
                data["xyz"]
            ):
                continue
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            conf = np.asarray(data["conf"], dtype=np.float32)
            source = np.asarray(data[source_field], dtype=np.int32)
            if source_min is not None or source_max is not None:
                keep = np.ones(len(source), dtype=bool)
                if source_min is not None:
                    keep &= source >= source_min
                if source_max is not None:
                    keep &= source <= source_max
                xyz, conf, source = xyz[keep], conf[keep], source[keep]
            if not len(xyz):
                continue
            xyz_parts.append(xyz)
            conf_parts.append(conf)
            source_parts.append(source)
    if not xyz_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    xyz = np.concatenate(xyz_parts)
    conf = np.concatenate(conf_parts)
    source = np.concatenate(source_parts)
    if max_points > 0 and len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz, conf, source = xyz[indices], conf[indices], source[indices]
    return xyz, conf, source


def load_source_point_blocks(
    blocks: Sequence[Dict[str, np.ndarray]],
    *,
    frame_count: int,
    max_points: int,
    source_field: str,
    source_min: Optional[int] = None,
    source_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read incrementally appended point blocks without touching artifact shards."""
    xyz_parts: List[np.ndarray] = []
    conf_parts: List[np.ndarray] = []
    source_parts: List[np.ndarray] = []
    for block in blocks:
        if not {"xyz", "conf", source_field}.issubset(block):
            continue
        frame_ids = np.asarray(block.get("source_frame_id", []), dtype=np.int32)
        if len(frame_ids) and int(frame_ids[0]) >= int(frame_count):
            continue
        xyz = np.asarray(block["xyz"], dtype=np.float32)
        if not len(xyz):
            continue
        conf = np.asarray(block["conf"], dtype=np.float32)
        source = np.asarray(block[source_field], dtype=np.int32)
        if source_min is not None or source_max is not None:
            keep = np.ones(len(source), dtype=bool)
            if source_min is not None:
                keep &= source >= source_min
            if source_max is not None:
                keep &= source <= source_max
            xyz, conf, source = xyz[keep], conf[keep], source[keep]
        if len(xyz):
            xyz_parts.append(xyz)
            conf_parts.append(conf)
            source_parts.append(source)
    if not xyz_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    xyz = np.concatenate(xyz_parts)
    conf = np.concatenate(conf_parts)
    source = np.concatenate(source_parts)
    if max_points > 0 and len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz, conf, source = xyz[indices], conf[indices], source[indices]
    return xyz, conf, source


def _runtime_available_frames(
    runtime: Dict[str, object], current_frame_idx: int
) -> int:
    frame_count = runtime.get("lingbot_frame_count")
    if frame_count is None:
        depth = runtime.get("depth")
        frame_count = 0 if depth is None else len(depth)
    return min(
        int(current_frame_idx),
        int(frame_count),
        len(runtime["lingbot_w2c"]),
        len(runtime["lingbot_Ks"]),
    )


def _runtime_image_hw(runtime: Dict[str, object]) -> Tuple[int, int]:
    image_hw = runtime.get("lingbot_image_hw")
    if image_hw is not None:
        return int(image_hw[0]), int(image_hw[1])
    depth = runtime.get("depth")
    if depth is None or not len(depth):
        raise ValueError("LingBot image size is unavailable")
    return tuple(int(v) for v in _numpy(depth)[0].shape[-2:])


def _project(
    xyz: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    config: PointCloudRetrievalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1))], axis=1)
    camera = (np.asarray(w2c, dtype=np.float64) @ points.T).T[:, :3]
    depth = camera[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    K = _pixel_intrinsic(K, height, width)
    u = K[0, 0] * camera[:, 0] / safe_depth + K[0, 2]
    v = K[1, 1] * camera[:, 1] / safe_depth + K[1, 2]
    valid = (depth > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if config.far is not None:
        valid &= depth < float(config.far)
    return valid, np.stack([u, v], axis=1), depth


def _zbuffer(
    visible: np.ndarray,
    pixels: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
    config: PointCloudRetrievalConfig,
) -> np.ndarray:
    indices = np.flatnonzero(visible)
    if not len(indices):
        return visible
    cell = max(1, config.zbuffer_cell_size)
    grid_width = (width + cell - 1) // cell
    u = np.clip((pixels[indices, 0] // cell).astype(np.int64), 0, grid_width - 1)
    v = np.clip(
        (pixels[indices, 1] // cell).astype(np.int64),
        0,
        (height + cell - 1) // cell - 1,
    )
    keys = v * grid_width + u
    nearest = np.full(((height + cell - 1) // cell) * grid_width, np.inf)
    np.minimum.at(nearest, keys, depth[indices])
    tolerance = np.maximum(config.depth_abs_tol, config.depth_rel_tol * nearest[keys])
    result = np.zeros_like(visible)
    result[indices[depth[indices] <= nearest[keys] + tolerance]] = True
    return result


def _zbuffer_indices(
    indices: np.ndarray,
    pixels: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
    config: PointCloudRetrievalConfig,
) -> np.ndarray:
    """Apply z-buffering to one candidate without allocating a full point mask."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return indices
    cell = max(1, config.zbuffer_cell_size)
    grid_width = (width + cell - 1) // cell
    grid_height = (height + cell - 1) // cell
    u = np.clip((pixels[indices, 0] // cell).astype(np.int64), 0, grid_width - 1)
    v = np.clip((pixels[indices, 1] // cell).astype(np.int64), 0, grid_height - 1)
    keys = v * grid_width + u
    nearest = np.full(grid_height * grid_width, np.inf)
    np.minimum.at(nearest, keys, depth[indices])
    tolerance = np.maximum(config.depth_abs_tol, config.depth_rel_tol * nearest[keys])
    return indices[depth[indices] <= nearest[keys] + tolerance]


def _sample_pixels(pixels: np.ndarray, max_samples: int) -> List[List[float]]:
    if max_samples <= 0 or len(pixels) == 0:
        return []
    count = min(int(max_samples), len(pixels))
    indices = np.linspace(0, len(pixels) - 1, count, dtype=np.int64)
    sampled = pixels[indices]
    return [[float(x), float(y)] for x, y in sampled]


def _bbox(pixels: np.ndarray) -> Optional[List[float]]:
    if len(pixels) == 0:
        return None
    return [
        float(np.min(pixels[:, 0])),
        float(np.min(pixels[:, 1])),
        float(np.max(pixels[:, 0])),
        float(np.max(pixels[:, 1])),
    ]


def _tile_count(
    pixels: np.ndarray, height: int, width: int, config: PointCloudRetrievalConfig
) -> int:
    if len(pixels) == 0:
        return 0
    cols = max(1, int(config.tile_cols))
    rows = max(1, int(config.tile_rows))
    u = np.clip((pixels[:, 0] * cols / max(1, width)).astype(np.int64), 0, cols - 1)
    v = np.clip((pixels[:, 1] * rows / max(1, height)).astype(np.int64), 0, rows - 1)
    return int(len(set((v * cols + u).tolist())))


def _projection_debug_for_chunks(
    xyz: np.ndarray,
    source_chunks: np.ndarray,
    chunks: Sequence[int],
    query_c2w: Sequence[np.ndarray],
    query_K: np.ndarray,
    height: int,
    width: int,
    config: PointCloudRetrievalConfig,
) -> List[Dict[str, object]]:
    """Summarize exact runtime world-point projections for retrieval top-k chunks."""
    summaries: List[Dict[str, object]] = []
    if config.debug_projection_top_k <= 0:
        return summaries
    samples_per_chunk = int(config.debug_projection_samples_per_chunk)
    samples_per_query = max(0, samples_per_chunk // max(1, len(query_c2w)))
    for chunk in list(chunks)[: int(config.debug_projection_top_k)]:
        chunk = int(chunk)
        chunk_mask = source_chunks == chunk
        chunk_xyz = xyz[chunk_mask]
        query_entries: List[Dict[str, object]] = []
        raw_pixels_all: List[np.ndarray] = []
        zbuffer_pixels_all: List[np.ndarray] = []
        for query_offset, c2w in enumerate(query_c2w):
            visible, pixels, projected_depth = _project(
                chunk_xyz, np.linalg.inv(c2w), query_K, height, width, config
            )
            raw_pixels = pixels[np.flatnonzero(visible)]
            zbuffer_visible = _zbuffer(
                visible.copy(), pixels, projected_depth, height, width, config
            )
            zbuffer_pixels = pixels[np.flatnonzero(zbuffer_visible)]
            if len(raw_pixels):
                raw_pixels_all.append(raw_pixels)
            if len(zbuffer_pixels):
                zbuffer_pixels_all.append(zbuffer_pixels)
            query_entries.append(
                {
                    "query_offset": int(query_offset),
                    "raw_count": int(len(raw_pixels)),
                    "zbuffer_count": int(len(zbuffer_pixels)),
                    "raw_tile_count": _tile_count(raw_pixels, height, width, config),
                    "zbuffer_tile_count": _tile_count(
                        zbuffer_pixels, height, width, config
                    ),
                    "raw_bbox": _bbox(raw_pixels),
                    "zbuffer_bbox": _bbox(zbuffer_pixels),
                    "raw_samples": _sample_pixels(raw_pixels, samples_per_query),
                    "zbuffer_samples": _sample_pixels(
                        zbuffer_pixels, samples_per_query
                    ),
                }
            )
        raw_all = (
            np.concatenate(raw_pixels_all, axis=0)
            if raw_pixels_all
            else np.empty((0, 2))
        )
        zbuffer_all = (
            np.concatenate(zbuffer_pixels_all, axis=0)
            if zbuffer_pixels_all
            else np.empty((0, 2))
        )
        summaries.append(
            {
                "chunk_id": int(chunk),
                "projection_image_size": [int(width), int(height)],
                "source_point_count": int(len(chunk_xyz)),
                "raw_count": int(len(raw_all)),
                "zbuffer_count": int(len(zbuffer_all)),
                "raw_tile_count": _tile_count(raw_all, height, width, config),
                "zbuffer_tile_count": _tile_count(zbuffer_all, height, width, config),
                "raw_bbox": _bbox(raw_all),
                "zbuffer_bbox": _bbox(zbuffer_all),
                "queries": query_entries,
            }
        )
    return summaries


def _layout(
    current_frame_idx: int, config: PointCloudRetrievalConfig
) -> Tuple[List[int], List[int]]:
    initial_end = min(current_frame_idx, config.initial_chunks * config.chunk_size)
    initial = list(range(initial_end))
    temporal_start = max(
        initial_end, current_frame_idx - config.temporal_chunks * config.chunk_size
    )
    temporal = list(range(temporal_start, current_frame_idx))
    return initial, temporal


def assign_unique_latents_per_query(
    ranked_by_query: Sequence[Sequence[int]],
    latents_per_query: int,
    *,
    used_sources: Optional[Set[int]] = None,
    candidate_is_usable: Optional[Callable[[int, int], bool]] = None,
) -> List[List[int]]:
    """Fill per-query slots from ranked candidates without reusing source latents."""
    selected: List[List[int]] = [[] for _ in ranked_by_query]
    used = {int(source) for source in (used_sources or set())}
    for _ in range(int(latents_per_query)):
        for query_index, ranked in enumerate(ranked_by_query):
            candidate = None
            for latent in ranked:
                latent = int(latent)
                if latent in used:
                    continue
                if candidate_is_usable is not None and not candidate_is_usable(
                    query_index, latent
                ):
                    continue
                candidate = latent
                break
            if candidate is None:
                continue
            selected[query_index].append(candidate)
            used.add(candidate)
    return selected


def select_temporal_ucm_sources(
    scores_by_query: Sequence[Dict[int, float]],
    *,
    previous_source: Optional[int] = None,
    switch_penalty: float = 0.0,
) -> List[Optional[int]]:
    """Select one source per query while penalizing source changes over time."""
    penalty = float(switch_penalty)
    if penalty < 0:
        raise ValueError("switch_penalty must be non-negative")

    # source -> (objective, path). The source key is the last non-empty choice,
    # so an uncovered query does not erase continuity at the next query.
    states: Dict[Optional[int], Tuple[float, List[Optional[int]]]] = {
        None if previous_source is None else int(previous_source): (0.0, [])
    }
    for query_scores in scores_by_query:
        candidates = {
            int(source): float(score) for source, score in query_scores.items()
        }
        if not candidates:
            states = {
                source: (objective, path + [None])
                for source, (objective, path) in states.items()
            }
            continue

        next_states: Dict[int, Tuple[float, List[Optional[int]]]] = {}
        for source, unary in sorted(candidates.items()):
            best: Optional[Tuple[float, List[Optional[int]]]] = None
            for prior_source, (objective, path) in states.items():
                transition = (
                    penalty
                    if prior_source is not None and int(prior_source) != source
                    else 0.0
                )
                proposal = (objective + unary - transition, path + [source])
                if best is None or proposal[0] > best[0]:
                    best = proposal
            assert best is not None
            next_states[source] = best
        states = next_states

    if not states:
        return [None for _ in scores_by_query]
    _, best_path = max(
        states.values(),
        key=lambda item: (
            item[0],
            tuple(-1 if x is None else -int(x) for x in item[1]),
        ),
    )
    return best_path


def select_temporal_ucm_source_sets(
    scores_by_query: Sequence[Dict[int, float]],
    *,
    sources_per_query: int,
    previous_sources: Optional[Sequence[int]] = None,
    switch_penalty: float = 0.0,
    covered_tokens_by_query: Optional[Sequence[Dict[int, Set[int]]]] = None,
    total_tokens_by_query: Optional[Sequence[int]] = None,
    union_coverage_weight: float = 0.0,
) -> List[List[int]]:
    """Jointly select a fixed-size UCM source set for each adjacent query."""
    source_count = int(sources_per_query)
    penalty = float(switch_penalty)
    if source_count <= 0:
        raise ValueError("sources_per_query must be positive")
    if penalty < 0:
        raise ValueError("switch_penalty must be non-negative")
    union_weight = float(union_coverage_weight)
    if union_weight < 0 or union_weight > 1:
        raise ValueError("union_coverage_weight must be in [0, 1]")
    if covered_tokens_by_query is not None and len(covered_tokens_by_query) != len(
        scores_by_query
    ):
        raise ValueError("covered token maps must match query score count")
    if total_tokens_by_query is not None and len(total_tokens_by_query) != len(
        scores_by_query
    ):
        raise ValueError("target token counts must match query score count")
    if source_count == 1:
        previous = next(iter(previous_sources or []), None)
        return [
            [] if source is None else [int(source)]
            for source in select_temporal_ucm_sources(
                scores_by_query,
                previous_source=previous,
                switch_penalty=penalty,
            )
        ]

    previous_set = tuple(sorted({int(source) for source in (previous_sources or [])}))
    states: Dict[Tuple[int, ...], Tuple[float, List[Tuple[int, ...]]]] = {
        previous_set: (0.0, [])
    }
    for query_index, query_scores in enumerate(scores_by_query):
        scores = {int(source): float(score) for source, score in query_scores.items()}
        if not scores:
            states = {
                source_set: (objective, path + [tuple()])
                for source_set, (objective, path) in states.items()
            }
            continue
        selected_count = min(source_count, len(scores))
        candidate_sets = list(combinations(sorted(scores), selected_count))
        next_states: Dict[Tuple[int, ...], Tuple[float, List[Tuple[int, ...]]]] = {}
        for source_set in candidate_sets:
            unary = sum(scores[source] for source in source_set)
            coverage_map = (
                covered_tokens_by_query[query_index]
                if covered_tokens_by_query is not None
                else None
            )
            if union_weight > 0 and coverage_map is not None:
                covered = set().union(
                    *(coverage_map.get(source, set()) for source in source_set)
                )
                total_tokens = max(
                    1,
                    int(total_tokens_by_query[query_index])
                    if total_tokens_by_query is not None
                    else len(covered),
                )
                union_ratio = len(covered) / total_tokens
                mean_quality = unary / max(1, len(source_set))
                unary = union_weight * union_ratio + (1.0 - union_weight) * mean_quality
            best = None
            current = set(source_set)
            for prior_set, (objective, path) in states.items():
                replaced = len(set(prior_set) - current) if prior_set else 0
                proposal = (
                    objective + unary - penalty * replaced,
                    path + [source_set],
                )
                if best is None or proposal[0] > best[0]:
                    best = proposal
            assert best is not None
            next_states[source_set] = best
        states = next_states

    if not states:
        return [[] for _ in scores_by_query]
    _, best_path = max(states.values(), key=lambda item: (item[0], item[1]))
    ordered_path = []
    for index, source_set in enumerate(best_path):
        coverage_map = (
            covered_tokens_by_query[index]
            if covered_tokens_by_query is not None
            else None
        )
        if union_weight <= 0 or coverage_map is None:
            ordered_path.append(
                sorted(
                    source_set,
                    key=lambda source: (-scores_by_query[index][source], source),
                )
            )
            continue
        remaining = set(source_set)
        covered: Set[int] = set()
        ordered = []
        while remaining:
            best = max(
                remaining,
                key=lambda source: (
                    len(coverage_map.get(source, set()) - covered),
                    scores_by_query[index][source],
                    -source,
                ),
            )
            ordered.append(best)
            covered.update(coverage_map.get(best, set()))
            remaining.remove(best)
        ordered_path.append(ordered)
    return ordered_path


def context_local_rope_indices(
    attention_target_indices: Sequence[int],
    use_warp_rope: Sequence[bool],
    *,
    current_frame_idx: int,
    query_frame_count: int,
) -> List[int]:
    """Map global UCM attention targets onto the rebuilt local context/query timeline."""
    if len(attention_target_indices) != len(use_warp_rope):
        raise ValueError("attention targets and warp flags must have the same length")
    context_size = len(attention_target_indices)
    local_indices: List[int] = []
    for context_index, (target_index, use_warp) in enumerate(
        zip(attention_target_indices, use_warp_rope)
    ):
        if not use_warp:
            local_indices.append(context_index)
            continue
        query_offset = int(target_index) - int(current_frame_idx)
        if query_offset < 0 or query_offset >= int(query_frame_count):
            raise ValueError(
                f"UCM target {target_index} is outside query range "
                f"[{current_frame_idx}, {current_frame_idx + query_frame_count})"
            )
        local_indices.append(context_size + query_offset)
    return local_indices


def select_point_cloud_latent_memory(
    runtime: Dict[str, object],
    trajectory: TrajectoryState,
    *,
    current_frame_idx: int,
    fov_selected: Sequence[int],
    config: PointCloudRetrievalConfig,
) -> Tuple[Optional[List[int]], Dict[str, object]]:
    """Return initial + temporal + per-query-latent spatial memory."""
    debug: Dict[str, object] = {
        "status": "fallback",
        "fallback_reason": None,
        "current_frame_idx": int(current_frame_idx),
        "retrieval_unit": "latent",
        "latents_per_query": int(config.latents_per_query),
        "ucm_p2l": {
            "sources_per_query": int(config.ucm_sources_per_query),
            "shortlist_size": int(config.ucm_shortlist_size),
            "switch_penalty": float(config.ucm_switch_penalty),
            "coverage_weight": float(config.ucm_coverage_weight),
            "valid_source_weight": float(config.ucm_valid_source_weight),
        },
        "ucm_warp_rope": {
            "enabled": bool(config.ucm_warp_rope_enabled),
            "min_target_cover": float(config.ucm_warp_rope_min_target_cover),
            "conf_threshold": float(config.ucm_warp_rope_conf_threshold),
            "depth_scale": float(config.ucm_warp_rope_depth_scale),
        },
        "ablation": config.ablation,
        "lingbot_pose_convention": str(runtime.get("lingbot_pose_convention", "w2c"))
        .strip()
        .lower(),
        "fov_selected_latents": [int(i) for i in fov_selected],
    }
    initial, temporal = _layout(current_frame_idx, config)
    if config.ablation not in {
        "baseline",
        "zbuffer",
        "temporal_overlap",
        "tile_coverage",
        "fused_coverage",
    }:
        debug["fallback_reason"] = "unsupported_ablation_for_latent_retrieval"
        return None, debug
    historical_min_latent = config.initial_chunks * config.chunk_size
    historical_max_latent = (
        current_frame_idx - config.temporal_chunks * config.chunk_size - 1
    )
    if historical_max_latent < historical_min_latent:
        debug["fallback_reason"] = "point_cloud_warmup"
        return None, debug
    required = {"lingbot_w2c", "lingbot_Ks"}
    if not required.issubset(runtime):
        debug["fallback_reason"] = "lingbot_runtime_not_ready"
        return None, debug
    available = _runtime_available_frames(runtime, current_frame_idx)
    if available <= 0:
        debug["fallback_reason"] = "lingbot_history_empty"
        return None, debug
    trajectory.update_lingbot_poses(
        runtime["lingbot_w2c"],
        available,
        convention=runtime.get("lingbot_pose_convention", "w2c"),
    )
    query_c2w, alignment = trajectory.query_c2w_in_lingbot_space(
        current_frame_idx,
        config.chunk_size,
        alignment_mode=config.query_alignment_mode,
        local_window=config.query_alignment_window,
        scale_min=config.query_alignment_scale_min,
        scale_max=config.query_alignment_scale_max,
        min_command_step=config.query_alignment_min_command_step,
        orientation_weight=config.query_alignment_orientation_weight,
        min_pairs=config.query_alignment_min_pairs,
        anchor_max_translation_step=config.query_alignment_anchor_max_translation_step,
        anchor_max_rotation_step_deg=config.query_alignment_anchor_max_rotation_step_deg,
    )
    if not query_c2w:
        debug["fallback_reason"] = "query_pose_empty"
        return None, debug
    point_blocks = runtime.get("lingbot_point_blocks")
    points_dir = runtime.get("lingbot_points_dir")
    if runtime.get("world_points") is None and not point_blocks and not points_dir:
        debug["fallback_reason"] = "official_unprojected_points_not_ready"
        return None, debug

    if point_blocks:
        xyz, confidence, source_latents = load_source_point_blocks(
            point_blocks,
            frame_count=available,
            max_points=config.max_points,
            source_field="source_frame_id",
            source_min=(
                historical_min_latent if config.ablation == "fused_coverage" else None
            ),
            source_max=(
                historical_max_latent
                if config.ablation in {"temporal_overlap", "fused_coverage"}
                else None
            ),
        )
        point_cloud_source = "lingbot_incremental_memory_blocks"
    elif points_dir:
        xyz, confidence, source_latents = load_source_point_shards(
            points_dir,
            frame_count=available,
            max_points=config.max_points,
            source_field="source_frame_id",
            source_min=(
                historical_min_latent if config.ablation == "fused_coverage" else None
            ),
            source_max=(
                historical_max_latent
                if config.ablation in {"temporal_overlap", "fused_coverage"}
                else None
            ),
        )
        point_cloud_source = "wmfactory_source_map_online_shards"
    else:
        xyz, confidence, source_latents = build_source_world_point_latent_cloud(
            runtime["world_points"],
            runtime.get("world_points_conf"),
            frame_count=available,
            config=config,
            source_latent_min=(
                historical_min_latent if config.ablation == "fused_coverage" else None
            ),
            source_latent_max=(
                historical_max_latent
                if config.ablation in {"temporal_overlap", "fused_coverage"}
                else None
            ),
        )
        point_cloud_source = "gctbase_unproject_depth_to_world"
    if not len(xyz):
        debug["fallback_reason"] = "source_points_empty"
        return None, debug
    height, width = _runtime_image_hw(runtime)
    query_K = _numpy(runtime["lingbot_Ks"])[available - 1]
    candidate_source = (source_latents >= historical_min_latent) & (
        source_latents <= historical_max_latent
    )

    query_results: List[Dict[str, object]] = []
    ranked_by_query: List[List[int]] = []
    quality_by_query: List[Dict[int, float]] = []
    total_candidate_points = 0
    min_query_candidate_points: Optional[int] = None

    for query_offset, c2w in enumerate(query_c2w):
        query_latent = int(current_frame_idx + query_offset)
        visible, pixels, projected_depth = _project(
            xyz, np.linalg.inv(c2w), query_K, height, width, config
        )
        if config.ablation == "zbuffer":
            visible = _zbuffer(visible, pixels, projected_depth, height, width, config)
        indices = np.flatnonzero(visible & candidate_source)
        if config.ablation == "fused_coverage" and len(indices):
            kept = []
            indexed_sources = source_latents[indices]
            for source in np.unique(indexed_sources):
                source_indices = indices[indexed_sources == source]
                kept.append(
                    _zbuffer_indices(
                        source_indices, pixels, projected_depth, height, width, config
                    )
                )
            indices = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
        candidate_points = int(len(indices))
        total_candidate_points += candidate_points
        min_query_candidate_points = (
            candidate_points
            if min_query_candidate_points is None
            else min(min_query_candidate_points, candidate_points)
        )
        weighted_scores: Dict[int, float] = {}
        tile_weights: Dict[int, Dict[int, float]] = {}
        counts: Dict[int, int] = {}
        if candidate_points:
            latents = source_latents[indices].astype(np.int64)
            weights = confidence[indices].astype(np.float64)
            weighted = np.bincount(latents, weights=weights)
            count = np.bincount(latents)
            for latent in np.flatnonzero(weighted > 0):
                weighted_scores[int(latent)] = float(weighted[latent])
                counts[int(latent)] = int(count[latent])
            if config.ablation in {"tile_coverage", "fused_coverage"}:
                positive_depth = projected_depth[indices]
                depth_reference = (
                    float(np.median(positive_depth)) if len(positive_depth) else 1.0
                )
                for point_idx in indices:
                    latent = int(source_latents[point_idx])
                    u = min(
                        config.tile_cols - 1,
                        int(pixels[point_idx, 0] * config.tile_cols / width),
                    )
                    v = min(
                        config.tile_rows - 1,
                        int(pixels[point_idx, 1] * config.tile_rows / height),
                    )
                    tile = v * config.tile_cols + u
                    latent_tiles = tile_weights.setdefault(latent, {})
                    weight = float(confidence[point_idx])
                    if config.ablation == "fused_coverage":
                        point_depth = max(float(projected_depth[point_idx]), 1e-6)
                        weight *= 1.0 + depth_reference / (
                            depth_reference + point_depth
                        )
                    latent_tiles[tile] = max(
                        latent_tiles.get(tile, 0.0),
                        weight,
                    )
        scores = dict(weighted_scores)
        if config.ablation in {"tile_coverage", "fused_coverage"}:
            scores = {
                latent: float(sum(values.values()))
                for latent, values in tile_weights.items()
            }
        greedy_gain: Dict[int, float] = {}
        if config.ablation == "fused_coverage":
            remaining = set(scores)
            covered_weights: Dict[int, float] = {}
            ranked_latents = []
            while remaining:

                def marginal_gain(latent: int) -> float:
                    return float(
                        sum(
                            max(0.0, weight - covered_weights.get(tile, 0.0))
                            for tile, weight in tile_weights.get(latent, {}).items()
                        )
                    )

                best = max(
                    remaining,
                    key=lambda latent: (
                        marginal_gain(latent),
                        scores[latent],
                        weighted_scores.get(latent, 0.0),
                        -latent,
                    ),
                )
                greedy_gain[best] = marginal_gain(best)
                ranked_latents.append(best)
                for tile, weight in tile_weights.get(best, {}).items():
                    covered_weights[tile] = max(covered_weights.get(tile, 0.0), weight)
                remaining.remove(best)
        else:
            ranked_latents = sorted(
                scores,
                key=lambda latent: (
                    scores[latent],
                    weighted_scores.get(latent, 0.0),
                    -latent,
                ),
                reverse=True,
            )
        ranked_by_query.append([int(latent) for latent in ranked_latents])
        quality_by_query.append(
            {
                int(latent): max(
                    0.0,
                    min(
                        1.0,
                        len(tile_weights.get(latent, {}))
                        / max(1, config.tile_rows * config.tile_cols),
                    ),
                )
                for latent in ranked_latents
            }
        )
        query_results.append(
            {
                "query_offset": int(query_offset),
                "query_latent": query_latent,
                "candidate_points": candidate_points,
                "selected_latents": [],
                "ranked_latents": [int(latent) for latent in ranked_latents],
                "topk": [
                    {
                        "rank": rank + 1,
                        "latent_id": int(latent),
                        "chunk_id": int(latent) // config.chunk_size,
                        "score": float(scores[latent]),
                        "baseline_score": float(weighted_scores.get(latent, 0.0)),
                        "count": int(counts.get(latent, 0)),
                        "covered_tiles": (
                            int(len(tile_weights.get(latent, {})))
                            if config.ablation in {"tile_coverage", "fused_coverage"}
                            else None
                        ),
                        "tile_coverage_ratio": (
                            float(
                                len(tile_weights.get(latent, {}))
                                / (config.tile_rows * config.tile_cols)
                            )
                            if config.ablation in {"tile_coverage", "fused_coverage"}
                            else None
                        ),
                        "greedy_gain": greedy_gain.get(latent),
                        "selected": False,
                    }
                    for rank, latent in enumerate(ranked_latents[: config.top_k])
                ],
            }
        )

    spatial_by_query = assign_unique_latents_per_query(
        ranked_by_query,
        config.latents_per_query,
    )
    spatial_flat: List[int] = []
    spatial_slot_targets: List[int] = []
    spatial_slot_records: List[Dict[str, int]] = []
    spatial_slot_quality: List[float] = []
    for query_offset, selected in enumerate(spatial_by_query):
        query_latent = int(current_frame_idx + query_offset)
        query_results[query_offset]["selected_latents"] = selected
        selected_set = set(selected)
        for item in query_results[query_offset]["topk"]:
            item["selected"] = int(item["latent_id"]) in selected_set
        spatial_flat.extend(selected)
        spatial_slot_targets.extend([query_latent for _ in selected])
        spatial_slot_quality.extend(
            quality_by_query[query_offset].get(int(latent), 0.0) for latent in selected
        )
        spatial_slot_records.extend(
            {
                "query_offset": int(query_offset),
                "query_latent": query_latent,
                "source_latent": int(latent),
            }
            for latent in selected
        )

    debug.update(
        {
            "available_lingbot_latents": int(available),
            "point_cloud_source": point_cloud_source,
            "source_point_count": int(len(xyz)),
            "candidate_points": int(total_candidate_points),
            "min_query_candidate_points": int(min_query_candidate_points or 0),
            "historical_min_latent": int(historical_min_latent),
            "historical_max_latent": int(historical_max_latent),
            "historical_min_chunk": int(historical_min_latent // config.chunk_size),
            "historical_max_chunk": int(historical_max_latent // config.chunk_size),
            "query_alignment": alignment,
            "query_results": query_results,
            "spatial_latents_by_query": spatial_by_query,
            "spatial_sources_unique": len(spatial_flat) == len(set(spatial_flat)),
            "retrieval_point_policy": (
                "exclude_temporal_sources_before_point_cap"
                if config.ablation == "fused_coverage"
                else "exclude_temporal_sources"
                if config.ablation in {"temporal_overlap", "s2_full"}
                else "all_available_sources"
            ),
            "fusion": (
                {
                    "zbuffer_scope": "per_source_latent",
                    "source_score": "near_weighted_tile_coverage",
                    "ranking": "greedy_marginal_tile_quality",
                }
                if config.ablation == "fused_coverage"
                else None
            ),
        }
    )
    if total_candidate_points < config.min_visible_points:
        debug["fallback_reason"] = "too_few_candidate_points"
        return None, debug
    if any(len(selected) < config.latents_per_query for selected in spatial_by_query):
        debug["fallback_reason"] = "too_few_unique_ranked_latents_for_query"
        return None, debug

    selected_history, selected_targets, _ = order_context_slots_by_source(
        list(initial) + list(temporal) + spatial_flat,
        [int(current_frame_idx) for _ in initial]
        + [int(current_frame_idx) for _ in temporal]
        + spatial_slot_targets,
    )
    quality_by_pair = {
        (int(source), int(target)): float(quality)
        for source, target, quality in zip(
            spatial_flat, spatial_slot_targets, spatial_slot_quality
        )
    }
    selected_slot_spatial_quality = [
        quality_by_pair.get((int(source), int(target)))
        for source, target in zip(selected_history, selected_targets)
    ]
    debug.update(
        {
            "status": "ok",
            "fallback_reason": None,
            "initial_latents": initial,
            "temporal_latents": temporal,
            "spatial_latents": spatial_flat,
            "spatial_slot_targets": spatial_slot_targets,
            "spatial_slot_records": spatial_slot_records,
            "spatial_source_scores": {
                int(item["latent_id"]): float(
                    item.get(
                        "ucm_score", item.get("score", item.get("baseline_score", 0.0))
                    )
                )
                for query in query_results
                for item in query.get("topk", [])
                if item.get("selected")
            },
            "selected_latents": selected_history,
            "selected_target_latents": selected_targets,
            "selected_slot_spatial_quality": selected_slot_spatial_quality,
            "context_slot_order": "source_chronological_latest_temporal_last",
            "selected_latents_unique": sorted(set(selected_history)),
            "selected_spatial_slot_count": int(len(spatial_flat)),
            "selected_total_slot_count": int(len(selected_history)),
        }
    )
    return selected_history, debug


def select_point_cloud_memory(
    runtime: Dict[str, object],
    trajectory: TrajectoryState,
    *,
    current_frame_idx: int,
    fov_selected: Sequence[int],
    config: PointCloudRetrievalConfig,
) -> Tuple[Optional[List[int]], Dict[str, object]]:
    """Return initial + temporal + p2c spatial latents, or None with fallback debug."""
    debug: Dict[str, object] = {
        "status": "fallback",
        "fallback_reason": None,
        "current_frame_idx": int(current_frame_idx),
        "ablation": config.ablation,
        "lingbot_pose_convention": str(runtime.get("lingbot_pose_convention", "w2c"))
        .strip()
        .lower(),
        "fov_selected_latents": [int(i) for i in fov_selected],
    }
    initial, temporal = _layout(current_frame_idx, config)
    current_chunk = current_frame_idx // config.chunk_size
    historical_min_chunk = config.initial_chunks
    historical_max_chunk = current_chunk - config.temporal_chunks - 1
    if historical_max_chunk < historical_min_chunk:
        debug["fallback_reason"] = "point_cloud_warmup"
        return None, debug
    required = {"lingbot_w2c", "lingbot_Ks"}
    if not required.issubset(runtime):
        debug["fallback_reason"] = "lingbot_runtime_not_ready"
        return None, debug
    available = _runtime_available_frames(runtime, current_frame_idx)
    if available <= 0:
        debug["fallback_reason"] = "lingbot_history_empty"
        return None, debug
    trajectory.update_lingbot_poses(
        runtime["lingbot_w2c"],
        available,
        convention=runtime.get("lingbot_pose_convention", "w2c"),
    )
    query_c2w, alignment = trajectory.query_c2w_in_lingbot_space(
        current_frame_idx,
        config.chunk_size,
        alignment_mode=config.query_alignment_mode,
        local_window=config.query_alignment_window,
        scale_min=config.query_alignment_scale_min,
        scale_max=config.query_alignment_scale_max,
        min_command_step=config.query_alignment_min_command_step,
        orientation_weight=config.query_alignment_orientation_weight,
        min_pairs=config.query_alignment_min_pairs,
        anchor_max_translation_step=config.query_alignment_anchor_max_translation_step,
        anchor_max_rotation_step_deg=config.query_alignment_anchor_max_rotation_step_deg,
    )
    if not query_c2w:
        debug["fallback_reason"] = "query_pose_empty"
        return None, debug
    point_blocks = runtime.get("lingbot_point_blocks")
    points_dir = runtime.get("lingbot_points_dir")
    if runtime.get("world_points") is None and not point_blocks and not points_dir:
        debug["fallback_reason"] = "official_unprojected_points_not_ready"
        return None, debug
    if point_blocks:
        xyz, confidence, source_chunks = load_source_point_blocks(
            point_blocks,
            frame_count=available,
            max_points=config.max_points,
            source_field="source_chunk_id",
            source_min=(historical_min_chunk if config.ablation == "s2_full" else None),
            source_max=(
                historical_max_chunk
                if config.ablation in {"temporal_overlap", "s2_full"}
                else None
            ),
        )
        point_cloud_source = "lingbot_incremental_memory_blocks"
    elif points_dir:
        xyz, confidence, source_chunks = load_source_point_shards(
            points_dir,
            frame_count=available,
            max_points=config.max_points,
            source_field="source_chunk_id",
            source_min=(historical_min_chunk if config.ablation == "s2_full" else None),
            source_max=(
                historical_max_chunk
                if config.ablation in {"temporal_overlap", "s2_full"}
                else None
            ),
        )
        point_cloud_source = "wmfactory_source_map_online_shards"
    else:
        xyz, confidence, source_chunks = build_source_world_point_cloud(
            runtime["world_points"],
            runtime.get("world_points_conf"),
            frame_count=available,
            config=config,
            source_chunk_min=(
                historical_min_chunk if config.ablation == "s2_full" else None
            ),
            source_chunk_max=(
                historical_max_chunk if config.ablation == "temporal_overlap" else None
            ),
        )
        point_cloud_source = "gctbase_unproject_depth_to_world"
    if not len(xyz):
        debug["fallback_reason"] = "source_points_empty"
        return None, debug
    height, width = _runtime_image_hw(runtime)
    query_K = _numpy(runtime["lingbot_Ks"])[available - 1]
    candidate_source = (source_chunks >= historical_min_chunk) & (
        source_chunks <= historical_max_chunk
    )
    weighted_scores: Dict[int, float] = {}
    count_scores: Dict[int, int] = {}
    cell_weights: Dict[int, Dict[int, float]] = {}
    tile_weights: Dict[int, Dict[int, float]] = {}
    candidate_points = 0
    for query_idx, c2w in enumerate(query_c2w):
        visible, pixels, projected_depth = _project(
            xyz, np.linalg.inv(c2w), query_K, height, width, config
        )
        if config.ablation == "zbuffer":
            visible = _zbuffer(visible, pixels, projected_depth, height, width, config)
        indices = np.flatnonzero(visible & candidate_source)
        if config.ablation == "s2_full" and len(indices):
            kept = []
            indexed_sources = source_chunks[indices]
            for source in np.unique(indexed_sources):
                source_indices = indices[indexed_sources == source]
                kept.append(
                    _zbuffer_indices(
                        source_indices, pixels, projected_depth, height, width, config
                    )
                )
            indices = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
        candidate_points += len(indices)
        if not len(indices):
            continue
        chunks = source_chunks[indices].astype(np.int64)
        weights = confidence[indices].astype(np.float64)
        weighted = np.bincount(chunks, weights=weights)
        counts = np.bincount(chunks)
        for chunk in np.flatnonzero(weighted > 0):
            weighted_scores[int(chunk)] = weighted_scores.get(int(chunk), 0.0) + float(
                weighted[chunk]
            )
            count_scores[int(chunk)] = count_scores.get(int(chunk), 0) + int(
                counts[chunk]
            )
        if config.ablation in {"baseline", "zbuffer", "temporal_overlap"}:
            continue
        for point_idx in indices:
            chunk = int(source_chunks[point_idx])
            weight = float(confidence[point_idx])
            if config.ablation == "greedy_submodular":
                gw = (width + config.greedy_cell_size - 1) // config.greedy_cell_size
                gh = (height + config.greedy_cell_size - 1) // config.greedy_cell_size
                key = (
                    query_idx * gw * gh
                    + int(pixels[point_idx, 1] // config.greedy_cell_size) * gw
                    + int(pixels[point_idx, 0] // config.greedy_cell_size)
                )
                chunk_cells = cell_weights.setdefault(chunk, {})
                chunk_cells[key] = max(chunk_cells.get(key, 0.0), weight)
            if config.ablation in {"tile_coverage", "s2_full"}:
                u = min(
                    config.tile_cols - 1,
                    int(pixels[point_idx, 0] * config.tile_cols / width),
                )
                v = min(
                    config.tile_rows - 1,
                    int(pixels[point_idx, 1] * config.tile_rows / height),
                )
                key = (
                    query_idx * config.tile_cols * config.tile_rows
                    + v * config.tile_cols
                    + u
                )
                chunk_tiles = tile_weights.setdefault(chunk, {})
                chunk_tiles[key] = max(chunk_tiles.get(key, 0.0), weight)
    debug.update(
        {
            "available_lingbot_latents": int(available),
            "point_cloud_source": point_cloud_source,
            "source_point_count": int(len(xyz)),
            "candidate_points": int(candidate_points),
            "historical_min_chunk": int(historical_min_chunk),
            "historical_max_chunk": int(historical_max_chunk),
            "retrieval_point_policy": (
                "exclude_temporal_sources_before_point_cap"
                if config.ablation == "s2_full"
                else "exclude_temporal_sources"
                if config.ablation == "temporal_overlap"
                else "all_available_sources"
            ),
            "query_alignment": alignment,
            "fusion": (
                {
                    "temporal_prefilter": True,
                    "zbuffer_scope": "per_candidate_chunk",
                    "chunk_score": "tile_coverage",
                    "pair_selector": "greedy_marginal_tile_coverage",
                }
                if config.ablation == "s2_full"
                else None
            ),
        }
    )
    if candidate_points < config.min_visible_points:
        debug["fallback_reason"] = "too_few_candidate_points"
        return None, debug
    scores = dict(weighted_scores)
    if config.ablation in {"tile_coverage", "s2_full"}:
        scores = {
            chunk: float(sum(values.values())) for chunk, values in tile_weights.items()
        }
    selected: List[int] = []
    greedy_gain: Dict[int, float] = {}
    if config.ablation in {"greedy_submodular", "s2_full"}:
        remaining = set(weighted_scores)
        covered = set()
        while remaining and len(selected) < config.spatial_chunks:
            best = max(
                remaining,
                key=lambda chunk: (
                    sum(
                        weight
                        for key, weight in (
                            tile_weights.get(chunk, {})
                            if config.ablation == "s2_full"
                            else cell_weights.get(chunk, {})
                        ).items()
                        if key not in covered
                    ),
                    weighted_scores[chunk],
                    -chunk,
                ),
            )
            coverage_weights = (
                tile_weights.get(best, {})
                if config.ablation == "s2_full"
                else cell_weights.get(best, {})
            )
            greedy_gain[best] = float(
                sum(
                    weight
                    for key, weight in coverage_weights.items()
                    if key not in covered
                )
            )
            selected.append(best)
            covered.update(coverage_weights)
            remaining.remove(best)
        ranked = selected + sorted(
            remaining, key=lambda chunk: (weighted_scores[chunk], -chunk), reverse=True
        )
    else:
        ranked = sorted(
            scores,
            key=lambda chunk: (scores[chunk], weighted_scores.get(chunk, 0.0), -chunk),
            reverse=True,
        )
        selected = ranked[: config.spatial_chunks]
    if not selected:
        debug["fallback_reason"] = "no_scored_chunks"
        return None, debug
    spatial = [
        latent
        for chunk in selected
        for latent in range(chunk * config.chunk_size, (chunk + 1) * config.chunk_size)
    ]
    spatial_source_scores = {
        int(latent): float(scores.get(chunk, weighted_scores.get(chunk, 0.0)))
        for chunk in selected
        for latent in range(chunk * config.chunk_size, (chunk + 1) * config.chunk_size)
    }
    result = sorted(set(initial + temporal + spatial))
    spatial_set = set(spatial)
    spatial_target_by_source = {
        int(source): _p2c_target_latent(source, current_frame_idx, config.chunk_size)
        for source in spatial_set
    }
    selected_targets = [
        spatial_target_by_source.get(int(source), -1) for source in result
    ]
    total_query_tiles = max(1, len(query_c2w) * config.tile_cols * config.tile_rows)
    spatial_quality_by_source = {
        int(latent): max(
            0.0,
            min(1.0, len(tile_weights.get(chunk, {})) / total_query_tiles),
        )
        for chunk in selected
        for latent in range(chunk * config.chunk_size, (chunk + 1) * config.chunk_size)
    }
    debug.update(
        {
            "status": "ok",
            "fallback_reason": None,
            "selected_chunks": [int(i) for i in selected],
            "initial_latents": initial,
            "temporal_latents": temporal,
            "spatial_latents": spatial,
            "spatial_source_scores": spatial_source_scores,
            "selected_latents": result,
            "spatial_slot_targets": [
                spatial_target_by_source[int(source)] for source in spatial
            ],
            "spatial_slot_records": [
                {
                    "query_offset": int(source) % config.chunk_size,
                    "query_latent": spatial_target_by_source[int(source)],
                    "source_latent": int(source),
                }
                for source in spatial
            ],
            "selected_target_latents": selected_targets,
            "selected_slot_spatial_quality": [
                spatial_quality_by_source.get(int(source))
                if int(source) in spatial_set
                else None
                for source in result
            ],
            "union_covered_tiles": (
                int(len(covered)) if config.ablation == "s2_full" else None
            ),
            "union_tile_coverage_ratio": (
                float(
                    len(covered)
                    / max(1, len(query_c2w) * config.tile_cols * config.tile_rows)
                )
                if config.ablation == "s2_full"
                else None
            ),
            "topk": [
                {
                    "rank": rank + 1,
                    "chunk_id": int(chunk),
                    "score": float(scores.get(chunk, weighted_scores.get(chunk, 0.0))),
                    "baseline_score": float(weighted_scores.get(chunk, 0.0)),
                    "count": int(count_scores.get(chunk, 0)),
                    "covered_tiles": (
                        int(len(tile_weights.get(chunk, {})))
                        if config.ablation in {"tile_coverage", "s2_full"}
                        else None
                    ),
                    "tile_coverage_ratio": (
                        float(
                            len(tile_weights.get(chunk, {}))
                            / max(
                                1, len(query_c2w) * config.tile_cols * config.tile_rows
                            )
                        )
                        if config.ablation in {"tile_coverage", "s2_full"}
                        else None
                    ),
                    "greedy_gain": greedy_gain.get(chunk),
                    "selected": chunk in selected,
                }
                for rank, chunk in enumerate(ranked[: config.top_k])
            ],
        }
    )
    if config.debug_projection_top_k > 0:
        debug["topk_projection_image_size"] = [int(width), int(height)]
        debug["topk_projection_query_K"] = _pixel_intrinsic(
            query_K, height, width
        ).tolist()
        debug["topk_projection_query_c2w"] = [
            np.asarray(c2w, dtype=np.float64).tolist() for c2w in query_c2w
        ]
        debug["topk_projection_debug"] = _projection_debug_for_chunks(
            xyz,
            source_chunks,
            ranked[: config.top_k],
            query_c2w,
            query_K,
            height,
            width,
            config,
        )
    return result, debug
