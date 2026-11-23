# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Loader and validator for the frozen UnrealZoo memory cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ContextBundle


_UNREAL_POSE_FORMAT = {
    "source": "unreal_engine",
    "field": "primary_pose",
    "order": [
        "x_cm",
        "y_cm",
        "z_cm",
        "pitch_deg",
        "yaw_deg",
        "roll_deg",
    ],
    "position_units": "cm",
    "rotation_units": "deg",
    "coordinate_system": "unreal_engine",
    "handedness": "left",
    "euler_order": "pitch_yaw_roll",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _camera_matrix(row: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
    value = row.get("camera_to_world")
    if value is None:
        raise ValueError(f"pose row {row.get('frame_id')} has no camera_to_world")
    if len(value) != 4 or any(len(item) != 4 for item in value):
        raise ValueError("camera_to_world must be a 4x4 matrix")
    return tuple(tuple(float(component) for component in item) for item in value)


def _context_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _with_inferred_pose_format(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fill the pose contract omitted by early UnrealZoo short-clip exports."""

    if manifest.get("pose_format"):
        return manifest
    if str(manifest.get("dataset", "")).strip() != "unrealzoo_memory_short_clips":
        return manifest
    normalized = dict(manifest)
    normalized["pose_format"] = {**_UNREAL_POSE_FORMAT, "inferred": True}
    return normalized


def load_context_bundle(case_dir: str | Path) -> ContextBundle:
    """Load and validate all GT Context RGB/Pose rows for one case."""

    root = Path(case_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    poses = _read_json(root / "pose" / "poses.json")
    if not isinstance(manifest, dict) or not isinstance(poses, list):
        raise ValueError(f"invalid memory case files under {root}")
    manifest = _with_inferred_pose_format(manifest)
    context = manifest.get("split", {}).get("context")
    eval_range = manifest.get("split", {}).get("eval")
    if not context or not eval_range:
        raise ValueError(f"manifest split is incomplete: {root / 'manifest.json'}")
    context_start, context_end = (int(context[0]), int(context[1]))
    eval_start, eval_end = (int(eval_range[0]), int(eval_range[1]))
    if len(poses) < context_end:
        raise ValueError(
            f"pose rows ({len(poses)}) do not cover context end ({context_end})"
        )

    rows = poses[context_start:context_end]
    frame_ids = tuple(int(row["frame_id"]) for row in rows)
    if frame_ids != tuple(range(context_start, context_end)):
        raise ValueError("pose frame ids do not match the manifest context range")
    rgb_paths: list[Path] = []
    timestamps: list[float] = []
    ue_poses: list[tuple[float, ...]] = []
    camera_poses: list[tuple[tuple[float, ...], ...]] = []
    for row in rows:
        relative_rgb = str(row.get("rgb") or row.get("files", {}).get("rgb") or "")
        if not relative_rgb:
            raise ValueError(f"pose row {row.get('frame_id')} has no RGB path")
        rgb_path = root / relative_rgb
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        rgb_paths.append(rgb_path)
        timestamps.append(float(row["timestamp"]))
        pose = row.get("primary_pose") or row.get("pose_unreal")
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError(f"pose row {row.get('frame_id')} has invalid primary_pose")
        ue_poses.append(tuple(float(item) for item in pose))
        camera_poses.append(_camera_matrix(row))

    resolution = manifest.get("resolution") or manifest.get("camera", {}).get(
        "resolution"
    )
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("manifest resolution must be [width, height]")
    return ContextBundle(
        case_id=str(manifest["case_id"]),
        frame_ids=frame_ids,
        rgb_paths=tuple(rgb_paths),
        timestamps_sec=tuple(timestamps),
        ue_poses=tuple(ue_poses),
        camera_to_world=tuple(camera_poses),
        context_range=(context_start, context_end),
        eval_range=(eval_start, eval_end),
        fps=float(manifest["fps"]),
        width=int(resolution[0]),
        height=int(resolution[1]),
        horizontal_fov_deg=float(
            manifest.get("camera", {}).get("horizontal_fov_deg", 90.0)
        ),
        context_sha256=_context_hash(tuple(rgb_paths)),
        metadata=manifest,
    )


__all__ = ["load_context_bundle"]
