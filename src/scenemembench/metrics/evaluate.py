"""Single-pair evaluation; Seen searches Query, Novel uses its endpoint."""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scenemembench.io import read_frames, sha256, write_json
from .image import (
    RotationMetricConfig,
    compute_pixel_metrics,
    DINOv2Scorer,
    LPIPSScorer,
    native_gt_frame,
)
from .selection import (
    PoseToleranceConfig,
    terminal_pose_candidates,
    SIFTCorrespondenceBackend,
    select_reference_frame,
)
from .geometry import endpoint_scale_invariant_depth_metrics, camera_aware_normal_map


def pose_matrix(row):
    # Source convenience matrices may discard pitch; use complete UE Euler pose.
    raw = row.get("primary_pose", row.get("pose_unreal"))
    if raw is None:
        return np.asarray(row["camera_to_world"], dtype=float)
    x, y, z, pitch, yaw, roll = raw
    p, y, r = np.deg2rad([pitch, yaw, roll])
    cp, sp, cy, sy, cr, sr = (
        np.cos(p),
        np.sin(p),
        np.cos(y),
        np.sin(y),
        np.cos(r),
        np.sin(r),
    )
    m = np.eye(4)
    m[:3, :3] = [
        [cp * cy, sr * sp * cy - cr * sy, -(cr * sp * cy + sr * sy)],
        [cp * sy, sr * sp * sy + cr * cy, cy * sr - cr * sp * sy],
        [sp, -sr * cp, cr * cp],
    ]
    m[:3, 3] = [x, y, z]
    return m


def geometry_pair(reference_depth, generated_depth, intrinsics):
    if reference_depth.shape != generated_depth.shape:
        raise ValueError("Depth sizes differ")
    result = endpoint_scale_invariant_depth_metrics(reference_depth, generated_depth)
    rn, rv = camera_aware_normal_map(reference_depth, intrinsics)
    gn, gv = camera_aware_normal_map(generated_depth, intrinsics)
    valid = rv & gv
    result["camera_aware_normal_cosine"] = (
        float(np.mean(np.sum(rn[valid] * gn[valid], axis=-1))) if valid.any() else None
    )
    return result


def infer_depth(image, model):
    # Independent inference: no reference/generated joint batch and no input pose.
    result = model.inference(
        [image],
        process_res=504,
        process_res_method="upper_bound_resize",
        align_to_input_ext_scale=False,
    )
    depth = np.asarray(result.depth[0], dtype=np.float32)
    return cv2.resize(
        depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )


def evaluate(
    case_dir,
    prediction,
    output,
    *,
    task,
    model_id="generic",
    max_rotation_deg=None,
    metrics="pixel",
    device="cpu",
    dino_model="facebook/dinov2-base",
    da3_model=None,
):
    root = Path(case_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "pose/poses.json").read_text())
    start, end = manifest["split"]["eval"]
    if not 0 <= start < end <= len(rows):
        raise ValueError("Invalid query range")
    if manifest.get("protocol") == "STRAIGHT_LOOP":
        if manifest.get("phase_ranges", {}).get("eval_turnaround") != [start, end]:
            raise ValueError(
                "Straight query must equal eval_turnaround; repair stale assets first"
            )
    generated, video = read_frames(prediction)
    generated = generated[video["frames"] - 1]
    poses = [pose_matrix(r) for r in rows[start:end]]
    config = PoseToleranceConfig(
        max_rotation_deg=max_rotation_deg,
        max_translation_m=float("inf"),
        max_candidate_frames=end - start,
    )
    if task == "seen":
        candidates = terminal_pose_candidates(
            poses, eval_start_source_frame=start, config=config
        )
    elif task == "novel":
        candidates = []
    else:
        raise ValueError("task must be seen or novel")
    query_video = root / "eval.mp4"
    wanted = (
        [c.eval_frame_index for c in candidates] if candidates else [end - start - 1]
    )
    if query_video.exists():
        refs, info = read_frames(query_video, wanted)
        if info["frames"] != end - start:
            raise ValueError(
                "eval.mp4 length does not match canonical query range; no implicit tail alignment"
            )
    else:
        refs = {
            i: np.asarray(
                Image.open(
                    root
                    / (rows[start + i].get("rgb") or rows[start + i]["files"]["rgb"])
                ).convert("RGB")
            )
            for i in wanted
        }
    selection = {
        "task": task,
        "rule": ("seen-query-search-v1" if max_rotation_deg is None else "seen-terminal-angular-v1")
        if task == "seen"
        else "novel-fixed-endpoint-v1",
    }
    index = end - start - 1
    if candidates:
        selector_refs = {
            i: native_gt_frame(image, model_id) if model_id != "generic" else image
            for i, image in refs.items()
        }
        selected, score, audit, reason = select_reference_frame(
            generated,
            candidates,
            selector_refs,
            SIFTCorrespondenceBackend(config),
            config,
        )
        index = selected.eval_frame_index
        selection.update(
            max_rotation_deg=max_rotation_deg,
            candidate_source_ids=[c.source_frame_id for c in candidates],
            selected_rotation_deg=selected.rotation_offset_deg,
            fallback_reason=reason,
            score=score.to_dict(),
        )
    reference = refs[index]
    metric_reference = (
        native_gt_frame(reference, model_id) if model_id != "generic" else reference
    )
    mc = RotationMetricConfig(dino_model=dino_model)
    values = {
        k: v[0]
        for k, v in compute_pixel_metrics([metric_reference], [generated], mc).items()
    }
    if metrics == "all":
        values["dino_v2"] = DINOv2Scorer(mc, device=device).score(
            [metric_reference], [generated]
        )[0]
        values["lpips"] = LPIPSScorer(mc, device=device).score(
            [metric_reference], [generated]
        )[0]
        if not da3_model:
            raise ValueError("--da3-model is required for all six metrics")
        from depth_anything_3.api import DepthAnything3

        depth_model = DepthAnything3.from_pretrained(da3_model).to(device).eval()
        ref = cv2.resize(
            reference,
            (generated.shape[1], generated.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        w, h = manifest["resolution"]
        f = w / (
            2
            * np.tan(
                np.deg2rad(manifest.get("camera", {}).get("horizontal_fov_deg", 90)) / 2
            )
        )
        k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
        k[0] *= generated.shape[1] / w
        k[1] *= generated.shape[0] / h
        values.update(
            geometry_pair(
                infer_depth(ref, depth_model), infer_depth(generated, depth_model), k
            )
        )
    result = {
        "schema_version": 1,
        "case_id": manifest["case_id"],
        "prediction_sha256": sha256(prediction),
        "generated_endpoint": video["frames"] - 1,
        "reference_source_frame": start + index,
        "selection": selection,
        "metrics_requested": metrics,
        "metrics": values,
    }
    write_json(output, result)
    return result
