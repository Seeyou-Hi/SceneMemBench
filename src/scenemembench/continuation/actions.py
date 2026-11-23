# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Native action/pose planning extracted from the original continuation launcher."""

from __future__ import annotations
import json
import os
import statistics
from pathlib import Path
from typing import Any
import numpy as np


def _infer_eval_rotation_speed(case_dir: Path, context: Any) -> float | None:
    with (case_dir / "pose" / "poses.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    rotation_axis = (
        str(context.metadata.get("task", {}).get("rotation_axis", "yaw"))
        .strip()
        .lower()
    )

    def _relative_rotation(row: Any) -> Any:
        """Accept both legacy top-level and collected nested yaw fields."""
        action = row.get("action", {}) if isinstance(row, dict) else {}
        if not isinstance(action, dict):
            action = {}
        if rotation_axis in {"pitch", "x"}:
            pose = row.get("primary_pose") or row.get("pose_unreal")
            if isinstance(pose, list) and len(pose) >= 4:
                return float(pose[3])
            value = action.get("pitch")
            if value is not None:
                return value
        value = row.get("relative_yaw_unwrapped_deg")
        if value is None:
            value = action.get("relative_yaw_unwrapped_deg")
        if value is None:
            value = action.get("relative_angle_unwrapped_deg")
        return value

    eval_start, eval_end = context.eval_range
    rates: list[float] = []
    for previous, current in zip(
        rows[max(eval_start - 1, 0) : eval_end - 1], rows[eval_start:eval_end]
    ):
        action_type = str(current.get("action", {}).get("type", "")).strip().lower()
        if action_type not in {"in_place_360", "rotate", "turn", "turn_in_place"}:
            continue
        previous_yaw = _relative_rotation(previous)
        current_yaw = _relative_rotation(current)
        delta_time = float(current["timestamp"]) - float(previous["timestamp"])
        if previous_yaw is None or current_yaw is None or delta_time <= 0.0:
            continue
        rate = abs(float(current_yaw) - float(previous_yaw)) / delta_time
        if rate > 0.0:
            rates.append(rate)
    return float(statistics.median(rates)) if rates else None


def _set_rotation_environment(model_id: str, rotation_speed: float | None) -> None:
    if rotation_speed is None:
        return
    if rotation_speed <= 0.0:
        raise ValueError("--rotation-degrees-per-second must be positive")
    environment_names = {
        "lingbot-world": "SCENEMEMBENCH_LINGBOT_ROTATION_DEGREES_PER_SECOND",
        "lingbot-world-v2": "SCENEMEMBENCH_LINGBOT_ROTATION_DEGREES_PER_SECOND",
        "dreamx-world-5b": "SCENEMEMBENCH_DREAMX_AR_ROTATION_DEGREES_PER_SECOND",
        "matrix-game-3": "SCENEMEMBENCH_MATRIX_GAME3_ROTATION_DEGREES_PER_SECOND",
        "matrix-game-3.5": "SCENEMEMBENCH_MATRIX_GAME35_ROTATION_DEGREES_PER_SECOND",
        "sana-wm-streaming": "SCENEMEMBENCH_SANA_WM_ROTATION_DEG_PER_FRAME",
        "sana-wm": "SCENEMEMBENCH_SANA_WM_ROTATION_DEG_PER_FRAME",
    }
    environment_name = environment_names.get(model_id)
    if environment_name is not None:
        value = (
            rotation_speed / 16.0
            if model_id in {"sana-wm", "sana-wm-streaming"}
            else rotation_speed
        )
        os.environ[environment_name] = str(value)


def _configure_pipeline_rotation(
    model_id: str, pipeline: Any, rotation_speed: float | None
) -> None:
    if rotation_speed is None or model_id not in {
        "hy-worldplay",
        "hy-worldplay-pointcloud",
    }:
        return
    # HY applies yaw once per four-pixel-frame causal latent motion.
    pipeline.operators.yaw_speed_deg = rotation_speed * 4.0 / 24.0


def _dataset_eval_actions(
    case_dir: Path,
    context: Any,
    pipeline: Any,
    capabilities: Any,
    *,
    gamecraft_control_mode: str = "native",
    sana_eval_chunks: int = 0,
    sana_eval_idle_frames: int = 0,
):
    """Lazily derive native actions after prefill cadence becomes available."""

    from scenemembench.continuation.controller.open_loop import (
        canonical_action_for_row,
        iter_open_loop_native_actions,
    )
    from scenemembench.continuation.contracts import ActionCommand

    with (case_dir / "pose" / "poses.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if gamecraft_control_mode not in {"native", "prescribed"}:
        raise ValueError(
            f"unsupported GameCraft control mode: {gamecraft_control_mode}"
        )
    pitch_case = str(
        context.metadata.get("task", {}).get("rotation_axis", "")
    ).strip().lower() in {
        "pitch",
        "x",
    }
    prescribed_for_model = capabilities.model_id in {
        "matrix-game-3.5",
        "sana-wm",
        "sana-wm-streaming",
    } or (
        capabilities.model_id == "hunyuan-gamecraft"
        and gamecraft_control_mode == "prescribed"
    )
    # MG3.5's native camera controller is the stable path for pitch.  Its
    # prescribed-pose path is retained for yaw and for explicit pose tests.
    if capabilities.model_id == "matrix-game-3.5" and pitch_case:
        prescribed_for_model = False
    rows_by_frame = {int(row["frame_id"]): row for row in rows}
    native_actions = list(
        iter_open_loop_native_actions(
            context,
            rows,
            native_fps=capabilities.native_fps,
            next_output_frames=pipeline.realtime_next_output_frames,
            tail_idle_frames=(
                sana_eval_idle_frames
                if capabilities.model_id in {"sana-wm", "sana-wm-streaming"}
                else 0
            ),
        )
    )
    if capabilities.model_id == "zing-world-0.5":
        from scenemembench._native.synthesis.visual_generation.zing_world.realtime import (
            VISUAL_CALIBRATED_90_STEPS,
        )

        turn_names = {"turn_left", "turn_right", "look_up", "look_down"}
        eval_start, eval_end = context.eval_range
        turn_rows: list[tuple[int, str]] = []
        for frame_id in range(eval_start, eval_end):
            current = rows_by_frame[frame_id]
            canonical = canonical_action_for_row(
                current,
                rows_by_frame.get(frame_id - 1),
            )
            if canonical in turn_names:
                turn_rows.append((frame_id, canonical))
        if not turn_rows:
            raise ValueError(
                "Zing's calibrated 90-degree Eval route requires at least one "
                "turn action in manifest split.eval"
            )
        from collections import Counter

        canonical_action = Counter(action for _, action in turn_rows).most_common(1)[0][
            0
        ]
        source_frame_ids = [
            frame_id for frame_id, action in turn_rows if action == canonical_action
        ]
        output_frames = int(pipeline.realtime_next_output_frames())
        if output_frames != int(capabilities.native_chunk_frames):
            raise ValueError(
                "Zing realtime cadence changed during calibrated Eval planning: "
                f"{output_frames} != {capabilities.native_chunk_frames}"
            )
        total_frames = VISUAL_CALIBRATED_90_STEPS * output_frames
        sampled_source_ids = [
            source_frame_ids[index]
            for index in np.rint(
                np.linspace(0, len(source_frame_ids) - 1, total_frames)
            ).astype(int)
        ]
        phase_names = tuple(
            dict.fromkeys(
                phase
                for action in native_actions
                if any(name == canonical_action for name, _ in action.action_segments)
                for phase in action.phase_names
            )
        )
        for step_id in range(VISUAL_CALIBRATED_90_STEPS):
            start = step_id * output_frames
            end = start + output_frames
            action = ActionCommand(
                step_id=step_id,
                canonical_action=canonical_action,
                duration_frames=output_frames,
                action_segments=((canonical_action, float(output_frames)),),
                trajectory_source_frame_ids=tuple(sampled_source_ids[start:end]),
                phase_names=phase_names,
                conditioning_only=(False,) * output_frames,
                trajectory_mode="canonical_action",
                source="open_loop_native_action_v0",
            )
            print(
                json.dumps(
                    {
                        "event": "eval_chunk_action",
                        "step_id": step_id,
                        "model_output_frames": output_frames,
                        "canonical_action": canonical_action,
                        "trajectory_mode": "canonical_action",
                        "control_mode": "zing_visual_calibrated_90deg",
                        "target_eval_steps": VISUAL_CALIBRATED_90_STEPS,
                        "target_eval_frames": total_frames,
                        "source": action.source,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            yield action
        return
    if capabilities.model_id in {"sana-wm", "sana-wm-streaming"} and sana_eval_chunks:
        if sana_eval_chunks < 1:
            raise ValueError("--sana-eval-chunks must be positive or zero")
        output_frames = int(pipeline.realtime_next_output_frames())
        source_ids = [
            frame_id
            for action in native_actions
            for frame_id, padding in zip(
                action.trajectory_source_frame_ids,
                action.conditioning_only,
            )
            if not padding
        ]
        source_poses = np.asarray(
            [rows_by_frame[frame_id]["camera_to_world"] for frame_id in source_ids],
            dtype=np.float64,
        )
        target_count = sana_eval_chunks * output_frames
        target_positions = np.linspace(0.0, len(source_poses) - 1, target_count)
        target_poses: list[np.ndarray] = []
        target_source_ids: list[int] = []
        for position in target_positions:
            left = int(np.floor(position))
            right = min(left + 1, len(source_poses) - 1)
            fraction = float(position - left)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = (1.0 - fraction) * source_poses[
                left, :3, 3
            ] + fraction * source_poses[right, :3, 3]
            rotation_blend = (1.0 - fraction) * source_poses[
                left, :3, :3
            ] + fraction * source_poses[right, :3, :3]
            u, _, vt = np.linalg.svd(rotation_blend)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
            pose[:3, :3] = rotation
            target_poses.append(pose)
            target_source_ids.append(source_ids[int(np.floor(position + 0.5))])
        target_poses[0] = source_poses[0].copy()
        target_poses[-1] = source_poses[-1].copy()
        phase_names = tuple(
            dict.fromkeys(
                phase for action in native_actions for phase in action.phase_names
            )
        )
        canonical_action = native_actions[0].canonical_action
        for step_id in range(sana_eval_chunks):
            start = step_id * output_frames
            end = start + output_frames
            action = ActionCommand(
                step_id=step_id,
                canonical_action=canonical_action,
                duration_frames=output_frames,
                action_segments=((canonical_action, float(output_frames)),),
                camera_trajectory_ue=tuple(
                    tuple(tuple(float(value) for value in row) for row in pose)
                    for pose in target_poses[start:end]
                ),
                trajectory_source_frame_ids=tuple(target_source_ids[start:end]),
                phase_names=phase_names,
                conditioning_only=(False,) * output_frames,
                trajectory_mode="prescribed_pose",
                source="open_loop_fixed_horizon_pose_v0",
            )
            print(
                json.dumps(
                    {
                        "event": "eval_chunk_action",
                        "step_id": step_id,
                        "model_output_frames": output_frames,
                        "canonical_action": canonical_action,
                        "trajectory_mode": action.trajectory_mode,
                        "control_mode": "prescribed_fixed_horizon",
                        "target_eval_chunks": sana_eval_chunks,
                        "source": action.source,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            yield action
        return

    for action in native_actions:
        if (
            capabilities.model_id == "hunyuan-gamecraft"
            and gamecraft_control_mode == "native"
            and not any(
                str(segment_action).strip().lower().startswith("turn_")
                for segment_action, _ in action.action_segments
            )
        ):
            # GameCraft has no native idle action.  Once the prescribed turn
            # has reached its target, stop emitting the padding/hold chunk;
            # the native comparison should contain only actual turn chunks.
            continue
        if prescribed_for_model:
            trajectory = tuple(
                tuple(
                    tuple(float(value) for value in coordinate)
                    for coordinate in rows_by_frame[frame_id]["camera_to_world"]
                )
                for frame_id in action.trajectory_source_frame_ids
            )
            action = ActionCommand(
                step_id=action.step_id,
                canonical_action=action.canonical_action,
                duration_frames=action.duration_frames,
                action_segments=action.action_segments,
                camera_trajectory_ue=trajectory,
                trajectory_source_frame_ids=action.trajectory_source_frame_ids,
                phase_names=action.phase_names,
                conditioning_only=action.conditioning_only,
                trajectory_mode="prescribed_pose",
                motion_scale=action.motion_scale,
                source=action.source,
            )
        if capabilities.model_id == "hunyuan-gamecraft":
            # GameCraft's native rotation action takes a total degree value,
            # not a normalized speed.  Preserve the GT yaw delta represented
            # by this native chunk so a mirror/rotation route is not reduced to
            # the upstream demo default of 0.2 degrees.
            source_ids = action.trajectory_source_frame_ids
            action = ActionCommand(
                step_id=action.step_id,
                canonical_action=action.canonical_action,
                duration_frames=action.duration_frames,
                action_segments=action.action_segments,
                camera_trajectory_ue=(
                    tuple(
                        tuple(
                            tuple(float(value) for value in coordinate)
                            for coordinate in rows_by_frame[frame_id]["camera_to_world"]
                        )
                        for frame_id in source_ids
                    )
                    if gamecraft_control_mode == "prescribed"
                    else ()
                ),
                trajectory_source_frame_ids=action.trajectory_source_frame_ids,
                phase_names=action.phase_names,
                conditioning_only=action.conditioning_only,
                trajectory_mode=(
                    "prescribed_pose"
                    if gamecraft_control_mode == "prescribed"
                    else "canonical_action"
                ),
                motion_scale=action.motion_scale,
                source=action.source,
            )
            source_ids = action.trajectory_source_frame_ids
            if source_ids:
                # Hold frames in MIRROR/VIEW_REVISIT routes may reset the
                # stored yaw to zero after the turn target.  Only use the
                # rows that are themselves turn actions; otherwise a turn
                # chunk followed by hold would look like a large reverse turn.
                turn_ids = [
                    frame_id
                    for frame_id in source_ids
                    if canonical_action_for_row(
                        rows_by_frame[frame_id],
                        rows_by_frame.get(frame_id - 1),
                    ).startswith("turn_")
                ]
                if not turn_ids:
                    turn_ids = []
            else:
                turn_ids = []
            if turn_ids:
                # Include the first turn increment from the frame immediately
                # before the chunk.  The pose row at turn_ids[0] already
                # contains that increment, so anchoring on it undercounts the
                # route (49.5 + 40.5 = 90.0, rather than 48 + 39 = 87).
                if turn_ids[0] == context.eval_range[0]:
                    first_yaw = 0.0
                else:
                    first_yaw = float(
                        rows_by_frame.get(
                            turn_ids[0] - 1, rows_by_frame[turn_ids[0]]
                        ).get("relative_yaw_unwrapped_deg", 0.0)
                    )
                last_row = rows_by_frame[turn_ids[-1]]
                last_yaw = last_row.get("relative_yaw_unwrapped_deg")
                if first_yaw is not None and last_yaw is not None:
                    yaw_delta = float(last_yaw) - float(first_yaw)
                    if abs(yaw_delta) > 1e-6:
                        action = ActionCommand(
                            step_id=action.step_id,
                            canonical_action=action.canonical_action,
                            duration_frames=action.duration_frames,
                            action_segments=action.action_segments,
                            camera_trajectory_ue=action.camera_trajectory_ue,
                            trajectory_source_frame_ids=action.trajectory_source_frame_ids,
                            phase_names=action.phase_names,
                            conditioning_only=action.conditioning_only,
                            trajectory_mode=action.trajectory_mode,
                            motion_scale=action.motion_scale,
                            rotation_deg=(yaw_delta, 0.0, 0.0),
                            source=action.source,
                        )
        if int(os.environ.get("RANK", "0") or "0") == 0:
            source_ids = action.trajectory_source_frame_ids
            print(
                json.dumps(
                    {
                        "event": "eval_chunk_action",
                        "step_id": action.step_id,
                        "source_frame_range": [min(source_ids), max(source_ids) + 1],
                        "model_output_frames": action.duration_frames,
                        "padded_native_frames": sum(action.conditioning_only),
                        "canonical_action": action.canonical_action,
                        "trajectory_mode": action.trajectory_mode,
                        "control_mode": (
                            gamecraft_control_mode
                            if capabilities.model_id == "hunyuan-gamecraft"
                            else "prescribed"
                            if action.trajectory_mode == "prescribed_pose"
                            else "native"
                        ),
                        "action_segments": [
                            list(segment) for segment in action.action_segments
                        ],
                        "phase_names": list(action.phase_names),
                        "source": action.source,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        yield action
