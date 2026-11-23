# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Open-loop native-action baseline derived from the frozen Eval timeline."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import ActionCommand, ContextBundle


ACTION_SOURCE = "open_loop_native_action_v0"
_HOLD_TYPES = {
    "boundary",
    "hold",
    "hold_view",
    "idle",
    "look_memory",
    "observe",
    "stop",
    "nomove",
}
_TURN_TYPES = {"in_place_360", "rotate", "turn", "turn_in_place"}
_ACTION_ALIASES = {
    "move_forward": "forward",
    "move_backward": "backward",
    "strafe_left": "left",
    "strafe_right": "right",
    "camera_left": "turn_left",
    "camera_right": "turn_right",
}


@dataclass(frozen=True, slots=True)
class NativeActionFrame:
    native_index: int
    source_frame_id: int
    phase_name: str
    canonical_action: str
    conditioning_only: bool = False


def _ranges(value: Any):
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        yield int(value[0]), int(value[1])
    elif isinstance(value, list):
        for item in value:
            yield from _ranges(item)


def _eval_phase_by_frame(context: ContextBundle) -> dict[int, str]:
    eval_start, eval_end = context.eval_range
    result: dict[int, str] = {}
    for name, raw_ranges in context.metadata.get("phase_ranges", {}).items():
        for start, end in _ranges(raw_ranges):
            for frame_id in range(max(start, eval_start), min(end, eval_end)):
                if frame_id in result:
                    raise ValueError(
                        f"overlapping Eval phases at source frame {frame_id}"
                    )
                result[frame_id] = str(name)
    missing = [
        frame_id for frame_id in range(eval_start, eval_end) if frame_id not in result
    ]
    if missing:
        raise ValueError(
            f"Eval phase coverage is incomplete at source frame {missing[0]}"
        )
    return result


def _turn_action(row: Mapping[str, Any], previous_row: Mapping[str, Any] | None) -> str:
    action = row.get("action", {})
    if not isinstance(action, Mapping):
        action = {}
    axis = str(action.get("axis", "yaw")).strip().lower()
    sign = action.get("turn_direction_sign")
    if sign is None:
        signed = action.get("signed_turn_deg")
        if signed is not None:
            sign = float(signed)
    if sign is None and previous_row is not None:
        previous = previous_row.get(
            "relative_yaw_unwrapped_deg",
            previous_row.get("action", {}).get("relative_yaw_unwrapped_deg"),
        )
        current = row.get(
            "relative_yaw_unwrapped_deg",
            action.get("relative_yaw_unwrapped_deg"),
        )
        if previous is not None and current is not None:
            sign = float(current) - float(previous)
    if axis in {"pitch", "x"}:
        return "look_up" if sign is None or float(sign) >= 0.0 else "look_down"
    return "turn_right" if sign is None or float(sign) >= 0.0 else "turn_left"


def canonical_action_for_row(
    row: Mapping[str, Any],
    previous_row: Mapping[str, Any] | None = None,
) -> str:
    action = row.get("action", {})
    if not isinstance(action, Mapping):
        action = {}
    action_type = str(action.get("type", "")).strip().lower()
    if action_type in _HOLD_TYPES:
        return "idle"
    if action_type in _TURN_TYPES:
        return _turn_action(row, previous_row)
    if action_type.startswith(("l_shaped_", "straight_")) or action_type in {
        "displaced_move",
        "forward",
        "walk_forward",
        "path_forward",
    }:
        return "forward"
    alias = _ACTION_ALIASES.get(action_type)
    if alias is not None:
        return alias
    raise ValueError(f"unsupported Eval action type: {action_type!r}")


def build_native_action_timeline(
    context: ContextBundle,
    pose_rows: Sequence[Mapping[str, Any]],
    *,
    native_fps: float,
) -> tuple[NativeActionFrame, ...]:
    """Resample source action labels once over the complete Eval duration."""

    if native_fps <= 0:
        raise ValueError("native_fps must be positive")
    eval_start, eval_end = context.eval_range
    if len(pose_rows) < eval_end:
        raise ValueError(
            f"pose rows ({len(pose_rows)}) do not cover Eval end ({eval_end})"
        )
    phase_by_frame = _eval_phase_by_frame(context)
    source_count = eval_end - eval_start
    native_count = max(1, int(round(source_count * native_fps / context.fps)))
    timeline: list[NativeActionFrame] = []
    for native_index in range(native_count):
        source_offset = min(
            source_count - 1,
            int(math.floor(native_index * context.fps / native_fps + 0.5)),
        )
        source_frame_id = eval_start + source_offset
        previous = pose_rows[source_frame_id - 1] if source_frame_id > 0 else None
        timeline.append(
            NativeActionFrame(
                native_index=native_index,
                source_frame_id=source_frame_id,
                phase_name=phase_by_frame[source_frame_id],
                canonical_action=canonical_action_for_row(
                    pose_rows[source_frame_id], previous
                ),
            )
        )
    return tuple(timeline)


def _segments(frames: Sequence[NativeActionFrame]) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    for frame in frames:
        if result and result[-1][0] == frame.canonical_action:
            name, duration = result[-1]
            result[-1] = (name, duration + 1.0)
        else:
            result.append((frame.canonical_action, 1.0))
    return tuple(result)


def iter_open_loop_native_actions(
    context: ContextBundle,
    pose_rows: Sequence[Mapping[str, Any]],
    *,
    native_fps: float,
    next_output_frames: Callable[[], int],
    tail_idle_frames: int = 0,
) -> Iterator[ActionCommand]:
    """Yield one dynamically-cadenced command at a time after resident prefill."""

    if tail_idle_frames < 0:
        raise ValueError("tail_idle_frames must be non-negative")
    timeline = list(
        build_native_action_timeline(context, pose_rows, native_fps=native_fps)
    )
    if tail_idle_frames:
        terminal = timeline[-1]
        for _ in range(tail_idle_frames):
            timeline.append(
                NativeActionFrame(
                    native_index=len(timeline),
                    source_frame_id=terminal.source_frame_id,
                    phase_name=terminal.phase_name,
                    canonical_action="idle",
                )
            )
    cursor = 0
    step_id = 0
    while cursor < len(timeline):
        output_frames = int(next_output_frames())
        if output_frames < 1:
            raise ValueError(
                "realtime_next_output_frames() must return a positive count"
            )
        frames = list(timeline[cursor : cursor + output_frames])
        padding = output_frames - len(frames)
        if padding:
            last = frames[-1]
            frames.extend(
                NativeActionFrame(
                    native_index=last.native_index,
                    source_frame_id=last.source_frame_id,
                    phase_name=last.phase_name,
                    canonical_action=last.canonical_action,
                    conditioning_only=True,
                )
                for _ in range(padding)
            )
        segments = _segments(frames)
        canonical_action = Counter(
            frame.canonical_action for frame in frames
        ).most_common(1)[0][0]
        yield ActionCommand(
            step_id=step_id,
            canonical_action=canonical_action,
            duration_frames=output_frames,
            action_segments=segments,
            trajectory_source_frame_ids=tuple(
                frame.source_frame_id for frame in frames
            ),
            phase_names=tuple(dict.fromkeys(frame.phase_name for frame in frames)),
            conditioning_only=tuple(frame.conditioning_only for frame in frames),
            trajectory_mode="canonical_action",
            source=ACTION_SOURCE,
        )
        cursor += output_frames
        step_id += 1


__all__ = [
    "ACTION_SOURCE",
    "NativeActionFrame",
    "build_native_action_timeline",
    "canonical_action_for_row",
    "iter_open_loop_native_actions",
]
