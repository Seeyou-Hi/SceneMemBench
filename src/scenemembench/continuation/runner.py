# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Offline orchestration for one resident Context continuation case."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from scenemembench.io import write_json, write_jsonl
from scenemembench.io import coerce_video_frames, write_video

from .contracts import ActionCommand, ContextBundle, ContextualWorldModelAdapter


def _continuation_target_frames(
    context: ContextBundle,
    *,
    native_fps: int,
    full_continuation: bool,
    extra_continuation_frames: int,
) -> int | None:
    if extra_continuation_frames < 0:
        raise ValueError("extra_continuation_frames must be non-negative")
    if full_continuation:
        return None
    return (
        round(
            (context.eval_range[1] - context.eval_range[0]) * native_fps / context.fps
        )
        + extra_continuation_frames
    )


def run_context_continuation(
    adapter: ContextualWorldModelAdapter,
    context: ContextBundle,
    actions: Iterable[ActionCommand],
    *,
    prompt: str = "",
    seed: int = 42,
    output_dir: str | Path | None = None,
    print_memory_selection: bool = False,
    full_continuation: bool = False,
    extra_continuation_frames: int = 0,
) -> dict:
    """Run prefill plus resident steps and write an auditable trace."""

    target_frames = _continuation_target_frames(
        context,
        native_fps=adapter.capabilities().native_fps,
        full_continuation=full_continuation,
        extra_continuation_frames=extra_continuation_frames,
    )

    root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "capability.json", adapter.capabilities().to_dict())
        write_json(root / "context_bundle.json", context.to_dict())

    try:
        adapter.begin_case(context, prompt=prompt, seed=seed)
        prefill = adapter.prefill_context(context)
        if prefill.status == "rejected":
            raise ValueError("Backend rejected Context; refusing Query generation")
        if root is not None:
            write_json(root / "prefill_report.json", prefill.to_dict())
        finalize = getattr(
            getattr(adapter, "pipeline", None), "finalize_realtime", None
        )
        if callable(finalize):
            set_defer = getattr(
                getattr(adapter, "pipeline", None), "set_realtime_defer_decode", None
            )
            if callable(set_defer):
                set_defer(True)
        trace = []
        action_trace = []
        generated_chunks = []
        for action in actions:
            action_trace.append(action.to_dict())
            result = adapter.step(action)
            if root is not None and result.video is not None:
                generated_chunks.append(coerce_video_frames(result.video))
                write_video(
                    result.video,
                    root / f"step-{result.step_id:03d}.mp4",
                    fps=adapter.capabilities().native_fps,
                    quality=9,
                )
            trace.append(result.to_dict())
            if print_memory_selection:
                print(
                    json.dumps(
                        {
                            "event": "memory_selection",
                            "step_id": result.step_id,
                            "action": result.canonical_action,
                            "history_cursor": [
                                result.history_cursor_before,
                                result.history_cursor_after,
                            ],
                            **dict(result.memory_selection),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if root is not None:
                write_jsonl(root / "action_log.jsonl", action_trace)
                write_jsonl(root / "state_trace.jsonl", trace)
        continuation_frame_count = 0
        continuation_path = None
        if target_frames is None:
            target_frames = sum(
                int(step.get("generated_frame_count", 0)) for step in trace
            )
        # Zing's resident preview decodes each block independently.  Its
        # temporal VAE cache is reset between calls, which can create a seam at
        # every step boundary.  Prefer the model-owned full-Eval decode when
        # available; other adapters retain the existing concatenation path.
        finalized_video = None
        if root is not None and int(os.environ.get("RANK", "0") or "0") == 0:
            if callable(finalize):
                finalized_video = finalize()
        if finalized_video is not None:
            continuation = coerce_video_frames(finalized_video)
            if not full_continuation:
                continuation = continuation[:target_frames]
            continuation_frame_count = int(continuation.shape[0])
            continuation_path = str(root / "continuation.mp4")
            write_video(
                continuation,
                continuation_path,
                fps=adapter.capabilities().native_fps,
                quality=9,
            )
        elif root is not None and generated_chunks:
            import numpy as np

            continuation = np.concatenate(generated_chunks, axis=0)
            if not full_continuation:
                continuation = continuation[:target_frames]
            continuation_frame_count = int(continuation.shape[0])
            continuation_path = str(root / "continuation.mp4")
            write_video(
                continuation,
                continuation_path,
                fps=adapter.capabilities().native_fps,
                quality=9,
            )
        observed_reads = [
            bool(step.get("memory_selection", {}).get("memory_bank_read"))
            for step in trace
            if "memory_bank_read" in step.get("memory_selection", {})
        ]
        continuation_checks = {
            "prefill_committed": (
                prefill.status in {"accepted", "partial"}
                and prefill.native_latent_count > 0
                and prefill.history_cursor_after > prefill.history_cursor_before
            ),
            "all_steps_generated_frames": bool(trace)
            and all(int(step["generated_frame_count"]) > 0 for step in trace),
            "all_steps_advanced_history": bool(trace)
            and all(
                int(step["history_cursor_after"]) > int(step["history_cursor_before"])
                for step in trace
            ),
            "eval_frame_count_matched": continuation_frame_count == target_frames,
        }
        failed_continuation_checks = [
            name for name, passed in continuation_checks.items() if not passed
        ]
        summary = {
            "case_id": context.case_id,
            "model_id": adapter.capabilities().model_id,
            "generation_config": {
                "prompt": str(prompt),
                "seed": int(seed),
                "full_continuation": bool(full_continuation),
                "extra_continuation_frames": int(extra_continuation_frames),
            },
            "prefill": prefill.to_dict(),
            "steps": trace,
            "action_source": sorted({item["source"] for item in action_trace}),
            "controller_used": False,
            "vlm_used": False,
            "prescribed_pose_used": any(
                bool(item["camera_trajectory_ue"]) for item in action_trace
            ),
            "continuation_path": continuation_path,
            "continuation_frame_count": continuation_frame_count,
            "expected_continuation_frame_count": target_frames,
            "continuation_checks": continuation_checks,
            "continuation_verified": not failed_continuation_checks,
            "continuation_verification_reason": (
                "Context was committed and every resident step advanced model history."
                if not failed_continuation_checks
                else "Failed checks: " + ", ".join(failed_continuation_checks)
            ),
            "memory_bank_read_observed": any(observed_reads),
            "memory_verified": False,
            "memory_status": (
                "no_probe"
                if str(context.metadata.get("protocol", "")) == "L_LOOP"
                else "not_scored"
            ),
            "verification_reason": (
                "Selected GT Context entries are logged, but causal memory verification "
                "requires a separate ablation/probe."
            ),
        }
        if root is not None:
            write_json(root / "run_report.json", summary)
        return summary
    finally:
        adapter.end_case()


__all__ = ["run_context_continuation"]
