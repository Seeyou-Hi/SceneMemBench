# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldFoundry Context adapter for Zing-0.5."""

from __future__ import annotations

import json

from ..contracts import ContextBundle, ModelCapabilities, PrefillResult
from ..controller.open_loop import canonical_action_for_row
from .base import (
    ResidentMemoryAdapter,
    _native_sample_indices,
    prefill_context_kwargs,
)


class ZingWorldMemoryAdapter(ResidentMemoryAdapter):
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id="zing-world-0.5",
            prefill_mode="partial_native_video_prefill",
            native_chunk_frames=16,
            native_fps=24,
            native_pose_format="keyboard_action_only",
            supports_gt_video_prefill=True,
            supports_persistent_step=True,
            supports_intra_step_action_segments=True,
            supports_action_motion_scale=False,
            state_observables=(
                "causal_kv_cache",
                "action_history",
                "prompt_context",
                "causal_vae_state",
            ),
            limitations=(
                "Official Zing exposes keyboard actions rather than exact c2w camera conditioning.",
                "Context tail is aligned to Zing's 1+4N temporal VAE cadence.",
            ),
        )

    def prefill_context(self, context: ContextBundle) -> PrefillResult:
        """Prefill native clean chunks with their exact dataset action rows."""

        if self.context is None or self.context.case_id != context.case_id:
            raise RuntimeError("begin_case must be called before prefill_context")
        prefill = getattr(self.pipeline, "prefill_context_realtime", None)
        if not callable(prefill):
            raise RuntimeError(
                "zing-world-0.5 pipeline does not expose prefill_context_realtime"
            )
        pose_path = context.rgb_paths[0].parents[1] / "pose" / "poses.json"
        with pose_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        rows_by_frame = {int(row["frame_id"]): row for row in rows}
        native_indices = _native_sample_indices(context, self.capabilities().native_fps)
        canonical_actions: list[str] = []
        for source_index in native_indices:
            frame_id = int(context.frame_ids[source_index])
            row = rows_by_frame[frame_id]
            previous = rows_by_frame.get(frame_id - 1)
            canonical_actions.append(canonical_action_for_row(row, previous))
        return PrefillResult(
            **prefill_context_kwargs(
                prefill,
                context,
                native_fps=self.capabilities().native_fps,
                extra_kwargs={"canonical_actions": canonical_actions},
            )
        )


__all__ = ["ZingWorldMemoryAdapter"]
