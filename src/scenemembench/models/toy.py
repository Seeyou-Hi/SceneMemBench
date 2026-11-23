"""CPU plumbing fixture, NOT a learned world model or scientific baseline.

Extrapolates pixels from the last Context image; it never reads Query RGB.
"""

from PIL import Image
import numpy as np
from scenemembench.continuation.contracts import (
    ModelCapabilities,
    PrefillResult,
    StepResult,
)


class ToyAdapter:
    def capabilities(self):
        return ModelCapabilities(
            "toy",
            "native_video_prefill",
            4,
            24,
            "toy pixel motion",
            True,
            True,
            limitations=("Synthetic plumbing fixture; no learned scene memory.",),
        )

    def begin_case(self, context, *, prompt="", seed=42):
        self.context = context
        self.cursor = 0
        self.frame = None

    def prefill_context(self, context):
        # Exercise all Context paths, never use a held-out Query image.
        for path in context.rgb_paths:
            with Image.open(path) as im:
                self.frame = np.asarray(im.convert("RGB")).copy()
        self.cursor = len(context.frame_ids)
        return PrefillResult(
            "accepted",
            context.frame_ids,
            native_latent_count=self.cursor,
            history_cursor_after=self.cursor,
            warnings=("Toy pixel-state count, not learned latents.",),
        )

    def step(self, action):
        before = self.cursor
        video = []
        sign = -1 if action.canonical_action in ("turn_right", "look_up") else 1
        axis = 0 if action.canonical_action.startswith("look_") else 1
        for _ in range(action.duration_frames):
            self.frame = np.roll(self.frame, sign, axis=axis)
            video.append(self.frame.copy())
        self.cursor += len(video)
        return StepResult(
            action.step_id,
            len(video),
            action.canonical_action,
            {"toy": True},
            before,
            self.cursor,
            video=np.stack(video),
        )

    def end_case(self):
        self.frame = None
        self.context = None
