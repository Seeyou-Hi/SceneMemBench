# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Resident WorldFoundry bridge for the official Zing-0.5 runtime.

The upstream Zing repository remains the source of the model implementation and
its CLI.  This module only adds a session boundary around the same generator,
cache, scheduler, VAE, and text encoder so WorldFoundry can prefill Context and
advance one causal block at a time.
"""

from __future__ import annotations

import importlib
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


DEFAULT_ZING_ROOT = Path("./external/zing-world-model")
DEFAULT_CHECKPOINT = Path("./checkpoints/Zing-0.5")
NATIVE_FPS = 24
TEMPORAL_SCALE = 4
LATENT_FRAMES_PER_BLOCK = 4
VISUAL_CALIBRATED_90_STEPS = 6
# Six steps is the short GT-anchored continuation calibration. Long free
# rollout accumulates substantially less visual yaw per native step, so its
# mirror timing is planned independently by run_rollout100.py and free 360
# uses the manually verified 58-step setting. The environment override keeps
# that free-rollout protocol explicit and reproducible on C3.
VISUAL_CALIBRATED_360_STEPS = int(
    os.getenv("SCENEMEMBENCH_ZING_VISUAL_360_STEPS", "58")
)
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1280
ACTION_KEYS = ("w", "a", "s", "d", "i", "j", "k", "l")
ACTION_ALIASES = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "camera_l": "j",
    "camera_left": "j",
    "turn_left": "j",
    "camera_r": "l",
    "camera_right": "l",
    "turn_right": "l",
    "camera_up": "i",
    "look_up": "i",
    "camera_down": "k",
    "look_down": "k",
}


def _official_modules(root: str | Path | None = None) -> dict[str, Any]:
    """Import official Zing modules without copying or modifying them."""

    source = (
        Path(root or os.getenv("SCENEMEMBENCH_ZING_SOURCE", DEFAULT_ZING_ROOT)) / "src"
    )
    if not source.is_dir():
        raise FileNotFoundError(
            f"Zing source directory is missing: {source}. "
            "Clone seedleap/zing-world-model or set SCENEMEMBENCH_ZING_SOURCE."
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {
        "config": importlib.import_module("zing_v0_5.config"),
        "model": importlib.import_module("zing_v0_5.model"),
        "scheduler": importlib.import_module("zing_v0_5.scheduler"),
    }


def _as_uint8_frames(
    image: Image.Image | np.ndarray | torch.Tensor, height: int, width: int
) -> torch.Tensor:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    value = torch.as_tensor(image)
    if value.ndim == 4 and value.shape[-1] == 3:
        # WorldFoundry video tensors commonly use [T, H, W, C].
        value = value.permute(3, 0, 1, 2).unsqueeze(0)
    if value.ndim == 4 and value.shape[1] == 3:
        # A single image in [B, C, H, W] form is one temporal frame.
        value = value.unsqueeze(2)
    if value.ndim != 5 or value.shape[1] != 3:
        raise ValueError(
            f"expected RGB frames shaped [T,H,W,3] or [B,3,T,H,W], got {tuple(value.shape)}"
        )
    if value.shape[-2:] != (height, width):
        value = (
            torch.nn.functional.interpolate(
                value.permute(0, 2, 1, 3, 4)
                .reshape(-1, 3, value.shape[-2], value.shape[-1])
                .float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            .reshape(value.shape[0], value.shape[2], 3, height, width)
            .permute(0, 2, 1, 3, 4)
        )
    return value.to(dtype=torch.uint8, device="cpu")


def _sample_temporal_indices(frame_count: int) -> tuple[list[int], int]:
    """Return contiguous indices padded to Zing's required ``1 + 4N`` cadence."""

    if frame_count < 1:
        raise ValueError("at least one frame is required")
    count = int(frame_count)
    padded = ((count - 1 + TEMPORAL_SCALE - 1) // TEMPORAL_SCALE) * TEMPORAL_SCALE + 1
    return list(range(count)) + [count - 1] * (padded - count), padded - count


def _infer_context_action_tensor(
    camera_to_world: Sequence[Sequence[Sequence[float]]] | None,
    latent_count: int,
) -> torch.Tensor | None:
    """Recover per-frame keyboard actions for GT cache prefill.

    Zing's native cache stores a short action history alongside K/V states.  A
    video-only Context therefore needs the same action rows that produced the
    observed camera motion; otherwise the first generated block starts with an
    empty action history even when the Context was rotating or translating.
    """

    if (
        not camera_to_world
        or len(camera_to_world) < 1 + (latent_count - 1) * TEMPORAL_SCALE
    ):
        return None
    transition_count = (latent_count - 1) * TEMPORAL_SCALE
    values = torch.zeros(
        (latent_count, TEMPORAL_SCALE, len(ACTION_KEYS)), dtype=torch.float32
    )
    matrices = np.asarray(camera_to_world[: transition_count + 1], dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        return None
    yaws = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    translations = matrices[:, :3, 3]
    for transition in range(transition_count):
        yaw_delta = float(yaws[transition + 1] - yaws[transition])
        if abs(np.degrees(yaw_delta)) >= 0.25:
            key_index = ACTION_KEYS.index("l" if yaw_delta > 0.0 else "j")
        elif (
            float(
                np.linalg.norm(translations[transition + 1] - translations[transition])
            )
            >= 1e-3
        ):
            key_index = ACTION_KEYS.index("w")
        else:
            continue
        latent_index = 1 + transition // TEMPORAL_SCALE
        subframe = transition % TEMPORAL_SCALE
        values[latent_index, subframe, key_index] = 1.0
    return values.unsqueeze(0)


def _context_action_tensor(
    canonical_actions: Sequence[str],
    latent_count: int,
) -> torch.Tensor:
    """Pack per-pixel GT actions with the official reference-frame alignment."""

    expected_pixel_frames = 1 + (latent_count - 1) * TEMPORAL_SCALE
    if len(canonical_actions) != expected_pixel_frames:
        raise ValueError(
            "canonical context actions must align one-to-one with padded RGB "
            f"frames ({len(canonical_actions)} != {expected_pixel_frames})"
        )
    values = torch.zeros(
        (latent_count, TEMPORAL_SCALE, len(ACTION_KEYS)),
        dtype=torch.float32,
    )
    # The first RGB frame is Zing's reference frame.  Just like the official
    # MessageProcessor._action path, its action slot remains zero and the
    # action attached to each later RGB frame describes that transition.
    for transition, canonical_action in enumerate(canonical_actions[1:]):
        key = ACTION_ALIASES.get(
            str(canonical_action).strip().lower(),
            str(canonical_action).strip().lower(),
        )
        if key not in ACTION_KEYS:
            continue
        latent_index = 1 + transition // TEMPORAL_SCALE
        subframe = transition % TEMPORAL_SCALE
        values[latent_index, subframe, ACTION_KEYS.index(key)] = 1.0
    return values.unsqueeze(0)


class ZingWorldRealtimeSession:
    """Persistent Zing session using the official causal cache semantics."""

    def __init__(
        self,
        checkpoint_source: str | Path | None = None,
        *,
        pretrained_dir: str | Path | None = None,
        source_root: str | Path | None = None,
        device: str = "cuda",
        local_attn_size: int | None = None,
        sink_size: int | None = None,
    ) -> None:
        modules = _official_modules(source_root)
        self._config_module = modules["config"]
        self._model_module = modules["model"]
        self._scheduler_module = modules["scheduler"]
        requested_device = str(device)
        if requested_device == "cuda":
            requested_device = "cuda:0"
        elif not requested_device.startswith("cuda"):
            requested_device = "cuda:0"
        self.device = torch.device(requested_device)
        if not torch.cuda.is_available():
            raise RuntimeError("Zing-0.5 realtime inference requires CUDA.")
        torch.cuda.set_device(self.device)

        checkpoint = Path(
            checkpoint_source
            or os.getenv("SCENEMEMBENCH_ZING_CHECKPOINT", DEFAULT_CHECKPOINT)
        ).expanduser()
        pretrained = Path(
            pretrained_dir
            or os.getenv("SCENEMEMBENCH_ZING_PRETRAINED", checkpoint / "pretrained")
        ).expanduser()
        config_path = (
            Path(
                source_root or os.getenv("SCENEMEMBENCH_ZING_SOURCE", DEFAULT_ZING_ROOT)
            )
            / "config"
            / "zing.yaml"
        )
        config = self._config_module.load_config(config_path)
        requested_window = local_attn_size
        requested_sink = sink_size
        if requested_window is None:
            requested_window = int(
                os.getenv(
                    "SCENEMEMBENCH_ZING_LOCAL_ATTN_SIZE",
                    str(config.generator.local_attn_size),
                )
            )
        if requested_sink is None:
            requested_sink = int(
                os.getenv(
                    "SCENEMEMBENCH_ZING_SINK_SIZE", str(config.generator.sink_size)
                )
            )
        config = self._config_module.with_cache_window(
            config, requested_window, requested_sink
        )
        self.config = config
        self.height = int(os.getenv("SCENEMEMBENCH_ZING_HEIGHT", str(DEFAULT_HEIGHT)))
        self.width = int(os.getenv("SCENEMEMBENCH_ZING_WIDTH", str(DEFAULT_WIDTH)))
        if self.height % 32 or self.width % 32:
            raise ValueError("Zing realtime height and width must be divisible by 32.")
        if not (checkpoint / "generator" / "model.pt").is_file():
            raise FileNotFoundError(
                f"Zing generator checkpoint is missing: {checkpoint / 'generator/model.pt'}"
            )
        for name in ("text_encoder", "tokenizer", "vae"):
            if not (pretrained / name).is_dir():
                raise FileNotFoundError(
                    f"Zing pretrained component is missing: {pretrained / name}"
                )

        generator_cls = self._model_module.WanModel
        text_encoder_cls = self._model_module.WanTextEncoder
        vae_cls = self._model_module.WanVAE
        state = torch.load(
            checkpoint / "generator" / "model.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        with torch.device("meta"):
            generator = generator_cls(config)
        generator.load_state_dict(state, strict=True, assign=True)
        del state
        self.generator = (
            generator.eval().requires_grad_(False).to(self.device, dtype=torch.bfloat16)
        )
        self.text_encoder = (
            text_encoder_cls(pretrained, config.text_encoder.max_length)
            .eval()
            .requires_grad_(False)
            .to("cpu", dtype=torch.bfloat16)
        )
        self.vae = (
            vae_cls(pretrained)
            .eval()
            .requires_grad_(False)
            .to("cpu", dtype=torch.bfloat16)
        )
        self._scheduler = self._scheduler_module.DmdScheduler(config.inference).to(
            self.device
        )
        self._prompt_context: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cache: Any = None
        self._last_latent: torch.Tensor | None = None
        self._generated_latents: list[torch.Tensor] = []
        self._defer_decode = False
        self._initial_latent: torch.Tensor | None = None
        self._context_latents_cpu: torch.Tensor | None = None
        self._configured = False
        self._seed = 42
        self._step_id = 0
        self._history_cursor = 0
        self._prompt = ""

    def _state_digest(self) -> str:
        """Summarize resident state without copying the full KV tensors."""

        digest = hashlib.sha256()
        digest.update(str(self._history_cursor).encode("ascii"))
        digest.update(str(self._step_id).encode("ascii"))
        digest.update(self._prompt.encode("utf-8"))
        cache = self._cache
        if cache is not None:
            for name in ("positions", "block_ids"):
                value = getattr(cache, name, None)
                if value is None:
                    digest.update(f"{name}:none".encode("ascii"))
                    continue
                flat = value.reshape(-1)
                sample = (
                    flat[:4].detach().cpu().tolist() + flat[-4:].detach().cpu().tolist()
                )
                digest.update(f"{name}:{tuple(value.shape)}:{sample}".encode("utf-8"))
            action_history = getattr(cache, "action_history", None)
            if action_history is not None:
                digest.update(f"action:{tuple(action_history.shape)}".encode("ascii"))
                digest.update(
                    action_history[:, -1:]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                    .tobytes()
                )
        return f"zing:{digest.hexdigest()[:16]}"

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.text_encoder.to(self.device)
        try:
            context, lengths = self.text_encoder.encode([prompt])
            return context.to(self.device, dtype=torch.bfloat16), lengths.to(
                self.device
            )
        finally:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _model_flow(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor | None,
        cache_mode: str | None,
        prompt_switch: bool = False,
    ) -> torch.Tensor:
        if self._prompt_context is None or self._cache is None:
            raise RuntimeError("Zing session is not configured or prefilled.")
        value = latents.permute(0, 2, 1, 3, 4)
        output = self.generator(
            value,
            timestep,
            self._prompt_context[0],
            self._prompt_context[1],
            self._cache,
            cache_mode,
            action=action,
            prompt_switch=prompt_switch,
        )
        return output.permute(0, 2, 1, 3, 4)

    def _reserve_cache_for_latents(self, latents: torch.Tensor) -> None:
        """Mirror the official pipeline's reservation before a kept chunk."""

        if self._cache is None:
            raise RuntimeError("Zing cache is not initialized")
        patch = self.config.generator.patch_size
        token_count = (
            int(latents.shape[1])
            // int(patch[0])
            * (int(latents.shape[3]) // int(patch[1]))
            * (int(latents.shape[4]) // int(patch[2]))
        )
        self._cache.reserve(token_count)

    def configure(
        self,
        image: Image.Image | str | Path,
        *,
        prompt: str = "",
        seed: int = 42,
        fps: int = NATIVE_FPS,
        **_: Any,
    ) -> dict[str, Any]:
        if fps != NATIVE_FPS:
            raise ValueError(f"Zing native FPS is {NATIVE_FPS}, got {fps}")
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        if not isinstance(image, Image.Image):
            raise ValueError("Zing realtime requires a PIL image or image path.")
        self._prompt = str(prompt or "").strip()
        if not self._prompt:
            raise ValueError("Zing realtime requires a non-empty prompt.")
        self._seed = int(seed)
        # Offline evaluation reuses one resident session for multiple cases.
        # finalize_generated_video() moves the generator to CPU so the VAE can
        # decode a long rollout without an avoidable memory peak.  Restore the
        # generator before configuring the next case; otherwise CUDA latents
        # reach CPU bfloat16 weights after the first completed rollout.
        self.generator.to(self.device, dtype=torch.bfloat16)
        self._prompt_context = self._encode_prompt(self._prompt)
        self._cache = self.generator.make_kv_cache()
        pixels = _as_uint8_frames(image, self.height, self.width)
        self.vae.to(self.device)
        try:
            self._initial_latent = self.vae.encode(pixels).to(
                self.device, dtype=torch.bfloat16
            )
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()
        if self._initial_latent.shape[1] != 1:
            raise RuntimeError(
                f"Zing reference image produced {self._initial_latent.shape[1]} latent frames, expected 1"
            )
        zero = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
        self._reserve_cache_for_latents(self._initial_latent)
        self._model_flow(self._initial_latent, zero, None, "final", prompt_switch=False)
        self._last_latent = self._initial_latent.detach()
        self._decode_anchor_latent = self._initial_latent.detach()
        self._context_latents_cpu = self._initial_latent.detach().cpu()
        self._generated_latents = []
        self._configured = True
        self._step_id = 0
        self._history_cursor = 0
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "model_id": "zing-world-0.5",
            "native_fps": NATIVE_FPS,
            "native_chunk_frames": TEMPORAL_SCALE * LATENT_FRAMES_PER_BLOCK,
            "temporal_scale": TEMPORAL_SCALE,
            "frames_per_block": LATENT_FRAMES_PER_BLOCK,
            "height": self.height,
            "width": self.width,
            "local_attn_size": self.config.generator.local_attn_size,
            "sink_size": self.config.generator.sink_size,
            "visual_calibrated_90_steps": VISUAL_CALIBRATED_90_STEPS,
            "visual_calibrated_90_frames": (
                VISUAL_CALIBRATED_90_STEPS * TEMPORAL_SCALE * LATENT_FRAMES_PER_BLOCK
            ),
            "visual_calibrated_360_steps": VISUAL_CALIBRATED_360_STEPS,
            "visual_calibrated_360_frames": (
                VISUAL_CALIBRATED_360_STEPS * TEMPORAL_SCALE * LATENT_FRAMES_PER_BLOCK
            ),
        }

    def set_defer_decode(self, enabled: bool = True) -> None:
        """Defer per-step VAE decoding until ``finalize_generated_video``."""

        self._defer_decode = bool(enabled)

    @torch.no_grad()
    def prefill_context(
        self,
        *,
        rgb_paths: Sequence[str | Path],
        frame_ids: Sequence[int] | None = None,
        camera_to_world: Sequence[Sequence[Sequence[float]]] | None = None,
        canonical_actions: Sequence[str] | None = None,
        fps: int = NATIVE_FPS,
        **_: Any,
    ) -> dict[str, Any]:
        if not self._configured:
            raise RuntimeError("configure must be called before prefill_context")
        if fps != NATIVE_FPS:
            raise ValueError(f"Zing native FPS is {NATIVE_FPS}, got {fps}")
        paths = [Path(path) for path in rgb_paths]
        indices, padded_count = _sample_temporal_indices(len(paths))
        frames: list[torch.Tensor] = []
        for index in indices:
            with Image.open(paths[index]) as source:
                frames.append(
                    _as_uint8_frames(source.convert("RGB"), self.height, self.width)
                )
        pixels = torch.cat(frames, dim=2)
        self.vae.to(self.device)
        try:
            clean = self.vae.encode(pixels).to(self.device, dtype=torch.bfloat16)
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

        # Context prefill replaces the one-frame native seed state.
        self._cache = self.generator.make_kv_cache()
        latent_count = int(clean.shape[1])
        if latent_count < 1:
            raise RuntimeError("Zing VAE returned no latent frames")
        padded_camera = None
        if camera_to_world:
            padded_camera = list(camera_to_world) + [camera_to_world[-1]] * padded_count
        padded_actions = None
        if canonical_actions is not None:
            if len(canonical_actions) != len(paths):
                raise ValueError(
                    "canonical_actions must align one-to-one with context RGB paths"
                )
            # Repeated cadence padding is a stationary frame, not a repeated
            # movement command.
            padded_actions = list(canonical_actions) + ["idle"] * padded_count
        context_action = (
            _context_action_tensor(padded_actions, latent_count)
            if padded_actions is not None
            else _infer_context_action_tensor(padded_camera, latent_count)
        )
        zero = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
        spans = [(0, 1)] + [
            (start, min(start + LATENT_FRAMES_PER_BLOCK, latent_count))
            for start in range(1, latent_count, LATENT_FRAMES_PER_BLOCK)
        ]
        for start, end in spans:
            current = clean[:, start:end]
            timestep = zero.expand(1, end - start)
            action = (
                None
                if context_action is None
                else context_action[:, start:end].to(self.device)
            )
            self._reserve_cache_for_latents(current)
            self._model_flow(current, timestep, action, "final", prompt_switch=False)
        self._decode_anchor_latent = clean[:, -1:].detach()
        # Keep all clean Context latents on CPU.  During final decode they are
        # replayed through the causal VAE before Eval latents, reproducing the
        # decoder feature-cache state of one native end-to-end sequence.
        self._context_latents_cpu = clean.detach().cpu()
        self._last_latent = self._decode_anchor_latent
        self._history_cursor = latent_count
        # Eval generation starts after the prefilled context.  Keep the
        # generated latent blocks so the caller can decode the whole Eval
        # sequence in one VAE pass, avoiding decoder-cache resets at steps.
        self._generated_latents = []
        source_ids = frame_ids or list(range(len(paths)))
        accepted = [int(source_ids[index]) for index in range(len(paths))]
        return {
            "status": "accepted",
            "accepted_frame_ids": accepted,
            "rejected_frame_ids": [],
            "native_latent_count": latent_count,
            "native_chunk_count": len(spans),
            "history_cursor_before": 0,
            "history_cursor_after": latent_count,
            "boundary_mapping": {
                "source_frame_count": len(paths),
                "accepted_source_indices": list(range(len(paths))),
                "sampled_source_indices": indices,
                "padded_source_frame_count": padded_count,
                "temporal_scale": TEMPORAL_SCALE,
                "frames_per_block": LATENT_FRAMES_PER_BLOCK,
                "context_action_history_prefilled": context_action is not None,
                "context_action_history_source": (
                    "dataset_canonical_actions"
                    if padded_actions is not None
                    else "camera_pose_inference"
                    if context_action is not None
                    else "unavailable"
                ),
                "causal_decode_context_latent_count": latent_count,
            },
            "last_context_pose": {},
            "warnings": (
                [
                    f"Context tail padded by repeating the final frame {padded_count} time(s) for Zing VAE cadence."
                ]
                if padded_count
                else []
            ),
        }

    @staticmethod
    def _action_tensor(
        segments: Sequence[Mapping[str, Any]] | None,
        interactions: Sequence[str] | None,
    ) -> torch.Tensor:
        pixel_frames = TEMPORAL_SCALE * LATENT_FRAMES_PER_BLOCK
        values = torch.zeros((pixel_frames, len(ACTION_KEYS)), dtype=torch.float32)
        normalized = list(segments or ())
        if not normalized:
            keys = {
                ACTION_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
                for item in (interactions or ())
            }
            for key in keys:
                if key in ACTION_KEYS:
                    values[:, ACTION_KEYS.index(key)] = 1.0
            return values.view(
                1, LATENT_FRAMES_PER_BLOCK, TEMPORAL_SCALE, len(ACTION_KEYS)
            )
        total = sum(max(float(item.get("duration", 0.0)), 0.0) for item in normalized)
        if total <= 0.0:
            return values.view(
                1, LATENT_FRAMES_PER_BLOCK, TEMPORAL_SCALE, len(ACTION_KEYS)
            )
        cursor = 0
        for index, item in enumerate(normalized):
            duration = max(float(item.get("duration", 0.0)), 0.0)
            end = (
                pixel_frames
                if index == len(normalized) - 1
                else min(
                    pixel_frames,
                    cursor + max(1, round(duration / total * pixel_frames)),
                )
            )
            keys = {
                ACTION_ALIASES.get(str(key).strip().lower(), str(key).strip().lower())
                for key in item.get("keys", ())
            }
            for key in keys:
                if key in ACTION_KEYS:
                    values[cursor:end, ACTION_KEYS.index(key)] = 1.0
            cursor = end
            if cursor >= pixel_frames:
                break
        return values.view(1, LATENT_FRAMES_PER_BLOCK, TEMPORAL_SCALE, len(ACTION_KEYS))

    @torch.no_grad()
    def generate(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not self._configured or self._cache is None:
            raise RuntimeError(
                "Zing session must be configured and prefilled before generate"
            )
        if (
            prompt is not None
            and str(prompt).strip()
            and str(prompt).strip() != self._prompt
        ):
            self._prompt = str(prompt).strip()
            self._prompt_context = self._encode_prompt(self._prompt)
            self._cache.reset_cross()
        if seed is not None:
            self._seed = int(seed)
        step_seed = self._seed + self._step_id
        random.seed(step_seed)
        np.random.seed(step_seed)
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        action = self._action_tensor(control_segments, interactions).to(self.device)
        clean = torch.zeros(
            (
                1,
                LATENT_FRAMES_PER_BLOCK,
                int(self.config.vae.z_dim),
                self.height // int(self.config.vae.spatial_scale),
                self.width // int(self.config.vae.spatial_scale),
            ),
            device=self.device,
            dtype=torch.bfloat16,
        )
        current = torch.randn_like(clean)
        self._reserve_cache_for_latents(current)
        scheduler = self._scheduler_module.DmdScheduler(self.config.inference).to(
            self.device
        )
        for timestep in scheduler.timesteps:
            prediction = self._model_flow(
                current,
                timestep
                * torch.ones(
                    (1, LATENT_FRAMES_PER_BLOCK),
                    device=self.device,
                    dtype=torch.float32,
                ),
                action,
                "active",
                prompt_switch=False,
            )
            current, _ = scheduler.step(prediction, current)
        zero = torch.zeros(
            (1, LATENT_FRAMES_PER_BLOCK), device=self.device, dtype=torch.float32
        )
        self._model_flow(current, zero, action, "final", prompt_switch=False)
        previous_latent = (
            self._last_latent if self._last_latent is not None else current[:, :1]
        )
        self._last_latent = current[:, -1:].detach()
        self._generated_latents.append(current.detach())
        video = None
        if not self._defer_decode:
            # Decode using the preceding latent as the causal temporal anchor.
            decode_input = torch.cat((previous_latent, current), dim=1)
            self.vae.to(self.device)
            try:
                decoded = self.vae.decode(decode_input)
            finally:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
            video = (decoded[:, 1:] * 0.5 + 0.5).clamp(0, 1).cpu()
        step_id = self._step_id
        history_cursor_before = self._history_cursor
        self._step_id += 1
        self._history_cursor += LATENT_FRAMES_PER_BLOCK
        return {
            "video": None if video is None else video[0].permute(0, 2, 3, 1),
            "frames": None if video is None else video[0].permute(0, 2, 3, 1),
            "realtime_metrics": {
                "history_cursor_before": history_cursor_before,
                "history_cursor_after": self._history_cursor,
                "state_digest": self._state_digest(),
                "native_cache_policy": {
                    "local_attn_size": self.config.generator.local_attn_size,
                    "sink_size": self.config.generator.sink_size,
                },
                "generated_frame_count": TEMPORAL_SCALE * LATENT_FRAMES_PER_BLOCK,
                "decode_deferred": self._defer_decode,
            },
        }

    @torch.no_grad()
    def _decode_eval_with_context(
        self,
        context_latents: torch.Tensor,
        generated_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Prime the native causal decoder with Context and retain only Eval."""

        # This is the official WanVAE.decode_streaming loop with one deliberate
        # storage change: decoded Context frames warm the feature cache but are
        # not accumulated in host memory.  The decoder operations themselves,
        # latent order, cache lifetime, and spatial unpatchify are unchanged.
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        latent = torch.cat(
            (
                context_latents.to(self.device, dtype=torch.bfloat16),
                generated_latents.to(self.device, dtype=torch.bfloat16),
            ),
            dim=1,
        )
        dtype = next(self.vae.model.parameters()).dtype
        latent = latent.to(device=next(self.vae.model.parameters()).device, dtype=dtype)
        mean, std = self.vae._stats(latent)
        value = latent.permute(0, 2, 1, 3, 4) * std + mean
        self.vae.model.clear_cache()
        try:
            value = self.vae.model.post_quant_conv(value)
            decoded_frames = []
            context_latent_count = int(context_latents.shape[1])
            for index in range(value.shape[2]):
                self.vae.model._conv_idx = [0]
                frame = self.vae.model.decoder(
                    value[:, :, index : index + 1, :, :],
                    feat_cache=self.vae.model._feat_map,
                    feat_idx=self.vae.model._conv_idx,
                    first_chunk=(index == 0),
                )
                if index >= context_latent_count:
                    decoded_frames.append(frame.float().cpu())
            decoded = torch.cat(decoded_frames, dim=2)
            if self.vae.model.config.patch_size is not None:
                decoded = unpatchify(
                    decoded,
                    patch_size=self.vae.model.config.patch_size,
                )
            return decoded.clamp(-1.0, 1.0).permute(0, 2, 1, 3, 4)
        finally:
            self.vae.model.clear_cache()

    @torch.no_grad()
    def finalize_generated_video(self) -> torch.Tensor | None:
        """Decode all resident Eval latents in one causal VAE pass.

        Per-step decoding is useful for interactive preview, but resetting the
        Wan VAE temporal feature cache at every call creates visible seams.
        This method is intended for offline Eval output and returns
        ``[T, H, W, C]`` frames without the context anchor frame.
        """

        if not self._generated_latents:
            return None
        generated = torch.cat(self._generated_latents, dim=1)
        context = self._context_latents_cpu
        if context is None:
            context = (
                getattr(
                    self,
                    "_decode_anchor_latent",
                    self._generated_latents[0][:, :1],
                )
                .detach()
                .cpu()
            )
        # The resident generator and its KV cache are no longer needed once
        # Eval sampling is complete.  Release them before moving the VAE to
        # CUDA; otherwise long 720p outputs can OOM before decoding starts.
        self._cache = None
        self._prompt_context = None
        self.generator.to("cpu")
        torch.cuda.empty_cache()
        self.vae.to(self.device)
        try:
            decoded = self._decode_eval_with_context(context, generated)
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()
        video = (decoded * 0.5 + 0.5).clamp(0, 1)
        return video[0].permute(0, 2, 3, 1)

    def reset(self) -> None:
        self._prompt_context = None
        self._cache = None
        self._last_latent = None
        self._generated_latents = []
        self._context_latents_cpu = None
        self._defer_decode = False
        if hasattr(self, "_decode_anchor_latent"):
            del self._decode_anchor_latent
        self._initial_latent = None
        self._configured = False
        self._step_id = 0
        self._history_cursor = 0
