# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Resident adapter for the official chunk-causal SANA-WM Streaming model.

GT and generated context are represented as clean stage-1 latents.  Each call
replays those latents through the official Self-Forcing cache update path, so
GDN recurrent state, softmax sink/recent KV, refiner KV, and causal-VAE state
follow the model's native history policy.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from scenemembench._native.core.io.paths import checkpoint_root_path
from scenemembench._native.core.realtime import RealtimeSpec
from scenemembench._native.runtime.local_checkpoint_cache import (
    stage_checkpoint_for_realtime,
)

_TOKEN_TO_KEY = {
    "forward": "w",
    "backward": "s",
    "left": "j",
    "right": "l",
    "camera_up": "i",
    "camera_down": "k",
    # SANA-WM's native DSL uses a/d for yaw and j/l for strafe.
    "camera_l": "a",
    "camera_r": "d",
}

# WorldFoundry's public realtime controls use WASD for translation and IJKL
# for camera rotation.  SANA-WM's native DSL swaps the horizontal pairs: A/D
# are yaw and J/L are strafe.  Translate segment keys at this adapter boundary
# so Studio and offline realtime-aligned rollouts share the public contract.
_PUBLIC_SEGMENT_KEY_TO_NATIVE = {
    "w": "w",
    "s": "s",
    "a": "j",
    "d": "l",
    "i": "i",
    "k": "k",
    "j": "a",
    "l": "d",
}

_REQUIRED_CHECKPOINT_PATHS = (
    "sana_dit/model.pt",
    "ltx2_causal_vae/config.json",
    "ltx2_causal_vae/diffusion_pytorch_model.safetensors",
    "refiner_diffusers/transformer/config.json",
    "refiner_diffusers/transformer/diffusion_pytorch_model.safetensors",
    "refiner_diffusers/connectors/config.json",
    "refiner_diffusers/connectors/diffusion_pytorch_model.safetensors",
    "gemma3_12b/config.json",
    "gemma3_12b/model.safetensors.index.json",
)

DEFAULT_REALTIME_WINDOW_FRAMES = 25
DEFAULT_SAMPLING_STEPS = 4
DEFAULT_CFG_SCALE = 1.0
DEFAULT_DENOISING_STEP_LIST = (1000, 960, 889, 727, 0)
NATIVE_PIXEL_FRAMES_PER_BLOCK = 24
NATIVE_LATENT_FRAMES_PER_BLOCK = 3


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), minimum)
    except ValueError:
        return max(int(default), minimum)


def _runtime_import_paths() -> tuple[Path, ...]:
    source_root = Path(__file__).resolve().parents[4]
    return (
        source_root / "_native/base_models/diffusion_model/image/sana",
        source_root / "_native/base_models/diffusion_model/video/wan",
        source_root / "_native/base_models/three_dimensions/point_clouds/pi3",
        source_root.parent,
    )


def ensure_sana_import_paths() -> None:
    """Expose the in-tree runtime's historical top-level package names."""

    for path in reversed(_runtime_import_paths()):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _resolve_checkpoint(source: str | Path | None) -> Path:
    candidates = []
    if source:
        candidates.append(Path(source).expanduser())
    candidates.extend(
        (
            checkpoint_root_path("SANA-WM_streaming"),
            checkpoint_root_path("hfd", "Efficient-Large-Model--SANA-WM_streaming"),
        )
    )
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / item).exists() for item in _REQUIRED_CHECKPOINT_PATHS
        ):
            return candidate.resolve()
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"SANA-WM checkpoint is missing or incomplete. Searched: {searched}"
    )


def _snap_num_frames(value: int) -> int:
    """Use the nearest complete 3-latent/24-pixel streaming block layout."""

    value = max(int(value), NATIVE_PIXEL_FRAMES_PER_BLOCK + 1)
    lower = value - ((value - 1) % NATIVE_PIXEL_FRAMES_PER_BLOCK)
    upper = lower + NATIVE_PIXEL_FRAMES_PER_BLOCK
    return lower if value - lower < upper - value else upper


def _device_topology() -> dict[str, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("SANA-WM realtime requires CUDA.")
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No visible CUDA device is available for SANA-WM.")

    raw = os.getenv("SCENEMEMBENCH_SANA_WM_DEVICE", "0").strip().lower()
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SCENEMEMBENCH_SANA_WM_DEVICE must be a CUDA index, got {raw!r}."
        ) from exc
    if not 0 <= index < count:
        raise ValueError(
            f"SCENEMEMBENCH_SANA_WM_DEVICE={index} but only {count} CUDA devices are visible."
        )
    device = torch.device(f"cuda:{index}")
    return {"stage1": device, "refiner": device, "vae": device, "text": device}


def _frames_for_segments(
    segments: Sequence[Mapping[str, Any]] | None,
    interactions: Sequence[str],
    *,
    frame_count: int,
    fps: int,
) -> list[frozenset[str]]:
    fallback = frozenset(
        _TOKEN_TO_KEY[token]
        for item in interactions
        if (token := str(item).strip().lower()) in _TOKEN_TO_KEY
    )
    if not segments:
        return [fallback] * frame_count

    rows: list[tuple[float, frozenset[str]]] = []
    for segment in segments:
        duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
        keys = frozenset(
            _PUBLIC_SEGMENT_KEY_TO_NATIVE[str(item).lower()]
            for item in (segment.get("keys") or ())
            if str(item).lower() in _PUBLIC_SEGMENT_KEY_TO_NATIVE
        )
        if duration:
            rows.append((duration, keys))
    if not rows:
        return [fallback] * frame_count

    total = sum(duration for duration, _ in rows)
    scale = (frame_count / float(fps)) / total if total > 0.0 else 1.0
    boundaries: list[tuple[float, frozenset[str]]] = []
    elapsed = 0.0
    for duration, keys in rows:
        elapsed += duration * scale
        boundaries.append((elapsed, keys))
    result: list[frozenset[str]] = []
    row_index = 0
    for frame_index in range(frame_count):
        timestamp = (frame_index + 1) / float(fps)
        while (
            row_index + 1 < len(boundaries)
            and timestamp > boundaries[row_index][0] + 1e-8
        ):
            row_index += 1
        result.append(boundaries[row_index][1])
    return result


def _compress_action_frames(frames: Sequence[frozenset[str]]) -> str:
    if not frames:
        return "none-1"
    segments: list[str] = []
    previous = frames[0]
    length = 1
    for keys in frames[1:]:
        if keys == previous:
            length += 1
            continue
        segments.append(f"{''.join(sorted(previous)) or 'none'}-{length}")
        previous = keys
        length = 1
    segments.append(f"{''.join(sorted(previous)) or 'none'}-{length}")
    return ",".join(segments)


def _validate_output_video(value: Any, *, expected_frames: int) -> np.ndarray:
    video = np.ascontiguousarray(value, dtype=np.uint8)
    if video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0:
        raise RuntimeError(f"SANA-WM returned an invalid video array: {video.shape}")
    if len(video) != expected_frames:
        raise RuntimeError(
            f"SANA-WM returned {len(video)} frames; expected {expected_frames} "
            "for the configured native window."
        )
    return video


class SanaWMRealtimeSession:
    """One persistent high-quality SANA-WM exploration session."""

    fps = 16

    def __init__(self, checkpoint_source: str | Path | None = None) -> None:
        ensure_sana_import_paths()
        import pyrallis

        from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.inference_sana_wm import (
            GenerationParams,
            InferenceConfig,
            RefinerSettings,
            SanaWMPipeline,
        )

        source = _resolve_checkpoint(checkpoint_source)
        self.checkpoint = (
            stage_checkpoint_for_realtime(
                source,
                required_paths=_REQUIRED_CHECKPOINT_PATHS,
            )
            if _env_flag("SCENEMEMBENCH_SANA_WM_STAGE_CHECKPOINT", False)
            else source
        )
        self.devices = _device_topology()
        torch.cuda.set_device(self.devices["stage1"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        device_memory = torch.cuda.get_device_properties(
            self.devices["stage1"]
        ).total_memory
        sequential_default = device_memory < 128 * 1024**3
        if "SANA_WM_STREAMING_SEQUENTIAL_OFFLOAD" in os.environ:
            self._sequential_offload = _env_flag(
                "SANA_WM_STREAMING_SEQUENTIAL_OFFLOAD",
                sequential_default,
            )
        else:
            self._sequential_offload = _env_flag(
                "SCENEMEMBENCH_SANA_WM_SEQUENTIAL_OFFLOAD",
                sequential_default,
            )
            os.environ["SANA_WM_STREAMING_SEQUENTIAL_OFFLOAD"] = (
                "1" if self._sequential_offload else "0"
            )
        if self._sequential_offload:
            os.environ.setdefault("SANA_WM_STREAMING_LAZY_VAE_DECODER", "1")

        # One call emits complete 3-latent blocks (24 pixels per block).
        self.num_frames = _snap_num_frames(
            _env_int(
                "SCENEMEMBENCH_SANA_WM_NUM_FRAMES",
                DEFAULT_REALTIME_WINDOW_FRAMES,
                minimum=9,
            )
        )
        self.output_frames = self.num_frames - 1
        self.logger = logging.getLogger("scenemembench._native.sana_wm.realtime")
        config: InferenceConfig = pyrallis.parse(
            config_class=InferenceConfig,
            config_path=str(
                Path(__file__).resolve().parents[3]
                / "base_models/diffusion_model/image/sana/configs/sana_wm/sana_wm_streaming_1600m_720p.yaml"
            ),
            args=[],
        )
        config.vae.vae_pretrained = str(self.checkpoint / "ltx2_causal_vae")
        refiner = RefinerSettings(
            root=self.checkpoint / "refiner_diffusers",
            gemma_root=self.checkpoint / "gemma3_12b",
            sink_size=1,
            seed=_env_int("SCENEMEMBENCH_SANA_WM_REFINER_SEED", 42, minimum=0),
            block_size=NATIVE_LATENT_FRAMES_PER_BLOCK,
            kv_max_frames=_env_int("SCENEMEMBENCH_SANA_WM_REFINER_KV_MAX_FRAMES", 11),
        )
        self.pipeline = SanaWMPipeline(
            config=config,
            model_path=self.checkpoint / "sana_dit/model.pt",
            device=self.devices["stage1"],
            refiner=refiner,
            offload_vae=False,
            offload_refiner=False,
            offload_text_encoder=False,
            logger=self.logger,
        )
        self.params_cls = GenerationParams
        self._image: Image.Image | None = None
        self._prompt = ""
        self._intrinsics: np.ndarray | None = None
        self._camera_history: np.ndarray | None = None
        self._context_latents: torch.Tensor | None = None
        self._refiner_context_latents: torch.Tensor | None = None
        # Pose frame used by the resident session.  Dataset GT is UE by
        # contract; action-generated poses are already in OpenCV coordinates.
        self._pose_format = "opencv"
        self._pose_origin_cv = np.eye(4, dtype=np.float32)
        self._horizontal_fov_deg = 55.0
        self._world_pose = np.eye(4, dtype=np.float32)
        self._history_cursor = 0
        self._configured = False
        self._chunk_index = 0
        self._seed = 42
        self._steps = DEFAULT_SAMPLING_STEPS
        self._cfg_scale = DEFAULT_CFG_SCALE
        self.last_metrics: dict[str, float] = {}
        self._state_lock = threading.RLock()

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.output_frames,
            steady_chunk_frames=self.output_frames,
        )

    def runtime_info(self) -> dict[str, Any]:
        """Describe actual resident component placement without claiming parallelism."""

        return {
            "component_devices": {
                name: str(device) for name, device in self.devices.items()
            },
            "execution": (
                "native-self-forcing-cache-replay-sequential"
                if self._sequential_offload
                else "native-self-forcing-cache-replay-pipelined"
            ),
            "tensor_parallel": False,
            "sequential_offload": self._sequential_offload,
            "future_pixel_frames": self.output_frames,
            "stage1_cached_blocks": int(
                os.getenv("SCENEMEMBENCH_SANA_WM_CACHED_BLOCKS", "2")
            ),
            "stage1_sink_token": True,
            "refiner_kv_max_frames": int(self.pipeline.refiner_settings.kv_max_frames),
            "sampling_steps": self._steps,
            "refiner_enabled": True,
            "prompt_updates": "chunk-boundary",
            "text_conditioning": ("stage1", "refiner"),
        }

    def _prepare_prompt_conditions(self, prompt: str) -> None:
        """Encode both text paths transactionally without touching world state."""

        self.pipeline._get_streaming_stage1_prompt(prompt, "")
        self.pipeline._get_streaming_refiner_prompt(prompt)

    def update_prompt(self, prompt: str) -> bool:
        """Apply new text conditioning to the next complete native chunk.

        Only the stage-1 and refiner prompt caches are replaced. The current
        anchor frame, camera pose, intrinsics, seed policy, and chunk index are
        deliberately retained so the generated world remains continuous.
        """

        normalized = str(prompt or "").strip()
        if not normalized:
            raise ValueError("SANA-WM prompt updates require a non-empty prompt.")
        with self._state_lock:
            if not self._configured:
                raise RuntimeError(
                    "Configure the SANA-WM realtime session before updating its prompt."
                )
            if normalized == self._prompt:
                return False
            started = time.perf_counter()
            self._prepare_prompt_conditions(normalized)
            self._prompt = normalized
            self.last_metrics = {
                "condition_ms": (time.perf_counter() - started) * 1000.0,
            }
            return True

    def configure(
        self,
        *,
        image: Image.Image,
        prompt: str,
        seed: int = 42,
        fps: int = 16,
        num_frames: int | None = None,
        step: int = DEFAULT_SAMPLING_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        horizontal_fov_deg: float = 55.0,
    ) -> dict[str, Any]:
        if int(fps) != self.fps:
            raise ValueError(
                f"SANA-WM realtime runs at the checkpoint-native {self.fps} FPS."
            )
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("SANA-WM realtime requires a user-provided text prompt.")
        if num_frames is not None:
            self.num_frames = _snap_num_frames(max(int(num_frames), 9))
            self.output_frames = self.num_frames - 1
        self._steps = max(int(step), 1)
        self._cfg_scale = float(cfg_scale)
        self._seed = int(seed)
        from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.inference_sana_wm import (
            TARGET_HEIGHT,
            TARGET_WIDTH,
            estimate_intrinsics_with_pi3x,
            resize_and_center_crop,
            transform_intrinsics_for_crop,
        )

        original = image.convert("RGB")
        cropped, source_size, resized_size, crop_offset = resize_and_center_crop(
            original,
            target_h=TARGET_HEIGHT,
            target_w=TARGET_WIDTH,
        )
        horizontal_fov_deg = float(horizontal_fov_deg)
        if not 0.0 < horizontal_fov_deg < 180.0:
            raise ValueError(
                f"SANA-WM horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}."
            )
        intrinsics_started = time.perf_counter()
        intrinsics_source: np.ndarray
        if _env_flag("SCENEMEMBENCH_SANA_WM_USE_PI3X", False):
            try:
                intrinsics_source = estimate_intrinsics_with_pi3x(
                    original,
                    self.devices["stage1"],
                    self.logger,
                )
            except Exception as exc:
                self.logger.warning(
                    "Pi3X intrinsics failed; using the supplied %.3f-degree FOV: %s",
                    horizontal_fov_deg,
                    exc,
                )
                width, height = original.size
                focal = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) / 2.0)
                intrinsics_source = np.asarray(
                    (focal, focal, width / 2.0, height / 2.0), dtype=np.float32
                )
        else:
            width, height = original.size
            focal = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) / 2.0)
            intrinsics_source = np.asarray(
                (focal, focal, width / 2.0, height / 2.0), dtype=np.float32
            )
        intrinsics = transform_intrinsics_for_crop(
            intrinsics_source,
            source_size,
            resized_size,
            crop_offset,
        )

        self._image = cropped
        self._prompt = prompt
        self._intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self._horizontal_fov_deg = horizontal_fov_deg
        self._camera_history = np.eye(4, dtype=np.float32)[None]
        self._context_latents = self._encode_context_images([cropped])
        self._refiner_context_latents = self._context_latents.clone()
        self._pose_format = "opencv"
        self._pose_origin_cv = np.eye(4, dtype=np.float32)
        self._world_pose = np.eye(4, dtype=np.float32)
        self._history_cursor = 0
        self._chunk_index = 0

        prompt_started = time.perf_counter()
        self._prepare_prompt_conditions(self._prompt)
        self._configured = True
        self.last_metrics = {
            "intrinsics_ms": (prompt_started - intrinsics_started) * 1000.0,
            "condition_ms": (time.perf_counter() - prompt_started) * 1000.0,
            "horizontal_fov_deg": horizontal_fov_deg,
        }
        return {
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def prefill_context(
        self,
        *,
        rgb_paths: Sequence[str | Path],
        camera_to_world: Sequence[Sequence[Sequence[float]]],
        frame_ids: Sequence[int],
        fps: float = 16.0,
        position_units: str = "m",
        coordinate_system: str = "",
        handedness: str = "",
        pose_format: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        """Encode GT video and teacher-force it through native streaming caches."""

        with self._state_lock:
            if not self._configured:
                raise RuntimeError("Configure SANA-WM before Context prefill.")
            if not rgb_paths:
                raise ValueError("SANA-WM Context prefill requires at least one frame.")
            if len(rgb_paths) != len(frame_ids) or len(camera_to_world) != len(
                frame_ids
            ):
                raise ValueError(
                    "SANA-WM Context RGB, pose, and frame-id lengths must match."
                )
            if abs(float(fps) - self.fps) > 1e-3:
                raise ValueError(f"SANA-WM expects {self.fps} FPS, got {fps}.")

            raw_poses = np.asarray(camera_to_world, dtype=np.float32)
            if raw_poses.ndim != 3 or raw_poses.shape[1:] != (4, 4):
                raise ValueError(
                    "SANA-WM Context poses must have shape (F, 4, 4) camera-to-world matrices."
                )
            requested_format = (
                str(
                    pose_format
                    or os.getenv("SCENEMEMBENCH_SANA_WM_POSE_FORMAT", "auto")
                )
                .strip()
                .lower()
            )
            if requested_format in {"", "auto", "metadata"}:
                normalized_coordinate = str(coordinate_system or "").strip().lower()
                normalized_handedness = str(handedness or "").strip().lower()
                if normalized_coordinate in {
                    "ue",
                    "unreal",
                    "unreal_engine",
                    "unrealengine",
                }:
                    requested_format = "ue"
                elif normalized_handedness in {"left", "left_handed", "lefthanded"}:
                    requested_format = "ue"
                else:
                    requested_format = "opencv"
            if requested_format in {"ue", "unreal", "unreal_engine", "unrealengine"}:
                from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.camera_control import (
                    ue_c2w_to_opencv_c2w,
                )

                converted_poses = ue_c2w_to_opencv_c2w(raw_poses)
                resolved_pose_format = "ue_to_opencv"
            elif requested_format in {"opencv", "cv", "opencv_c2w"}:
                converted_poses = raw_poses.copy()
                resolved_pose_format = "opencv"
            else:
                raise ValueError(
                    "SANA-WM pose_format must be auto, ue, or opencv; "
                    f"got {requested_format!r}."
                )
            self._pose_format = resolved_pose_format
            self._pose_origin_cv = converted_poses[0].copy()
            context_pose_inverse = np.linalg.inv(converted_poses[0]).astype(np.float32)
            normalized_units = str(position_units or "m").strip().lower()
            position_scale = (
                0.01
                if normalized_units
                in {"cm", "centimeter", "centimetres", "centimeters"}
                else 1.0
            )
            relative_poses = np.stack(
                [context_pose_inverse @ pose for pose in converted_poses],
                axis=0,
            ).astype(np.float32)
            relative_poses[:, :3, 3] *= position_scale

            aligned_count = (
                math.ceil((len(rgb_paths) - 1) / NATIVE_PIXEL_FRAMES_PER_BLOCK)
                * NATIVE_PIXEL_FRAMES_PER_BLOCK
                + 1
            )
            padding_count = aligned_count - len(rgb_paths)
            images = [self._load_context_image(path) for path in rgb_paths]
            # Native SANA blocks are aligned on the pixel-frame cadence.  If
            # the dataset context ends between block boundaries, extend the
            # *tail* with the final clean frame/pose.  This keeps the actual
            # GT context prefix intact and makes the causal state at the
            # continuation boundary explicitly anchored to context[-1].
            encoded_images = images + [images[-1].copy() for _ in range(padding_count)]
            padded_poses = np.concatenate(
                [
                    relative_poses,
                    np.repeat(relative_poses[-1:], padding_count, axis=0),
                ],
                axis=0,
            )
            context_latents = self._encode_context_images(encoded_images)
            expected_latents = (aligned_count - 1) // 8 + 1
            if int(context_latents.shape[2]) != expected_latents:
                raise RuntimeError(
                    "SANA-WM causal VAE produced an unexpected temporal layout: "
                    f"got {context_latents.shape[2]}, expected {expected_latents}."
                )

            history_before = self._history_cursor
            self._image = images[-1]
            self._world_pose = relative_poses[-1].copy()
            self._camera_history = padded_poses
            self._context_latents = context_latents
            self._refiner_context_latents = context_latents.clone()
            self._history_cursor = history_before + len(frame_ids)
            accepted_frame_ids = [int(item) for item in frame_ids]
            native_chunk_count = max(
                0, (expected_latents - 1) // NATIVE_LATENT_FRAMES_PER_BLOCK
            )
            return {
                "status": "accepted",
                "accepted_frame_ids": accepted_frame_ids,
                "rejected_frame_ids": [],
                "native_latent_count": expected_latents,
                "native_chunk_count": native_chunk_count,
                "history_cursor_before": history_before,
                "history_cursor_after": self._history_cursor,
                "state_digest_before": f"sana:{history_before}",
                "state_digest_after": f"sana:{self._history_cursor}:{expected_latents}:self_forcing",
                "last_context_pose": {"camera_to_world": camera_to_world[-1]},
                "boundary_mapping": {
                    "dataset_context_frames": [
                        int(frame_ids[0]),
                        int(frame_ids[-1]) + 1,
                    ],
                    "native_history_policy": "gdn_state_plus_sink_and_recent_kv",
                    "stage1_cached_blocks": int(
                        os.getenv("SCENEMEMBENCH_SANA_WM_CACHED_BLOCKS", "2")
                    ),
                    "refiner_kv_max_frames": int(
                        self.pipeline.refiner_settings.kv_max_frames
                    ),
                    "tail_padding_frames": padding_count,
                    "continuation_boundary": "context_tail_after_tail_padding",
                    "horizontal_fov_deg": self._horizontal_fov_deg,
                    "intrinsics_vec4_after_crop": self._intrinsics.tolist(),
                    "pose_input_format": resolved_pose_format,
                    "pose_conversion": (
                        "UE(+X forward,+Y right,+Z up) -> OpenCV(+X right,+Y down,+Z forward)"
                        if resolved_pose_format == "ue_to_opencv"
                        else "none"
                    ),
                },
                "warnings": (
                    [
                        "GT context was tail-padded with its last frame to complete the native 24-frame block cadence."
                    ]
                    if padding_count
                    else []
                ),
            }

    def _encode_context_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.inference_sana_wm import (
            vae_encode,
        )

        if not images:
            raise ValueError("SANA-WM context encoding requires at least one image.")
        pixels = np.stack(
            [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images]
        )
        video = torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).float()
        video = video.div_(127.5).sub_(1.0)
        self.pipeline.vae.to(self.pipeline.device)
        latents = vae_encode(
            self.pipeline.config.vae.vae_type,
            self.pipeline.vae,
            video.to(self.pipeline.device, dtype=self.pipeline.vae_dtype),
            device=self.pipeline.device,
        )
        return latents.to(dtype=self.pipeline.weight_dtype).detach().cpu()

    @staticmethod
    def _load_context_image(path: str | Path) -> Image.Image:
        from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.inference_sana_wm import (
            TARGET_HEIGHT,
            TARGET_WIDTH,
            resize_and_center_crop,
        )

        with Image.open(path) as source:
            cropped, _, _, _ = resize_and_center_crop(
                source.convert("RGB"), target_h=TARGET_HEIGHT, target_w=TARGET_WIDTH
            )
        return cropped.copy()

    def generate(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
        camera_trajectory_ue: Sequence[Sequence[Sequence[float]]] | None = None,
        trajectory_mode: str | None = None,
        position_units: str = "m",
    ) -> dict[str, Any]:
        """Advance one chunk, applying an optional prompt at its boundary."""

        with self._state_lock:
            condition_ms: float | None = None
            if prompt is not None and self.update_prompt(prompt):
                condition_ms = self.last_metrics.get("condition_ms")
            result = self._generate_current_prompt(
                interactions=interactions,
                control_segments=control_segments,
                seed=seed,
                camera_trajectory_ue=camera_trajectory_ue,
                trajectory_mode=trajectory_mode,
                position_units=position_units,
            )
            if condition_ms is not None:
                self.last_metrics["condition_ms"] = condition_ms
                result["realtime_metrics"]["condition_ms"] = condition_ms
            return result

    def _generate_current_prompt(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        camera_trajectory_ue: Sequence[Sequence[Sequence[float]]] | None = None,
        trajectory_mode: str | None = None,
        position_units: str = "m",
    ) -> dict[str, Any]:
        if (
            not self._configured
            or self._image is None
            or self._intrinsics is None
            or self._camera_history is None
            or self._context_latents is None
        ):
            raise RuntimeError("SANA-WM realtime session is not configured.")
        from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.inference_sana_wm import (
            action_string_to_c2w,
        )

        frame_keys = _frames_for_segments(
            control_segments,
            list(interactions or ()),
            frame_count=self.output_frames,
            fps=self.fps,
        )
        if camera_trajectory_ue is not None:
            prescribed = np.asarray(camera_trajectory_ue, dtype=np.float32)
            if prescribed.ndim != 3 or prescribed.shape[1:] != (4, 4):
                raise ValueError(
                    "SANA-WM prescribed camera trajectory must have shape (F, 4, 4)."
                )
            if len(prescribed) != self.output_frames:
                raise ValueError(
                    "SANA-WM prescribed trajectory must contain exactly one pose per "
                    f"future frame ({self.output_frames}), got {len(prescribed)}."
                )
            if self._pose_format == "ue_to_opencv":
                from scenemembench._native.base_models.diffusion_model.image.sana.inference_video_scripts.wm.camera_control import (
                    ue_c2w_to_opencv_c2w,
                )

                prescribed = ue_c2w_to_opencv_c2w(prescribed)
            absolute_future = np.einsum(
                "ij,fjk->fik",
                np.linalg.inv(self._pose_origin_cv).astype(np.float32),
                prescribed,
            ).astype(np.float32)
            normalized_units = str(position_units or "m").strip().lower()
            if normalized_units in {"cm", "centimeter", "centimetres", "centimeters"}:
                absolute_future[..., :3, 3] *= 0.01
            future_c2w = absolute_future
            trajectory_source = "prescribed_gt_pose"
        else:
            relative_future = action_string_to_c2w(
                _compress_action_frames(frame_keys),
                translation_speed=float(
                    os.getenv("SCENEMEMBENCH_SANA_WM_TRANSLATION_PER_FRAME", "0.025")
                ),
                rotation_speed_deg=float(
                    os.getenv("SCENEMEMBENCH_SANA_WM_ROTATION_DEG_PER_FRAME", "1.2")
                ),
            )
            absolute_future = np.einsum(
                "ij,fjk->fik",
                self._world_pose,
                relative_future,
            ).astype(np.float32)
            future_c2w = absolute_future[1:]
            trajectory_source = "native_action"
        c2w = np.concatenate([self._camera_history, future_c2w], axis=0)
        intrinsics = np.broadcast_to(self._intrinsics, (len(c2w), 4)).copy()
        cached_blocks = int(os.getenv("SCENEMEMBENCH_SANA_WM_CACHED_BLOCKS", "2"))
        if cached_blocks == 0 or cached_blocks < -1:
            raise ValueError(
                "SCENEMEMBENCH_SANA_WM_CACHED_BLOCKS must be -1 or a positive integer."
            )
        params = self.params_cls(
            num_frames=len(c2w),
            fps=self.fps,
            step=self._steps,
            cfg_scale=self._cfg_scale,
            flow_shift=8.0,
            # Each call reconstructs the full native timeline and overwrites
            # its observed prefix. Reusing the case seed keeps future noise at
            # the same absolute offsets as one-shot native inference; changing
            # the seed per resident step would restart an unrelated trajectory.
            seed=self._seed,
            negative_prompt="",
            sampling_algo="self_forcing",
            num_cached_blocks=cached_blocks,
            sink_token=True,
            num_frame_per_block=NATIVE_LATENT_FRAMES_PER_BLOCK,
            denoising_step_list=list(DEFAULT_DENOISING_STEP_LIST),
        )
        started = time.perf_counter()
        history_cursor_before = self._history_cursor
        context_latent_count = int(self._context_latents.shape[2])
        decoded_chunks: list[np.ndarray] = []

        def collect_chunk(
            frames: np.ndarray, _frame_base: int, _chunk_index: int
        ) -> None:
            decoded_chunks.append(np.ascontiguousarray(frames, dtype=np.uint8))

        result = self.pipeline.generate_streaming(
            self._image,
            self._prompt,
            c2w,
            intrinsics,
            params,
            output_path=Path("/tmp/worldfoundry-sana-wm-streaming.mp4"),
            output_mode="cpu",
            decoded_chunk_callback=collect_chunk,
            context_latents=self._context_latents,
            refiner_context_latents=self._refiner_context_latents,
        )
        if not decoded_chunks:
            raise RuntimeError("SANA-WM Streaming decoded no output chunks.")
        all_video = np.concatenate(decoded_chunks, axis=0)
        video = _validate_output_video(
            all_video[-self.output_frames :],
            expected_frames=self.output_frames,
        )
        self._image = Image.fromarray(video[-1].copy(), mode="RGB")
        self._world_pose = c2w[-1].copy()
        self._camera_history = c2w
        self._context_latents = result["latent"]
        self._refiner_context_latents = result.get("refiner_latent")
        if self._refiner_context_latents is None:
            raise RuntimeError(
                "SANA-WM streaming did not return refiner history latents."
            )
        self._chunk_index += 1
        self._history_cursor += self.output_frames
        self.last_metrics = dict(
            generation_ms=(time.perf_counter() - started) * 1000.0,
            history_cursor_before=history_cursor_before,
            history_cursor_after=self._history_cursor,
            state_digest=f"sana:{self._history_cursor}:{self._chunk_index}:self_forcing",
            context_latent_count=context_latent_count,
            context_latent_count_after=int(self._context_latents.shape[2]),
            stage1_cached_blocks=cached_blocks,
            trajectory_source=trajectory_source,
            trajectory_mode=trajectory_mode or trajectory_source,
            case_seed=self._seed,
            requested_step_seed=None if seed is None else int(seed),
            stage1_sink_token=True,
            refiner_kv_max_frames=int(self.pipeline.refiner_settings.kv_max_frames),
            refiner_noise_block_offset=int(result.get("refiner_noise_block_offset", 0)),
            memory_bank_read=context_latent_count > 0,
        )
        return {
            "frames": video,
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def reset(self) -> None:
        with self._state_lock:
            self._image = None
            self._intrinsics = None
            self._camera_history = None
            self._context_latents = None
            self._refiner_context_latents = None
            self._prompt = ""
            self._world_pose = np.eye(4, dtype=np.float32)
            self._history_cursor = 0
            self._configured = False
            self._chunk_index = 0
            self.last_metrics = {}


__all__ = ["SanaWMRealtimeSession", "ensure_sana_import_paths"]
