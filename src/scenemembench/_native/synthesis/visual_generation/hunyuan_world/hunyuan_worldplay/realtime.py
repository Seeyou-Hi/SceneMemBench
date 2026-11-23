# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Resident autoregressive inference for HY-WorldPlay.

The released pipeline already denoises four latent frames at a time.  This
adapter keeps that native block size and retains the expensive state that the
offline entry point normally recreates: prompt/text KV, the random stream,
generated latent history, camera/action history, geometry-selected vision KV,
and the causal VAE decoder feature maps.
"""

from __future__ import annotations

import time
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import distributed as dist

from scenemembench._native.core.realtime import RealtimeSpec
from scenemembench._native.core.camera_trajectory import (
    unreal_camera_poses_to_worldplay_motions,
)

from .commons import auto_offload_model
from .commons.infer_state import get_infer_state
from .commons.pose_utils import pose_to_input
from .generate_custom_trajectory import generate_camera_trajectory_local
from .pipelines.pipeline_utils import retrieve_timesteps
from .utils.retrieval_context import generate_points_in_sphere

_KEY_TO_CONTROL = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "i": "camera_up",
    "k": "camera_down",
    "j": "camera_l",
    "l": "camera_r",
}
_CONTROL_ALIASES = {
    "back": "backward",
    "camera_left": "camera_l",
    "camera_right": "camera_r",
}


def _rotation_scaled_query_viewmats(
    viewmats: torch.Tensor,
    *,
    start_idx: int,
    count: int,
    scale: float,
    accumulated_extra_yaw: float,
) -> tuple[torch.Tensor, float]:
    """Scale only new query yaw while leaving the stored camera trajectory intact."""
    if start_idx <= 0 or count <= 0:
        raise ValueError(
            "rotation-scaled query poses require prior context and new poses"
        )
    original = viewmats[0]
    if start_idx + count > original.shape[0]:
        raise ValueError("query pose range exceeds available viewmats")
    output: list[torch.Tensor] = []
    extra_yaw = float(accumulated_extra_yaw)
    for index in range(start_idx, start_idx + count):
        previous_c2w = torch.linalg.inv(original[index - 1])
        current_c2w = torch.linalg.inv(original[index])
        relative_rotation = previous_c2w[:3, :3].transpose(0, 1) @ current_c2w[:3, :3]
        yaw_delta = torch.atan2(relative_rotation[0, 2], relative_rotation[2, 2])
        extra_yaw += (float(scale) - 1.0) * float(yaw_delta.item())
        angle = current_c2w.new_tensor(extra_yaw)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        yaw_rotation = torch.eye(3, device=current_c2w.device, dtype=current_c2w.dtype)
        yaw_rotation[0, 0] = cosine
        yaw_rotation[0, 2] = sine
        yaw_rotation[2, 0] = -sine
        yaw_rotation[2, 2] = cosine
        scaled_c2w = current_c2w.clone()
        scaled_c2w[:3, :3] = current_c2w[:3, :3] @ yaw_rotation
        output.append(torch.linalg.inv(scaled_c2w))
    return torch.stack(output, dim=0).unsqueeze(0), extra_yaw


class HunyuanWorldPlayRealtimeSession:
    """One quality-preserving resident HY-WorldPlay rollout."""

    fps = 24
    latent_frames_per_chunk = 4
    first_output_frames = 13
    steady_output_frames = 16

    def __init__(self, runtime: Any, operator: Any) -> None:
        self.runtime = runtime
        self.core = runtime.model
        self.operator = operator
        self.device = torch.device(self.core.execution_device)
        self.dtype = self.core.target_dtype
        self.rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self.configured = False
        self.configure_metrics: dict[str, float] = {}
        self._generator: torch.Generator | None = None
        self._timesteps: torch.Tensor | None = None
        self._latents: torch.Tensor | None = None
        self._cond_latents: torch.Tensor | None = None
        self._image_condition: torch.Tensor | None = None
        self._motions: list[dict[str, float]] = []
        self._height = 0
        self._width = 0
        self._latent_height = 0
        self._latent_width = 0
        self._num_inference_steps = 0
        self._context_latent_count = 0
        self._context_pose_source_frame_ids: tuple[int, ...] = ()
        self._context_motion_diagnostics: dict[str, float | int] = {}
        self._last_camera_condition_metrics: dict[str, object] = {}
        self._point_cloud_config: Mapping[str, Any] | Any | None = None
        self._point_cloud_runtime: dict[str, object] | None = None
        self._lingbot_runtime: Any | None = None
        self._last_prescribed_ue_pose: np.ndarray | None = None
        self._rotation_conditioning_extra_yaw = 0.0

    def _maybe_create_lingbot_runtime(self) -> None:
        """Start the optional HY LingBot worker used by point-cloud retrieval."""

        if self._lingbot_runtime is not None or self._point_cloud_runtime is not None:
            return
        checkpoint = os.environ.get("SCENEMEMBENCH_LINGBOT_CHECKPOINT", "").strip()
        worker_devices = os.environ.get(
            "SCENEMEMBENCH_LINGBOT_WORKER_CUDA_VISIBLE_DEVICES", ""
        ).strip()
        if not checkpoint or not worker_devices:
            return
        if self.world_size != 1 and self.rank != 0:
            # Rank 0 owns the external LingBot process.  Context selection is
            # broadcast before the distributed vision rebuild below.
            return
        hy_root = (
            Path(
                os.environ.get(
                    "SCENEMEMBENCH_HY_WORLDPLAY_ROOT",
                    "./external/HY-WorldPlay",
                )
            )
            .expanduser()
            .resolve()
        )
        if not (
            hy_root / "hyvideo" / "worldplay_memory" / "lingbot_external_runtime.py"
        ).is_file():
            raise FileNotFoundError(f"HY-WorldPlay runtime not found under {hy_root}")
        root_text = str(hy_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        python_path = os.environ.get("PYTHONPATH", "")
        if root_text not in python_path.split(os.pathsep):
            os.environ["PYTHONPATH"] = root_text + (
                os.pathsep + python_path if python_path else ""
            )
        from hyvideo.worldplay_memory.lingbot_external_runtime import (
            ExternalLingBotWorkerRuntime,
        )

        self._lingbot_runtime = ExternalLingBotWorkerRuntime(
            cuda_visible_devices=worker_devices,
            lingbot_root=os.environ.get(
                "SCENEMEMBENCH_LINGBOT_ROOT",
                str(hy_root / "third_party" / "lingbot_map"),
            ),
            checkpoint_path=checkpoint,
            image_size=int(os.environ.get("SCENEMEMBENCH_LINGBOT_IMAGE_SIZE", "518")),
            patch_size=int(os.environ.get("SCENEMEMBENCH_LINGBOT_PATCH_SIZE", "14")),
            num_scale_frames=8,
            max_frame_num=int(
                os.environ.get("SCENEMEMBENCH_LINGBOT_MAX_FRAME_NUM", "1024")
            ),
            kv_cache_sliding_window=int(
                os.environ.get("SCENEMEMBENCH_LINGBOT_KV_CACHE_WINDOW", "64")
            ),
            camera_num_iterations=int(
                os.environ.get("SCENEMEMBENCH_LINGBOT_CAMERA_ITERATIONS", "4")
            ),
            use_sdpa=True,
            enable_3d_rope=True,
            keyframe_mode=os.environ.get(
                "SCENEMEMBENCH_LINGBOT_KEYFRAME_MODE", "fixed"
            ),
            keyframe_interval=1,
            retain_dense_history=True,
        )

    def _refresh_lingbot_runtime(self) -> None:
        if self._lingbot_runtime is None:
            return
        # The worker writes source-tagged point shards asynchronously.  Wait
        # before taking the snapshot so retrieval sees the same history as the
        # pose/depth arrays, and expose the shard directory to the selector.
        wait_for_writes = getattr(self._lingbot_runtime, "wait_for_writes", None)
        if callable(wait_for_writes):
            wait_for_writes()
        snapshot = self._lingbot_runtime.snapshot()
        if not snapshot:
            return
        runtime: dict[str, object] = dict(snapshot)
        points_dir = getattr(self._lingbot_runtime, "points_dir", None)
        if points_dir is not None:
            runtime["lingbot_points_dir"] = str(points_dir)
        for key in ("depth", "depth_conf"):
            value = runtime.get(key)
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[-1] == 1:
                runtime[key] = value.squeeze(-1)
        frame_count = int(runtime["lingbot_w2c"].shape[0])
        runtime["lingbot_frame_count"] = frame_count
        runtime["pose_indices"] = list(range(frame_count))
        runtime["lingbot_pose_convention"] = "w2c"
        if self._lingbot_runtime.input_hw is not None:
            runtime["lingbot_image_hw"] = tuple(self._lingbot_runtime.input_hw)
        self._point_cloud_runtime = runtime
        self.core.update_point_cloud_runtime(runtime)

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
            controls=(
                "forward",
                "backward",
                "left",
                "right",
                "camera_up",
                "camera_down",
                "camera_l",
                "camera_r",
            ),
        )

    def next_output_frames(self) -> int:
        return (
            self.first_output_frames
            if self.generated_latent_frames == 0
            else self.steady_output_frames
        )

    @property
    def generated_latent_frames(self) -> int:
        return 0 if self._latents is None else int(self._latents.shape[2])

    @staticmethod
    def _normalize_controls(controls: Sequence[str]) -> list[str]:
        result = []
        for control in controls:
            value = str(control).strip().lower()
            value = _CONTROL_ALIASES.get(value, value)
            if value:
                result.append(value)
        return result

    def _controls_for_latents(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        *,
        count: int,
    ) -> list[list[str]]:
        fallback = self._normalize_controls(interactions)
        segments: list[tuple[float, list[str]]] = []
        for segment in control_segments or ():
            duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
            controls = [
                _KEY_TO_CONTROL[key]
                for key in (str(item).lower() for item in segment.get("keys", ()))
                if key in _KEY_TO_CONTROL
            ]
            if duration > 0.0:
                segments.append((duration, controls))
        if not segments:
            return [fallback] * count
        total = sum(duration for duration, _ in segments)
        if total <= 0.0:
            return [fallback] * count
        result: list[list[str]] = []
        segment_index = 0
        segment_end = segments[0][0]
        for index in range(count):
            sample_time = (index + 0.5) * total / count
            while segment_index + 1 < len(segments) and sample_time > segment_end:
                segment_index += 1
                segment_end += segments[segment_index][0]
            result.append(segments[segment_index][1])
        return result

    def _motion(self, controls: Sequence[str]) -> dict[str, float]:
        controls = self._normalize_controls(controls)
        forward = float("forward" in controls) - float("backward" in controls)
        right = float("right" in controls) - float("left" in controls)
        yaw = float("camera_r" in controls) - float("camera_l" in controls)
        pitch = float("camera_up" in controls) - float("camera_down" in controls)
        motion: dict[str, float] = {}
        if forward:
            motion["forward"] = forward * float(self.operator.forward_speed)
        if right:
            motion["right"] = right * float(self.operator.forward_speed)
        if yaw:
            motion["yaw"] = yaw * np.deg2rad(float(self.operator.yaw_speed_deg))
        if pitch:
            motion["pitch"] = pitch * np.deg2rad(float(self.operator.pitch_speed_deg))
        return motion

    def _camera_conditions(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        camera_trajectory_ue: Sequence[Sequence[Sequence[float]]] | None = None,
        action_motion_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The first chunk includes the identity pose, so it needs only three new
        # motions. Every continuation contributes four new latent poses.
        new_motion_count = (
            self.latent_frames_per_chunk - 1
            if self.generated_latent_frames == 0
            else self.latent_frames_per_chunk
        )
        if camera_trajectory_ue:
            # Controller routes are supplied at pixel cadence.  WorldPlay's
            # action conditioner consumes one pose per four-frame latent step;
            # convert the prescribed route into the same local motion history
            # used by keyboard actions so route replay and live control share a
            # single state representation.
            poses = np.asarray(camera_trajectory_ue, dtype=np.float64)
            if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                raise ValueError("camera_trajectory_ue must contain 4x4 poses")
            endpoint_indices = np.minimum(
                (np.arange(new_motion_count) + 1) * self.latent_frames_per_chunk - 1,
                len(poses) - 1,
            )
            if self._last_prescribed_ue_pose is None:
                sampled = poses[np.concatenate(([0], endpoint_indices))]
            else:
                sampled = np.concatenate(
                    (self._last_prescribed_ue_pose[None], poses[endpoint_indices]),
                    axis=0,
                )
            converted = unreal_camera_poses_to_worldplay_motions(
                sampled,
                latent_count=len(sampled),
                temporal_stride=1,
                translation_step=float(self.operator.forward_speed),
            )
            self._motions.extend(
                {
                    key: float(value) * float(action_motion_scale)
                    for key, value in motion.items()
                }
                for motion in converted
            )
            self._last_prescribed_ue_pose = poses[-1].copy()
        else:
            controls = self._controls_for_latents(
                interactions,
                control_segments,
                count=new_motion_count,
            )
            self._motions.extend(
                {
                    key: float(value) * float(action_motion_scale)
                    for key, value in self._motion(items).items()
                }
                for items in controls
            )
        poses = generate_camera_trajectory_local(self._motions)
        intrinsic = [
            [969.6969696969696, 0.0, 960.0],
            [0.0, 969.6969696969696, 540.0],
            [0.0, 0.0, 1.0],
        ]
        pose_data = {
            str(index): {"extrinsic": pose.tolist(), "K": intrinsic}
            for index, pose in enumerate(poses)
        }
        viewmats, intrinsics, action = pose_to_input(pose_data, len(poses))
        query_poses = poses[-(new_motion_count + 1) :]
        query_yaws = [
            float(np.degrees(np.arctan2(float(pose[0, 2]), float(pose[2, 2]))))
            for pose in query_poses
        ]
        self._last_camera_condition_metrics = {
            "new_motion_count": int(new_motion_count),
            "query_yaw_deg": query_yaws,
            "query_action_labels": [
                int(value) for value in action[-new_motion_count:].tolist()
            ],
            "operator_yaw_speed_deg": float(self.operator.yaw_speed_deg),
            "source": "prescribed_pose" if camera_trajectory_ue else "canonical_action",
        }
        return viewmats.unsqueeze(0), intrinsics.unsqueeze(0), action.unsqueeze(0)

    def _condition_block(self) -> torch.Tensor:
        assert self._image_condition is not None
        channels = int(self.core.transformer.config.in_channels)
        condition = torch.zeros(
            (
                1,
                channels,
                self.latent_frames_per_chunk,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            # Match the released offline path: the fp16 image latent is
            # concatenated with an fp32 task mask, producing fp32 conditioning.
            dtype=torch.float32,
        )
        mask = torch.zeros(
            (
                1,
                1,
                self.latent_frames_per_chunk,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        if self.generated_latent_frames == 0:
            condition[:, :, :1] = self._image_condition.to(
                device=self.device, dtype=self.dtype
            )
            mask[:, :, :1] = 1
        return torch.cat((condition, mask), dim=1)

    @torch.inference_mode()
    def configure(
        self,
        image: Image.Image,
        *,
        prompt: str,
        seed: int = 1,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        flow_shift: float | None = None,
        guidance_scale: float | None = None,
        few_step: bool = True,
        user_height: int | None = None,
        user_width: int | None = None,
        point_cloud_config: Mapping[str, Any] | Any | None = None,
        point_cloud_runtime: Mapping[str, object] | None = None,
    ) -> dict[str, Any] | None:
        """Encode invariant conditions and initialize resident model caches."""

        if not isinstance(image, Image.Image):
            raise TypeError("HY-WorldPlay realtime configuration requires a PIL image.")
        infer_state = get_infer_state()
        if infer_state is not None and infer_state.use_vae_parallel:
            raise RuntimeError(
                "HY-WorldPlay realtime does not support tile-parallel VAE decode: "
                "persistent causal feat_map state is tile-local. Disable use_vae_parallel."
            )
        self.reset()
        started = time.perf_counter()
        core = self.core
        image = image.convert("RGB")
        target_resolution = core.ideal_resolution
        height, width = core.get_closest_resolution_given_reference_image(
            image, target_resolution
        )
        self._height = int(user_height or height)
        self._width = int(user_width or width)
        _, self._latent_height, self._latent_width = core.get_latent_size(
            self.first_output_frames,
            self._height,
            self._width,
        )
        self._num_inference_steps = int(num_inference_steps)
        core.scheduler = core._create_scheduler(
            core.config.flow_shift if flow_shift is None else flow_shift
        )
        resolved_guidance = (
            core.config.guidance_scale if guidance_scale is None else guidance_scale
        )
        core._guidance_scale = 1.0 if few_step else resolved_guidance
        core._guidance_rescale = 0.0
        core._clip_skip = None
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))

        prompt_started = time.perf_counter()
        with auto_offload_model(
            core.text_encoder, core.execution_device, enabled=core.enable_offloading
        ):
            prompt_embeds, negative_embeds, prompt_mask, negative_mask = (
                core.encode_prompt(
                    prompt,
                    self.device,
                    1,
                    core.do_classifier_free_guidance,
                    negative_prompt,
                    clip_skip=None,
                    data_type="video",
                )
            )
        extra_kwargs = {}
        if core.config.glyph_byT5_v2:
            with auto_offload_model(
                core.byt5_model, core.execution_device, enabled=core.enable_offloading
            ):
                extra_kwargs = core._prepare_byt5_embeddings(prompt, self.device)
        if core.do_classifier_free_guidance:
            prompt_embeds = torch.cat((negative_embeds, prompt_embeds))
            if prompt_mask is not None:
                prompt_mask = torch.cat((negative_mask, prompt_mask))
        prompt_ms = (time.perf_counter() - prompt_started) * 1000.0

        latent_template = torch.zeros(
            (
                1,
                int(core.transformer.config.in_channels),
                self.latent_frames_per_chunk,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        n_tokens = (
            self.latent_frames_per_chunk * self._latent_height * self._latent_width
        )
        timestep_kwargs = core.prepare_extra_func_kwargs(
            core.scheduler.set_timesteps,
            {"n_tokens": n_tokens},
        )
        self._timesteps, self._num_inference_steps = retrieve_timesteps(
            core.scheduler,
            self._num_inference_steps,
            self.device,
            **timestep_kwargs,
        )
        core.num_warmup_steps = (
            len(self._timesteps) - self._num_inference_steps * core.scheduler.order
        )
        core._num_timesteps = len(self._timesteps)
        core.num_inference_steps = self._num_inference_steps
        core.chunk_latent_frames = self.latent_frames_per_chunk
        core.points_local = generate_points_in_sphere(50000, 8.0).to(self.device)
        if self.world_size > 1:
            # Geometry retrieval must choose identical context indices on every
            # sequence-parallel rank.
            dist.broadcast(core.points_local, src=0)
        self._point_cloud_config = point_cloud_config
        self._point_cloud_runtime = (
            dict(point_cloud_runtime) if point_cloud_runtime is not None else None
        )
        if self._point_cloud_config is not None and self._point_cloud_runtime is None:
            self._maybe_create_lingbot_runtime()
        core.configure_point_cloud_retrieval(
            point_cloud_config,
            runtime=self._point_cloud_runtime,
        )

        visual_started = time.perf_counter()
        with auto_offload_model(
            core.vae, core.execution_device, enabled=core.enable_offloading
        ):
            self._image_condition = core.get_image_condition_latents(
                "i2v", image, self._height, self._width
            )
        with auto_offload_model(
            core.vision_encoder, core.execution_device, enabled=core.enable_offloading
        ):
            vision_states = core._prepare_vision_states(
                np.asarray(image), target_resolution, latent_template, self.device
            )
        visual_ms = (time.perf_counter() - visual_started) * 1000.0

        cache_started = time.perf_counter()
        core.prepare_ar_text_cache(
            latents=latent_template,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            vision_states=vision_states,
            task_type="i2v",
            extra_kwargs=extra_kwargs,
            device=self.device,
        )
        cache_ms = (time.perf_counter() - cache_started) * 1000.0
        self._latents = latent_template[:, :, :0].clone()
        self._cond_latents = torch.zeros(
            (
                1,
                latent_template.shape[1] + 1,
                0,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        if self.rank == 0:
            core.vae.begin_stream_decode()
        self.configured = True
        self.configure_metrics = {
            "prompt_ms": prompt_ms,
            "visual_condition_ms": visual_ms,
            "text_cache_ms": cache_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "world_size": float(self.world_size),
        }
        if self.rank != 0:
            return None
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.configure_metrics),
        }

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        camera_trajectory_ue: Sequence[Sequence[Sequence[float]]] | None = None,
        point_cloud_runtime: Mapping[str, object] | None = None,
        point_cloud_config: Mapping[str, Any] | Any | None = None,
        action_motion_scale: float = 1.0,
        **_: Any,
    ) -> dict[str, Any] | None:
        """Advance exactly one four-latent block on the persistent rollout."""

        if not self.configured or self._generator is None or self._timesteps is None:
            raise RuntimeError("HY-WorldPlay realtime session is not configured.")
        assert self._latents is not None and self._cond_latents is not None
        started = time.perf_counter()
        condition_started = started
        start_idx = self.generated_latent_frames
        viewmats, intrinsics, action = self._camera_conditions(
            interactions,
            control_segments,
            camera_trajectory_ue=camera_trajectory_ue,
            action_motion_scale=action_motion_scale,
        )
        if point_cloud_runtime is not None:
            self._point_cloud_runtime = dict(point_cloud_runtime)
            self.core.update_point_cloud_runtime(self._point_cloud_runtime)
        if point_cloud_config is not None:
            self._point_cloud_config = point_cloud_config
            self.core.configure_point_cloud_retrieval(
                point_cloud_config,
                runtime=self._point_cloud_runtime,
            )
        query_viewmats_override = None
        retrieval_config = self.core.point_cloud_retrieval_config
        if (
            start_idx > 0
            and retrieval_config is not None
            and retrieval_config.cache_policy == "rotation_following"
            and float(retrieval_config.rotation_query_pose_scale) != 1.0
        ):
            query_viewmats_override, self._rotation_conditioning_extra_yaw = (
                _rotation_scaled_query_viewmats(
                    viewmats,
                    start_idx=start_idx,
                    count=self.latent_frames_per_chunk,
                    scale=float(retrieval_config.rotation_query_pose_scale),
                    accumulated_extra_yaw=self._rotation_conditioning_extra_yaw,
                )
            )
            self._last_camera_condition_metrics["rotation_query_pose_scale"] = float(
                retrieval_config.rotation_query_pose_scale
            )
            self._last_camera_condition_metrics["rotation_query_extra_yaw_deg"] = float(
                np.degrees(self._rotation_conditioning_extra_yaw)
            )
        condition_block = self._condition_block()
        noise = self.core.prepare_latents(
            1,
            int(self.core.transformer.config.in_channels),
            self._latent_height,
            self._latent_width,
            self.latent_frames_per_chunk,
            self.dtype,
            self.device,
            self._generator,
        )
        self._latents = torch.cat((self._latents, noise), dim=2)
        self._cond_latents = torch.cat((self._cond_latents, condition_block), dim=2)
        selected: list[int] = []
        selection = {
            "selected_history_latent_indices": [],
            "anchor_memory_latent_indices": [],
            "retrieved_memory_latent_indices": [],
            "temporal_context_latent_indices": [],
        }
        # ``None`` is the native/FOV-only path. Only the explicitly configured
        # point-cloud model may replace it with warped-RoPE coordinates below.
        # The first realtime chunk has no history branch, so it must retain this
        # disabled sentinel instead of referencing an unbound local.
        ucm_context: dict[str, torch.Tensor] | None = None
        if start_idx > 0:
            context_target_latent_indices = None
            context_slot_attention_weights = None
            if self.world_size > 1:
                if self.rank == 0:
                    selected, selection = self.core.select_ar_context_indices(
                        viewmats=viewmats,
                        Ks=intrinsics,
                        current_frame_idx=start_idx,
                        chunk_latent_frames=self.latent_frames_per_chunk,
                        latent_hw=(self._latent_height, self._latent_width),
                        device=self.device,
                        return_diagnostics=True,
                    )
                    context_target_latent_indices = (
                        self.core.context_target_latent_indices(
                            selected_frame_indices=selected,
                            selection=selection,
                            current_frame_idx=start_idx,
                        )
                    )
                    selected_payload: list[object] = [
                        {
                            "selected": list(selected),
                            "context_target_latent_indices": context_target_latent_indices,
                            "context_slot_attention_weights": selection.get(
                                "context_slot_attention_weights"
                            ),
                        }
                    ]
                else:
                    selected_payload = [None]
                dist.broadcast_object_list(selected_payload, src=0)
                payload = selected_payload[0] or {}
                selected = [int(index) for index in payload.get("selected", [])]
                raw_context_targets = payload.get("context_target_latent_indices")
                context_target_latent_indices = (
                    None
                    if raw_context_targets is None
                    else [int(index) for index in raw_context_targets]
                )
                raw_slot_weights = payload.get("context_slot_attention_weights")
                context_slot_attention_weights = (
                    None
                    if raw_slot_weights is None
                    else [float(value) for value in raw_slot_weights]
                )
                if self.rank != 0:
                    selection = {
                        "selected_history_latent_indices": selected,
                        "anchor_memory_latent_indices": [],
                        "retrieved_memory_latent_indices": [],
                        "temporal_context_latent_indices": [],
                    }
            else:
                selected, selection = self.core.select_ar_context_indices(
                    viewmats=viewmats,
                    Ks=intrinsics,
                    current_frame_idx=start_idx,
                    chunk_latent_frames=self.latent_frames_per_chunk,
                    latent_hw=(self._latent_height, self._latent_width),
                    device=self.device,
                    return_diagnostics=True,
                )
                context_target_latent_indices = self.core.context_target_latent_indices(
                    selected_frame_indices=selected,
                    selection=selection,
                    current_frame_idx=start_idx,
                )
                context_slot_attention_weights = selection.get(
                    "context_slot_attention_weights"
                )
            if self.core.point_cloud_retrieval_config is not None and (
                self.core.point_cloud_retrieval_config.ucm_warp_rope_enabled
            ):
                if self.world_size > 1:
                    if self.rank == 0:
                        ucm_context = self.core.prepare_ucm_context(
                            selected_frame_indices=selected,
                            selection=selection,
                            viewmats=viewmats,
                            Ks=intrinsics,
                            current_frame_idx=start_idx,
                            chunk_latent_frames=self.latent_frames_per_chunk,
                            latent_hw=(self._latent_height, self._latent_width),
                            device=self.device,
                        )
                    payload = [ucm_context] if self.rank == 0 else [None]
                    dist.broadcast_object_list(payload, src=0)
                    ucm_context = payload[0]
                    if ucm_context is not None:
                        ucm_context = {
                            key: value.to(self.device)
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in ucm_context.items()
                        }
                else:
                    ucm_context = self.core.prepare_ucm_context(
                        selected_frame_indices=selected,
                        selection=selection,
                        viewmats=viewmats,
                        Ks=intrinsics,
                        current_frame_idx=start_idx,
                        chunk_latent_frames=self.latent_frames_per_chunk,
                        latent_hw=(self._latent_height, self._latent_width),
                        device=self.device,
                    )
            if ucm_context is not None:
                context_target_latent_indices = [
                    int(index)
                    for index in ucm_context.get("context_target_latent_indices", [])
                ]
            if self.rank == 0 and isinstance(
                selection.get("point_cloud_retrieval"), dict
            ):
                selection["point_cloud_retrieval"]["ucm_warp_rope_runtime"] = dict(
                    self.core._last_ucm_warp_rope_metrics
                )
            self.core.rebuild_ar_vision_cache(
                latents=self._latents,
                cond_latents=self._cond_latents,
                viewmats=viewmats,
                Ks=intrinsics,
                action=action,
                selected_frame_indices=selected,
                timesteps=self._timesteps,
                task_type="i2v",
                device=self.device,
                ucm_context=ucm_context,
                context_target_latent_indices=context_target_latent_indices,
                context_slot_attention_weights=context_slot_attention_weights,
            )
        condition_ms = (time.perf_counter() - condition_started) * 1000.0

        # ``retrieve_timesteps`` configured the first block with its n_tokens
        # scheduler argument. Offline AR resets only continuation blocks.
        if start_idx > 0:
            self.core.scheduler.set_timesteps(
                self._num_inference_steps, device=self.device
            )
        model_started = time.perf_counter()
        block = self.core.denoise_ar_chunk(
            latents=self._latents,
            cond_latents=self._cond_latents,
            timesteps=self._timesteps,
            task_type="i2v",
            viewmats=viewmats,
            Ks=intrinsics,
            action=action,
            start_idx=start_idx,
            chunk_latent_frames=self.latent_frames_per_chunk,
            selected_frame_indices=selected,
            device=self.device,
            show_progress=False,
            use_ucm_attention_mask=ucm_context is not None,
            query_viewmats_override=query_viewmats_override,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter() - model_started) * 1000.0

        if self.rank != 0:
            return None
        decode_started = time.perf_counter()
        if getattr(self.core.vae.config, "shift_factor", None):
            decode_latents = (
                block / self.core.vae.config.scaling_factor
                + self.core.vae.config.shift_factor
            )
        else:
            decode_latents = block / self.core.vae.config.scaling_factor
        with (
            torch.autocast(
                device_type="cuda",
                dtype=self.core.vae_dtype,
                enabled=self.core.vae_autocast_enabled,
            ),
            auto_offload_model(
                self.core.vae,
                self.core.execution_device,
                enabled=self.core.enable_offloading,
            ),
        ):
            video = self.core.vae.decode_stream(decode_latents, return_dict=False)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        frames = (
            (video / 2 + 0.5)
            .clamp_(0, 1)
            .mul_(255)
            .to(device="cpu", dtype=torch.uint8)
            .permute(0, 2, 3, 4, 1)
            .numpy()[0]
        )
        if self._lingbot_runtime is not None:
            latent_images = [Image.fromarray(frame) for frame in frames[::4]]
            self._lingbot_runtime.process_images(latent_images)
            self._refresh_lingbot_runtime()
        context_selected = [
            index for index in selected if index < self._context_latent_count
        ]
        generated_selected = [
            index for index in selected if index >= self._context_latent_count
        ]
        memory_selected = sorted(
            set(selection["anchor_memory_latent_indices"])
            | set(selection["retrieved_memory_latent_indices"])
        )
        context_memory_selected = [
            index for index in memory_selected if index < self._context_latent_count
        ]
        retrieved_context_selected = [
            index
            for index in selection["retrieved_memory_latent_indices"]
            if index < self._context_latent_count
        ]

        def source_frame_ids(indices: Sequence[int]) -> list[int]:
            return [
                self._context_pose_source_frame_ids[index]
                for index in indices
                if 0 <= index < len(self._context_pose_source_frame_ids)
            ]

        return {
            "video": np.ascontiguousarray(frames),
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                "condition_ms": condition_ms,
                "model_ms": model_ms,
                "decode_ms": decode_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "context_latent_frames": len(selected),
                "generated_latent_frames": self.generated_latent_frames,
                "history_cursor_before": start_idx,
                "history_cursor_after": self.generated_latent_frames,
                "world_size": self.world_size,
                **selection,
                "selected_context_latent_indices": context_selected,
                "selected_generated_latent_indices": generated_selected,
                "selected_context_pose_source_frame_ids": source_frame_ids(
                    context_selected
                ),
                "memory_bank_context_latent_indices": context_memory_selected,
                "memory_bank_context_pose_source_frame_ids": source_frame_ids(
                    context_memory_selected
                ),
                "retrieved_context_memory_latent_indices": retrieved_context_selected,
                "retrieved_context_memory_pose_source_frame_ids": source_frame_ids(
                    retrieved_context_selected
                ),
                "memory_bank_read": bool(context_memory_selected),
                "retrieved_context_memory_bank_hit": bool(retrieved_context_selected),
                "context_latent_count": self._context_latent_count,
                "point_cloud_retrieval": selection.get(
                    "point_cloud_retrieval", {"status": "disabled"}
                ),
                "trajectory_mode": "prescribed_pose"
                if camera_trajectory_ue
                else "canonical_action",
                "camera_condition": dict(self._last_camera_condition_metrics),
                **getattr(self.core, "_last_ar_vision_cache_metrics", {}),
            },
        }

    @staticmethod
    def _load_context_pixels(
        rgb_paths: Sequence[str | Path],
        *,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Load GT RGB into the VAE's [B,C,T,H,W] layout without MP4 roundtrips."""

        if not rgb_paths:
            raise ValueError("HY Context prefill requires at least one RGB frame.")
        frames: list[np.ndarray] = []
        for raw_path in rgb_paths:
            with Image.open(raw_path) as source:
                image = source.convert("RGB")
                scale = max(width / image.width, height / image.height)
                resized = image.resize(
                    (
                        max(width, int(round(image.width * scale))),
                        max(height, int(round(image.height * scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
                left = max((resized.width - width) // 2, 0)
                top = max((resized.height - height) // 2, 0)
                frames.append(
                    np.asarray(
                        resized.crop((left, top, left + width, top + height)),
                        dtype=np.float32,
                    )
                )
        pixels = torch.from_numpy(np.stack(frames, axis=0)).permute(3, 0, 1, 2)
        return (
            (pixels / 127.5 - 1.0).unsqueeze(0).to(device=device, dtype=torch.float32)
        )

    @torch.inference_mode()
    def prefill_context(
        self,
        *,
        rgb_paths: Sequence[str | Path],
        camera_to_world: Sequence[Sequence[Sequence[float]]],
        frame_ids: Sequence[int],
        fps: float = 24.0,
        **_: Any,
    ) -> dict[str, Any]:
        """Encode GT Context into HY latent history before the first Eval step."""

        if not self.configured:
            raise RuntimeError("Configure HY-WorldPlay before Context prefill.")
        if len(rgb_paths) != len(frame_ids) or len(camera_to_world) != len(frame_ids):
            raise ValueError("HY Context RGB, pose, and frame-id lengths must match.")
        if abs(float(fps) - self.fps) > 1e-3:
            raise ValueError(f"HY-WorldPlay expects {self.fps} FPS, got {fps}.")
        before = self.generated_latent_frames
        pixels = self._load_context_pixels(
            rgb_paths,
            height=self._height,
            width=self._width,
            device=self.device,
        )
        # HY's causal VAE encodes one seed frame followed by groups of four
        # frames. Repeat the tail so the final source frames are represented
        # instead of being silently dropped by the encoder's floor slicing.
        pixel_padding = (1 - int(pixels.shape[2])) % 4
        if pixel_padding:
            pixels = torch.cat(
                (
                    pixels,
                    pixels[:, :, -1:].expand(-1, -1, pixel_padding, -1, -1),
                ),
                dim=2,
            )
        with auto_offload_model(
            self.core.vae,
            self.core.execution_device,
            enabled=self.core.enable_offloading,
        ):
            encoded = self.core.vae.encode(pixels).latent_dist.mode()
            encoded = encoded * self.core.vae.config.scaling_factor
        encoded = encoded.to(device=self.device, dtype=self.dtype).contiguous()
        latent_count = int(encoded.shape[2])
        self._context_latent_count = latent_count
        self._context_pose_source_frame_ids = tuple(
            int(frame_ids[min(index * 4, len(frame_ids) - 1)])
            for index in range(latent_count)
        )
        self._latents = encoded
        self._cond_latents = torch.zeros(
            (1, encoded.shape[1] + 1, latent_count, encoded.shape[3], encoded.shape[4]),
            device=self.device,
            dtype=torch.float32,
        )
        self._image_condition = encoded[:, :, :1].contiguous()
        # A native rollout conditions its first latent on the seed image and
        # marks that slot in the I2V mask. Preserve the same condition history
        # when the observed Context is teacher-forced into the latent bank.
        self._cond_latents[:, : encoded.shape[1], :1] = self._image_condition
        self._cond_latents[:, -1:, :1] = 1.0
        self._motions, self._context_motion_diagnostics = (
            unreal_camera_poses_to_worldplay_motions(
                camera_to_world,
                latent_count=latent_count,
                temporal_stride=4,
                translation_step=float(self.operator.forward_speed),
                return_diagnostics=True,
            )
        )
        if self._lingbot_runtime is not None:
            latent_paths = [
                rgb_paths[min(index * 4, len(rgb_paths) - 1)]
                for index in range(latent_count)
            ]
            self._lingbot_runtime.process_paths(latent_paths)
            self._refresh_lingbot_runtime()
        if self.rank == 0:
            self.core.vae.reset_stream_decode()
            self.core.vae.begin_stream_decode()
            # Prime the causal decoder with GT Context; discard decoded pixels
            # but retain its resident feature-map state for Eval continuation.
            if getattr(self.core.vae.config, "shift_factor", None):
                decode_context = (
                    encoded / self.core.vae.config.scaling_factor
                    + self.core.vae.config.shift_factor
                )
            else:
                decode_context = encoded / self.core.vae.config.scaling_factor
            with (
                torch.autocast(
                    device_type="cuda",
                    dtype=self.core.vae_dtype,
                    enabled=self.core.vae_autocast_enabled,
                ),
                auto_offload_model(
                    self.core.vae,
                    self.core.execution_device,
                    enabled=self.core.enable_offloading,
                ),
            ):
                for latent_chunk in decode_context.split(
                    self.latent_frames_per_chunk, dim=2
                ):
                    self.core.vae.decode_stream(latent_chunk, return_dict=False)
        warnings = []
        if pixel_padding:
            warnings.append(f"padded_context_pixel_frames={pixel_padding}")
        return {
            "status": "partial" if warnings else "accepted",
            "accepted_frame_ids": [int(item) for item in frame_ids],
            "rejected_frame_ids": [],
            "native_latent_count": latent_count,
            "native_chunk_count": (latent_count + self.latent_frames_per_chunk - 1)
            // self.latent_frames_per_chunk,
            "history_cursor_before": before,
            "history_cursor_after": latent_count,
            "state_digest_before": f"hy:{before}",
            "state_digest_after": f"hy:{latent_count}:{tuple(encoded.shape)}",
            "last_context_pose": {"camera_to_world": camera_to_world[-1]},
            "boundary_mapping": {
                "dataset_context_frames": [int(frame_ids[0]), int(frame_ids[-1]) + 1],
                "native_latent_frames": [0, latent_count],
                "eval_starts_from_native_latent": latent_count,
                "padded_context_pixel_frames": pixel_padding,
                "native_latent_pose_source_frame_ids": list(
                    self._context_pose_source_frame_ids
                ),
                "context_motion_conversion": dict(self._context_motion_diagnostics),
            },
            "warnings": warnings,
        }

    def reset(self) -> None:
        """Drop rollout state while keeping all weights resident."""

        vae = getattr(self.core, "vae", None)
        if self.rank == 0 and vae is not None and hasattr(vae, "reset_stream_decode"):
            vae.reset_stream_decode()
        if self._lingbot_runtime is not None:
            self._lingbot_runtime.close()
            self._lingbot_runtime = None
        self.configured = False
        self.configure_metrics = {}
        self._generator = None
        self._timesteps = None
        self._latents = None
        self._cond_latents = None
        self._image_condition = None
        self._motions = []
        self._height = self._width = 0
        self._latent_height = self._latent_width = 0
        self._num_inference_steps = 0
        self._context_latent_count = 0
        self._context_pose_source_frame_ids = ()
        self._context_motion_diagnostics = {}
        self._point_cloud_runtime = None
        self._point_cloud_config = None
        self._last_prescribed_ue_pose = None
        self._rotation_conditioning_extra_yaw = 0.0
        if hasattr(self.core, "configure_point_cloud_retrieval"):
            self.core.configure_point_cloud_retrieval(None)
        for name in ("_kv_cache", "_kv_cache_neg"):
            if hasattr(self.core, name):
                setattr(self.core, name, None)
        if hasattr(self.core, "_last_ar_vision_cache_metrics"):
            self.core._last_ar_vision_cache_metrics = {}


__all__ = ["HunyuanWorldPlayRealtimeSession"]
