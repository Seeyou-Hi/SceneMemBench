# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Native WorldFoundry driver for Matrix-Game 3.5 distilled inference.

The driver deliberately lives next to the MG3.5 code.  WorldFoundry imports it
only inside the model's dedicated Conda child, so the large Wan/DA3 dependency
stack never leaks into the Studio process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


_ACTION_ALIASES = {
    "back": "backward",
    "camera_left": "camera_l",
    "camera_right": "camera_r",
}
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

# UE camera coordinates are +X forward, +Y right, +Z up. MG3.5's bundled
# metric trajectories use +X right, +Y up, +Z forward. This proper rotation
# also maps UE yaw around +Z to MG3.5 yaw around +Y without changing its sign.
_UE_RELATIVE_TO_MG35_BASIS = np.asarray(
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    dtype=np.float32,
)

_MG35_NATIVE_SELECTION_CONTRACT = {
    "latent_window_size": 21,
    "chunk_size": 3,
    "context_chunks": 7,
    "context_pose_pool_size": 5,
    "candidates_per_query_group_val": 5,
    "memory_vae_encode_input_frames": 1,
}


def _dispatch_model_parallel_dit(model: Any) -> dict[str, Any] | None:
    """Shard the Wan DIT blocks across two visible GPUs in one process.

    ``torchrun`` is intentionally not used for this mode: two distributed
    ranks would each construct a complete model.  Explicit forward relays are
    used instead of relying on outer Accelerate hooks because MG3.5 invokes
    DIT blocks through gradient-checkpoint wrappers. This keeps hidden states,
    weights, and runtime KV tensors on GPU memory.
    """
    raw = (
        os.environ.get("SCENEMEMBENCH_MATRIX_GAME35_MODEL_PARALLEL", "").strip().lower()
    )
    if raw in {"", "0", "false", "off", "no"}:
        return None
    if raw not in {"2", "true", "on", "yes"}:
        raise ValueError(
            "SCENEMEMBENCH_MATRIX_GAME35_MODEL_PARALLEL must be unset/0 or 2"
        )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "MG3.5 model parallel mode requires two visible CUDA devices; "
            f"found {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    blocks = getattr(dit, "blocks", None)
    if dit is None or blocks is None or len(blocks) < 2:
        raise RuntimeError("could not find at least two Wan DIT blocks to shard")

    primary, secondary = "cuda:0", "cuda:1"
    split = (len(blocks) + 1) // 2

    def move_tree(value: Any, device: str) -> Any:
        if torch.is_tensor(value):
            return value if value.device == torch.device(device) else value.to(device)
        if isinstance(value, tuple):
            return tuple(move_tree(item, device) for item in value)
        if isinstance(value, list):
            # Preserve list identity: causal KV metadata owns mutable cache
            # dictionaries that the attention kernel fills in-place.
            for index, item in enumerate(value):
                value[index] = move_tree(item, device)
            return value
        if isinstance(value, dict):
            # Preserve dict identity for the same write-back cache contract.
            for key, item in list(value.items()):
                value[key] = move_tree(item, device)
            return value
        return value

    def relay(module: Any, device: str) -> None:
        module.to(device)
        original_forward = module.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            return original_forward(
                *move_tree(args, device),
                **move_tree(kwargs, device),
            )

        module.forward = forward

    # The model is initially resident on cuda:0. Move the second block group
    # and relay every block boundary explicitly; the final head must also
    # accept the output of a cuda:1 block.
    for index, block in enumerate(blocks):
        relay(block, primary if index < split else secondary)
    relay(dit.head, primary)
    return {
        "enabled": True,
        "devices": [primary, secondary],
        "dit_blocks": len(blocks),
        "split": split,
        "offload_buffers": False,
        "dispatch": "explicit_forward_relay",
    }


def _validate_native_selection_contract(config: Any, args: Any) -> None:
    """Reject WorldFoundry runs that alter released MG3.5 selection budgets."""
    observed = {
        "latent_window_size": int(config.latent_window_size),
        "chunk_size": int(config.chunk_size),
        "context_chunks": int(config.context_chunks),
        "context_pose_pool_size": int(config.context_pose_pool_size),
        "candidates_per_query_group_val": int(args.candidates_per_query_group_val),
        "memory_vae_encode_input_frames": int(args.memory_vae_encode_input_frames),
    }
    mismatches = {
        name: (observed[name], expected)
        for name, expected in _MG35_NATIVE_SELECTION_CONTRACT.items()
        if observed[name] != expected
    }
    if bool(getattr(args, "memory_neighbor_window_val", False)):
        mismatches["memory_neighbor_window_val"] = (True, False)
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(
            "WorldFoundry MG3.5 must preserve the released selection budgets: "
            + details
        )


def _parse_bool_option(value: Any, *, name: str, default: bool = False) -> bool:
    """Parse driver options without treating a non-empty string as truthy."""
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _resolve_memory_contract(options: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve native-compatible memory defaults and explicit ablation knobs.

    Native MG3.5 validation starts from the current clean anchor and publishes
    generated memory as the rollout advances.  Full Context registration and
    pre-C0 history are useful benchmark ablations, but must never be injected
    by the continuation facade implicitly.
    """
    full_context = _parse_bool_option(
        options.get(
            "full_context_memory",
            os.environ.get("SCENEMEMBENCH_MATRIX_GAME35_FULL_CONTEXT_MEMORY", "0"),
        ),
        name="full_context_memory",
    )
    context_as_generated_history = _parse_bool_option(
        options.get(
            "context_as_generated_history",
            os.environ.get(
                "SCENEMEMBENCH_MATRIX_GAME35_CONTEXT_AS_GENERATED_HISTORY", "0"
            ),
        ),
        name="context_as_generated_history",
    )
    bootstrap = (
        str(
            options.get(
                "memory_bootstrap_mode",
                os.environ.get(
                    "SCENEMEMBENCH_MATRIX_GAME35_MEMORY_BOOTSTRAP", "c0_only"
                ),
            )
        )
        .strip()
        .lower()
    )
    if bootstrap not in {"c0_only", "history"}:
        raise ValueError(
            f"memory_bootstrap_mode must be 'c0_only' or 'history', got {bootstrap!r}"
        )
    return {
        "full_context_memory": full_context,
        "context_as_generated_history": context_as_generated_history,
        "memory_bootstrap_mode": bootstrap,
    }


def _as_action_set(interactions: Sequence[str]) -> set[str]:
    return {
        _ACTION_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
        for value in interactions
        if str(value).strip()
    }


def _actions_for_frames(
    interactions: Sequence[str],
    control_segments: Sequence[Mapping[str, Any]] | None,
    *,
    count: int,
) -> list[set[str]]:
    fallback = _as_action_set(interactions)
    if not control_segments:
        return [fallback] * count
    rows: list[tuple[float, set[str]]] = []
    for segment in control_segments:
        duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
        controls = {
            _KEY_TO_CONTROL[key]
            for key in (str(item).strip().lower() for item in segment.get("keys", ()))
            if key in _KEY_TO_CONTROL
        }
        if duration > 0.0:
            rows.append((duration, controls))
    if not rows:
        return [fallback] * count
    total = sum(duration for duration, _ in rows)
    result: list[set[str]] = []
    row_index = 0
    boundary = rows[0][0]
    for frame_index in range(count):
        sample = (frame_index + 0.5) * total / count
        while row_index + 1 < len(rows) and sample > boundary:
            row_index += 1
            boundary += rows[row_index][0]
        result.append(rows[row_index][1])
    return result


def _camera_chunk(
    previous: np.ndarray,
    interactions: Sequence[str],
    *,
    frames: int,
    motion_scale: float,
    intrinsics: np.ndarray,
    rotation_degrees_per_second: float = 30.0,
    fps: float = 17.0,
    translation_meters_per_frame: float = 0.04,
    control_segments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Build a small metric C2W trajectory from keyboard/camera controls."""
    if frames <= 0:
        raise ValueError("camera chunk must contain at least one frame")
    pose = np.asarray(previous, dtype=np.float32).reshape(4, 4).copy()
    step = float(translation_meters_per_frame) * float(motion_scale)
    yaw = np.deg2rad(
        float(rotation_degrees_per_second) / max(float(fps), 1e-6) * float(motion_scale)
    )
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.asarray(
        ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float32
    )
    pitch = np.deg2rad(
        float(rotation_degrees_per_second) / max(float(fps), 1e-6) * float(motion_scale)
    )
    cp, sp = np.cos(pitch), np.sin(pitch)
    # MG3.5 local axes are +X right, +Y up, +Z forward. Positive rotation
    # around local X moves +Z toward -Y, i.e. looks down.
    pitch_rotation = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)), dtype=np.float32
    )
    frame_actions = _actions_for_frames(
        interactions,
        control_segments,
        count=max(int(frames) - 1, 0),
    )
    extrinsics = []
    for frame_index in range(int(frames)):
        # Keep the first generated RGB frame on the Context/Eval boundary;
        # apply one native action increment before each later RGB frame.
        if frame_index > 0:
            actions = frame_actions[frame_index - 1]
            direction = (
                1.0 if "camera_r" in actions else -1.0 if "camera_l" in actions else 0.0
            )
            pitch_direction = (
                1.0
                if "camera_down" in actions
                else -1.0
                if "camera_up" in actions
                else 0.0
            )
            # Movement controls are camera-local.  `pose` is C2W, so its
            # rotation maps MG3.5 local axes (+X right, +Y up, +Z forward)
            # into world coordinates.  Applying deltas to world X/Z makes a
            # rotated camera strafe when the requested action is forward.
            local_delta = np.zeros(3, dtype=np.float32)
            if "forward" in actions:
                local_delta[2] += step
            if "backward" in actions:
                local_delta[2] -= step
            if "left" in actions:
                local_delta[0] -= step
            if "right" in actions:
                local_delta[0] += step
            # Camera up/down are view rotations, not vertical translations.
            pose[:3, 3] += pose[:3, :3] @ local_delta
            if direction:
                pose[:3, :3] = pose[:3, :3] @ (
                    yaw_rotation if direction > 0 else yaw_rotation.T
                )
            if pitch_direction:
                pose[:3, :3] = pose[:3, :3] @ (
                    pitch_rotation if pitch_direction > 0 else pitch_rotation.T
                )
        extrinsics.append(pose.copy())
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.repeat(
        np.asarray(intrinsics, dtype=np.float32)[None], int(frames), axis=0
    )
    return {"intrinsics": intrinsics, "extrinsics": extrinsics, "last_pose": pose}


def _normalize_metric_c2w(
    camera_to_world: Sequence[np.ndarray],
    *,
    position_units: str = "m",
    reference_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Rebase C2W poses and convert declared UE centimetres to metres.

    MG3.5's camera contract is metric translation.  UnrealZoo stores absolute
    Unreal Engine positions in centimetres, so feeding those values directly
    would expose kilometre-scale coordinates to PRoPE.  Re-centering at the
    first Context camera also matches the local trajectory convention used by
    the bundled MG3.5 samples.
    """
    poses = np.asarray(camera_to_world, dtype=np.float32).reshape(-1, 4, 4)
    if len(poses) == 0:
        raise ValueError("camera trajectory must contain at least one pose")
    reference = (
        poses[0]
        if reference_pose is None
        else np.asarray(reference_pose, dtype=np.float32).reshape(4, 4)
    )
    first = np.linalg.inv(reference.astype(np.float64)).astype(np.float32)
    relative = np.einsum("ij,njk->nik", first, poses).astype(np.float32)
    units = str(position_units or "m").strip().lower()
    if units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        relative[:, :3, 3] /= 100.0
    elif units not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError(
            "MG3.5 requires metric camera translations; unsupported position "
            f"units: {position_units!r}"
        )
    basis = _UE_RELATIVE_TO_MG35_BASIS
    normalized = relative.copy()
    normalized[:, :3, 3] = np.einsum("ij,nj->ni", basis, relative[:, :3, 3])
    normalized[:, :3, :3] = np.einsum(
        "ij,njk,lk->nil", basis, relative[:, :3, :3], basis
    )
    return normalized


def _c2w_to_w2c(camera_to_world: Sequence[np.ndarray]) -> np.ndarray:
    """Convert driver-owned C2W poses to MG3.5 resident camera extrinsics."""
    poses = np.asarray(camera_to_world, dtype=np.float32).reshape(-1, 4, 4)
    return np.linalg.inv(poses.astype(np.float64)).astype(np.float32)


class MatrixGame35Driver:
    """One model-owned, resumable distilled rollout."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        options: Mapping[str, Any] | None = None,
    ):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.device = str(device)
        self.options = dict(options or {})
        self._configured = False
        self._prefilled = False
        self._resident = None
        self._accelerator = None
        self._model_parallel_info: dict[str, Any] | None = None
        self._workspace: Path | None = None
        self._last_pose = np.eye(4, dtype=np.float32)
        self._intrinsics = np.eye(3, dtype=np.float32)
        self._step_id = 0
        self._history_cursor = 0
        self._prompt = ""
        self._seed = 42
        self._memory_contract = _resolve_memory_contract(self.options)
        self._rotation_degrees_per_second = float(
            os.environ.get(
                "SCENEMEMBENCH_MATRIX_GAME35_ROTATION_DEGREES_PER_SECOND", "30"
            )
        )
        self._translation_meters_per_frame = float(
            os.environ.get(
                "SCENEMEMBENCH_MATRIX_GAME35_TRANSLATION_METERS_PER_FRAME", "0.04"
            )
        )

    def _config_path(self) -> Path:
        return Path(
            self.options.get("config")
            or (Path(__file__).parent / "configs/infer_distilled.yaml")
        )

    def configure(
        self, image: Image.Image, *, prompt: str, seed: int, **kwargs: Any
    ) -> Mapping[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("MG3.5 driver.configure requires a PIL image")
        self.reset()
        self._image = image.convert("RGB")
        self._prompt = str(prompt or "")
        if not self._prompt.strip():
            raise ValueError(
                "MG3.5 distilled resident inference requires a non-empty prompt"
            )
        self._seed = int(seed)
        self._configured = True
        digest = hashlib.sha256(self._image.tobytes()).hexdigest()[:16]
        return {"state_digest": f"configured:{digest}:{self._seed}"}

    @staticmethod
    def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
        import cv2

        if not frames:
            raise ValueError("MG3.5 Context requires at least one RGB frame")
        first = np.asarray(frames[0])
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open Context video writer: {path}")
        try:
            for frame in frames:
                arr = np.asarray(frame).astype(np.uint8, copy=False)
                writer.write(arr[..., ::-1])
        finally:
            writer.release()

    def _build_workspace(
        self,
        rgb_paths,
        camera_to_world,
        frame_ids,
        fps,
        args,
        *,
        position_units="m",
        horizontal_fov_deg=90.0,
    ):
        root = Path(
            tempfile.mkdtemp(prefix="mg35_wf_", dir=self.options.get("work_dir"))
        )
        scene = root / "workspace" / "scenes" / "case"
        for name in ("rgb", "pose", "intrinsics", "depth"):
            (scene / name).mkdir(parents=True, exist_ok=True)
        frames = []
        for raw in rgb_paths:
            with Image.open(raw) as source:
                frames.append(
                    np.asarray(source.convert("RGB").resize((args.width, args.height)))
                )
        # The dataset requires one full 84-frame Eval window. Repeat the final
        # Context frame only as a camera/data carrier; generated pixels are not
        # consumed as GT memory by the resident rollout.
        required = len(frames) + 85
        frames.extend([frames[-1].copy()] * max(0, required - len(frames)))
        self._write_video(scene / "rgb" / "video.mp4", frames, fps)
        poses = _normalize_metric_c2w(camera_to_world, position_units=position_units)
        poses = np.concatenate(
            (poses, np.repeat(poses[-1:], len(frames) - len(poses), axis=0)), axis=0
        )
        np.savez_compressed(scene / "pose" / "video.npz", data=poses)
        height, width = int(args.height), int(args.width)
        hfov = np.deg2rad(float(horizontal_fov_deg))
        if not np.isfinite(hfov) or not 0.0 < hfov < np.pi:
            raise ValueError(
                f"horizontal_fov_deg must lie in (0, 180), got {horizontal_fov_deg!r}"
            )
        fx = fy = float(width / (2.0 * np.tan(hfov / 2.0)))
        K = np.asarray(
            ((fx, 0.0, width / 2.0), (0.0, fy, height / 2.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        np.savez_compressed(
            scene / "intrinsics" / "video.npz",
            data=np.repeat(K[None], len(frames), axis=0),
        )
        # UnrealZoo memory contains RGB/pose only.  MG3.5's dataset scanner
        # requires a parseable depth artifact, but validation never consumes
        # these values: FrustumHandler runs DA3 on the RGB frames at runtime.
        placeholder_depth = np.full((height, width), 5.0, np.float32)
        np.savez_compressed(
            scene / "depth" / "video.npz",
            **{
                f"frame_{index:05d}": placeholder_depth
                for index in range(len(frame_ids))
            },
        )
        (scene / "depth" / "metadata.json").write_text(
            json.dumps({"format": "npz_float"}), encoding="utf-8"
        )
        caption = root / "workspace" / "caption" / "case"
        caption.mkdir(parents=True, exist_ok=True)
        (caption / "video.json").write_text(
            json.dumps(
                {"detailed": {"0": {"start": "", "dynamic": self._prompt}}},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        return root, frames

    def _ensure_model(self, args, workspace: Path):
        da3_model = self.options.get("da3_model")
        if da3_model:
            da3_path = Path(da3_model).expanduser().resolve()
            if not (da3_path / "config.json").is_file():
                raise FileNotFoundError(f"MG3.5 DA3 checkpoint is missing: {da3_path}")
            os.environ["DA3_MODEL_PATH"] = str(da3_path)
        import accelerate
        from distilled_config import load_inference_config
        from examples.wanvideo.pipeline.mosaic.causal_inference import (
            DistilledResidentInference,
        )
        from examples.wanvideo.pipeline.mosaic.causal_config import build_runtime_args
        from examples.wanvideo.pipeline.mosaic.datasets import (
            build_mosaic_validation_dataset,
        )
        from tools.run_distilled_inference import build_model
        import infer

        config = load_inference_config(self._config_path())
        wan_dir = Path(
            self.options.get("wan_dir") or os.environ.get("WAN22_TI2V_5B_DIR", "")
        )
        tokenizer_dir = Path(
            self.options.get("tokenizer_dir")
            or os.environ.get("UMT5_TOKENIZER_DIR", "")
        )
        if not self.model_path or not self.model_path.is_file():
            raise FileNotFoundError(
                f"MG3.5 distilled checkpoint is missing: {self.model_path}"
            )
        if not wan_dir.is_dir() or not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                "set WAN22_TI2V_5B_DIR and UMT5_TOKENIZER_DIR for MG3.5 native driver"
            )
        index = infer.build_index(
            str(workspace / "workspace"), os.sys.executable, os.environ.copy()
        )
        args = build_runtime_args(
            config,
            checkpoint=self.model_path,
            dataset_index=index,
            workspace=workspace / "workspace",
            wan_dir=wan_dir,
            tokenizer_dir=tokenizer_dir,
            memory_cache_dir=workspace / "memory_latents",
        )
        args.validation_min_anchor_frame_idx = max(0, int(self._context_count) - 1)
        args.num_validation_blocks = int(
            self.options.get(
                "num_blocks",
                os.environ.get("SCENEMEMBENCH_MATRIX_GAME35_NUM_BLOCKS", 64),
            )
        )
        _validate_native_selection_contract(config, args)
        args.causal_memory_visualization_path = str(
            self.options.get("memory_visualization")
            or os.environ.get("SCENEMEMBENCH_MATRIX_GAME35_MEMORY_VISUALIZATION", "")
        )
        memory_contract = _resolve_memory_contract(self.options)
        # Keep the released native contract by default: only the current clean
        # anchor is registered before generation, then generated chunks are
        # published through the normal write-back path. Full Context and
        # pre-C0 history remain explicit benchmark ablations.
        args.causal_validation_full_context_memory = memory_contract[
            "full_context_memory"
        ]
        args.causal_validation_context_as_generated_history = memory_contract[
            "context_as_generated_history"
        ]
        # Calibrate DA3 metric depth to the dataset W2C translation scale when
        # the registered sequence contains enough camera motion. Pure-rotation
        # generated chunks reuse the scale established by the full Context.
        args.causal_dmd_self_memory_da3_align_input_pose_scale = True
        args.causal_memory_bootstrap_mode = memory_contract["memory_bootstrap_mode"]
        args.causal_validation_memory_bootstrap_mode = memory_contract[
            "memory_bootstrap_mode"
        ]
        self._memory_contract = memory_contract
        args.causal_memory_context_selection_policy = str(
            self.options.get("memory_context_selection_policy", "legacy")
        )
        args.causal_dynamic_context_selection_policy = str(
            self.options.get("dynamic_context_selection_policy", "oldest")
        )
        args.rank = 0
        accelerator = accelerate.Accelerator()
        dataset = build_mosaic_validation_dataset(args, train_dataset=None)
        # G is the first Eval RGB frame, immediately after Context frame C-1.
        dataset.SFgenerating_first_idx = int(self._context_count)
        model = build_model(args, accelerator)
        self._model_parallel_info = _dispatch_model_parallel_dit(model)
        self._accelerator, self._args, self._dataset, self._model = (
            accelerator,
            args,
            dataset,
            model,
        )
        self._resident = DistilledResidentInference(
            accelerator, dataset, model, args, interactive_camera=True
        )

    def prefill_context(
        self,
        *,
        rgb_paths,
        camera_to_world,
        frame_ids,
        fps=17.0,
        position_units="m",
        horizontal_fov_deg=90.0,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if not self._configured:
            raise RuntimeError("configure must run before Context prefill")
        if len(rgb_paths) != len(frame_ids) or len(camera_to_world) != len(frame_ids):
            raise ValueError("Context RGB, camera, and frame-id lengths must match")
        if len(frame_ids) < 10:
            raise ValueError("MG3.5 distilled Context requires at least 10 frames")
        self._context_count = len(frame_ids)
        self._context_reference_pose = (
            np.asarray(camera_to_world[0], dtype=np.float32).reshape(4, 4).copy()
        )
        metric_poses = _normalize_metric_c2w(
            camera_to_world,
            position_units=position_units,
            reference_pose=self._context_reference_pose,
        )
        self._last_pose = metric_poses[-1].copy()
        self._workspace, frames = self._build_workspace(
            rgb_paths,
            camera_to_world,
            frame_ids,
            fps,
            self._args_for_workspace(),
            position_units=position_units,
            horizontal_fov_deg=horizontal_fov_deg,
        )
        workspace_args = self._args_for_workspace()
        focal = float(
            workspace_args.width
            / (2.0 * np.tan(np.deg2rad(float(horizontal_fov_deg)) / 2.0))
        )
        self._intrinsics = np.asarray(
            (
                (focal, 0.0, workspace_args.width / 2.0),
                (0.0, focal, workspace_args.height / 2.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        self._ensure_model(workspace_args, self._workspace)
        self._prefilled = True
        self._history_cursor = len(frame_ids)
        causal_context_latent_count = max(0, (int(self._context_count) - 1 + 3) // 4)
        accepted_context_chunk_count = (
            causal_context_latent_count
            + _MG35_NATIVE_SELECTION_CONTRACT["chunk_size"]
            - 1
        ) // _MG35_NATIVE_SELECTION_CONTRACT["chunk_size"]
        padded_causal_context_latent_count = (
            accepted_context_chunk_count * _MG35_NATIVE_SELECTION_CONTRACT["chunk_size"]
        )
        context_as_generated_history = self._memory_contract[
            "context_as_generated_history"
        ]
        registered_context_latent_count = (
            int(self._context_count)
            if self._memory_contract["full_context_memory"]
            or context_as_generated_history
            else 1
        )
        registered_context_chunk_count = (
            accepted_context_chunk_count if context_as_generated_history else 1
        )
        native_selection_contract = {
            **_MG35_NATIVE_SELECTION_CONTRACT,
            "mosaic_query_latents_per_chunk": 3,
            "mosaic_memory_rgb_frames_per_latent": 1,
            "dynamic_context_rgb_frames_per_latent": 5,
            "dynamic_context_selected_per_chunk": 1,
            "rolling_kv_max_latent_frames": 19,
        }
        return {
            "status": "accepted",
            "accepted_frame_ids": list(map(int, frame_ids)),
            "native_latent_count": int(1 + padded_causal_context_latent_count),
            "native_chunk_count": accepted_context_chunk_count,
            "history_cursor_before": 0,
            "history_cursor_after": self._history_cursor,
            "model_parallel": self._model_parallel_info,
            "last_context_pose": {
                "camera_to_world": metric_poses[-1].tolist(),
                "position_units": "m",
                "rebased_at_context_frame": int(frame_ids[0]),
            },
            "boundary_mapping": {
                "context_rgb_frames": [
                    int(frame_ids[0]),
                    int(frame_ids[-1]) + 1,
                ],
                "memory_bank": (
                    "teacher_forced_context_then_generated_writeback"
                    if context_as_generated_history
                    else "all_context_frames_registered_then_frustum_selected"
                    if getattr(self, "_memory_contract", {}).get(
                        "full_context_memory", False
                    )
                    else "current_context_anchor_then_generated_writeback"
                ),
                "rolling_kv": (
                    "teacher_forced_context_then_recent_generated_chunks"
                    if context_as_generated_history
                    else "native_boundary_anchor_plus_recent_generated_chunks"
                ),
                "accepted_context_latent_count": int(1 + causal_context_latent_count),
                "padded_context_latent_count": int(
                    1 + padded_causal_context_latent_count
                ),
                "accepted_context_chunk_count": accepted_context_chunk_count,
                "context_memory_latent_count": registered_context_latent_count,
                "context_memory_chunk_count": registered_context_chunk_count,
                "full_context_memory": self._memory_contract["full_context_memory"],
                "memory_bootstrap_mode": self._memory_contract["memory_bootstrap_mode"],
                "context_as_generated_history": self._memory_contract[
                    "context_as_generated_history"
                ],
                "source_position_units": str(position_units),
                "pose_translation": "UE cm converted to metres and rebased at first Context pose",
                "pose_axes": "UE camera (+X forward,+Y right,+Z up) converted to MG3.5 (+X right,+Y up,+Z forward)",
                "resident_extrinsics": "W2C (interactive updates match dataset _load_extrinsics)",
                "da3_depth_scale_alignment": "aligned to the metric Context input trajectory and reused for degenerate chunks",
                "native_selection_contract": native_selection_contract,
            },
            "warnings": (
                [
                    "Full Context expands the Patch/Mosaic candidate bank; native "
                    "per-query and rolling-KV selection budgets are unchanged."
                ]
                if getattr(self, "_memory_contract", {}).get(
                    "full_context_memory", False
                )
                else [
                    "GT Context was teacher-forced through native clean cache-fill; "
                    "this approximates generated causal state but does not reproduce "
                    "the model's denoised historical latents."
                ]
                if context_as_generated_history
                else []
            ),
            "state_digest_after": f"prefill:{self._history_cursor}",
        }

    def _args_for_workspace(self):
        from distilled_config import load_inference_config

        config = load_inference_config(self._config_path())
        return SimpleNamespace(height=config.height, width=config.width)

    def generate(
        self,
        *,
        interactions,
        control_segments=None,
        motion_scale=1.0,
        step_id=0,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if not self._prefilled or self._resident is None:
            raise RuntimeError("Context prefill is required before generation")
        prescribed = kwargs.get("camera_trajectory_ue")
        if prescribed:
            metric_poses = _normalize_metric_c2w(
                prescribed,
                position_units=str(kwargs.get("position_units", "cm")),
                reference_pose=self._context_reference_pose,
            )
            if len(metric_poses) != 12:
                raise ValueError(
                    "MG3.5 prescribed camera trajectory must contain exactly "
                    f"12 frames, got {len(metric_poses)}"
                )
            chunk = {
                "intrinsics": np.repeat(self._intrinsics[None], 12, axis=0),
                "extrinsics": metric_poses,
                "last_pose": metric_poses[-1].copy(),
            }
        else:
            chunk = _camera_chunk(
                self._last_pose,
                interactions,
                frames=12,
                motion_scale=motion_scale,
                intrinsics=self._intrinsics,
                rotation_degrees_per_second=self._rotation_degrees_per_second,
                translation_meters_per_frame=self._translation_meters_per_frame,
                control_segments=control_segments,
            )
        self._last_pose = chunk.pop("last_pose")
        # The on-disk dataset accepts C2W and inverts it in _load_extrinsics.
        # Interactive updates bypass that loader and therefore must already be
        # W2C before replacing the resident rollout's full_noisy_extr tensor.
        chunk["extrinsics"] = _c2w_to_w2c(chunk["extrinsics"])
        result = self._resident.step(camera_update=chunk)
        video = result["video"]
        generated = int(result["generated_frame_count"])
        # The native rollout cursor is latent-space (one step = three
        # generated latent chunks). WorldFoundry's public cursor is RGB frame
        # space, so keep that boundary conversion in the driver.
        self._history_cursor += generated
        memory_selection = dict(result.get("memory_selection", {}))
        memory_selection["native_selection_contract"] = {
            **_MG35_NATIVE_SELECTION_CONTRACT,
            "mosaic_query_latents_per_chunk": 3,
            "mosaic_memory_rgb_frames_per_latent": 1,
            "dynamic_context_rgb_frames_per_latent": 5,
            "dynamic_context_selected_per_chunk": 1,
            "rolling_kv_max_latent_frames": 19,
        }
        memory_selection["memory_contract"] = dict(self._memory_contract)
        return {
            "video": video,
            "generated_frame_count": generated,
            "history_cursor_after": self._history_cursor,
            "model_parallel": self._model_parallel_info,
            "memory_selection": memory_selection,
            "state_digest": result.get(
                "state_digest", f"step:{step_id}:{self._history_cursor}"
            ),
        }

    def reset(self) -> None:
        if self._resident is not None:
            self._resident.close()
        self._resident = None
        self._accelerator = None
        self._model_parallel_info = None
        self._prefilled = False
        self._configured = False
        self._step_id = 0
        self._history_cursor = 0
        self._memory_contract = _resolve_memory_contract(self.options)
        if self._workspace is not None:
            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace = None


def create_driver(
    model_path: str = "", device: str = "cuda", options: Mapping[str, Any] | None = None
) -> MatrixGame35Driver:
    return MatrixGame35Driver(model_path=model_path, device=device, options=options)


__all__ = ["MatrixGame35Driver", "create_driver", "_camera_chunk"]
