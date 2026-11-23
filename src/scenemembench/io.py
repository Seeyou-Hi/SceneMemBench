"""Portable media IO. Video readers retain only explicitly requested frames."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, allow_nan=False) + "\n")


def coerce_video_frames(video):
    if hasattr(video, "detach"):
        video = video.detach().cpu().numpy()
    a = np.asarray(video)
    if a.ndim == 5 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4:
        raise ValueError(f"Expected video with four axes, got {a.shape}")
    if a.shape[-1] not in (1, 3, 4):
        if a.shape[0] in (1, 3, 4):
            a = a.transpose(1, 2, 3, 0)
        elif a.shape[1] in (1, 3, 4):
            a = a.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Cannot identify RGB channels: {a.shape}")
    if np.issubdtype(a.dtype, np.floating):
        if not np.isfinite(a).all():
            raise ValueError("Non-finite video pixels")
        if a.min() < 0:
            a = (a + 1) / 2
        a = np.rint(np.clip(a, 0, 1) * 255)
    a = a.astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return a[..., :3]


def write_video(video, path, fps=24, quality=9, **kwargs):
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=quality, macro_block_size=1
    ) as writer:
        for frame in coerce_video_frames(video):
            writer.append_data(frame)
    return str(path)


def read_frames(path, indices=None):
    """Decode sequentially; return selected RGB frames (None means first and last)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    wanted = None if indices is None else set(indices)
    if wanted is not None and (not wanted or min(wanted) < 0):
        cap.release()
        raise ValueError("Frame indices must be nonempty and nonnegative")
    frames = {}
    count = 0
    last = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if wanted is None or count in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if wanted is None:
                    if count == 0:
                        frames[0] = rgb
                    last = rgb
                else:
                    frames[count] = rgb
            count += 1
    finally:
        cap.release()
    if count == 0:
        raise ValueError(f"Empty video: {path}")
    if wanted is None:
        frames[count - 1] = last
    elif wanted - frames.keys():
        raise ValueError(f"Missing frames: {sorted(wanted - frames.keys())}")
    return frames, {"frames": count, "fps": fps}
