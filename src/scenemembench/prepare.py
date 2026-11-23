"""Materialize Context images from a video-based mini, without reading Query RGB."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import cv2

from .io import sha256, write_json


def prepare(case_dir):
    root = Path(case_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("rgb_storage", {}).get("mode") != "decode_context_video":
        raise ValueError("Case is not a video-based mini; existing RGB is left unchanged")
    rows = json.loads((root / "pose/poses.json").read_text())
    start, end = manifest["split"]["context"]
    if start != 0 or not 0 < end <= len(rows):
        raise ValueError("Expected a zero-based Context covered by pose rows")
    for i, row in enumerate(rows[:end]):
        if row["frame_id"] != i or row["rgb"] != f"rgb/{i:06d}.png":
            raise ValueError("Unexpected Context frame ID or image path")
    identity = {"video_sha256": sha256(root / "context.mp4"), "frames": end,
                "manifest_sha256": sha256(root / "manifest.json")}
    marker = root / "context_prepared.json"
    destination = root / "rgb"
    if destination.exists():
        if (marker.exists() and json.loads(marker.read_text()) == identity
                and len(list(destination.glob("*.png"))) == end
                and all((root / r["rgb"]).is_file() for r in rows[:end])):
            return {"case_id": manifest["case_id"], "context_frames": end, "status": "already_prepared"}
        raise FileExistsError("Existing RGB does not match the preparation record; use a fresh extracted case")
    temporary = Path(tempfile.mkdtemp(prefix=".context-rgb-", dir=root))
    cap = cv2.VideoCapture(str(root / "context.mp4"))
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open context.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if abs(fps - manifest["fps"]) > 1e-3:
            raise ValueError("Context FPS differs from manifest")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count >= end or [frame.shape[1], frame.shape[0]] != manifest["resolution"]:
                raise ValueError("Context frame count or resolution differs from manifest")
            if not cv2.imwrite(str(temporary / f"{count:06d}.png"), frame,
                               [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError("Failed to save Context image")
            count += 1
        if count != end:
            raise ValueError(f"Expected {end} Context frames, decoded {count}")
        temporary.rename(destination)
        write_json(marker, identity)
    finally:
        cap.release()
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"case_id": manifest["case_id"], "context_frames": count, "status": "prepared"}
