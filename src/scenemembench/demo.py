"""Procedurally rendered, freely redistributable pipeline fixtures."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scenemembench.io import write_json, write_video, sha256
from scenemembench.metrics.evaluate import pose_matrix

REVISITS = {
    "reversed_rotation": "MIRROR_VIEW",
    "full_rotation": "ROTATION_360",
    "forward_look_back": "STRAIGHT_LOOP",
    "new_viewpoint": "VIEW_REVISIT",
}


def create(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Demo destination must be empty")
    for revisit, protocol in REVISITS.items():
        case = root / revisit
        (case / "rgb").mkdir(parents=True)
        (case / "pose").mkdir()
        canvas = Image.new("RGB", (256, 144), (150, 190, 230))
        d = ImageDraw.Draw(canvas)
        d.rectangle((0, 85, 256, 144), fill=(90, 130, 75))
        for i in range(12):
            x = i * 22
            d.rectangle(
                (x, 35 + (i % 3) * 8, x + 16, 100),
                fill=(40 + i * 15, 60 + i * 10, 160 - i * 6),
            )
            d.line((x, 35, x + 16, 100), fill="white", width=2)
        texture = np.asarray(canvas)
        rows = []
        frames = []
        for i in range(49):
            yaw = (
                i * 360 / 48
                if revisit == "full_rotation"
                else (i * 90 / 32 if i < 33 else 90 - (i - 32) * 90 / 16)
            )
            raw = [
                float(
                    i * 8 if revisit in ("forward_look_back", "new_viewpoint") else 0
                ),
                0,
                200,
                0,
                yaw,
                0,
            ]
            frame = np.roll(texture, i * 2, axis=1).copy()
            path = case / "rgb" / f"{i:06d}.png"
            Image.fromarray(frame).save(path)
            frames.append(frame)
            row = {
                "frame_id": i,
                "timestamp": i / 24,
                "rgb": f"rgb/{i:06d}.png",
                "primary_pose": raw,
                "action": {
                    "type": "rotate",
                    "axis": "yaw",
                    "signed_turn_deg": -5.625 if revisit != "full_rotation" else 7.5,
                },
            }
            row["camera_to_world"] = pose_matrix(row).tolist()
            rows.append(row)
        write_json(case / "pose/poses.json", rows)
        manifest = {
            "dataset": "scenemembench_synthetic_demo",
            "case_id": revisit,
            "protocol": protocol,
            "revisit": revisit,
            "synthetic": True,
            "description": "Synthetic IO/control fixture, not a physical UE trajectory or scientific benchmark sample.",
            "fps": 24,
            "resolution": [256, 144],
            "camera": {"horizontal_fov_deg": 90},
            "split": {"context": [0, 33], "eval": [33, 49]},
            "phase_ranges": {"context": [0, 33], "eval_turnaround": [33, 49]},
            "task": {"return_segment_in_eval": False},
            "prompt": "A stylized street with colorful geometric buildings.",
        }
        write_json(case / "manifest.json", manifest)
        write_video(frames[:33], case / "context.mp4")
        write_video(frames[33:], case / "eval.mp4")
    hashes = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    write_json(root / "checksums.json", hashes)
    return {"cases": list(REVISITS), "root": str(root), "synthetic": True}
