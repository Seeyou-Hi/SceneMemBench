#!/usr/bin/env python3
"""Prepare an official, pinned MG3.5 source tree plus the published native patch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--destination", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "locks/matrix-game-3.5.json").read_text())
patch = root / lock["patch"]
if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
    raise RuntimeError("Patch checksum mismatch")
if a.destination.exists():
    raise FileExistsError(
        "Destination must not exist; existing checkouts are never modified"
    )
subprocess.run(["git", "clone", lock["repository"], str(a.destination)], check=True)
subprocess.run(
    ["git", "-C", str(a.destination), "checkout", "--detach", lock["revision"]],
    check=True,
)
subprocess.run(
    ["git", "-C", str(a.destination), "apply", "--check", str(patch)], check=True
)
subprocess.run(["git", "-C", str(a.destination), "apply", str(patch)], check=True)
print("Prepared native source:", a.destination.resolve())
