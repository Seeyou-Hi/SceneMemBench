"""SceneMemBench standalone command line."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import yaml


def run(args):
    """Initialize LingBot's native collectives when launched with torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return _run(args)
    if args.model != "lingbot-world-v2":
        raise ValueError("The multi-process CLI currently supports LingBot v2 only")
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", init_method="env://")
    try:
        return _run(args, rank=dist.get_rank(), distributed=dist)
    finally:
        if owns_group:
            dist.destroy_process_group()


def _run(args, *, rank=0, distributed=None):
    from .continuation.dataset import load_context_bundle
    from .continuation.controller.open_loop import iter_open_loop_native_actions
    from .continuation.runner import run_context_continuation
    from .models.registry import load
    from .io import write_json, sha256

    context = load_context_bundle(args.case)
    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    config = config or {}
    out = Path(args.output)
    output_occupied = [out.exists() and any(out.iterdir()) if rank == 0 else None]
    if distributed is not None:
        distributed.broadcast_object_list(output_occupied, src=0)
    if output_occupied[0]:
        raise FileExistsError("Use an empty output directory")
    rows = json.loads((Path(args.case) / "pose/poses.json").read_text())
    # Pose/action rows only; query RGB paths are removed before control construction.
    controls = [
        {k: v for k, v in row.items() if k not in ("rgb", "files")} for row in rows
    ]
    from .continuation.actions import (
        _infer_eval_rotation_speed,
        _set_rotation_environment,
        _configure_pipeline_rotation,
        _dataset_eval_actions,
    )

    speed = config.get("rotation_degrees_per_second")
    if speed is None:
        speed = _infer_eval_rotation_speed(Path(args.case), context)
    _set_rotation_environment(args.model, speed)
    adapter = load(args.model, config, args.device)
    pipeline = getattr(adapter, "pipeline", None)
    if pipeline is None:
        actions = iter_open_loop_native_actions(
            context,
            controls,
            native_fps=adapter.capabilities().native_fps,
            next_output_frames=lambda: 4,
        )
    else:
        _configure_pipeline_rotation(args.model, pipeline, speed)
        actions = _dataset_eval_actions(
            Path(args.case),
            context,
            pipeline,
            adapter.capabilities(),
            sana_eval_chunks=config.get("sana_eval_chunks", 0),
            sana_eval_idle_frames=config.get("sana_eval_idle_frames", 0),
        )
    prompt = (
        args.prompt
        if args.prompt is not None
        else context.metadata.get("prompt", context.metadata.get("scene_prompt", ""))
    )
    result = run_context_continuation(
        adapter,
        context,
        actions,
        prompt=prompt,
        seed=args.seed,
        output_dir=out if rank == 0 else None,
        full_continuation=config.get(
            "full_continuation", args.model == "zing-world-0.5"
        ),
    )
    if rank == 0:
        write_json(
            out / "invocation.json",
            {
                "model": args.model,
                "device": args.device,
                "seed": args.seed,
                "config": config,
                "manifest_sha256": sha256(Path(args.case) / "manifest.json"),
            },
        )
    return {
        "model": args.model,
        "output": str(out),
        "continuation_verified": result["continuation_verified"],
        "frame_count": result["continuation_frame_count"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="scenemembench")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("demo")
    q.add_argument("--output", required=True)
    q = sub.add_parser("validate")
    q.add_argument("--case", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--case", required=True)
    q = sub.add_parser("run")
    q.add_argument("--case", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--config")
    q.add_argument("--device", default="cuda")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--prompt")
    q = sub.add_parser("evaluate")
    q.add_argument("--case", required=True)
    q.add_argument("--prediction", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--task", choices=["seen", "novel"], required=True)
    q.add_argument("--model-id", default="generic")
    q.add_argument(
        "--max-rotation-deg", type=float, default=None,
        help="Optional endpoint angular bound for Seen search",
    )
    q.add_argument("--metrics", choices=["pixel", "all"], default="pixel")
    q.add_argument("--device", default="cpu")
    q.add_argument("--dino-model", default="facebook/dinov2-base")
    q.add_argument("--da3-model")
    q = sub.add_parser("vqa")
    q.add_argument("--video", required=True)
    q.add_argument("--output", required=True)
    q.add_argument(
        "--revisit",
        choices=[
            "reversed_rotation",
            "full_rotation",
            "forward_look_back",
            "new_viewpoint",
        ],
        required=True,
    )
    q.add_argument("--endpoint", type=int)
    q.add_argument("--model")
    q.add_argument("--base-url")
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("models")
    a = p.parse_args(argv)
    if a.command == "prepare":
        from .prepare import prepare

        result = prepare(a.case)
    elif a.command == "demo":
        from .demo import create

        result = create(a.output)
    elif a.command == "validate":
        from .continuation.dataset import load_context_bundle

        c = load_context_bundle(a.case)
        result = {
            "case_id": c.case_id,
            "context_frames": c.context_frame_count,
            "context_sha256": c.context_sha256,
        }
    elif a.command == "run":
        result = run(a)
    elif a.command == "evaluate":
        from .metrics.evaluate import evaluate

        result = evaluate(
            a.case,
            a.prediction,
            a.output,
            task=a.task,
            model_id=a.model_id,
            max_rotation_deg=a.max_rotation_deg,
            metrics=a.metrics,
            device=a.device,
            dino_model=a.dino_model,
            da3_model=a.da3_model,
        )
    elif a.command == "vqa":
        from .judges.vqa import evaluate

        result = evaluate(
            a.video,
            a.output,
            revisit=a.revisit,
            endpoint=a.endpoint,
            model=a.model,
            base_url=a.base_url,
            dry_run=a.dry_run,
        )
    else:
        from .models.registry import MODELS

        result = {"native": list(MODELS), "fixtures": ["toy"]}
    if a.command != "run" or int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
