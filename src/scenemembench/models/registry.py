"""Lazy registry. Selecting a model loads only its native backend."""

from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path

MODELS = {
    "hy-worldplay": (
        "hunyuan_world.pipeline_hunyuan_worldplay",
        "HunyuanWorldPlayPipeline",
        "hunyuan_worldplay",
        "HunyuanWorldPlayMemoryAdapter",
    ),
    "lingbot-world-v2": (
        "lingbot_world_v2.pipeline_lingbot_world_v2",
        "LingBotWorldV2Pipeline",
        "lingbot_world_v2",
        "LingBotWorldV2MemoryAdapter",
    ),
    "matrix-game-3": (
        "matrix_game.pipeline_matrix_game_3",
        "MatrixGame3Pipeline",
        "matrix_game_3",
        "MatrixGame3MemoryAdapter",
    ),
    "matrix-game-3.5": (
        "matrix_game.pipeline_matrix_game_35",
        "MatrixGame35Pipeline",
        "matrix_game_35",
        "MatrixGame35MemoryAdapter",
    ),
    "sana-wm-streaming": (
        "sana_wm.pipeline_sana_wm",
        "SanaWMPipeline",
        "sana_wm",
        "SanaWMStreamingMemoryAdapter",
    ),
    "zing-world-0.5": (
        "zing_world.pipeline_zing_world",
        "ZingWorldPipeline",
        "zing_world",
        "ZingWorldMemoryAdapter",
    ),
}


def load(model, config, device):
    if model == "toy":
        from .toy import ToyAdapter

        return ToyAdapter()
    if model not in MODELS:
        raise ValueError(f"Unknown model {model}")
    if not config:
        raise ValueError(
            "Native models require --config with explicit checkpoint settings"
        )

    def expand(value):
        if isinstance(value, str):
            result = os.path.expandvars(os.path.expanduser(value))
            if "${" in result:
                raise ValueError(f"Unconfigured environment variable in {value}")
            return result
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    config = expand(config)
    options = dict(config.get("pipeline", {}))
    if not options.get("model_path"):
        raise ValueError("pipeline.model_path must identify the checkpoint")
    for key, value in config.get("environment", {}).items():
        if not key.startswith(("SCENEMEMBENCH_", "WAN22_", "UMT5_")):
            raise ValueError(f"Unsupported environment setting: {key}")
        os.environ[key] = os.path.expandvars(os.path.expanduser(str(value)))
    # MG3.5 and Zing keep separately installed upstream code; neither uses an old benchmark repository.
    if model == "matrix-game-3.5":
        root = Path(config["upstream_source"]).expanduser().resolve()
        if not (root / "infer.py").exists():
            raise ValueError("MG3.5 upstream_source must contain infer.py")
        sys.path.insert(0, str(root))
        components = dict(options.get("required_components", {}))
        components["driver"] = "scenemembench.models.mg35_driver:create_driver"
        components.setdefault("config", str(root / "configs/infer_distilled.yaml"))
        options["required_components"] = components
    module, cls, adapter_module, adapter_cls = MODELS[model]
    pipeline_cls = getattr(
        importlib.import_module("scenemembench._native.pipelines." + module), cls
    )
    if model == "sana-wm-streaming":
        options.setdefault("model_id", model)
    options = {
        k: os.path.expandvars(os.path.expanduser(v)) if isinstance(v, str) else v
        for k, v in options.items()
    }
    pipeline = pipeline_cls.from_pretrained(device=device, **options)
    adapter = getattr(
        importlib.import_module(
            "scenemembench.continuation.adapters." + adapter_module
        ),
        adapter_cls,
    )(pipeline)
    return adapter
