# Native backend setup

Model code is imported lazily: installing the default package and running VQA or pixel metrics does not load a world model. Run each native backend in its own environment when Torch, Diffusers or attention-kernel versions conflict.

The `.[native]` extra is a common dependency starting point, not a universal lockfile. Do not install every attention extension by default. Native GPU validation status is tracked in `validation.md`.

## Paths and model configuration

Copy the relevant `configs/models/*.yaml` to an ignored local file if you need machine-specific paths. Set `SCENEMEMBENCH_CKPT_DIR` to a weight directory; files are read from there but never copied into Git. Nested config values expand environment variables; missing `${VARIABLE}` values cause an error.

- HY: `HY-WorldPlay` and `HunyuanVideo-1.5`; the HY directory must include the distilled action checkpoint. Group offload is enabled in the example.
- LingBot v2: `lingbot-world-v2-14b-causal-fast`, including VAE and text components, plus `taehv/lighttaew2_1.pth` under the checkpoint root (or set `SCENEMEMBENCH_LINGBOT_LIGHTTAE_PATH`). Native streaming attention may require its matching flash-attention build.
- MG3: `Matrix-Game-3.0`, including `base_distilled_model`, the UMT5 components and native VAE. Preserve the backend's native resolution, quantization and overlap configuration.
- MG3.5: set `SCENEMEMBENCH_MG35_SOURCE` to the official Matrix-Game-3.5 checkout; set `WAN22_TI2V_5B_DIR` and `UMT5_TOKENIZER_DIR`. Configure `required_components.da3_model` with the local DA3 checkpoint directory. The included driver bridges its native resident inference to the continuation lifecycle. Run `python scripts/prepare_mg35.py --destination /your/Matrix-Game-3.5` to fetch the fixed official revision and apply `backends/matrix-game-3.5/native-history.patch`. The patch has been checked against a clean `fa6d2b628ac9b0f1657dc24689536d74bfeb51da` checkout; this establishes a reconstructible code baseline, not proof of historical execution. Install the upstream dependencies in the backend environment.
- SANA: `SANA-WM_streaming` with stage-1, causal VAE, refiner and Gemma components. Keep the Stage-1 `Efficient-Large-Model--gemma-2-2b-it` encoder under `$SCENEMEMBENCH_CKPT_DIR/hfd/` (or set `SCENEMEMBENCH_HFD_ROOT`); the refiner uses the checkpoint's `gemma3_12b`. Native code also needs packages such as `pyrallis`, `timm`, `einops`, and a Diffusers build with LTX2 support. Optional Transformer Engine paths are not mandatory unless enabled.
- Zing: set `SCENEMEMBENCH_ZING_SOURCE` to the official `seedleap/zing-world-model` source; its `src/` and `config/zing.yaml` must exist. The weight tree contains the generator and pretrained VAE/text components.

The MG3.5 and Zing upstream checkouts are **model dependencies**, not dependencies on WorldFoundry or PlayWorld. Model weights and third-party sources are not fetched automatically by the core CLI.

## 48 GB GPU configurations

The real-mini MG3.5 check uses two 48 GB GPUs with its native resolution:

```bash
CUDA_VISIBLE_DEVICES=0,1 SCENEMEMBENCH_MATRIX_GAME35_MODEL_PARALLEL=2 \
  scenemembench run --model matrix-game-3.5 --device cuda:0 \
  --config configs/models/matrix-game-3.5.yaml \
  --case /path/to/case --output outputs/mg35
```

LingBot v2 uses two worker processes with DiT sharding and CPU text encoding:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m scenemembench.cli run --model lingbot-world-v2 --device cuda:0 \
  --config configs/models/lingbot-world-v2.yaml \
  --case /path/to/case --output outputs/lingbot
```

Use an empty output directory. Only rank zero writes videos and reports.

## Lifecycle

`begin_case → prefill_context → Query steps → final decode → end_case`

The runner never resets between prefill and Query. A rejected prefill aborts generation, and cleanup runs on failure. Context input paths exclude Query images; Query pose/action rows may be used for controls. Native adapters retain accepted frame IDs, temporal resampling and padding reports. `continuation_verified` checks state/frame bookkeeping, not scene-memory quality.

Action planning is extracted from the existing launcher: HY angular speed conversion, per-model native FPS, MG3.5 prescribed yaw/native pitch, SANA pose mapping, and Zing's calibrated six chunks remain explicit. Do not interpret Zing keyboard controls as exact UE poses.

Outputs are local `continuation.mp4`, per-step videos, configuration, prefill and action/state traces. They are ignored by Git and are not part of the release data.
