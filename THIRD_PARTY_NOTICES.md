# Third-party sources and license boundaries

The top-level Apache-2.0 license covers SceneMemBench-authored code and extracted components already under those terms. It does **not** relicense the native model implementations, their derivatives or model weights. This repository is a development extraction with mixed component licenses, not a uniformly permissive model-code distribution.

| Component | Included adaptation / source | Retained terms |
| --- | --- | --- |
| WorldFoundry | Continuation contracts, adapters, action planning, metrics and selected native runtime dependencies; [source repository](https://github.com/Seeyou-Hi/Memory-Benchmark) | Apache-2.0 for original framework code; nested third-party exceptions remain |
| HY-WorldPlay | `_native/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/`, related HY runtime and adapters; [upstream](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | [Tencent community license](licenses/HY-WorldPlay.txt); includes territorial and other conditions, not Apache-2.0 |
| HunyuanVideo-1.5 | Related Hunyuan native components; [upstream](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | [Upstream license](licenses/HunyuanVideo-1.5.txt); external weights retain their own terms |
| LingBot-World-v2 | V2 DiT, causal inference and causal RoPE attention; [fixed source revision](https://github.com/Robbyant/lingbot-world-v2/tree/94f43115de8d4a4f9f282126528c300a0b232c5f) | [CC BY-NC-SA 4.0](licenses/LingBot-World-v2.txt), including these adaptations; not Apache-2.0 |
| Matrix-Game-3.0 | MG3 memory runtime and related components; [upstream](https://github.com/SkyworkAI/Matrix-Game) | [Upstream license](licenses/Matrix-Game-3.0.txt); retain per-file exceptions |
| Matrix-Game-3.5 | Standalone bridge and native history patch against a fixed official source revision; [upstream](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) | [Apache-2.0](licenses/Matrix-Game-3.5.txt) |
| SANA / SANA-WM | Streaming runtime, model/config subsets; [upstream](https://github.com/NVlabs/Sana) | [Apache-2.0](licenses/SANA.txt), with retained embedded notices |
| Wan | Common DiT/VAE/text/runtime components; [upstream](https://github.com/Wan-Video/Wan2.2) | [Apache-2.0](licenses/Wan.txt), with retained embedded notices |
| Zing | Adapter and runtime bridge; official model code is separately installed; [upstream](https://github.com/seedleap/zing-world-model) | [Apache-2.0](licenses/Zing-World.txt); checkpoint terms remain separate |
| SGLang / LingBot-VLA | Shared GroupNorm/SiLU and routed-MoE kernels inherited through WorldFoundry | Apache-2.0; see retained headers and [kernel notices](licenses/kernel-NOTICES.md) |

The VQA client and prompts are independently implemented for this release. No PlayWorld source is copied. Prompt wording is separately versioned and not presented as an exact replay of historical judge responses.

DINOv2, LPIPS and Depth Anything 3 are separately installed metric dependencies; their code and weights are not vendored here. Native checkpoints, Unreal assets, private configuration and experimental results are excluded.

Modified extracted Python files carry a modification notice. `docs/source_manifest.json` records the original file hash and current distributed-file hash. Namespace conversion, path/config portability, removal of platform-only dependencies and formatting are part of the adaptation; model-specific state/history behavior is retained. A source snapshot hash identifies code, not proof that it produced historical paper results.

Before a tagged public release, verify component redistribution requirements against the exact code and checkpoint versions selected. In particular, retain LingBot's noncommercial/share-alike boundary and Tencent's community-license terms; a root Apache license does not remove either restriction.
