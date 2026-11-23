# Retained notices for included kernels

## LingBot-VLA v2 packed routed MoE

`_native/core/kernels/triton_moe.py` is adapted from the grouped routed-MoE inference path in Robbyant LingBot-VLA v2 via WorldFoundry. It exposes a packed SwiGLU interface and a portable PyTorch fallback.

Source: https://github.com/Robbyant/lingbot-vla-v2

Copyright 2026 Robbyant Team and/or its affiliates. Licensed under the Apache License 2.0 (see the repository LICENSE).

## SGLang GroupNorm + SiLU

`_native/core/kernels/triton_group_norm_silu.py` is adapted from `sglang/jit_kernel/diffusion/triton/group_norm_silu.py` via WorldFoundry. The adaptation removes SGLang runtime coupling and exposes a local launcher.

Source: https://github.com/sgl-project/sglang

Licensed under the Apache License 2.0 (see the repository LICENSE). Original file headers remain in place.
