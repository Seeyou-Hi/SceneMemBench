# Evaluation contract

Continued evaluation compares the final generated frame with a GT reference. All requested metrics use the same image pair.

The reference is shared across DINOv2, LPIPS, SSIM, PSNR, depth SI-AbsRel and normal cosine. The matcher uses the extracted SIFT ratio/RANSAC/inlier-hull implementation and its fallback thresholds. Pixel preprocessing is 448×252 with fixed resize/center-crop rules. `--model-id` enables the model's native reference canvas transform.

Geometry uses two separate single-image DA3 inferences. There is no joint reference/generated inference or input-pose alignment. Depth scale is a trimmed median of GT/generated ratios. Generated-invalid, reference-valid pixels receive unit depth error. Normals are derived by back-projection and central differences with scaled intrinsics; undefined values remain null.

For the learned metrics:

```bash
pip install -e '.[metrics]'
# Install the official DA3 package in the metric environment:
pip install 'git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
```

Pin the DA3 code revision and checkpoint revision/hash in your own experiment environment. The initial development branch does not claim a historical evaluator lock. Provide `--da3-model` explicitly and optionally `--dino-model` as a local directory to avoid downloads. First LPIPS use may also fetch its AlexNet backbone unless already cached.

The saved evaluation JSON identifies the search rule, candidate source frame IDs, selected reference, generated endpoint, and prediction hash. This development release provides runnable evaluation tools and does not include paper result tables.
