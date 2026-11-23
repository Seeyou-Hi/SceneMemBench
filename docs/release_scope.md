# Small continuation release

This development release provides Context-prefilled Query generation for six
open-weight world models, endpoint evaluation, and small example inputs.

## Included

- HY-WorldPlay, LingBot-World-v2, Matrix-Game-3, Matrix-Game-3.5,
  SANA-WM streaming, and Zing-World-0.5 continuation entry points.
- Context loading, pose/action controls, generated Query video, and run reports.
- Seen and Novel endpoint evaluation with six optional metrics; the default
  lightweight installation computes PSNR and SSIM.
- Synthetic installation fixtures and a separately packaged Greek Island mini
  containing one real example per trajectory type.
- Installation, model-specific setup, data format, source and license notices.

Straight examples must stop at completion of the turn at B. No subsequent
return to A belongs in the distributed case, including its pose/RGB files.
Context images are inference inputs; Query images are evaluation references.

## Acceptance

The CPU quickstart must run from installation through generation and evaluation.
For each native model, record an actual Context-prefill and Query-generation
run in its supported environment. Record output frame count and any failure;
import success alone does not establish a working backend. Current verification
coverage is in [validation.md](validation.md).

Before bundling real examples, check that their videos, pose ranges and frame
IDs agree, paths are portable, checksums exist, and redistribution terms are
documented. Synthetic fixtures do not stand in for real benchmark examples.

Full Rollout generation/results, the full dataset, historical run archives and
paper leaderboards are outside this release. This is a runnable development
release rather than a complete reproduction of the paper's result tables.
