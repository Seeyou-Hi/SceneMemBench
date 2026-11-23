# Mini data

The first code release includes four procedurally generated input fixtures, approximately one tiny case per interface category. They are explicitly synthetic and not physical UE trajectories, benchmark cases, generated model outputs or published results. Their purpose is to make the installation and data contract runnable without downloading the large corpus.

Generate them using `scenemembench demo --output data/demo`, or extract `mini-synthetic-v1.tar.gz` into an empty directory. `checksums.json` contains hashes of the input files. No external image assets are used.

Each case has:

```text
case/
  manifest.json
  pose/poses.json
  rgb/000000.png ...
  context.mp4
  eval.mp4
```

`split.context` and `split.eval` are zero-based **half-open** source-frame ranges `[start,end)`. Rows are ordered by `frame_id`; timestamps are seconds. `primary_pose` is UE `[x_cm,y_cm,z_cm,pitch_deg,yaw_deg,roll_deg]`; `camera_to_world` is 4×4. `resolution` is `[width,height]`; horizontal FOV is degrees. Query actions and phase ranges are explicit. See a generated manifest for a complete example.

The continuation loader reads Context PNGs and does not read Query RGB. The evaluator can use an exactly sliced `eval.mp4` or Query RGB. For the video-based real mini, run `prepare` first to decode the Context frames.

Straight query boundaries must equal `phase_ranges.eval_turnaround`. Old source videos containing B→A after the turnaround are not accepted as canonical Query clips. Source video frame counts and manifest ranges must agree.

The committed archive is synthetic. The selected real examples are packaged separately as `scenemembench-greek-island-mini-v1.tar.gz`. Full Rollout's 60 initial images comprise both UE renders and AI-generated images; those inputs and their result packages are outside this continuation release. Model weights and UE source assets are distributed separately under their respective terms.

Selected Greek Island examples are listed in [mini-real-selection.json](mini-real-selection.json).

## Real mini: prepare and run

Extract `scenemembench-greek-island-mini-v1.tar.gz`, then:

```bash
scenemembench prepare --case /path/to/scenemembench-greek-island-mini-v1/greek_island_reversed_rotation
scenemembench validate --case /path/to/scenemembench-greek-island-mini-v1/greek_island_reversed_rotation
```

Use that case path with the README's `run` and `evaluate` commands. The package contains four cases with original Context/Query video bytes, pose/actions, prompts, source hashes and file checksums. It omits raw capture PNGs and full videos. `prepare` writes only Context PNGs, decoded from `context.mp4`; these are video-derived images. Query references remain in `eval.mp4`. Decoded images require additional local disk space.

Straight ends at source frame 517 inclusive: 457 Context frames and 61 Query frames. No return-to-A RGB, pose or full video is included.

Build from an existing source dataset using `scripts/release/build_real_mini.py --source-root <dataset> --selection data/mini-real-selection.json --output <new-directory>`. Original source files are unchanged.
