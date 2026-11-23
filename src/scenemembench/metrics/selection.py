# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Query reference selection for SceneMemBench, with optional pose bounds.

The selector is intentionally independent of the six reported benchmark
metrics.  It optionally restricts GT frames to a terminal SE(3) neighbourhood
and selects one frame using local feature correspondences, RANSAC geometry,
and symmetric inlier coverage.  All reported metrics must subsequently use
that one selected frame.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PoseToleranceConfig:
    """Frozen GT endpoint-neighbourhood and selector configuration."""

    max_rotation_deg: float | None = None
    max_translation_m: float = 0.5
    max_candidate_frames: int = 48
    position_units_per_meter: float = 100.0
    selector_max_dimension: int = 960
    ransac_reprojection_px: float = 4.0
    sift_ratio_threshold: float = 0.75
    sift_nfeatures: int = 4096
    minimum_inliers: int = 12
    minimum_coverage: float = 0.01
    duplicate_rotation_epsilon_deg: float = 0.05
    duplicate_translation_epsilon_m: float = 0.005

    def __post_init__(self) -> None:
        if (
            self.max_rotation_deg is not None
            and (not np.isfinite(self.max_rotation_deg) or self.max_rotation_deg < 0.0)
        ) or self.max_translation_m < 0.0:
            raise ValueError("pose tolerances must be non-negative")
        if self.max_candidate_frames < 1:
            raise ValueError("max_candidate_frames must be positive")
        if self.position_units_per_meter <= 0.0:
            raise ValueError("position_units_per_meter must be positive")
        if self.selector_max_dimension < 64:
            raise ValueError("selector_max_dimension must be at least 64")
        if self.ransac_reprojection_px <= 0.0:
            raise ValueError("ransac_reprojection_px must be positive")
        if not 0.0 < self.sift_ratio_threshold < 1.0:
            raise ValueError("sift_ratio_threshold must be in (0, 1)")
        if self.sift_nfeatures < 1 or self.minimum_inliers < 4:
            raise ValueError("invalid feature/inlier count")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in [0, 1]")
        if self.duplicate_rotation_epsilon_deg < 0.0:
            raise ValueError("duplicate_rotation_epsilon_deg must be non-negative")
        if self.duplicate_translation_epsilon_m < 0.0:
            raise ValueError("duplicate_translation_epsilon_m must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidatePose:
    """One GT eval frame admitted by the terminal pose gate."""

    eval_frame_index: int
    source_frame_id: int
    rotation_offset_deg: float
    translation_offset_m: float
    pose_group_start_eval_frame_index: int | None = None
    pose_group_size: int = 1


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Auditable selector score for one generated/GT endpoint pair."""

    score: float
    correspondence_count: int
    inlier_count: int
    generated_coverage: float
    gt_coverage: float
    spatial_coverage: float
    homography_found: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorrespondenceBackend(Protocol):
    """Feature backend returning pixel correspondences in two images."""

    name: str

    def correspondences(
        self, generated_rgb: np.ndarray, gt_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _selector_image(image: np.ndarray, max_dimension: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB HWC image, got {array.shape}")
    height, width = array.shape[:2]
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(array)


class SIFTCorrespondenceBackend:
    """Deterministic mutual-ratio SIFT correspondences."""

    name = "sift_mutual_ratio"

    def __init__(self, config: PoseToleranceConfig) -> None:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV was built without SIFT support")
        self.config = config
        self.extractor = cv2.SIFT_create(nfeatures=config.sift_nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._feature_cache: OrderedDict[
            tuple[tuple[int, ...], str], tuple[list[cv2.KeyPoint], np.ndarray | None]
        ] = OrderedDict()
        self._feature_cache_limit = max(config.max_candidate_frames + 1, 8)

    def _features(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        prepared = _selector_image(image, self.config.selector_max_dimension)
        digest = hashlib.blake2b(
            memoryview(prepared).cast("B"), digest_size=16
        ).hexdigest()
        key = (tuple(int(value) for value in prepared.shape), digest)
        cached = self._feature_cache.get(key)
        if cached is not None:
            self._feature_cache.move_to_end(key)
            return cached
        gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
        value = self.extractor.detectAndCompute(gray, None)
        self._feature_cache[key] = value
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self._feature_cache_limit:
            self._feature_cache.popitem(last=False)
        return value

    @staticmethod
    def _ratio_matches(
        pairs: Sequence[Sequence[cv2.DMatch]], threshold: float
    ) -> dict[int, int]:
        accepted: dict[int, int] = {}
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair[0], pair[1]
            if first.distance < threshold * second.distance:
                accepted[int(first.queryIdx)] = int(first.trainIdx)
        return accepted

    def correspondences(
        self, generated_rgb: np.ndarray, gt_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        generated_keypoints, generated_descriptors = self._features(generated_rgb)
        gt_keypoints, gt_descriptors = self._features(gt_rgb)
        if generated_descriptors is None or gt_descriptors is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy()
        forward = self._ratio_matches(
            self.matcher.knnMatch(generated_descriptors, gt_descriptors, k=2),
            self.config.sift_ratio_threshold,
        )
        reverse = self._ratio_matches(
            self.matcher.knnMatch(gt_descriptors, generated_descriptors, k=2),
            self.config.sift_ratio_threshold,
        )
        mutual = sorted(
            (generated_index, gt_index)
            for generated_index, gt_index in forward.items()
            if reverse.get(gt_index) == generated_index
        )
        if not mutual:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy()
        generated_points = np.asarray(
            [generated_keypoints[index].pt for index, _ in mutual], dtype=np.float32
        )
        gt_points = np.asarray(
            [gt_keypoints[index].pt for _, index in mutual], dtype=np.float32
        )
        return generated_points, gt_points


class LightGlueCorrespondenceBackend:
    """Optional SuperPoint+LightGlue correspondence backend.

    Imports are deliberately lazy so the default SIFT evaluator has no extra
    dependency.  The official ``lightglue`` package API is used when present.
    """

    name = "superpoint_lightglue"

    def __init__(self, config: PoseToleranceConfig, *, device: str = "cuda:0") -> None:
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LightGlue selection requires the lightglue package and torch"
            ) from exc
        self.config = config
        self.torch = torch
        self.device = torch.device(device)
        self.extractor = (
            SuperPoint(max_num_keypoints=config.sift_nfeatures).eval().to(self.device)
        )
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _tensor(self, image: np.ndarray):
        value = _selector_image(image, self.config.selector_max_dimension)
        return (
            self.torch.from_numpy(value)
            .to(self.device, dtype=self.torch.float32)
            .permute(2, 0, 1)
            .div(255.0)
        )

    def correspondences(
        self, generated_rgb: np.ndarray, gt_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.inference_mode():
            generated_features = self.extractor.extract(self._tensor(generated_rgb))
            gt_features = self.extractor.extract(self._tensor(gt_rgb))
            matched = self.matcher(
                {"image0": generated_features, "image1": gt_features}
            )
        matches = matched["matches"][0].detach().cpu().numpy()
        if not len(matches):
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy()
        generated_keypoints = generated_features["keypoints"][0].detach().cpu().numpy()
        gt_keypoints = gt_features["keypoints"][0].detach().cpu().numpy()
        return (
            np.asarray(generated_keypoints[matches[:, 0]], dtype=np.float32),
            np.asarray(gt_keypoints[matches[:, 1]], dtype=np.float32),
        )


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic SO(3) distance between two camera-to-world rotations."""

    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(
        second, dtype=np.float64
    )
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def terminal_pose_candidates(
    poses: Sequence[np.ndarray],
    *,
    eval_start_source_frame: int,
    config: PoseToleranceConfig,
) -> list[CandidatePose]:
    """Return the contiguous terminal GT pose neighbourhood.

    The scan starts at the exact endpoint and stops at the first frame outside
    either pose threshold.  This prevents a looping trajectory from admitting
    an earlier, disconnected visit to the same pose.
    """

    if not poses:
        raise ValueError("GT eval pose sequence is empty")
    matrices = [np.asarray(value, dtype=np.float64) for value in poses]
    if any(value.shape != (4, 4) for value in matrices):
        raise ValueError("every GT camera pose must be 4x4")
    endpoint = matrices[-1]
    selected: list[CandidatePose] = []
    lower = max(0, len(matrices) - config.max_candidate_frames)
    for index in range(len(matrices) - 1, lower - 1, -1):
        pose = matrices[index]
        rotation = rotation_distance_deg(pose[:3, :3], endpoint[:3, :3])
        translation = float(
            np.linalg.norm(pose[:3, 3] - endpoint[:3, 3])
            / config.position_units_per_meter
        )
        if (
            (config.max_rotation_deg is not None and rotation > config.max_rotation_deg + 1e-6)
            or translation > config.max_translation_m + 1e-9
        ):
            break
        selected.append(
            CandidatePose(
                eval_frame_index=index,
                source_frame_id=eval_start_source_frame + index,
                rotation_offset_deg=rotation,
                translation_offset_m=translation,
            )
        )
    selected.reverse()
    if not selected or selected[-1].eval_frame_index != len(matrices) - 1:
        raise AssertionError("exact GT endpoint was not retained")
    # A terminal hold can contain many encoded frames with one identical pose.
    # Treating each as an independent candidate gives that pose many chances to
    # win through compression or feature noise. Collapse every consecutive
    # near-identical pose group and retain its latest frame; for the endpoint
    # hold this is exactly the frozen endpoint frame.
    collapsed: list[CandidatePose] = []
    group_start = selected[0].eval_frame_index
    group_size = 0
    previous_index: int | None = None
    for candidate in selected:
        same_group = False
        if previous_index is not None:
            previous = matrices[previous_index]
            current = matrices[candidate.eval_frame_index]
            rotation_delta = rotation_distance_deg(previous[:3, :3], current[:3, :3])
            translation_delta = float(
                np.linalg.norm(previous[:3, 3] - current[:3, 3])
                / config.position_units_per_meter
            )
            same_group = (
                rotation_delta <= config.duplicate_rotation_epsilon_deg
                and translation_delta <= config.duplicate_translation_epsilon_m
            )
        if not same_group:
            group_start = candidate.eval_frame_index
            group_size = 1
            collapsed.append(
                CandidatePose(
                    eval_frame_index=candidate.eval_frame_index,
                    source_frame_id=candidate.source_frame_id,
                    rotation_offset_deg=candidate.rotation_offset_deg,
                    translation_offset_m=candidate.translation_offset_m,
                    pose_group_start_eval_frame_index=group_start,
                    pose_group_size=group_size,
                )
            )
        else:
            group_size += 1
            collapsed[-1] = CandidatePose(
                eval_frame_index=candidate.eval_frame_index,
                source_frame_id=candidate.source_frame_id,
                rotation_offset_deg=candidate.rotation_offset_deg,
                translation_offset_m=candidate.translation_offset_m,
                pose_group_start_eval_frame_index=group_start,
                pose_group_size=group_size,
            )
        previous_index = candidate.eval_frame_index
    if collapsed[-1].eval_frame_index != len(matrices) - 1:
        raise AssertionError("pose de-duplication removed the exact GT endpoint")
    return collapsed


def _convex_hull_coverage(points: np.ndarray, image: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
    height, width = image.shape[:2]
    return float(np.clip(area / max(float(height * width), 1.0), 0.0, 1.0))


def ransac_match_score(
    generated_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    backend: CorrespondenceBackend,
    config: PoseToleranceConfig,
) -> MatchScore:
    """Score correspondences with homography RANSAC and symmetric coverage."""

    generated = _selector_image(generated_rgb, config.selector_max_dimension)
    gt = _selector_image(gt_rgb, config.selector_max_dimension)
    generated_points, gt_points = backend.correspondences(generated, gt)
    correspondence_count = int(len(generated_points))
    if correspondence_count < 4:
        return MatchScore(0.0, correspondence_count, 0, 0.0, 0.0, 0.0, False)
    _, mask = cv2.findHomography(
        generated_points,
        gt_points,
        cv2.RANSAC,
        config.ransac_reprojection_px,
    )
    if mask is None:
        return MatchScore(0.0, correspondence_count, 0, 0.0, 0.0, 0.0, False)
    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    generated_coverage = _convex_hull_coverage(generated_points[inliers], generated)
    gt_coverage = _convex_hull_coverage(gt_points[inliers], gt)
    spatial_coverage = min(generated_coverage, gt_coverage)
    return MatchScore(
        score=float(inlier_count * spatial_coverage),
        correspondence_count=correspondence_count,
        inlier_count=inlier_count,
        generated_coverage=generated_coverage,
        gt_coverage=gt_coverage,
        spatial_coverage=spatial_coverage,
        homography_found=True,
    )


def select_reference_frame(
    generated_rgb: np.ndarray,
    candidates: Sequence[CandidatePose],
    candidate_images: dict[int, np.ndarray],
    backend: CorrespondenceBackend,
    config: PoseToleranceConfig,
) -> tuple[CandidatePose, MatchScore, list[dict[str, Any]], str | None]:
    """Select one GT frame, or conservatively fall back to the exact endpoint."""

    if not candidates:
        raise ValueError("candidate list is empty")
    scored: list[tuple[CandidatePose, MatchScore]] = []
    for candidate in candidates:
        image = candidate_images.get(candidate.eval_frame_index)
        if image is None:
            raise KeyError(f"missing GT candidate image {candidate.eval_frame_index}")
        scored.append(
            (candidate, ransac_match_score(generated_rgb, image, backend, config))
        )
    # Prefer score, then inlier support, then coverage.  The final key favours
    # the later GT frame under an exact tie, limiting unnecessary pose relief.
    best_candidate, best_score = max(
        scored,
        key=lambda item: (
            item[1].score,
            item[1].inlier_count,
            item[1].spatial_coverage,
            item[0].eval_frame_index,
        ),
    )
    fallback_reason = None
    if (
        best_score.inlier_count < config.minimum_inliers
        or best_score.spatial_coverage < config.minimum_coverage
    ):
        fallback_reason = "insufficient_selector_confidence"
        best_candidate = candidates[-1]
        best_score = next(
            score for candidate, score in scored if candidate == best_candidate
        )
    audit = [
        {
            **asdict(candidate),
            **score.to_dict(),
        }
        for candidate, score in scored
    ]
    return best_candidate, best_score, audit, fallback_reason


def load_eval_poses(case_dir: Path, eval_range: Sequence[int]) -> list[np.ndarray]:
    """Load GT eval camera-to-world matrices from either supported pose layout."""

    if len(eval_range) != 2:
        raise ValueError(f"invalid eval range: {eval_range}")
    start, end = (int(value) for value in eval_range)
    if end <= start:
        raise ValueError(f"empty eval range: {eval_range}")
    combined = case_dir / "pose" / "poses.json"
    rows_by_frame: dict[int, Any] = {}
    if combined.is_file():
        import json

        with combined.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        rows_by_frame = {int(row["frame_id"]): row for row in rows}
    output: list[np.ndarray] = []
    for frame_id in range(start, end):
        if rows_by_frame:
            row = rows_by_frame.get(frame_id)
            if row is None:
                raise KeyError(f"missing pose row for source frame {frame_id}")
        else:
            import json

            path = case_dir / "pose" / f"{frame_id:06d}.json"
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
        output.append(np.asarray(row["camera_to_world"], dtype=np.float64))
    return output


def read_video_frames(path: Path, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Decode selected zero-based frames from one video as RGB arrays."""

    requested = sorted(set(int(value) for value in indices))
    if not requested or requested[0] < 0:
        raise ValueError("frame indices must be non-empty and non-negative")
    wanted = set(requested)
    output: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frame_index = 0
    last_requested = requested[-1]
    while frame_index <= last_requested:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in wanted:
            output[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_index += 1
    cap.release()
    missing = [value for value in requested if value not in output]
    if missing:
        raise RuntimeError(f"video {path} is missing requested frames {missing}")
    return output


__all__ = [
    "CandidatePose",
    "CorrespondenceBackend",
    "LightGlueCorrespondenceBackend",
    "MatchScore",
    "PoseToleranceConfig",
    "SIFTCorrespondenceBackend",
    "load_eval_poses",
    "ransac_match_score",
    "read_video_frames",
    "rotation_distance_deg",
    "select_reference_frame",
    "terminal_pose_candidates",
]
