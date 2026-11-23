# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""Geometry helpers for UCM warped RoPE in the resident WorldPlay runtime."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _canonical_c2w_from_w2c(w2c: torch.Tensor, convention: str) -> np.ndarray:
    c2w = torch.linalg.inv(torch.as_tensor(w2c).float()).cpu().numpy()
    rotations = c2w[:, :3, :3]
    if str(convention).lower() == "unreal":
        rotations = np.stack(
            [rotations[:, :, 1], -rotations[:, :, 2], rotations[:, :, 0]], axis=2
        )
    result = np.tile(np.eye(4, dtype=np.float32), (c2w.shape[0], 1, 1))
    result[:, :3, :3] = rotations.astype(np.float32)
    result[:, :3, 3] = c2w[:, :3, 3].astype(np.float32)
    return result


def _fit_sim3(source_points, target_points, source_rotations, target_rotations):
    count = min(len(source_points), len(target_points))
    source = np.asarray(source_points[:count], dtype=np.float64)
    target = np.asarray(target_points[:count], dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source0 = source - source_mean
    target0 = target - target_mean
    corr = target0.T @ source0
    norm = float(np.linalg.norm(corr))
    corr = corr / norm if norm > 1e-10 else np.zeros((3, 3), dtype=np.float64)
    if source_rotations is not None and target_rotations is not None:
        orient = np.einsum(
            "nij,nkj->ik", target_rotations[:count], source_rotations[:count]
        )
        orient_norm = float(np.linalg.norm(orient))
        if orient_norm > 1e-10:
            corr += orient / orient_norm
    if float(np.linalg.norm(corr)) <= 1e-10:
        rotation = np.eye(3, dtype=np.float64)
    else:
        u, _, vt = np.linalg.svd(corr)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt
    denom = float((source0 * source0).sum())
    scale = (
        1.0
        if denom < 1e-10
        else max(float(((source0 @ rotation.T) * target0).sum() / denom), 1e-8)
    )
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def align_w2c_to_lingbot(
    source_w2c: torch.Tensor,
    source_convention: str,
    target_w2c: torch.Tensor,
    target_convention: str = "w2c",
    fit_count: int = 0,
) -> torch.Tensor:
    """Align command poses into the LingBot coordinate/scale convention."""
    source = _canonical_c2w_from_w2c(source_w2c, source_convention)
    target = _canonical_c2w_from_w2c(target_w2c, target_convention)
    count = min(len(source), len(target))
    if fit_count > 0:
        count = min(count, int(fit_count))
    scale, rotation, translation = _fit_sim3(
        source[:count, :3, 3],
        target[:count, :3, 3],
        source[:count, :3, :3],
        target[:count, :3, :3],
    )
    aligned = np.tile(np.eye(4, dtype=np.float32), (len(source), 1, 1))
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, source[:, :3, :3])
    aligned[:, :3, 3] = (
        scale * np.einsum("ij,nj->ni", rotation, source[:, :3, 3]) + translation
    )
    return torch.from_numpy(np.linalg.inv(aligned).astype(np.float32))


def anchor_w2c_to_observed_source(
    aligned_w2c: torch.Tensor,
    observed_w2c: torch.Tensor,
    source_index: int,
    observed_convention: str = "w2c",
) -> torch.Tensor:
    aligned_c2w = torch.linalg.inv(aligned_w2c.float()).cpu().numpy()
    observed_c2w = _canonical_c2w_from_w2c(observed_w2c, observed_convention)
    index = int(source_index)
    residual = observed_c2w[index] @ np.linalg.inv(aligned_c2w[index])
    anchored_c2w = np.einsum("ij,njk->nik", residual, aligned_c2w)
    return torch.from_numpy(np.linalg.inv(anchored_c2w).astype(np.float32))


def _scale_intrinsics(K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    K = K.clone()
    normalized = (
        K[..., 0, 2].abs().amax() <= 1.5
        and K[..., 1, 2].abs().amax() <= 1.5
        and K[..., 0, 0].abs().amax() <= 10.0
        and K[..., 1, 1].abs().amax() <= 10.0
    )
    if normalized:
        K[..., 0, 0] *= width
        K[..., 1, 1] *= height
        K[..., 0, 2] *= width
        K[..., 1, 2] *= height
    else:
        orig_w = (K[..., 0, 2].abs() * 2).clamp_min(1.0)
        orig_h = (K[..., 1, 2].abs() * 2).clamp_min(1.0)
        sx, sy = width / orig_w, height / orig_h
        K[..., 0, 0] *= sx
        K[..., 0, 2] *= sx
        K[..., 1, 1] *= sy
        K[..., 1, 2] *= sy
    return K


@torch.no_grad()
def project_token_coordinates(
    source_depth: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    latent_hw: tuple[int, int],
    source_conf: torch.Tensor | None = None,
    conf_threshold: float = 0.0,
    min_depth: float = 1e-4,
    depth_scale: float = 1.0,
    vae_spatial_scale: int = 16,
    camera_convention: str = "opencv",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project source latent-cell centers into target latent coordinates."""
    h_lat, w_lat = (int(latent_hw[0]), int(latent_hw[1]))
    if device is None:
        device = source_depth.device
    dtype = torch.float32
    depth = F.interpolate(
        source_depth[None, None].to(device=device, dtype=dtype),
        size=(h_lat, w_lat),
        mode="nearest",
    )[0, 0] * float(depth_scale)
    conf = None
    if source_conf is not None:
        conf = F.interpolate(
            source_conf[None, None].to(device=device, dtype=dtype),
            size=(h_lat, w_lat),
            mode="nearest",
        )[0, 0]
    ys, xs = torch.meshgrid(
        torch.arange(h_lat, device=device, dtype=dtype),
        torch.arange(w_lat, device=device, dtype=dtype),
        indexing="ij",
    )
    h_img, w_img = h_lat * int(vae_spatial_scale), w_lat * int(vae_spatial_scale)
    x_pix = (xs + 0.5) * w_img / w_lat - 0.5
    y_pix = (ys + 0.5) * h_img / h_lat - 0.5
    source_K = _scale_intrinsics(source_K.to(device=device, dtype=dtype), h_img, w_img)
    target_K = _scale_intrinsics(target_K.to(device=device, dtype=dtype), h_img, w_img)
    source_w2c = source_w2c.to(device=device, dtype=dtype)
    target_w2c = target_w2c.to(device=device, dtype=dtype)
    z_src = depth.reshape(-1)
    x_norm = ((x_pix - source_K[0, 2]) / source_K[0, 0]).reshape(-1)
    y_norm = ((y_pix - source_K[1, 2]) / source_K[1, 1]).reshape(-1)
    if camera_convention == "opencv":
        forward_axis, horizontal_axis, vertical_axis, vertical_sign = 2, 0, 1, 1.0
    elif camera_convention == "unreal":
        forward_axis, horizontal_axis, vertical_axis, vertical_sign = 0, 1, 2, -1.0
    else:
        raise ValueError(f"Unsupported camera convention: {camera_convention}")
    points_cam = torch.zeros((h_lat * w_lat, 3), device=device, dtype=dtype)
    points_cam[:, forward_axis] = z_src
    points_cam[:, horizontal_axis] = x_norm * z_src
    points_cam[:, vertical_axis] = y_norm * vertical_sign * z_src
    points_world = (
        torch.cat(
            [points_cam, torch.ones((len(points_cam), 1), device=device, dtype=dtype)],
            dim=-1,
        )
        @ torch.linalg.inv(source_w2c).T
    )
    points_target = (points_world @ target_w2c.T)[:, :3]
    z_tgt = points_target[:, forward_axis]
    denom = z_tgt.clamp_min(float(min_depth))
    u_img = target_K[0, 0] * points_target[:, horizontal_axis] / denom + target_K[0, 2]
    v_img = (
        target_K[1, 1] * points_target[:, vertical_axis] / denom * vertical_sign
        + target_K[1, 2]
    )
    u_lat = u_img / max(w_img - 1, 1) * max(w_lat - 1, 1)
    v_lat = v_img / max(h_img - 1, 1) * max(h_lat - 1, 1)
    valid = (z_src > min_depth) & (z_tgt > min_depth)
    valid &= (u_lat >= 0) & (u_lat <= w_lat - 1) & (v_lat >= 0) & (v_lat <= h_lat - 1)
    if conf is not None:
        valid &= conf.reshape(-1) > float(conf_threshold)
    return u_lat, v_lat, z_tgt, valid
