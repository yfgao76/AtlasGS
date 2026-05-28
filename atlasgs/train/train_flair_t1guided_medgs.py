import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F


def add_medgs_to_path(medgs_root):
    medgs_root = Path(medgs_root).resolve()
    sys.path.insert(0, str(medgs_root))
    return medgs_root


def find_latest_iteration(model_path):
    pc_root = Path(model_path) / "point_cloud"
    candidates = []
    for item in pc_root.glob("iteration_*"):
        try:
            it = int(item.name.split("_")[-1])
            candidates.append((it, item))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* found in {pc_root}")
    candidates.sort()
    return candidates[-1][0]


def resolve_model_ply(model_path, model_iter):
    if model_iter is None or model_iter == "latest":
        model_iter = find_latest_iteration(model_path)
    ply = Path(model_path) / "point_cloud" / f"iteration_{model_iter}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Model PLY not found: {ply}")
    return ply, int(model_iter)


def resolve_t1_ply(t1_model_path, t1_iter):
    return resolve_model_ply(t1_model_path, t1_iter)


def make_dataset_args(source_path, model_path, sh_degree, poly_degree, camera, distance, num_pts, white_background):
    return argparse.Namespace(
        sh_degree=sh_degree,
        source_path=os.path.abspath(source_path),
        model_path=os.path.abspath(model_path),
        images="images",
        depths="",
        resolution=-1,
        white_background=white_background,
        train_test_exp=False,
        data_device="cuda",
        eval=False,
        gs_type="gs",
        camera=camera,
        distance=distance,
        num_pts=num_pts,
        poly_degree=poly_degree,
    )


def write_cfg_args(model_path, dataset_args):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    cfg = argparse.Namespace(**vars(dataset_args))
    with open(model_path / "cfg_args", "w", encoding="utf-8") as f:
        f.write(str(cfg))


def freeze_params(gaussians, freeze_xyz=True, freeze_cov=True, freeze_time=True):
    if freeze_xyz:
        gaussians._xyz.requires_grad_(False)
    if freeze_cov:
        gaussians._scaling.requires_grad_(False)
        gaussians._rotation.requires_grad_(False)
    if freeze_time:
        gaussians.m.requires_grad_(False)
        gaussians.sigma.requires_grad_(False)
        gaussians._w1.requires_grad_(False)
        if hasattr(gaussians, "time_func"):
            try:
                gaussians.time_func.requires_grad_(False)
            except Exception:
                pass


def reset_time_func(gaussians, frames):
    if hasattr(gaussians, "time_func"):
        gaussians.time_func = torch.ones(frames, device="cuda") / frames


def apply_time_init_linear_map(gaussians, scale, shift, eps=1e-4):
    scale = float(scale)
    shift = float(shift)
    if abs(scale - 1.0) < 1e-8 and abs(shift) < 1e-8:
        return
    with torch.no_grad():
        m = torch.sigmoid(gaussians.m)
        m = scale * m + shift
        m = m.clamp(float(eps), 1.0 - float(eps))
        gaussians.m.data = torch.logit(m)
        if abs(scale) < 1e-6:
            return
        chunks = torch.chunk(gaussians._w1.data, chunks=gaussians.polynomial_degree, dim=-1)
        scaled = [w / (scale ** (idx + 1)) for idx, w in enumerate(chunks)]
        gaussians._w1.data = torch.cat(scaled, dim=-1)


def reset_appearance(gaussians):
    gaussians._features_dc.data.zero_()
    gaussians._features_rest.data.zero_()


def apply_preprune(gaussians, keep_ratio=1.0, min_opacity=0.0, min_keep=50_000):
    n_total = int(gaussians._xyz.shape[0])
    keep_ratio = float(keep_ratio)
    min_opacity = float(min_opacity)
    min_keep = int(min_keep)
    if n_total == 0:
        return 0, 0
    if keep_ratio >= 1.0 and min_opacity <= 0.0:
        return n_total, n_total

    with torch.no_grad():
        opacity = gaussians.get_opacity.squeeze()
        scales = gaussians.get_scaling[:, [0, 2]]
        size_score = torch.sqrt(scales[:, 0] * scales[:, 1] + 1e-12)
        score = opacity * size_score

        n_target = int(round(n_total * keep_ratio))
        n_target = max(1, min(n_total, max(min_keep, n_target)))

        valid_idx = torch.where(opacity >= min_opacity)[0]
        if valid_idx.numel() >= n_target:
            keep_local = torch.topk(score[valid_idx], k=n_target, largest=True).indices
            keep_idx = valid_idx[keep_local]
        else:
            keep_idx = torch.topk(score, k=n_target, largest=True).indices
        keep_idx = keep_idx.sort().values

        def subset(param):
            data = param.detach()[keep_idx].clone()
            return torch.nn.Parameter(data.requires_grad_(True))

        gaussians._xyz = subset(gaussians._xyz)
        gaussians._features_dc = subset(gaussians._features_dc)
        gaussians._features_rest = subset(gaussians._features_rest)
        gaussians._scaling = subset(gaussians._scaling)
        gaussians._rotation = subset(gaussians._rotation)
        gaussians._opacity = subset(gaussians._opacity)
        gaussians.m = subset(gaussians.m)
        gaussians.sigma = subset(gaussians.sigma)
        gaussians._w1 = subset(gaussians._w1)
        gaussians.max_radii2D = torch.zeros((keep_idx.numel(),), device=gaussians._xyz.device)

    return n_total, int(keep_idx.numel())


def ensure_state_size(vec, n, device=None):
    n = int(n)
    if vec is None:
        return torch.zeros((n,), device=device if device is not None else "cuda", dtype=torch.float32)
    if vec.shape[0] == n:
        return vec
    if vec.shape[0] > n:
        return vec[:n].contiguous()
    pad_n = n - vec.shape[0]
    return torch.cat([vec, torch.zeros((pad_n,), device=vec.device, dtype=vec.dtype)], dim=0)


def ensure_ref_size(ref, tensor):
    n = int(tensor.shape[0])
    if ref.shape[0] == n:
        return ref
    if ref.shape[0] > n:
        return ref[:n].contiguous()
    pad = tensor[ref.shape[0] :].detach().clone()
    return torch.cat([ref, pad], dim=0)


def compute_t1_gradients(t1_vol):
    t1_vol = np.nan_to_num(t1_vol, nan=0.0, posinf=0.0, neginf=0.0)
    gx = np.gradient(t1_vol, axis=0)
    gy = np.gradient(t1_vol, axis=1)
    gz = np.gradient(t1_vol, axis=2)
    gradmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    gradmag = np.nan_to_num(gradmag, nan=0.0, posinf=0.0, neginf=0.0)
    return gx, gy, gradmag


def normalize_slice_weight(gradmag_slice, p=95.0, gamma=2.0):
    if not np.isfinite(gradmag_slice).all():
        gradmag_slice = np.nan_to_num(gradmag_slice, nan=0.0, posinf=0.0, neginf=0.0)
    thresh = np.percentile(gradmag_slice, p)
    if thresh <= 0:
        return np.zeros_like(gradmag_slice, dtype=np.float32)
    w = np.clip(gradmag_slice / (thresh + 1e-8), 0, 1)
    if gamma != 1.0:
        w = w ** gamma
    return w.astype(np.float32)


def to_gray(img):
    if img.shape[0] == 1:
        return img
    return img.mean(dim=0, keepdim=True)


def grad2d(img):
    dx = img[:, :, 1:] - img[:, :, :-1]
    dy = img[:, 1:, :] - img[:, :-1, :]
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    return dx, dy


def compute_normals_from_cov(gaussians):
    rot = gaussians.get_rotation  # (N,4) quat
    scaling = gaussians.get_scaling  # (N,3)
    # build rotation matrix from quaternion (w,x,y,z)
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )  # (N,3,3)
    # normal is axis with smallest scale
    min_idx = torch.argmin(scaling, dim=1)  # (N,)
    normals = R[torch.arange(R.shape[0], device=R.device), :, min_idx]
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    return normals


def normal_alignment_loss(gaussians, k=8, sigma=2.0, sample_n=20000):
    xyz = gaussians.get_xyz
    n_pts = xyz.shape[0]
    if n_pts < 2:
        return torch.tensor(0.0, device=xyz.device)
    if sample_n < n_pts:
        idx = torch.randperm(n_pts, device=xyz.device)[:sample_n]
        xyz = xyz[idx]
    normals = compute_normals_from_cov(gaussians)
    if not torch.isfinite(normals).all():
        return torch.tensor(0.0, device=xyz.device)
    if sample_n < n_pts:
        normals = normals[idx]
    # pairwise distances (sampled)
    dist = torch.cdist(xyz, xyz)
    if not torch.isfinite(dist).all():
        return torch.tensor(0.0, device=xyz.device)
    k = min(k + 1, dist.shape[1])
    knn = torch.topk(dist, k=k, largest=False).indices[:, 1:]  # skip self
    n_i = normals.unsqueeze(1).expand(-1, knn.shape[1], -1)
    n_j = normals[knn]
    dot = (n_i * n_j).sum(dim=-1)
    align = 1.0 - dot.pow(2)
    align = torch.nan_to_num(align, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.exp(-(dist.gather(1, knn) ** 2) / (2 * sigma ** 2))
    loss = (w * align).mean()
    if not torch.isfinite(loss):
        return torch.tensor(0.0, device=xyz.device)
    return loss


def build_optimizer(gaussians, feature_lr, opacity_lr, train_opacity, cov_lr, xyz_lr, time_lr):
    xyz_group_lr = float(xyz_lr) if gaussians._xyz.requires_grad else 0.0
    fdc_group_lr = float(feature_lr) if gaussians._features_dc.requires_grad else 0.0
    frest_group_lr = float(feature_lr / 20.0) if gaussians._features_rest.requires_grad else 0.0
    scaling_group_lr = float(cov_lr) if gaussians._scaling.requires_grad else 0.0
    rotation_group_lr = float(cov_lr) if gaussians._rotation.requires_grad else 0.0
    opacity_group_lr = float(opacity_lr) if (train_opacity and gaussians._opacity.requires_grad) else 0.0
    m_group_lr = float(time_lr) if gaussians.m.requires_grad else 0.0
    sigma_group_lr = float(time_lr) if gaussians.sigma.requires_grad else 0.0
    w1_group_lr = float(time_lr) if gaussians._w1.requires_grad else 0.0
    params = []

    def maybe_add(param, lr, name):
        # Optimizer cannot hold non-leaf tensors (e.g., derived appearance tensors).
        if not isinstance(param, torch.Tensor):
            return
        if not param.is_leaf:
            return
        params.append({"params": [param], "lr": float(lr), "name": name})

    maybe_add(gaussians._xyz, xyz_group_lr, "xyz")
    maybe_add(gaussians._features_dc, fdc_group_lr, "f_dc")
    maybe_add(gaussians._features_rest, frest_group_lr, "f_rest")
    maybe_add(gaussians._opacity, opacity_group_lr, "opacity")
    maybe_add(gaussians._scaling, scaling_group_lr, "scaling")
    maybe_add(gaussians._rotation, rotation_group_lr, "rotation")
    maybe_add(gaussians.m, m_group_lr, "m")
    maybe_add(gaussians.sigma, sigma_group_lr, "sigma")
    maybe_add(gaussians._w1, w1_group_lr, "w1")

    if not params:
        raise RuntimeError("No valid leaf parameters found for optimizer.")
    optim = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    return optim


def regrow_clone_points(
    gaussians,
    add_ratio,
    max_points,
    opacity_init,
    jitter_scale,
    scale_shrink,
    topk_ratio,
    zero_features,
    chosen_idx=None,
):
    n_total = int(gaussians._xyz.shape[0])
    max_points = int(max_points)
    if n_total <= 0 or max_points <= n_total:
        return 0
    add_ratio = float(add_ratio)
    if add_ratio <= 0:
        return 0

    n_add = int(round(n_total * add_ratio))
    n_add = max(1, n_add)
    n_add = min(n_add, max_points - n_total)
    if n_add <= 0:
        return 0

    req = {
        "xyz": gaussians._xyz.requires_grad,
        "f_dc": gaussians._features_dc.requires_grad,
        "f_rest": gaussians._features_rest.requires_grad,
        "scaling": gaussians._scaling.requires_grad,
        "rotation": gaussians._rotation.requires_grad,
        "opacity": gaussians._opacity.requires_grad,
        "m": gaussians.m.requires_grad,
        "sigma": gaussians.sigma.requires_grad,
        "w1": gaussians._w1.requires_grad,
    }

    with torch.no_grad():
        opacity = gaussians.get_opacity.squeeze()
        scales = gaussians.get_scaling[:, [0, 2]]
        score = opacity * torch.sqrt(scales[:, 0] * scales[:, 1] + 1e-12)
        if chosen_idx is None:
            pool = int(round(n_total * float(topk_ratio)))
            pool = max(n_add, pool)
            pool = min(pool, n_total)
            top_idx = torch.topk(score, k=pool, largest=True).indices
            chosen = top_idx[torch.randint(0, pool, (n_add,), device=top_idx.device)]
        else:
            candidate = chosen_idx.long().to(score.device)
            candidate = candidate[(candidate >= 0) & (candidate < n_total)]
            if candidate.numel() == 0:
                pool = int(round(n_total * float(topk_ratio)))
                pool = max(n_add, pool)
                pool = min(pool, n_total)
                top_idx = torch.topk(score, k=pool, largest=True).indices
                chosen = top_idx[torch.randint(0, pool, (n_add,), device=top_idx.device)]
            else:
                if candidate.numel() < n_add:
                    fill_idx = torch.topk(score, k=min(n_total, n_add), largest=True).indices
                    candidate = torch.unique(torch.cat([candidate, fill_idx], dim=0))
                perm = torch.randperm(candidate.numel(), device=candidate.device)[:n_add]
                chosen = candidate[perm]

        src_xyz = gaussians._xyz.detach()[chosen]
        src_scaling_world = gaussians.get_scaling[chosen].detach()
        noise = torch.randn_like(src_xyz)
        noise[:, 1] = 0
        new_xyz = src_xyz + noise * src_scaling_world * float(jitter_scale)

        new_scaling = gaussians._scaling.detach()[chosen].clone()
        if scale_shrink is not None and float(scale_shrink) > 0 and float(scale_shrink) != 1.0:
            new_scaling = new_scaling + math.log(float(scale_shrink))
        new_rotation = gaussians._rotation.detach()[chosen].clone()

        if zero_features:
            new_features_dc = torch.zeros_like(gaussians._features_dc.detach()[chosen])
            new_features_rest = torch.zeros_like(gaussians._features_rest.detach()[chosen])
        else:
            new_features_dc = gaussians._features_dc.detach()[chosen].clone()
            new_features_rest = gaussians._features_rest.detach()[chosen].clone()

        if opacity_init is None:
            new_opacity = gaussians._opacity.detach()[chosen].clone()
        else:
            init_val = max(min(float(opacity_init), 0.99), 1e-4)
            new_opacity = torch.logit(torch.full_like(gaussians._opacity.detach()[chosen], init_val))

        new_m = gaussians.m.detach()[chosen].clone()
        new_sigma = gaussians.sigma.detach()[chosen].clone()
        new_w1 = gaussians._w1.detach()[chosen].clone()

        if getattr(gaussians, "optimizer", None) is not None:
            gaussians.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                new_m,
                new_sigma,
                new_w1,
            )
        else:
            # Densification can run before optimizer construction in some branches.
            gaussians._xyz = torch.nn.Parameter(
                torch.cat((gaussians._xyz.detach(), new_xyz), dim=0), requires_grad=req["xyz"]
            )
            gaussians._features_dc = torch.nn.Parameter(
                torch.cat((gaussians._features_dc.detach(), new_features_dc), dim=0), requires_grad=req["f_dc"]
            )
            gaussians._features_rest = torch.nn.Parameter(
                torch.cat((gaussians._features_rest.detach(), new_features_rest), dim=0), requires_grad=req["f_rest"]
            )
            gaussians._opacity = torch.nn.Parameter(
                torch.cat((gaussians._opacity.detach(), new_opacity), dim=0), requires_grad=req["opacity"]
            )
            gaussians._scaling = torch.nn.Parameter(
                torch.cat((gaussians._scaling.detach(), new_scaling), dim=0), requires_grad=req["scaling"]
            )
            gaussians._rotation = torch.nn.Parameter(
                torch.cat((gaussians._rotation.detach(), new_rotation), dim=0), requires_grad=req["rotation"]
            )
            gaussians.m = torch.nn.Parameter(
                torch.cat((gaussians.m.detach(), new_m), dim=0), requires_grad=req["m"]
            )
            gaussians.sigma = torch.nn.Parameter(
                torch.cat((gaussians.sigma.detach(), new_sigma), dim=0), requires_grad=req["sigma"]
            )
            gaussians._w1 = torch.nn.Parameter(
                torch.cat((gaussians._w1.detach(), new_w1), dim=0), requires_grad=req["w1"]
            )

            n_pts = int(gaussians._xyz.shape[0])
            device = gaussians._xyz.device
            gaussians.xyz_gradient_accum = torch.zeros((n_pts, 1), device=device)
            gaussians.denom = torch.zeros((n_pts, 1), device=device)
            gaussians.m_gradient_accum = torch.zeros((n_pts, 1), device=device)
            gaussians.m_denom = torch.zeros((n_pts, 1), device=device)
            gaussians.max_radii2D = torch.zeros((n_pts,), device=device)

        gaussians._xyz.requires_grad_(req["xyz"])
        gaussians._features_dc.requires_grad_(req["f_dc"])
        gaussians._features_rest.requires_grad_(req["f_rest"])
        gaussians._scaling.requires_grad_(req["scaling"])
        gaussians._rotation.requires_grad_(req["rotation"])
        gaussians._opacity.requires_grad_(req["opacity"])
        gaussians.m.requires_grad_(req["m"])
        gaussians.sigma.requires_grad_(req["sigma"])
        gaussians._w1.requires_grad_(req["w1"])

    return n_add


def compute_temporal_means3d(gaussians, camera_time, alpha=0.0, interp=1, interp_idx=0):
    xyz = gaussians.get_xyz
    time_func = gaussians.get_time
    time = torch.sum(time_func[:camera_time]).repeat(xyz.shape[0], 1)
    time_next = torch.sum(time_func[:camera_time + 1]).repeat(xyz.shape[0], 1)
    if alpha != 0:
        time = time + (time_next - time) * alpha
    else:
        time = time + (time_next - time) * interp_idx / max(interp, 1)
    poly_weights = torch.chunk(gaussians._w1, chunks=gaussians.polynomial_degree, dim=-1)
    means = xyz[:, [0, -1]]
    center = gaussians.get_m - time[0]
    for idx, poly_weight in enumerate(poly_weights):
        means = means + poly_weight * (center ** (idx + 1))
    means = torch.cat(
        [
            means[:, 0].unsqueeze(1),
            torch.zeros_like(means[:, 0]).unsqueeze(1),
            means[:, -1].unsqueeze(1),
        ],
        dim=1,
    )
    return means


def project_to_image_pixels(means3d, camera):
    ones = torch.ones((means3d.shape[0], 1), device=means3d.device, dtype=means3d.dtype)
    points_h = torch.cat([means3d, ones], dim=1)
    clip = torch.matmul(points_h, camera.full_proj_transform.unsqueeze(0)).squeeze(0)
    w = clip[:, 3:4].clamp_min(1e-8)
    ndc = clip[:, :3] / w
    width = int(camera.image_width)
    height = int(camera.image_height)
    x = ((ndc[:, 0] + 1.0) * 0.5) * max(width - 1, 1)
    y = ((1.0 - ndc[:, 1]) * 0.5) * max(height - 1, 1)
    return x, y


def accumulate_residual_votes(
    residual_votes,
    gaussians,
    camera,
    residual_map,
    visibility_filter,
    top_pct,
):
    with torch.no_grad():
        n_pts = gaussians._xyz.shape[0]
        if residual_votes is None or residual_votes.shape[0] != n_pts:
            residual_votes = torch.zeros((n_pts,), device=gaussians._xyz.device, dtype=torch.float32)
        if residual_map.ndim == 3:
            residual_map = residual_map.mean(dim=0)
        residual_map = residual_map.float()
        flat = residual_map.reshape(-1)
        if flat.numel() == 0:
            return residual_votes
        keep = max(min(float(top_pct), 100.0), 0.01)
        q = 1.0 - keep / 100.0
        thresh = torch.quantile(flat, q=q)
        focus = torch.clamp(residual_map - thresh, min=0.0)
        if focus.max() <= 0:
            return residual_votes

        means3d = compute_temporal_means3d(gaussians, camera.time)
        px, py = project_to_image_pixels(means3d, camera)
        xi = px.round().long()
        yi = py.round().long()
        h, w = focus.shape[-2], focus.shape[-1]
        valid = visibility_filter.clone()
        valid = valid & torch.isfinite(px) & torch.isfinite(py)
        valid = valid & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        idx = torch.where(valid)[0]
        if idx.numel() == 0:
            return residual_votes
        sampled = focus[yi[idx], xi[idx]]
        residual_votes[idx] += sampled
    return residual_votes


def pick_indices_from_votes(votes, n_add, topk_ratio):
    if votes is None or votes.numel() == 0:
        return None
    n_total = votes.shape[0]
    n_add = max(1, min(int(n_add), n_total))
    pool = int(round(n_total * float(topk_ratio)))
    pool = max(n_add, pool)
    pool = min(pool, n_total)
    top_idx = torch.topk(votes, k=pool, largest=True).indices
    top_votes = votes[top_idx]
    top_idx = top_idx[top_votes > 0]
    if top_idx.numel() == 0:
        return None
    if top_idx.numel() <= n_add:
        return top_idx
    perm = torch.randperm(top_idx.numel(), device=top_idx.device)[:n_add]
    return top_idx[perm]


def tiny_blob_loss(gaussians, scale_c):
    scales = gaussians.get_scaling[:, [0, 2]]
    sx = scales[:, 0]
    sz = scales[:, 1]
    opacity = gaussians.get_opacity.squeeze()
    c = max(float(scale_c), 1e-6)
    tiny_mask = torch.exp(-sx / c) * torch.exp(-sz / c)
    return (opacity * tiny_mask).mean()


def slab_alpha_schedule(num_samples):
    num_samples = max(1, int(num_samples))
    if num_samples == 1:
        return [0.0]
    step = 1.0 / float(num_samples)
    return [float((i + 0.5) * step) for i in range(num_samples)]


def render_view_with_slab_forward(render_fn, camera, gaussians, pipe, bg, iteration, train, alphas):
    if len(alphas) == 1 and float(alphas[0]) == 0.0:
        pkg = render_fn(camera, gaussians, pipe, bg, train=train, iter=iteration, alpha=0.0)
        return pkg["render"], pkg["visibility_filter"].detach(), pkg["radii"].detach()

    pred_sum = None
    vis_union = None
    radii_max = None
    for alpha in alphas:
        pkg = render_fn(camera, gaussians, pipe, bg, train=train, iter=iteration, alpha=float(alpha))
        pred = pkg["render"]
        pred_sum = pred if pred_sum is None else (pred_sum + pred)
        vis = pkg["visibility_filter"].detach()
        vis_union = vis if vis_union is None else (vis_union | vis)
        rad = pkg["radii"].detach()
        radii_max = rad if radii_max is None else torch.maximum(radii_max, rad)

    pred_avg = pred_sum / float(len(alphas))
    return pred_avg, vis_union, radii_max


def parse_int_list(text, default_values):
    if text is None:
        return list(default_values)
    vals = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals if vals else list(default_values)


def _lpvi_collect_intervals(simplex_tree, max_dim=2):
    intervals = []
    for dim in range(int(max_dim) + 1):
        try:
            part = simplex_tree.persistence_intervals_in_dimension(dim)
        except Exception:
            part = []
        if part is None or len(part) == 0:
            continue
        arr = np.asarray(part, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        if arr.size == 0:
            continue
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        if arr.size > 0:
            intervals.append(arr)
    if not intervals:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(intervals)


def lpvi_topological_similarity_measurement(
    points1,
    points2,
    complex_mode="rips",
    dist_model="W",
    max_edge_length=2.0,
):
    if points1.shape[0] < 2 or points2.shape[0] < 2:
        return np.inf
    if complex_mode != "rips":
        raise NotImplementedError(f"Unsupported complex_mode: {complex_mode}")
    try:
        import gudhi
        from gudhi.wasserstein import wasserstein_distance
    except Exception:
        return np.inf

    try:
        rips1 = gudhi.RipsComplex(points=points1, max_edge_length=float(max_edge_length))
        simplex_tree1 = rips1.create_simplex_tree(max_dimension=2)
        simplex_tree1.compute_persistence(homology_coeff_field=2, min_persistence=0)
        diag1 = _lpvi_collect_intervals(simplex_tree1, max_dim=2)

        rips2 = gudhi.RipsComplex(points=points2, max_edge_length=float(max_edge_length))
        simplex_tree2 = rips2.create_simplex_tree(max_dimension=2)
        simplex_tree2.compute_persistence(homology_coeff_field=2, min_persistence=0)
        diag2 = _lpvi_collect_intervals(simplex_tree2, max_dim=2)

        if diag1.shape[0] == 0 and diag2.shape[0] == 0:
            return 0.0

        if dist_model == "W":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cardinality of essential parts differs.",
                    category=UserWarning,
                )
                return float(
                    wasserstein_distance(diag1, diag2, matching=False, order=1, internal_p=2)
                )
        if dist_model == "B":
            return float(gudhi.bottleneck_distance(diag1, diag2))
        raise ValueError(f"Unknown dist_model: {dist_model}")
    except Exception:
        return np.inf


class MRIPersistLoss(torch.nn.Module):
    """Persistence-style loss adapted for MRI slices.

    If topologylayer is available, it computes topological barcodes on a
    point-cloud embedding [x, y, intensity]. Otherwise, it falls back to a
    differentiable pseudo-persistence signature over soft level sets.
    """

    def __init__(
        self,
        dims=None,
        ks=None,
        downsample=0.125,
        use_spatial=True,
        spatial_weight=0.35,
        intensity_weight=1.0,
        threshold_count=8,
        sigmoid_tau=0.05,
        fg_eps=0.02,
        min_fg_ratio=0.01,
        min_fg_points=8,
        min_std=0.01,
        skip_lowinfo=True,
    ):
        super().__init__()
        self.dims = list(dims if dims is not None else [0, 1])
        self.ks = list(ks if ks is not None else [64, 32])
        if len(self.dims) != len(self.ks):
            raise ValueError(f"persist dims/ks length mismatch: dims={self.dims}, ks={self.ks}")
        self.downsample = float(max(min(downsample, 1.0), 1e-3))
        self.use_spatial = bool(use_spatial)
        self.spatial_weight = float(spatial_weight)
        self.intensity_weight = float(intensity_weight)
        self.sigmoid_tau = float(max(sigmoid_tau, 1e-4))
        self.fg_eps = float(max(fg_eps, 0.0))
        self.min_fg_ratio = float(max(min_fg_ratio, 0.0))
        self.min_fg_points = int(max(min_fg_points, 0))
        self.min_std = float(max(min_std, 0.0))
        self.skip_lowinfo = bool(skip_lowinfo)
        self.mse = torch.nn.MSELoss()
        self.register_buffer("thresholds", torch.linspace(0.1, 0.9, steps=max(2, int(threshold_count))))
        self.has_topologylayer = False
        self.topo_error = ""
        try:
            from topologylayer.nn.alpha import AlphaLayer  # type: ignore
            from topologylayer.nn.features import TopKBarcodeLengths  # type: ignore

            self.layer = AlphaLayer(maxdim=max(self.dims))
            for i, (dim, k) in enumerate(zip(self.dims, self.ks)):
                self.add_module(f"feature_{i}", TopKBarcodeLengths(int(dim), int(k)))
            self.has_topologylayer = True
        except Exception as exc:
            self.layer = None
            self.topo_error = str(exc)

    def _to_gray(self, x):
        if x.ndim == 2:
            gray = x.unsqueeze(0)
        elif x.ndim == 3:
            gray = x.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Expected [H,W] or [C,H,W], got shape={tuple(x.shape)}")
        gray = gray.clamp(0.0, 1.0)
        if self.downsample < 1.0:
            gray = F.interpolate(
                gray.unsqueeze(0),
                scale_factor=self.downsample,
                mode="area",
            ).squeeze(0)
        return gray[0]

    def _build_point_embedding(self, gray, fg_only=False):
        h, w = gray.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=h, device=gray.device, dtype=gray.dtype),
            torch.linspace(0.0, 1.0, steps=w, device=gray.device, dtype=gray.dtype),
            indexing="ij",
        )
        if self.use_spatial:
            pts = torch.stack(
                [
                    xx * self.spatial_weight,
                    yy * self.spatial_weight,
                    gray * self.intensity_weight,
                ],
                dim=-1,
            )
        else:
            zero = torch.zeros_like(gray)
            pts = torch.stack(
                [zero, zero, gray * self.intensity_weight],
                dim=-1,
            )
        if fg_only:
            fg = gray > self.fg_eps
            if fg.any():
                pts = pts[fg]
            else:
                pts = pts.reshape(0, 3)
        else:
            pts = pts.reshape(-1, 3)
        return pts

    def _slice_stats(self, gray):
        fg = gray > self.fg_eps
        fg_count = int(fg.sum().item())
        fg_ratio = float(fg.float().mean().item())
        if fg_count > 1:
            std = float(gray[fg].std(unbiased=False).item())
        else:
            std = float(gray.std(unbiased=False).item())
        return {
            "fg_count": fg_count,
            "fg_ratio": fg_ratio,
            "std": std,
        }

    def is_informative_gray(self, gray):
        st = self._slice_stats(gray)
        ok = (
            st["fg_count"] >= self.min_fg_points
            and st["fg_ratio"] >= self.min_fg_ratio
            and st["std"] >= self.min_std
        )
        return ok, st

    def is_informative(self, x):
        gray = self._to_gray(x)
        return self.is_informative_gray(gray)

    def _signature_topologylayer(self, gray):
        points = self._build_point_embedding(gray, fg_only=self.skip_lowinfo)
        if points.shape[0] < 4:
            raise ValueError("Too few points for topologylayer alpha complex.")
        diags = self.layer(points.cpu())
        feats = []
        for i in range(len(self.dims)):
            feat = self.__getattr__(f"feature_{i}")(diags)
            feats.append(feat.reshape(-1).to(gray.device, dtype=gray.dtype))
        return torch.cat(feats, dim=0)

    def _signature_fallback(self, gray):
        thr = self.thresholds.to(gray.device, dtype=gray.dtype).view(-1, 1, 1)
        occ = torch.sigmoid((gray.unsqueeze(0) - thr) / self.sigmoid_tau)
        area = occ.mean(dim=(1, 2))

        dx = occ[:, :, 1:] - occ[:, :, :-1]
        dy = occ[:, 1:, :] - occ[:, :-1, :]
        boundary = dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2))

        lap = -4.0 * occ
        lap[:, :, 1:] += occ[:, :, :-1]
        lap[:, :, :-1] += occ[:, :, 1:]
        lap[:, 1:, :] += occ[:, :-1, :]
        lap[:, :-1, :] += occ[:, 1:, :]
        hole_proxy = torch.relu(lap).mean(dim=(1, 2))

        return torch.cat([area, boundary, hole_proxy], dim=0)

    def signature_with_mode(self, x, detach=False):
        gray = self._to_gray(x)
        if self.skip_lowinfo:
            info_ok, _ = self.is_informative_gray(gray)
            if not info_ok:
                sig = self._signature_fallback(gray)
                if detach:
                    sig = sig.detach()
                return sig, "fallback"
        mode = "fallback"
        if self.has_topologylayer:
            try:
                sig = self._signature_topologylayer(gray)
                mode = "topology"
            except Exception:
                sig = self._signature_fallback(gray)
                mode = "fallback"
        else:
            sig = self._signature_fallback(gray)
            mode = "fallback"
        if detach:
            sig = sig.detach()
        return sig, mode

    def signature(self, x, detach=False):
        sig, _ = self.signature_with_mode(x, detach=detach)
        return sig

    def signature_pair(self, x):
        """Precompute both topology/fallback signatures for stable matching."""
        gray = self._to_gray(x)
        topo_sig = None
        info_ok, _ = self.is_informative_gray(gray)
        if self.has_topologylayer and (not self.skip_lowinfo or info_ok):
            try:
                topo_sig = self._signature_topologylayer(gray).detach()
            except Exception:
                topo_sig = None
        fallback_sig = self._signature_fallback(gray).detach()
        return {"topology": topo_sig, "fallback": fallback_sig}

    def forward(self, pred, gt=None, gt_signature=None):
        sig_pred, pred_mode = self.signature_with_mode(pred, detach=False)
        if gt_signature is None:
            if gt is None:
                raise ValueError("Persist loss requires either gt or gt_signature.")
            sig_gt = self.signature(gt, detach=True)
        elif isinstance(gt_signature, dict):
            sig_gt = None
            if pred_mode == "topology":
                sig_gt = gt_signature.get("topology", None)
            if sig_gt is None:
                sig_gt = gt_signature.get("fallback", None)
            if sig_gt is None:
                if gt is None:
                    raise ValueError("Persist loss dict signature missing and gt is None.")
                sig_gt = self.signature(gt, detach=True)
            else:
                sig_gt = sig_gt.detach().to(sig_pred.device, dtype=sig_pred.dtype)
        else:
            sig_gt = gt_signature.detach().to(sig_pred.device, dtype=sig_pred.dtype)

        if sig_pred.shape != sig_gt.shape:
            # Keep topologylayer when possible, but hard-fallback safely if a branch fails.
            pred_fallback = self._signature_fallback(self._to_gray(pred))
            if isinstance(gt_signature, dict) and gt_signature.get("fallback", None) is not None:
                gt_fallback = gt_signature["fallback"].detach().to(
                    pred_fallback.device, dtype=pred_fallback.dtype
                )
            elif gt is not None:
                gt_fallback = self._signature_fallback(self._to_gray(gt)).detach().to(
                    pred_fallback.device, dtype=pred_fallback.dtype
                )
            else:
                m = min(int(sig_pred.numel()), int(sig_gt.numel()))
                return self.mse(sig_pred.reshape(-1)[:m], sig_gt.reshape(-1)[:m])
            return self.mse(pred_fallback, gt_fallback)
        return self.mse(sig_pred, sig_gt)


def lpvi_augment_gaussians(
    gaussians,
    anchor_samples=2048,
    k_max=8,
    k_min=4,
    threshold=0.25,
    max_new_points=120000,
    min_opacity=0.01,
    max_vertices_per_anchor=4,
    scale_shrink=0.9,
    use_topology_check=True,
    dist_model="W",
    complex_mode="rips",
    complex_max_edge=2.0,
    seed=0,
):
    """Topology-GS LPVI-style interpolation on (x, z, t) point clouds."""
    try:
        from scipy.spatial import Voronoi, cKDTree
    except Exception:
        return 0, "scipy_missing"

    n_total = int(gaussians._xyz.shape[0])
    if n_total <= 1 or max_new_points <= 0:
        return 0, "no_points"

    req = {
        "xyz": gaussians._xyz.requires_grad,
        "f_dc": gaussians._features_dc.requires_grad,
        "f_rest": gaussians._features_rest.requires_grad,
        "scaling": gaussians._scaling.requires_grad,
        "rotation": gaussians._rotation.requires_grad,
        "opacity": gaussians._opacity.requires_grad,
        "m": gaussians.m.requires_grad,
        "sigma": gaussians.sigma.requires_grad,
        "w1": gaussians._w1.requires_grad,
    }

    with torch.no_grad():
        xyz = gaussians._xyz.detach().cpu().numpy()
        t_center = gaussians.get_m.detach().cpu().numpy().reshape(-1, 1)
        xzt = np.concatenate([xyz[:, [0, 2]], t_center], axis=1).astype(np.float64)
        opacity = gaussians.get_opacity.squeeze().detach().cpu().numpy()

        valid = np.where(opacity >= float(min_opacity))[0]
        if valid.size == 0:
            valid = np.arange(n_total, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        n_anchor = int(min(max(1, anchor_samples), valid.size))
        anchor_idx = rng.choice(valid, size=n_anchor, replace=False)
        tree = cKDTree(xzt)
        visited = np.zeros((n_total,), dtype=np.bool_)

        k_max = int(max(3, min(k_max, n_total - 1)))
        k_min = int(max(2, min(k_min, k_max)))
        max_vertices_per_anchor = int(max(1, max_vertices_per_anchor))
        thr = float(max(threshold, 1e-6))
        use_topology = bool(use_topology_check)
        if use_topology:
            try:
                import gudhi  # noqa: F401
                from gudhi.wasserstein import wasserstein_distance  # noqa: F401
            except Exception:
                use_topology = False

        new_points_xzt = []
        for idx in anchor_idx:
            idx = int(idx)
            if idx < 0 or idx >= n_total:
                continue
            if visited[idx]:
                continue
            visited[idx] = True
            anchor = xzt[idx]
            _, neigh_idx = tree.query(anchor, k=k_max + 1)
            neigh_idx = np.atleast_1d(neigh_idx).astype(np.int64)
            neigh_idx = neigh_idx[(neigh_idx >= 0) & (neigh_idx < n_total)]
            local = xzt[neigh_idx]
            if local.shape[0] < 4:
                continue

            cand = np.zeros((0, 3), dtype=np.float64)
            use_small_scope = True
            try:
                vor = Voronoi(local)
                vertices = vor.vertices.astype(np.float64)
            except Exception:
                vertices = np.zeros((0, 3), dtype=np.float64)
            if vertices.shape[0] > 0:
                if use_topology:
                    local_new = np.vstack([local, vertices])
                    similarity = lpvi_topological_similarity_measurement(
                        local,
                        local_new,
                        complex_mode=complex_mode,
                        dist_model=dist_model,
                        max_edge_length=complex_max_edge,
                    )
                    use_small_scope = not (np.isfinite(similarity) and similarity < thr)
                else:
                    use_small_scope = False
                if not use_small_scope:
                    cand = vertices
                    visited[neigh_idx] = True

            if use_small_scope:
                _, neigh_idx_small = tree.query(anchor, k=k_min + 1)
                neigh_idx_small = np.atleast_1d(neigh_idx_small).astype(np.int64)[1:]
                neigh_idx_small = neigh_idx_small[(neigh_idx_small >= 0) & (neigh_idx_small < n_total)]
                if neigh_idx_small.size == 0:
                    continue
                neighbors = xzt[neigh_idx_small]
                if neighbors.shape[0] < 2:
                    continue
                pts = np.vstack([anchor[None, :], neighbors])
                pts_center = pts - pts.mean(axis=0, keepdims=True)
                try:
                    _, _, vh = np.linalg.svd(pts_center, full_matrices=False)
                except Exception:
                    continue
                if vh.shape[0] < 2:
                    continue
                basis = vh[:2, :]  # [2,3]
                projected = pts_center @ basis.T  # [N,2]
                try:
                    vor2 = Voronoi(projected)
                    vertices2 = vor2.vertices.astype(np.float64)
                except Exception:
                    vertices2 = np.zeros((0, 2), dtype=np.float64)
                if vertices2.shape[0] == 0:
                    continue
                cand = vertices2 @ basis + anchor[None, :]
                visited[neigh_idx_small] = True

            if cand.ndim == 1:
                cand = cand[None, :]
            if cand.shape[0] == 0:
                continue
            finite = np.isfinite(cand).all(axis=1)
            cand = cand[finite]
            if cand.shape[0] == 0:
                continue

            local_dist = np.linalg.norm(local - anchor[None, :], axis=1)
            r_med = float(np.median(local_dist))
            max_r = max((1.0 + thr) * r_med, 1e-6)
            dist = np.linalg.norm(cand - anchor[None, :], axis=1)
            keep = np.isfinite(dist) & (dist <= max_r)
            cand = cand[keep]
            if cand.shape[0] == 0:
                continue

            lo = local.min(axis=0)
            hi = local.max(axis=0)
            margin = (hi - lo) * thr
            keep = np.all((cand >= (lo - margin)) & (cand <= (hi + margin)), axis=1)
            cand = cand[keep]
            if cand.shape[0] == 0:
                continue

            if cand.shape[0] > max_vertices_per_anchor:
                pick = rng.choice(cand.shape[0], size=max_vertices_per_anchor, replace=False)
                cand = cand[pick]

            new_points_xzt.append(cand)
            if sum(p.shape[0] for p in new_points_xzt) >= int(max_new_points):
                break

        if not new_points_xzt:
            return 0, "no_candidates"

        new_xzt = np.concatenate(new_points_xzt, axis=0)
        if new_xzt.shape[0] > int(max_new_points):
            keep = rng.choice(new_xzt.shape[0], size=int(max_new_points), replace=False)
            new_xzt = new_xzt[keep]

        # Deduplicate nearby points for numerical stability.
        rounded = np.round(new_xzt, decimals=5)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        new_xzt = new_xzt[unique_idx]
        if new_xzt.shape[0] == 0:
            return 0, "dedup_empty"

        _, src_idx = tree.query(new_xzt, k=1)
        src_idx = np.atleast_1d(src_idx).astype(np.int64)
        src = torch.from_numpy(src_idx).to(gaussians._xyz.device)

        new_xyz = torch.zeros((new_xzt.shape[0], 3), device=gaussians._xyz.device, dtype=gaussians._xyz.dtype)
        new_xyz[:, 0] = torch.from_numpy(new_xzt[:, 0]).to(new_xyz.device, dtype=new_xyz.dtype)
        new_xyz[:, 1] = 0.0
        new_xyz[:, 2] = torch.from_numpy(new_xzt[:, 1]).to(new_xyz.device, dtype=new_xyz.dtype)
        new_t = np.clip(new_xzt[:, 2], 1e-4, 1.0 - 1e-4).astype(np.float32)

        new_features_dc = gaussians._features_dc.detach()[src].clone()
        new_features_rest = gaussians._features_rest.detach()[src].clone()
        new_opacity = gaussians._opacity.detach()[src].clone()
        new_scaling = gaussians._scaling.detach()[src].clone()
        if scale_shrink is not None and float(scale_shrink) > 0 and float(scale_shrink) != 1.0:
            new_scaling = new_scaling + math.log(float(scale_shrink))
        new_rotation = gaussians._rotation.detach()[src].clone()
        new_m = torch.logit(
            torch.from_numpy(new_t).to(gaussians._xyz.device, dtype=gaussians.m.dtype).unsqueeze(1)
        )
        new_sigma = gaussians.sigma.detach()[src].clone()
        new_w1 = gaussians._w1.detach()[src].clone()

        if getattr(gaussians, "optimizer", None) is not None:
            gaussians.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                new_m,
                new_sigma,
                new_w1,
            )
        else:
            # LPVI can run before optimizer is built; append tensors manually in that case.
            gaussians._xyz = torch.nn.Parameter(
                torch.cat((gaussians._xyz.detach(), new_xyz), dim=0), requires_grad=req["xyz"]
            )
            gaussians._features_dc = torch.nn.Parameter(
                torch.cat((gaussians._features_dc.detach(), new_features_dc), dim=0), requires_grad=req["f_dc"]
            )
            gaussians._features_rest = torch.nn.Parameter(
                torch.cat((gaussians._features_rest.detach(), new_features_rest), dim=0), requires_grad=req["f_rest"]
            )
            gaussians._opacity = torch.nn.Parameter(
                torch.cat((gaussians._opacity.detach(), new_opacity), dim=0), requires_grad=req["opacity"]
            )
            gaussians._scaling = torch.nn.Parameter(
                torch.cat((gaussians._scaling.detach(), new_scaling), dim=0), requires_grad=req["scaling"]
            )
            gaussians._rotation = torch.nn.Parameter(
                torch.cat((gaussians._rotation.detach(), new_rotation), dim=0), requires_grad=req["rotation"]
            )
            gaussians.m = torch.nn.Parameter(
                torch.cat((gaussians.m.detach(), new_m), dim=0), requires_grad=req["m"]
            )
            gaussians.sigma = torch.nn.Parameter(
                torch.cat((gaussians.sigma.detach(), new_sigma), dim=0), requires_grad=req["sigma"]
            )
            gaussians._w1 = torch.nn.Parameter(
                torch.cat((gaussians._w1.detach(), new_w1), dim=0), requires_grad=req["w1"]
            )

            n_pts = int(gaussians._xyz.shape[0])
            device = gaussians._xyz.device
            gaussians.xyz_gradient_accum = torch.zeros((n_pts, 1), device=device)
            gaussians.denom = torch.zeros((n_pts, 1), device=device)
            gaussians.m_gradient_accum = torch.zeros((n_pts, 1), device=device)
            gaussians.m_denom = torch.zeros((n_pts, 1), device=device)
            gaussians.max_radii2D = torch.zeros((n_pts,), device=device)

        gaussians._xyz.requires_grad_(req["xyz"])
        gaussians._features_dc.requires_grad_(req["f_dc"])
        gaussians._features_rest.requires_grad_(req["f_rest"])
        gaussians._scaling.requires_grad_(req["scaling"])
        gaussians._rotation.requires_grad_(req["rotation"])
        gaussians._opacity.requires_grad_(req["opacity"])
        gaussians.m.requires_grad_(req["m"])
        gaussians.sigma.requires_grad_(req["sigma"])
        gaussians._w1.requires_grad_(req["w1"])

    if use_topology_check and not use_topology:
        return int(new_xzt.shape[0]), "ok_no_gudhi"
    return int(new_xzt.shape[0]), "ok"


def flatten_appearance_features(gaussians):
    n = int(gaussians._features_dc.shape[0])
    dc = gaussians._features_dc.reshape(n, -1)
    rest = gaussians._features_rest.reshape(n, -1)
    flat = torch.cat([dc, rest], dim=1)
    return flat, tuple(gaussians._features_dc.shape), tuple(gaussians._features_rest.shape), int(dc.shape[1]), int(rest.shape[1])


def split_appearance_features(flat, dc_shape, rest_shape, dc_dim, rest_dim):
    n = int(flat.shape[0])
    dc = flat[:, :dc_dim].reshape((n, *dc_shape[1:]))
    rest = flat[:, dc_dim : dc_dim + rest_dim].reshape((n, *rest_shape[1:]))
    return dc, rest


def build_latent_decoder(latent_dim, hidden_dim, out_dim):
    if hidden_dim <= 0:
        return torch.nn.Linear(latent_dim, out_dim)
    return torch.nn.Sequential(
        torch.nn.Linear(latent_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )


def init_latent_appearance_state(gaussians, args):
    flat, dc_shape, rest_shape, dc_dim, rest_dim = flatten_appearance_features(gaussians)
    n_pts, feat_dim = flat.shape
    latent_dim = max(1, int(args.appearance_latent_dim))
    hidden_dim = int(args.appearance_latent_hidden)
    decoder = build_latent_decoder(latent_dim, hidden_dim, feat_dim).to(flat.device)
    theta0 = torch.zeros((n_pts, latent_dim), device=flat.device, dtype=flat.dtype)
    copy_dim = min(latent_dim, feat_dim)
    theta0[:, :copy_dim] = flat[:, :copy_dim]
    theta0 = theta0 + 0.01 * torch.randn_like(theta0)
    theta = torch.nn.Parameter(theta0)
    with torch.no_grad():
        if isinstance(decoder, torch.nn.Linear):
            decoder.weight.zero_()
            decoder.bias.zero_()
            for dim in range(min(feat_dim, latent_dim)):
                decoder.weight[dim, dim] = 1.0
        else:
            last = decoder[-1]
            if isinstance(last, torch.nn.Linear):
                last.weight.zero_()
                last.bias.copy_(flat.mean(dim=0))
    return {
        "mode": "latent",
        "theta": theta,
        "decoder": decoder,
        "dc_shape": dc_shape,
        "rest_shape": rest_shape,
        "dc_dim": dc_dim,
        "rest_dim": rest_dim,
        "feature_dim": int(feat_dim),
        "latent_dim": int(latent_dim),
    }


def add_appearance_optimizer_groups(optimizer, appearance_state, args):
    if appearance_state is None:
        return
    if appearance_state["mode"] == "latent":
        optimizer.add_param_group(
            {
                "params": [appearance_state["theta"]],
                "lr": float(args.appearance_theta_lr),
                "name": "app_theta",
            }
        )
        optimizer.add_param_group(
            {
                "params": list(appearance_state["decoder"].parameters()),
                "lr": float(args.appearance_head_lr),
                "name": "app_head",
            }
        )
    else:
        raise ValueError(f"Unknown appearance mode: {appearance_state['mode']}")


def apply_appearance_model(gaussians, appearance_state, args):
    if appearance_state is None:
        return {}
    if appearance_state["mode"] == "latent":
        theta = appearance_state["theta"]
        flat = appearance_state["decoder"](theta)
        dc, rest = split_appearance_features(
            flat,
            appearance_state["dc_shape"],
            appearance_state["rest_shape"],
            appearance_state["dc_dim"],
            appearance_state["rest_dim"],
        )
        gaussians._features_dc = dc
        gaussians._features_rest = rest
        return {"theta": theta, "flat": flat}
    raise ValueError(f"Unknown appearance mode: {appearance_state['mode']}")


def graph_smoothness_loss(values, xyz, k=8, sigma=2.0, sample_n=2000):
    n_pts = int(values.shape[0])
    if n_pts < 2:
        return torch.tensor(0.0, device=values.device)
    sample_n = max(2, min(int(sample_n), n_pts))
    if sample_n < n_pts:
        idx = torch.randperm(n_pts, device=values.device)[:sample_n]
        vals = values[idx]
        xyz_s = xyz[idx]
    else:
        vals = values
        xyz_s = xyz
    d = torch.cdist(xyz_s, xyz_s)
    k = min(max(1, int(k)), sample_n - 1)
    if k <= 0:
        return torch.tensor(0.0, device=values.device)
    knn = torch.topk(d, k=k + 1, largest=False).indices[:, 1:]
    diff = vals.unsqueeze(1) - vals[knn]
    ww = torch.exp(-(d.gather(1, knn) ** 2) / (2.0 * max(float(sigma), 1e-6) ** 2))
    return (ww.unsqueeze(-1) * diff.pow(2)).mean()


def appearance_regularization_loss(gaussians, appearance_state, appearance_stats, args):
    if appearance_state is None:
        return torch.tensor(0.0, device=gaussians._xyz.device), {}
    reg = torch.tensor(0.0, device=gaussians._xyz.device)
    logs = {}
    if appearance_state["mode"] == "latent":
        theta = appearance_stats["theta"]
        if args.appearance_lambda_smooth > 0:
            smooth = graph_smoothness_loss(
                theta,
                gaussians.get_xyz,
                k=args.appearance_smooth_k,
                sigma=args.appearance_smooth_sigma,
                sample_n=args.appearance_smooth_sample,
            )
            reg = reg + float(args.appearance_lambda_smooth) * smooth
            logs["appearance_smooth"] = float(smooth.detach().item())
        if args.appearance_lambda_theta_l2 > 0:
            theta_l2 = theta.pow(2).mean()
            reg = reg + float(args.appearance_lambda_theta_l2) * theta_l2
            logs["appearance_theta_l2"] = float(theta_l2.detach().item())
        if args.appearance_lambda_head_l2 > 0:
            head_l2 = torch.tensor(0.0, device=theta.device)
            for param in appearance_state["decoder"].parameters():
                head_l2 = head_l2 + param.pow(2).mean()
            reg = reg + float(args.appearance_lambda_head_l2) * head_l2
            logs["appearance_head_l2"] = float(head_l2.detach().item())
    return reg, logs


def main():
    parser = argparse.ArgumentParser(description="Train MedGS FLAIR using frozen T1 geometry.")
    parser.add_argument("--medgs-root", required=True, help="Path to MedGS repo root.")
    parser.add_argument("--t1-model", required=True, help="T1 MedGS model output dir.")
    parser.add_argument("--t1-iter", default="latest", help="Iteration number or 'latest'.")
    parser.add_argument("--single-model", default=None, help="Optional single-modality MedGS model for confidence-gated prior.")
    parser.add_argument("--single-iter", default="latest", help="Iteration for single model, or 'latest'.")
    parser.add_argument("--flair-dataset", required=True, help="FLAIR LR frames dir (with original/).")
    parser.add_argument("--out", required=True, help="Output model dir.")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--feature-lr", type=float, default=0.0025)
    parser.add_argument("--opacity-lr", type=float, default=0.025)
    parser.add_argument("--train-opacity", action="store_true")
    parser.add_argument("--feature-l2", type=float, default=0.0)
    parser.add_argument("--lambda-dssim", type=float, default=0.2)
    parser.add_argument("--use-interp-loss", action="store_true")
    parser.add_argument("--interp-weight", type=float, default=0.5)
    parser.add_argument(
        "--interp-lambda-dssim",
        type=float,
        default=-1.0,
        help="DSSIM weight for interpolation guidance; <0 uses --lambda-dssim.",
    )
    parser.add_argument("--opacity-init", type=float, default=None)
    parser.add_argument("--opacity-scale", type=float, default=None)
    parser.add_argument("--reset-appearance", action="store_true")
    parser.add_argument("--t1", default=None, help="T1 volume for boundary/grad-align losses.")
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--lambda-gradalign", type=float, default=0.0)
    parser.add_argument("--lambda-normal", type=float, default=0.0)
    parser.add_argument("--normal-k", type=int, default=8)
    parser.add_argument("--normal-sigma", type=float, default=2.0)
    parser.add_argument("--normal-sample", type=int, default=2000)
    parser.add_argument("--cov-lr", type=float, default=0.0)
    parser.add_argument("--xyz-lr", type=float, default=0.0)
    parser.add_argument("--lambda-xyz-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-xyz-zero-mean", type=float, default=0.0)
    parser.add_argument("--lambda-xyz-max", type=float, default=0.0)
    parser.add_argument("--xyz-max-mm", type=float, default=0.0)
    parser.add_argument("--xyz-delta-clamp-mm", type=float, default=0.0)
    parser.add_argument("--time-lr", type=float, default=0.0)
    parser.add_argument("--time-init-scale", type=float, default=1.0)
    parser.add_argument("--time-init-shift", type=float, default=0.0)
    parser.add_argument("--preprune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--preprune-min-opacity", type=float, default=0.0)
    parser.add_argument("--preprune-min-keep", type=int, default=50000)
    parser.add_argument("--regrow-interval", type=int, default=0)
    parser.add_argument("--regrow-add-ratio", type=float, default=0.0)
    parser.add_argument("--regrow-start-iter", type=int, default=0)
    parser.add_argument("--regrow-end-iter", type=int, default=0)
    parser.add_argument("--regrow-max-points", type=int, default=600000)
    parser.add_argument("--regrow-opacity-init", type=float, default=0.02)
    parser.add_argument("--regrow-jitter-scale", type=float, default=0.35)
    parser.add_argument("--regrow-scale-shrink", type=float, default=0.7)
    parser.add_argument("--regrow-topk-ratio", type=float, default=0.2)
    parser.add_argument("--regrow-zero-features", action="store_true")
    parser.add_argument("--regrow-mode", type=str, choices=["score", "residual"], default="score")
    parser.add_argument("--residual-top-pct", type=float, default=1.0)
    parser.add_argument("--residual-vote-interval", type=int, default=10)
    parser.add_argument("--lambda-blob", type=float, default=0.0)
    parser.add_argument("--blob-scale-c", type=float, default=0.003)
    parser.add_argument("--views-per-iter", type=int, default=1, help="Number of randomly sampled LR views per training iteration.")
    parser.add_argument("--slab-forward", dest="slab_forward", action="store_true", help="Average multiple alpha samples per LR frame to match thick-slab acquisition.")
    parser.add_argument("--no-slab-forward", dest="slab_forward", action="store_false")
    parser.set_defaults(slab_forward=True)
    parser.add_argument("--slab-samples", type=int, default=0, help="Number of samples within each LR slab (<=0: infer from factor-z or T1/LR frame ratio).")
    parser.add_argument("--lpvi-enable", action="store_true", help="Enable LPVI-style point interpolation before fitting.")
    parser.add_argument("--lpvi-anchor-samples", type=int, default=2048)
    parser.add_argument("--lpvi-k-max", type=int, default=8)
    parser.add_argument("--lpvi-k-min", type=int, default=4)
    parser.add_argument("--lpvi-threshold", type=float, default=0.25)
    parser.add_argument("--lpvi-max-new-points", type=int, default=120000)
    parser.add_argument("--lpvi-min-opacity", type=float, default=0.01)
    parser.add_argument("--lpvi-max-vertices-per-anchor", type=int, default=4)
    parser.add_argument("--lpvi-scale-shrink", type=float, default=0.9)
    parser.add_argument("--lpvi-dist-model", type=str, default="W", choices=["W", "B"])
    parser.add_argument("--lpvi-complex-max-edge", type=float, default=2.0)
    parser.add_argument("--lpvi-use-topology-check", dest="lpvi_use_topology_check", action="store_true")
    parser.add_argument("--lpvi-no-topology-check", dest="lpvi_use_topology_check", action="store_false")
    parser.set_defaults(lpvi_use_topology_check=True)
    parser.add_argument("--persist-lambda", type=float, default=0.0, help="Weight for MRI-adapted persistence-style loss.")
    parser.add_argument("--persist-dims", type=str, default="0,1")
    parser.add_argument("--persist-ks", type=str, default="64,32")
    parser.add_argument("--persist-downsample", type=float, default=0.125)
    parser.add_argument("--persist-use-spatial", dest="persist_use_spatial", action="store_true")
    parser.add_argument("--persist-no-spatial", dest="persist_use_spatial", action="store_false")
    parser.set_defaults(persist_use_spatial=True)
    parser.add_argument("--persist-spatial-weight", type=float, default=0.35)
    parser.add_argument("--persist-intensity-weight", type=float, default=1.0)
    parser.add_argument("--persist-threshold-count", type=int, default=8)
    parser.add_argument("--persist-sigmoid-tau", type=float, default=0.05)
    parser.add_argument("--persist-fg-eps", type=float, default=0.02, help="Foreground threshold for slice informativeness/topology points.")
    parser.add_argument("--persist-min-fg-ratio", type=float, default=0.01, help="Minimum foreground ratio to apply topology persist on a slice.")
    parser.add_argument("--persist-min-fg-points", type=int, default=8, help="Minimum foreground pixel count (after downsample) for topology persist.")
    parser.add_argument("--persist-min-std", type=float, default=0.01, help="Minimum intensity std to consider a slice informative.")
    parser.add_argument("--persist-skip-lowinfo", dest="persist_skip_lowinfo", action="store_true")
    parser.add_argument("--persist-keep-lowinfo", dest="persist_skip_lowinfo", action="store_false")
    parser.set_defaults(persist_skip_lowinfo=True)
    parser.add_argument("--appearance-mode", type=str, default="direct", choices=["direct", "latent"])
    parser.add_argument("--appearance-latent-dim", type=int, default=4, help="Per-gaussian latent dimension.")
    parser.add_argument("--appearance-latent-hidden", type=int, default=32, help="Hidden width for modality head (<=0 gives linear head).")
    parser.add_argument("--appearance-theta-lr", type=float, default=0.0025, help="LR for latent vectors.")
    parser.add_argument("--appearance-head-lr", type=float, default=0.001, help="LR for latent modality head.")
    parser.add_argument("--appearance-lambda-smooth", type=float, default=0.0, help="Graph smoothness regularization for coefficients/latents.")
    parser.add_argument("--appearance-smooth-k", type=int, default=8)
    parser.add_argument("--appearance-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--appearance-smooth-sample", type=int, default=2000)
    parser.add_argument("--appearance-lambda-theta-l2", type=float, default=0.0)
    parser.add_argument("--appearance-lambda-head-l2", type=float, default=0.0)
    parser.add_argument("--coverage-ema-decay", type=float, default=0.995)
    parser.add_argument("--coverage-low-thresh", type=float, default=0.02)
    parser.add_argument("--coverage-warmup-iters", type=int, default=1000)
    parser.add_argument("--lambda-coverage-opacity", type=float, default=0.0)
    parser.add_argument("--lambda-coverage-feature", type=float, default=0.0)
    parser.add_argument("--lambda-coverage-xyz-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-coverage-cov-anchor", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--lambda-single-prior", type=float, default=0.0)
    parser.add_argument("--single-prior-tau", type=float, default=8.0)
    parser.add_argument("--single-prior-wmin", type=float, default=0.1)
    parser.add_argument("--single-prior-wmax", type=float, default=0.9)
    parser.add_argument("--single-prior-start-iter", type=int, default=0)
    parser.add_argument("--scale-clamp-min", type=float, default=None)
    parser.add_argument("--scale-clamp-max", type=float, default=None)
    parser.add_argument("--rotation-clamp", type=float, default=None)
    parser.add_argument("--boundary-p", type=float, default=95.0)
    parser.add_argument("--boundary-gamma", type=float, default=2.0)
    parser.add_argument("--factor-z", type=float, default=None)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--poly-degree", type=int, default=7)
    parser.add_argument("--camera", type=str, default="mirror", choices=["mirror", "one"])
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num-pts", type=int, default=100_000)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--oneup-sh-every", type=int, default=1000)
    parser.add_argument("--no-freeze-cov", action="store_true")
    parser.add_argument("--no-freeze-xyz", action="store_true")
    parser.add_argument("--no-freeze-time", action="store_true")
    args = parser.parse_args()

    add_medgs_to_path(args.medgs_root)

    from gaussian_renderer import render
    from models import gaussianModel
    from scene import Scene
    from utils.loss_utils import l1_loss, ssim

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    t1_ply, t1_iter = resolve_t1_ply(args.t1_model, args.t1_iter)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = len(list(Path(args.flair_dataset).joinpath("original").glob("*.png")))
    if frames == 0:
        frames = len(list(Path(args.flair_dataset).glob("*.png")))
    if frames == 0:
        raise FileNotFoundError("No frames found in flair dataset.")

    dataset = make_dataset_args(
        args.flair_dataset,
        out_dir,
        args.sh_degree,
        args.poly_degree,
        args.camera,
        args.distance,
        args.num_pts,
        args.white_background,
    )
    write_cfg_args(out_dir, dataset)

    gaussians = gaussianModel["gs"](dataset.sh_degree, dataset.poly_degree, frames, use_dff=False)
    scene = Scene(dataset, gaussians, shuffle=False)

    gaussians.load_ply(str(t1_ply))
    lpvi_added = 0
    lpvi_status = "disabled"
    if args.lpvi_enable:
        lpvi_added, lpvi_status = lpvi_augment_gaussians(
            gaussians,
            anchor_samples=args.lpvi_anchor_samples,
            k_max=args.lpvi_k_max,
            k_min=args.lpvi_k_min,
            threshold=args.lpvi_threshold,
            max_new_points=args.lpvi_max_new_points,
            min_opacity=args.lpvi_min_opacity,
            max_vertices_per_anchor=args.lpvi_max_vertices_per_anchor,
            scale_shrink=args.lpvi_scale_shrink,
            use_topology_check=args.lpvi_use_topology_check,
            dist_model=args.lpvi_dist_model,
            complex_mode="rips",
            complex_max_edge=args.lpvi_complex_max_edge,
            seed=args.seed,
        )
        print(f"lpvi_enable=True lpvi_status={lpvi_status} lpvi_added={lpvi_added}")
    source_time_bins = int(gaussians.time_func.shape[0]) if hasattr(gaussians, "time_func") else int(frames)
    single_gaussians = None
    single_iter = None
    if args.single_model and args.lambda_single_prior > 0:
        single_ply, single_iter = resolve_model_ply(args.single_model, args.single_iter)
        single_gaussians = gaussianModel["gs"](dataset.sh_degree, dataset.poly_degree, frames, use_dff=False)
        single_gaussians.load_ply(str(single_ply))
    reset_time_func(gaussians, frames)
    if abs(float(args.time_init_scale) - 1.0) > 1e-8 or abs(float(args.time_init_shift)) > 1e-8:
        apply_time_init_linear_map(gaussians, args.time_init_scale, args.time_init_shift)
        print(
            f"time_init_linear scale={float(args.time_init_scale):.6f} "
            f"shift={float(args.time_init_shift):.6f}"
        )
    if args.reset_appearance:
        reset_appearance(gaussians)
    pruned_from, pruned_to = apply_preprune(
        gaussians,
        keep_ratio=args.preprune_keep_ratio,
        min_opacity=args.preprune_min_opacity,
        min_keep=args.preprune_min_keep,
    )
    if args.opacity_init is not None:
        init_val = float(args.opacity_init)
        init_val = max(min(init_val, 0.99), 1e-4)
        gaussians._opacity.data = torch.logit(torch.full_like(gaussians._opacity, init_val))
    if args.opacity_scale is not None:
        scale = float(args.opacity_scale)
        if scale != 1.0:
            op = gaussians.get_opacity * scale
            op = op.clamp(1e-4, 0.99)
            gaussians._opacity.data = torch.logit(op)

    xyz_reference = gaussians._xyz.detach().clone()
    scaling_reference = gaussians._scaling.detach().clone()
    rotation_reference = gaussians._rotation.detach().clone()
    coverage_ema = torch.zeros((gaussians._xyz.shape[0],), device=gaussians._xyz.device, dtype=torch.float32)

    freeze_params(
        gaussians,
        freeze_xyz=not args.no_freeze_xyz,
        freeze_cov=not args.no_freeze_cov,
        freeze_time=not args.no_freeze_time,
    )

    appearance_state = None
    if args.appearance_mode != "direct":
        if args.regrow_add_ratio > 0 and args.regrow_interval > 0:
            raise ValueError("appearance-mode latent currently does not support regrow; disable regrow for this branch.")
        gaussians._features_dc.requires_grad_(False)
        gaussians._features_rest.requires_grad_(False)
        if args.appearance_mode == "latent":
            appearance_state = init_latent_appearance_state(gaussians, args)
        else:
            raise ValueError(f"Unknown appearance mode: {args.appearance_mode}")
        _ = apply_appearance_model(gaussians, appearance_state, args)

    optimizer = build_optimizer(
        gaussians,
        feature_lr=args.feature_lr,
        opacity_lr=args.opacity_lr,
        train_opacity=args.train_opacity,
        cov_lr=args.cov_lr,
        xyz_lr=args.xyz_lr,
        time_lr=args.time_lr,
    )
    add_appearance_optimizer_groups(optimizer, appearance_state, args)
    gaussians.optimizer = optimizer

    bg = torch.tensor([1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0], device="cuda")
    pipe = argparse.Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )

    cameras = scene.getTrainCameras()
    gts = [cam.get_image(bg, random_background=False).cuda() for cam in cameras]

    inferred_factor = None
    if args.factor_z is not None and float(args.factor_z) > 0:
        inferred_factor = float(args.factor_z)
    elif len(cameras) > 0:
        inferred_factor = float(max(source_time_bins, 1)) / float(len(cameras))
    else:
        inferred_factor = 1.0
    if args.slab_samples > 0:
        slab_samples = int(args.slab_samples)
    else:
        slab_samples = int(round(max(inferred_factor, 1.0)))
    if not args.slab_forward:
        slab_samples = 1
    slab_alphas = slab_alpha_schedule(slab_samples)
    print(
        f"slab_forward={args.slab_forward} slab_samples={slab_samples} "
        f"inferred_factor={inferred_factor:.3f} source_time_bins={source_time_bins} views={len(cameras)}"
    )

    persist_loss_fn = None
    persist_gt_signatures = None
    persist_valid_views = None
    persist_valid_view_stats = None
    persist_has_topologylayer = False
    persist_topologylayer_error = ""
    if args.persist_lambda > 0:
        persist_dims = parse_int_list(args.persist_dims, default_values=[0, 1])
        persist_ks = parse_int_list(args.persist_ks, default_values=[64, 32])
        persist_loss_fn = MRIPersistLoss(
            dims=persist_dims,
            ks=persist_ks,
            downsample=args.persist_downsample,
            use_spatial=args.persist_use_spatial,
            spatial_weight=args.persist_spatial_weight,
            intensity_weight=args.persist_intensity_weight,
            threshold_count=args.persist_threshold_count,
            sigmoid_tau=args.persist_sigmoid_tau,
            fg_eps=args.persist_fg_eps,
            min_fg_ratio=args.persist_min_fg_ratio,
            min_fg_points=args.persist_min_fg_points,
            min_std=args.persist_min_std,
            skip_lowinfo=args.persist_skip_lowinfo,
        )
        persist_has_topologylayer = persist_loss_fn.has_topologylayer
        persist_topologylayer_error = persist_loss_fn.topo_error
        with torch.no_grad():
            persist_valid_views = []
            persist_valid_view_stats = []
            persist_gt_signatures = []
            for gt in gts:
                info_ok, info_st = persist_loss_fn.is_informative(gt)
                persist_valid_views.append(bool(info_ok))
                persist_valid_view_stats.append(info_st)
                if not info_ok and args.persist_skip_lowinfo:
                    persist_gt_signatures.append(None)
                elif persist_has_topologylayer:
                    # Cache both topology/fallback signatures so forward can match mode safely.
                    persist_gt_signatures.append(persist_loss_fn.signature_pair(gt))
                else:
                    persist_gt_signatures.append(persist_loss_fn.signature(gt, detach=True))
            n_valid = int(sum(1 for x in persist_valid_views if x))
            n_total = int(len(persist_valid_views))
            valid_frac = float(n_valid) / float(max(n_total, 1))
            n_lowinfo = n_total - n_valid
        print(
            f"persist_lambda={args.persist_lambda} persist_dims={persist_dims} "
            f"persist_ks={persist_ks} topologylayer={persist_has_topologylayer} "
            f"valid_slices={n_valid}/{n_total} ({valid_frac:.3f}) lowinfo={n_lowinfo}"
        )

    t1_grad = None
    t1_w = None
    t1_shape = None
    if args.t1 and (args.lambda_boundary > 0 or args.lambda_gradalign > 0):
        from atlasgs.ops.nifti_io import load_nii
        t1_vol, _, _ = load_nii(args.t1)
        t1_shape = t1_vol.shape
        gx, gy, gradmag = compute_t1_gradients(t1_vol)
        t1_grad = (gx, gy)
        t1_w = gradmag

    regrow_end_iter = args.regrow_end_iter if args.regrow_end_iter > 0 else args.iterations
    regrow_added_total = 0
    residual_votes = None

    for iteration in range(1, args.iterations + 1):
        if args.oneup_sh_every > 0 and iteration % args.oneup_sh_every == 0:
            gaussians.oneupSHdegree()

        n_pts = int(gaussians._xyz.shape[0])
        coverage_ema = ensure_state_size(coverage_ema, n_pts, device=gaussians._xyz.device)
        xyz_reference = ensure_ref_size(xyz_reference, gaussians._xyz.detach())
        scaling_reference = ensure_ref_size(scaling_reference, gaussians._scaling.detach())
        rotation_reference = ensure_ref_size(rotation_reference, gaussians._rotation.detach())

        view_repeats = max(1, int(args.views_per_iter))
        loss = None
        l1 = None
        prior_loss = None
        persist_loss = None
        app_reg_logs = {}
        valid_views = 0
        appearance_stats = apply_appearance_model(gaussians, appearance_state, args)
        for _ in range(view_repeats):
            idx = random.randint(0, len(cameras) - 1)
            cam = cameras[idx]
            gt = gts[idx]

            pred, vis_filter, _ = render_view_with_slab_forward(
                render_fn=render,
                camera=cam,
                gaussians=gaussians,
                pipe=pipe,
                bg=bg,
                iteration=iteration,
                train=True,
                alphas=slab_alphas,
            )
            if not torch.isfinite(pred).all():
                continue

            with torch.no_grad():
                decay = min(max(float(args.coverage_ema_decay), 0.0), 1.0)
                coverage_ema.mul_(decay)
                vis = vis_filter
                if vis.any():
                    coverage_ema[vis] = coverage_ema[vis] + (1.0 - decay)

            if (
                args.regrow_mode == "residual"
                and args.regrow_interval > 0
                and args.regrow_add_ratio > 0
                and args.residual_vote_interval > 0
                and iteration >= args.regrow_start_iter
                and iteration <= regrow_end_iter
                and iteration % args.residual_vote_interval == 0
            ):
                residual_map = (pred.detach() - gt.detach()).abs().mean(dim=0)
                residual_votes = accumulate_residual_votes(
                    residual_votes=residual_votes,
                    gaussians=gaussians,
                    camera=cam,
                    residual_map=residual_map,
                    visibility_filter=vis_filter,
                    top_pct=args.residual_top_pct,
                )

            l1_view = l1_loss(pred, gt)
            if args.lambda_dssim > 0:
                ssim_loss = 1.0 - ssim(pred, gt)
                view_loss = (1.0 - args.lambda_dssim) * l1_view + args.lambda_dssim * ssim_loss
            else:
                view_loss = l1_view
            prior_view = None
            if (
                single_gaussians is not None
                and args.lambda_single_prior > 0
                and iteration >= args.single_prior_start_iter
            ):
                with torch.no_grad():
                    single_pred, _, _ = render_view_with_slab_forward(
                        render_fn=render,
                        camera=cam,
                        gaussians=single_gaussians,
                        pipe=pipe,
                        bg=bg,
                        iteration=iteration,
                        train=False,
                        alphas=slab_alphas,
                    )
                if torch.isfinite(single_pred).all():
                    tau = max(float(args.single_prior_tau), 1e-6)
                    err_t = (pred.detach() - gt).abs().mean(dim=0, keepdim=True)
                    err_s = (single_pred - gt).abs().mean(dim=0, keepdim=True)
                    wt = torch.exp(-err_t / tau)
                    ws = torch.exp(-err_s / tau)
                    conf_t = wt / (wt + ws + 1e-6)
                    conf_t = conf_t.clamp(float(args.single_prior_wmin), float(args.single_prior_wmax))
                    prior_w = (1.0 - conf_t).detach()
                    prior_view = (prior_w * (pred - single_pred).abs().mean(dim=0, keepdim=True)).mean()
                    view_loss = view_loss + args.lambda_single_prior * prior_view

            persist_view = None
            if persist_loss_fn is not None and args.persist_lambda > 0:
                gt_valid = True
                if persist_valid_views is not None:
                    gt_valid = bool(persist_valid_views[idx])
                pred_valid = True
                if args.persist_skip_lowinfo:
                    pred_valid, _ = persist_loss_fn.is_informative(pred.detach())
                if gt_valid and pred_valid:
                    gt_sig = persist_gt_signatures[idx] if persist_gt_signatures is not None else None
                    persist_view = persist_loss_fn(pred, gt=gt, gt_signature=gt_sig)
                    if torch.isfinite(persist_view):
                        view_loss = view_loss + float(args.persist_lambda) * persist_view
                    else:
                        persist_view = None

            if args.use_interp_loss and len(cameras) > 1:
                prev_idx = max(idx - 1, 0)
                next_idx = min(idx + 1, len(cameras) - 1)
                if prev_idx != idx and next_idx != idx:
                    alpha = 0.5
                    interp_gt = (1 - alpha) * gts[prev_idx] + alpha * gts[next_idx]
                    interp_pkg = render(cam, gaussians, pipe, bg, train=True, iter=iteration, alpha=alpha)
                    interp_pred = interp_pkg["render"]
                    interp_l1 = l1_loss(interp_pred, interp_gt)
                    interp_lambda_dssim = float(args.interp_lambda_dssim)
                    if interp_lambda_dssim < 0:
                        interp_lambda_dssim = float(args.lambda_dssim)
                    interp_lambda_dssim = max(0.0, min(1.0, interp_lambda_dssim))
                    if interp_lambda_dssim > 0:
                        interp_ssim = 1.0 - ssim(interp_pred, interp_gt)
                        interp_mix = (1.0 - interp_lambda_dssim) * interp_l1 + interp_lambda_dssim * interp_ssim
                    else:
                        interp_mix = interp_l1
                    view_loss = view_loss + args.interp_weight * interp_mix
            if t1_w is not None and (args.lambda_boundary > 0 or args.lambda_gradalign > 0):
                if args.factor_z is None:
                    factor_z = t1_shape[2] / max(len(cameras), 1)
                else:
                    factor_z = args.factor_z
                z_idx = int(round(idx * factor_z))
                z_idx = max(0, min(t1_shape[2] - 1, z_idx))
                w_slice = normalize_slice_weight(t1_w[:, :, z_idx], p=args.boundary_p, gamma=args.boundary_gamma)
                w_t = torch.from_numpy(w_slice).to(pred.device).float().unsqueeze(0)
                pred_g = to_gray(pred)
                gt_g = to_gray(gt)
                if args.lambda_boundary > 0:
                    view_loss = view_loss + args.lambda_boundary * ((1.0 + w_t) * (pred_g - gt_g).abs()).mean()
                if args.lambda_gradalign > 0:
                    dx_p, dy_p = grad2d(pred_g)
                    dx_t = torch.from_numpy(t1_grad[0][:, :, z_idx]).to(pred.device).float().unsqueeze(0)
                    dy_t = torch.from_numpy(t1_grad[1][:, :, z_idx]).to(pred.device).float().unsqueeze(0)
                    denom = (dx_p.pow(2) + dy_p.pow(2)).sqrt() * (dx_t.pow(2) + dy_t.pow(2)).sqrt() + 1e-8
                    dot = (dx_p * dx_t + dy_p * dy_t) / denom
                    align = 1.0 - dot.pow(2)
                    align = torch.nan_to_num(align, nan=0.0, posinf=0.0, neginf=0.0)
                    view_loss = view_loss + args.lambda_gradalign * (w_t * align).mean()

            loss = view_loss if loss is None else (loss + view_loss)
            l1 = l1_view if l1 is None else (l1 + l1_view)
            if prior_view is not None:
                prior_loss = prior_view if prior_loss is None else (prior_loss + prior_view)
            if persist_view is not None:
                persist_loss = persist_view if persist_loss is None else (persist_loss + persist_view)
            valid_views += 1

        if valid_views == 0:
            optimizer.zero_grad(set_to_none=True)
            continue
        loss = loss / float(valid_views)
        l1 = l1 / float(valid_views)
        if prior_loss is not None:
            prior_loss = prior_loss / float(valid_views)
        if persist_loss is not None:
            persist_loss = persist_loss / float(valid_views)

        if args.lambda_normal > 0 and (gaussians._scaling.requires_grad or gaussians._rotation.requires_grad) and args.cov_lr > 0:
            loss = loss + args.lambda_normal * normal_alignment_loss(
                gaussians,
                k=args.normal_k,
                sigma=args.normal_sigma,
                sample_n=args.normal_sample,
            )
        if args.feature_l2 > 0:
            feat_l2 = gaussians._features_dc.pow(2).mean() + gaussians._features_rest.pow(2).mean()
            loss = loss + args.feature_l2 * feat_l2
        if appearance_state is not None:
            app_reg, app_reg_logs = appearance_regularization_loss(gaussians, appearance_state, appearance_stats, args)
            if app_reg is not None and torch.isfinite(app_reg):
                loss = loss + app_reg
        if args.lambda_blob > 0:
            loss = loss + args.lambda_blob * tiny_blob_loss(gaussians, args.blob_scale_c)
        if (
            (args.lambda_xyz_anchor > 0 or args.lambda_xyz_zero_mean > 0 or args.lambda_xyz_max > 0)
            and gaussians._xyz.requires_grad
        ):
            xyz_delta = gaussians._xyz - xyz_reference
            if args.lambda_xyz_anchor > 0:
                loss = loss + args.lambda_xyz_anchor * torch.sqrt((xyz_delta * xyz_delta).sum(dim=1) + 1e-8).mean()
            if args.lambda_xyz_zero_mean > 0:
                loss = loss + args.lambda_xyz_zero_mean * torch.sqrt((xyz_delta.mean(dim=0) ** 2).sum() + 1e-8)
            if args.lambda_xyz_max > 0 and args.xyz_max_mm > 0:
                delta_norm = torch.sqrt((xyz_delta * xyz_delta).sum(dim=1) + 1e-8)
                overshoot = torch.clamp(delta_norm - args.xyz_max_mm, min=0.0)
                loss = loss + args.lambda_xyz_max * overshoot.mean()
        if iteration >= int(args.coverage_warmup_iters):
            thresh = max(float(args.coverage_low_thresh), 1e-6)
            low_cov_w = torch.clamp((thresh - coverage_ema) / thresh, min=0.0, max=1.0)
            if args.lambda_coverage_opacity > 0:
                loss = loss + args.lambda_coverage_opacity * (low_cov_w * gaussians.get_opacity.squeeze()).mean()
            if args.lambda_coverage_feature > 0:
                feat_mag = gaussians._features_dc.pow(2).mean(dim=(1, 2)) + gaussians._features_rest.pow(2).mean(dim=(1, 2))
                loss = loss + args.lambda_coverage_feature * (low_cov_w * feat_mag).mean()
            if args.lambda_coverage_xyz_anchor > 0 and gaussians._xyz.requires_grad:
                xyz_delta = gaussians._xyz - xyz_reference
                xyz_norm = torch.sqrt((xyz_delta * xyz_delta).sum(dim=1) + 1e-8)
                loss = loss + args.lambda_coverage_xyz_anchor * (low_cov_w * xyz_norm).mean()
            if args.lambda_coverage_cov_anchor > 0 and (gaussians._scaling.requires_grad or gaussians._rotation.requires_grad):
                cov_delta = (gaussians._scaling - scaling_reference).pow(2).mean(dim=1)
                cov_delta = cov_delta + (gaussians._rotation - rotation_reference).pow(2).mean(dim=1)
                loss = loss + args.lambda_coverage_cov_anchor * (low_cov_w * cov_delta).mean()
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            params = []
            for group in optimizer.param_groups:
                params.extend([p for p in group["params"] if p.grad is not None])
            if params:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if args.xyz_delta_clamp_mm > 0 and gaussians._xyz.requires_grad:
            with torch.no_grad():
                xyz_delta = gaussians._xyz.data - xyz_reference
                delta_norm = torch.sqrt((xyz_delta * xyz_delta).sum(dim=1, keepdim=True) + 1e-8)
                scale = torch.clamp(float(args.xyz_delta_clamp_mm) / delta_norm, max=1.0)
                gaussians._xyz.data.copy_(xyz_reference + xyz_delta * scale)
        if args.cov_lr and args.cov_lr > 0:
            if args.scale_clamp_min is not None or args.scale_clamp_max is not None:
                lo = args.scale_clamp_min if args.scale_clamp_min is not None else gaussians._scaling.data.min().item()
                hi = args.scale_clamp_max if args.scale_clamp_max is not None else gaussians._scaling.data.max().item()
                gaussians._scaling.data.clamp_(lo, hi)
            if args.rotation_clamp is not None:
                gaussians._rotation.data.clamp_(-args.rotation_clamp, args.rotation_clamp)

        if (
            args.regrow_interval > 0
            and args.regrow_add_ratio > 0
            and iteration >= args.regrow_start_iter
            and iteration <= regrow_end_iter
            and iteration % args.regrow_interval == 0
        ):
            selected_idx = None
            if args.regrow_mode == "residual":
                n_current = int(gaussians._xyz.shape[0])
                n_add_target = int(round(n_current * float(args.regrow_add_ratio)))
                n_add_target = max(1, n_add_target)
                n_add_target = min(n_add_target, max(0, int(args.regrow_max_points) - n_current))
                if n_add_target > 0:
                    selected_idx = pick_indices_from_votes(
                        residual_votes,
                        n_add=n_add_target,
                        topk_ratio=args.regrow_topk_ratio,
                    )
            added = regrow_clone_points(
                gaussians,
                add_ratio=args.regrow_add_ratio,
                max_points=args.regrow_max_points,
                opacity_init=args.regrow_opacity_init,
                jitter_scale=args.regrow_jitter_scale,
                scale_shrink=args.regrow_scale_shrink,
                topk_ratio=args.regrow_topk_ratio,
                zero_features=args.regrow_zero_features,
                chosen_idx=selected_idx,
            )
            regrow_added_total += int(added)
            if added > 0:
                if residual_votes is not None:
                    n_new = int(gaussians._xyz.shape[0]) - int(residual_votes.shape[0])
                    if n_new > 0:
                        residual_votes = torch.cat(
                            [residual_votes, torch.zeros((n_new,), device=residual_votes.device)],
                            dim=0,
                        )
                print(
                    f"iter {iteration} regrow_added={added} "
                    f"points={gaussians._xyz.shape[0]}"
                )
        optimizer.zero_grad(set_to_none=True)

        if args.log_every > 0 and iteration % args.log_every == 0:
            app_msg = ""
            if app_reg_logs:
                app_msg = " " + " ".join([f"{k}={v:.4f}" for k, v in sorted(app_reg_logs.items())])
            persist_msg = f" persist={persist_loss.item():.4f}" if persist_loss is not None else ""
            if prior_loss is not None:
                print(
                    f"iter {iteration} loss={loss.item():.4f} "
                    f"l1={l1.item():.4f} prior={prior_loss.item():.4f}{persist_msg}{app_msg}"
                )
            else:
                print(f"iter {iteration} loss={loss.item():.4f} l1={l1.item():.4f}{persist_msg}{app_msg}")

        if args.save_every > 0 and iteration % args.save_every == 0:
            if appearance_state is not None:
                _ = apply_appearance_model(gaussians, appearance_state, args)
            scene.save(iteration)

    if appearance_state is not None:
        _ = apply_appearance_model(gaussians, appearance_state, args)
    scene.save(args.iterations)

    meta = {
        "t1_model": str(args.t1_model),
        "t1_iter": t1_iter,
        "single_model": str(args.single_model) if args.single_model else None,
        "single_iter": single_iter,
        "flair_dataset": str(args.flair_dataset),
        "iterations": args.iterations,
        "feature_lr": args.feature_lr,
        "opacity_lr": args.opacity_lr,
        "train_opacity": args.train_opacity,
        "lambda_dssim": args.lambda_dssim,
        "interp_lambda_dssim": args.interp_lambda_dssim,
        "lambda_boundary": args.lambda_boundary,
        "lambda_gradalign": args.lambda_gradalign,
        "lambda_normal": args.lambda_normal,
        "cov_lr": args.cov_lr,
        "xyz_lr": args.xyz_lr,
        "lambda_xyz_anchor": args.lambda_xyz_anchor,
        "lambda_xyz_zero_mean": args.lambda_xyz_zero_mean,
        "lambda_xyz_max": args.lambda_xyz_max,
        "xyz_max_mm": args.xyz_max_mm,
        "xyz_delta_clamp_mm": args.xyz_delta_clamp_mm,
        "time_lr": args.time_lr,
        "time_init_scale": args.time_init_scale,
        "time_init_shift": args.time_init_shift,
        "preprune_keep_ratio": args.preprune_keep_ratio,
        "preprune_min_opacity": args.preprune_min_opacity,
        "preprune_min_keep": args.preprune_min_keep,
        "regrow_interval": args.regrow_interval,
        "regrow_add_ratio": args.regrow_add_ratio,
        "regrow_start_iter": args.regrow_start_iter,
        "regrow_end_iter": regrow_end_iter,
        "regrow_max_points": args.regrow_max_points,
        "regrow_opacity_init": args.regrow_opacity_init,
        "regrow_jitter_scale": args.regrow_jitter_scale,
        "regrow_scale_shrink": args.regrow_scale_shrink,
        "regrow_topk_ratio": args.regrow_topk_ratio,
        "regrow_zero_features": args.regrow_zero_features,
        "regrow_mode": args.regrow_mode,
        "residual_top_pct": args.residual_top_pct,
        "residual_vote_interval": args.residual_vote_interval,
        "regrow_added_total": regrow_added_total,
        "pruned_from": pruned_from,
        "pruned_to": pruned_to,
        "lambda_blob": args.lambda_blob,
        "blob_scale_c": args.blob_scale_c,
        "views_per_iter": args.views_per_iter,
        "slab_forward": args.slab_forward,
        "slab_samples": slab_samples,
        "slab_alphas": slab_alphas,
        "slab_inferred_factor": inferred_factor,
        "source_time_bins": source_time_bins,
        "lpvi_enable": args.lpvi_enable,
        "lpvi_status": lpvi_status,
        "lpvi_added": lpvi_added,
        "lpvi_anchor_samples": args.lpvi_anchor_samples,
        "lpvi_k_max": args.lpvi_k_max,
        "lpvi_k_min": args.lpvi_k_min,
        "lpvi_threshold": args.lpvi_threshold,
        "lpvi_max_new_points": args.lpvi_max_new_points,
        "lpvi_min_opacity": args.lpvi_min_opacity,
        "lpvi_max_vertices_per_anchor": args.lpvi_max_vertices_per_anchor,
        "lpvi_scale_shrink": args.lpvi_scale_shrink,
        "lpvi_use_topology_check": args.lpvi_use_topology_check,
        "lpvi_dist_model": args.lpvi_dist_model,
        "lpvi_complex_max_edge": args.lpvi_complex_max_edge,
        "persist_lambda": args.persist_lambda,
        "persist_dims": parse_int_list(args.persist_dims, default_values=[0, 1]),
        "persist_ks": parse_int_list(args.persist_ks, default_values=[64, 32]),
        "persist_downsample": args.persist_downsample,
        "persist_use_spatial": args.persist_use_spatial,
        "persist_spatial_weight": args.persist_spatial_weight,
        "persist_intensity_weight": args.persist_intensity_weight,
        "persist_threshold_count": args.persist_threshold_count,
        "persist_sigmoid_tau": args.persist_sigmoid_tau,
        "persist_fg_eps": args.persist_fg_eps,
        "persist_min_fg_ratio": args.persist_min_fg_ratio,
        "persist_min_fg_points": args.persist_min_fg_points,
        "persist_min_std": args.persist_min_std,
        "persist_skip_lowinfo": args.persist_skip_lowinfo,
        "persist_valid_slice_count": (
            int(sum(1 for x in persist_valid_views if x)) if persist_valid_views is not None else None
        ),
        "persist_total_slice_count": (int(len(persist_valid_views)) if persist_valid_views is not None else None),
        "persist_valid_slice_fraction": (
            float(sum(1 for x in persist_valid_views if x)) / float(max(len(persist_valid_views), 1))
            if persist_valid_views is not None
            else None
        ),
        "persist_has_topologylayer": persist_has_topologylayer,
        "persist_topologylayer_error": persist_topologylayer_error,
        "appearance_mode": args.appearance_mode,
        "appearance_latent_dim": args.appearance_latent_dim,
        "appearance_latent_hidden": args.appearance_latent_hidden,
        "appearance_theta_lr": args.appearance_theta_lr,
        "appearance_head_lr": args.appearance_head_lr,
        "appearance_lambda_smooth": args.appearance_lambda_smooth,
        "appearance_smooth_k": args.appearance_smooth_k,
        "appearance_smooth_sigma": args.appearance_smooth_sigma,
        "appearance_smooth_sample": args.appearance_smooth_sample,
        "appearance_lambda_theta_l2": args.appearance_lambda_theta_l2,
        "appearance_lambda_head_l2": args.appearance_lambda_head_l2,
        "appearance_feature_dim": int(appearance_state["feature_dim"]) if appearance_state is not None else None,
        "appearance_component_dim": (
            int(appearance_state["latent_dim"])
            if appearance_state is not None and appearance_state["mode"] == "latent"
            else None
        ),
        "coverage_ema_decay": args.coverage_ema_decay,
        "coverage_low_thresh": args.coverage_low_thresh,
        "coverage_warmup_iters": args.coverage_warmup_iters,
        "lambda_coverage_opacity": args.lambda_coverage_opacity,
        "lambda_coverage_feature": args.lambda_coverage_feature,
        "lambda_coverage_xyz_anchor": args.lambda_coverage_xyz_anchor,
        "lambda_coverage_cov_anchor": args.lambda_coverage_cov_anchor,
        "coverage_ema_mean": float(coverage_ema.mean().item()) if coverage_ema.numel() > 0 else 0.0,
        "coverage_ema_lowfrac": float((coverage_ema < float(args.coverage_low_thresh)).float().mean().item()) if coverage_ema.numel() > 0 else 0.0,
        "lambda_single_prior": args.lambda_single_prior,
        "single_prior_tau": args.single_prior_tau,
        "single_prior_wmin": args.single_prior_wmin,
        "single_prior_wmax": args.single_prior_wmax,
        "single_prior_start_iter": args.single_prior_start_iter,
        "opacity_init": args.opacity_init,
        "opacity_scale": args.opacity_scale,
        "reset_appearance": args.reset_appearance,
        "sh_degree": args.sh_degree,
        "poly_degree": args.poly_degree,
        "freeze_xyz": not args.no_freeze_xyz,
        "freeze_cov": not args.no_freeze_cov,
        "freeze_time": not args.no_freeze_time,
    }
    with open(out_dir / "t1guided_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
