import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlasgs.ops.nifti_io import load_nii, save_nii


def add_alpine_to_path(alpine_root):
    alpine_root = Path(alpine_root).resolve()
    sys.path.insert(0, str(alpine_root))
    return alpine_root


def load_siren(alpine_root):
    add_alpine_to_path(alpine_root)
    try:
        from alpine.models.siren import Siren as AlpineSiren

        return AlpineSiren, "alpine"
    except Exception as exc:
        print(f"[A2] Alpine Siren import failed ({exc}). Falling back to local Siren.")

        class LocalSine(nn.Module):
            def __init__(self, omega=30.0):
                super().__init__()
                self.omega = float(omega)

            def forward(self, x):
                return torch.sin(self.omega * x)

        def local_linear(
            in_features,
            out_features,
            omega=30.0,
            bias=True,
            is_first=False,
            is_last=False,
        ):
            layer = nn.Linear(in_features, out_features, bias=bias)
            if is_first:
                layer.weight.data.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / max(float(omega), 1e-12)
                layer.weight.data.uniform_(-bound, bound)
            if is_last:
                bound = np.sqrt(6.0 / in_features) / max(float(omega), 1e-12)
                layer.weight.data.uniform_(-bound, bound)
            return layer

        class LocalSiren(nn.Module):
            def __init__(
                self,
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True,
                omegas=[30.0],
                bias=True,
            ):
                super().__init__()
                layers = []
                omega_list = (
                    list(omegas)
                    if len(omegas) == hidden_layers
                    else [float(omegas[0])] * int(hidden_layers)
                )
                layers.append(
                    local_linear(
                        in_features,
                        hidden_features,
                        omega=omega_list[0],
                        is_first=True,
                        bias=bias,
                    )
                )
                layers.append(LocalSine(omega=omega_list[0]))
                for layer_index in range(max(int(hidden_layers) - 2, 0)):
                    layers.append(
                        local_linear(
                            hidden_features,
                            hidden_features,
                            omega=omega_list[layer_index + 1],
                            bias=bias,
                        )
                    )
                    layers.append(LocalSine(omega=omega_list[layer_index + 1]))
                layers.append(
                    local_linear(
                        hidden_features,
                        out_features,
                        omega=omega_list[-1],
                        is_last=outermost_linear,
                        bias=bias,
                    )
                )
                if not outermost_linear:
                    layers.append(LocalSine(omega=omega_list[-1]))
                self.model = nn.ModuleList(layers)

            def forward(self, coords):
                output = coords
                for layer in self.model:
                    output = layer(output)
                return {"output": output}

        return LocalSiren, "local_fallback"


def robust_stats(volume, mask=None, pmin=1.0, pmax=99.0):
    if mask is None:
        vals = volume.reshape(-1)
    else:
        vals = volume[mask > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = volume.reshape(-1)
    lo = float(np.percentile(vals, pmin))
    hi = float(np.percentile(vals, pmax))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalize(volume, lo, hi):
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def denormalize(volume, lo, hi):
    return (volume * (hi - lo) + lo).astype(np.float32)


def make_coords(shape, device):
    x = torch.linspace(-1.0, 1.0, shape[0], device=device)
    y = torch.linspace(-1.0, 1.0, shape[1], device=device)
    z = torch.linspace(-1.0, 1.0, shape[2], device=device)
    grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def sample_indices(mask_flat, n, device):
    valid = torch.where(mask_flat > 0)[0]
    if valid.numel() == 0:
        valid = torch.arange(mask_flat.numel(), device=device)
    pick = valid[torch.randint(0, valid.numel(), (n,), device=device)]
    return pick


def gather_lr_points(lr_shape, batch_size, device):
    x = torch.randint(0, lr_shape[0], (batch_size,), device=device)
    y = torch.randint(0, lr_shape[1], (batch_size,), device=device)
    z = torch.randint(0, lr_shape[2], (batch_size,), device=device)
    return x, y, z


def build_hr_block_coords(x, y, z_lr, factor_z, hr_shape, device):
    offsets = torch.arange(int(factor_z), device=device).view(1, -1)
    z_hr = z_lr.view(-1, 1) * int(factor_z) + offsets
    z_hr = z_hr.clamp(max=hr_shape[2] - 1)
    wx = 2.0 * x.float().view(-1, 1) / max(hr_shape[0] - 1, 1) - 1.0
    wy = 2.0 * y.float().view(-1, 1) / max(hr_shape[1] - 1, 1) - 1.0
    wz = 2.0 * z_hr.float() / max(hr_shape[2] - 1, 1) - 1.0
    coords = torch.stack(
        [
            wx.expand_as(wz),
            wy.expand_as(wz),
            wz,
        ],
        dim=-1,
    )  # (B, F, 3)
    return coords


def predict_target_from_feats(feats, t1_pred, head, alpha, beta):
    residual = head(feats)
    pred = alpha * t1_pred + beta + residual
    return pred


def chunk_predict(coords, trunk, t1_head, target_head, alpha, beta, chunk):
    outs = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk):
            part = coords[start : start + chunk]
            feats = trunk(part)["output"]
            t1p = t1_head(feats)
            pred = predict_target_from_feats(feats, t1p, target_head, alpha, beta)
            outs.append(pred)
    return torch.cat(outs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="A2: ALPINE frozen T1 trunk + target residual head.")
    parser.add_argument("--alpine-root", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--target-gt", required=True)
    parser.add_argument("--target-lr", required=True)
    parser.add_argument("--target-pseudo-hr", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out-nifti", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--factor-z", type=int, required=True)
    parser.add_argument("--pretrain-iters", type=int, default=2000)
    parser.add_argument("--fit-iters", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--trunk-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-pseudo", type=float, default=0.1)
    parser.add_argument("--lambda-alpha-reg", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    Siren, siren_source = load_siren(args.alpine_root)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t1, _, _ = load_nii(args.t1)
    tgt_gt, tgt_aff, tgt_hdr = load_nii(args.target_gt)
    tgt_lr, _, _ = load_nii(args.target_lr)
    mask = None
    if args.mask:
        mask, _, _ = load_nii(args.mask)
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = (t1 > 0).astype(np.uint8)
    pseudo_hr = None
    if args.target_pseudo_hr:
        pseudo_hr, _, _ = load_nii(args.target_pseudo_hr)

    if t1.shape != tgt_gt.shape:
        raise ValueError(f"T1 shape {t1.shape} != target GT shape {tgt_gt.shape}")

    t1_lo, t1_hi = robust_stats(t1, mask=mask)
    tgt_lo, tgt_hi = robust_stats(tgt_gt, mask=mask)
    t1_n = normalize(t1, t1_lo, t1_hi)
    tgt_gt_n = normalize(tgt_gt, tgt_lo, tgt_hi)
    tgt_lr_n = normalize(tgt_lr, tgt_lo, tgt_hi)
    pseudo_n = normalize(pseudo_hr, tgt_lo, tgt_hi) if pseudo_hr is not None else None

    hr_shape = t1.shape
    lr_shape = tgt_lr.shape

    coords_all = make_coords(hr_shape, device=device)
    t1_flat = torch.from_numpy(t1_n.reshape(-1, 1)).to(device)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device)
    tgt_lr_t = torch.from_numpy(tgt_lr_n).to(device)
    if pseudo_n is not None:
        pseudo_flat = torch.from_numpy(pseudo_n.reshape(-1, 1)).to(device)
    else:
        pseudo_flat = None

    trunk = Siren(
        in_features=3,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=args.feature_dim,
        outermost_linear=True,
        omegas=[30.0],
    ).to(device)
    t1_head = nn.Linear(args.feature_dim, 1).to(device)
    target_head = nn.Sequential(
        nn.Linear(args.feature_dim, args.feature_dim),
        nn.ReLU(inplace=True),
        nn.Linear(args.feature_dim, 1),
    ).to(device)
    alpha = nn.Parameter(torch.tensor(1.0, device=device))
    beta = nn.Parameter(torch.tensor(0.0, device=device))

    optim_t1 = torch.optim.Adam(
        list(trunk.parameters()) + list(t1_head.parameters()),
        lr=args.trunk_lr,
    )
    for it in range(1, args.pretrain_iters + 1):
        idx = sample_indices(mask_flat, args.batch_size, device=device)
        c = coords_all[idx]
        gt = t1_flat[idx]
        pred = t1_head(trunk(c)["output"])
        loss = F.l1_loss(pred, gt)
        optim_t1.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(trunk.parameters()) + list(t1_head.parameters()), 1.0)
        optim_t1.step()
        if args.log_every > 0 and it % args.log_every == 0:
            print(f"[A2:T1] iter {it}/{args.pretrain_iters} loss={loss.item():.6f}")

    for param in trunk.parameters():
        param.requires_grad_(False)
    for param in t1_head.parameters():
        param.requires_grad_(False)
    trunk.eval()
    t1_head.eval()

    optim_fit = torch.optim.Adam(
        list(target_head.parameters()) + [alpha, beta],
        lr=args.head_lr,
    )
    for it in range(1, args.fit_iters + 1):
        bx, by, bz = gather_lr_points(lr_shape, args.batch_size, device=device)
        coords_block = build_hr_block_coords(bx, by, bz, args.factor_z, hr_shape, device=device)  # (B,F,3)
        bsz, fac, _ = coords_block.shape
        coords_flat = coords_block.reshape(-1, 3)

        with torch.no_grad():
            feats_flat = trunk(coords_flat)["output"]
            t1_flat_pred = t1_head(feats_flat)
        pred_flat = predict_target_from_feats(feats_flat, t1_flat_pred, target_head, alpha, beta)
        pred_block = pred_flat.view(bsz, fac, 1).squeeze(-1)
        pred_lr = pred_block.mean(dim=1)
        obs_lr = tgt_lr_t[bx, by, bz]
        loss_dc = F.l1_loss(pred_lr, obs_lr)

        loss = loss_dc
        if pseudo_flat is not None and args.lambda_pseudo > 0:
            zc = (bz * int(args.factor_z) + int(args.factor_z) // 2).clamp(max=hr_shape[2] - 1)
            center_idx = bx + by * hr_shape[0] + zc * hr_shape[0] * hr_shape[1]
            ccoords = coords_all[center_idx]
            with torch.no_grad():
                cfeat = trunk(ccoords)["output"]
                ct1 = t1_head(cfeat)
            cpred = predict_target_from_feats(cfeat, ct1, target_head, alpha, beta)
            cpseudo = pseudo_flat[center_idx]
            loss = loss + float(args.lambda_pseudo) * F.l1_loss(cpred, cpseudo)

        loss = loss + float(args.lambda_alpha_reg) * (alpha - 1.0).pow(2)
        optim_fit.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(target_head.parameters()) + [alpha, beta], 1.0)
        optim_fit.step()
        if args.log_every > 0 and it % args.log_every == 0:
            print(
                f"[A2:FIT] iter {it}/{args.fit_iters} "
                f"loss={loss.item():.6f} dc={loss_dc.item():.6f} alpha={alpha.item():.4f} beta={beta.item():.4f}"
            )

    pred_all = chunk_predict(
        coords=coords_all,
        trunk=trunk,
        t1_head=t1_head,
        target_head=target_head,
        alpha=alpha,
        beta=beta,
        chunk=args.chunk_size,
    ).reshape(hr_shape)
    pred_all = pred_all.detach().cpu().numpy().astype(np.float32)
    pred_all = np.clip(pred_all, 0.0, 1.0)
    pred_denorm = denormalize(pred_all, tgt_lo, tgt_hi)
    save_nii(args.out_nifti, pred_denorm, tgt_aff, header=tgt_hdr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "alpha": float(alpha.detach().cpu()),
        "beta": float(beta.detach().cpu()),
        "feature_dim": int(args.feature_dim),
        "hidden_features": int(args.hidden_features),
        "hidden_layers": int(args.hidden_layers),
        "factor_z": int(args.factor_z),
        "pretrain_iters": int(args.pretrain_iters),
        "fit_iters": int(args.fit_iters),
        "t1_stats": [t1_lo, t1_hi],
        "target_stats": [tgt_lo, tgt_hi],
        "trunk_state": trunk.state_dict(),
        "t1_head_state": t1_head.state_dict(),
        "target_head_state": target_head.state_dict(),
    }
    torch.save(ckpt, out_dir / "alpine_a2.pt")
    meta = {
        "method": "alpine_a2_frozen_t1_trunk_residual_head",
        "siren_source": siren_source,
        "device": device,
        "t1": str(args.t1),
        "target_gt": str(args.target_gt),
        "target_lr": str(args.target_lr),
        "target_pseudo_hr": str(args.target_pseudo_hr) if args.target_pseudo_hr else None,
        "mask": str(args.mask) if args.mask else None,
        "out_nifti": str(args.out_nifti),
        "factor_z": int(args.factor_z),
        "pretrain_iters": int(args.pretrain_iters),
        "fit_iters": int(args.fit_iters),
        "batch_size": int(args.batch_size),
        "feature_dim": int(args.feature_dim),
        "hidden_features": int(args.hidden_features),
        "hidden_layers": int(args.hidden_layers),
        "trunk_lr": float(args.trunk_lr),
        "head_lr": float(args.head_lr),
        "lambda_pseudo": float(args.lambda_pseudo),
        "lambda_alpha_reg": float(args.lambda_alpha_reg),
        "alpha": float(alpha.detach().cpu()),
        "beta": float(beta.detach().cpu()),
        "t1_stats": [t1_lo, t1_hi],
        "target_stats": [tgt_lo, tgt_hi],
    }
    with open(out_dir / "alpine_a2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
