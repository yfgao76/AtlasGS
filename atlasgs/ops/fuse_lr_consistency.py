import argparse

import numpy as np
try:
    from scipy.ndimage import gaussian_filter, zoom  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    gaussian_filter = None
    zoom = None

from .nifti_io import load_nii, save_nii


def maybe_resize_like(volume, target_shape):
    if volume.shape == target_shape:
        return volume.astype(np.float32)
    if HAVE_SCIPY:
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        return zoom(volume, factors, order=1).astype(np.float32)
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(volume.astype(np.float32))[None, None]
    out = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return out[0, 0].cpu().numpy().astype(np.float32)


def blur_along_z(volume, sigma_z):
    sigma_z = float(sigma_z)
    if sigma_z <= 0:
        return volume.astype(np.float32)
    if HAVE_SCIPY:
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        return gaussian_filter1d(volume, sigma=sigma_z, axis=2).astype(np.float32)
    radius = max(1, int(round(3.0 * sigma_z)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma_z * sigma_z))
    kernel /= np.sum(kernel)
    padded = np.pad(volume, ((0, 0), (0, 0), (radius, radius)), mode="edge").astype(np.float32)
    out = np.zeros_like(volume, dtype=np.float32)
    for z in range(volume.shape[2]):
        out[:, :, z] = np.tensordot(
            padded[:, :, z : z + 2 * radius + 1],
            kernel,
            axes=([2], [0]),
        )
    return out


def simulate_lr(hr, affine, factor_z, sigma_z, mode):
    del affine
    factor_z = int(factor_z)
    blurred = blur_along_z(hr, sigma_z=float(sigma_z))
    if mode == "avgpool":
        z = blurred.shape[2]
        z_trim = z - (z % factor_z)
        cropped = blurred[:, :, :z_trim]
        lr = cropped.reshape(
            cropped.shape[0],
            cropped.shape[1],
            z_trim // factor_z,
            factor_z,
        ).mean(axis=3)
    elif mode == "stride":
        lr = blurred[:, :, ::factor_z]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return lr.astype(np.float32)


def softmax_weight(err_t, err_s, tau, eps=1e-6):
    tau = max(float(tau), eps)
    wt = np.exp(-err_t / tau)
    ws = np.exp(-err_s / tau)
    return (wt / (wt + ws + eps)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Fuse HR predictions by LR consistency confidence.")
    parser.add_argument("--single-hr", required=True)
    parser.add_argument("--t1guided-hr", required=True)
    parser.add_argument("--lr-obs", required=True)
    parser.add_argument("--ref", required=True, help="Reference NIfTI for output affine/header.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor-z", type=int, required=True)
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="avgpool")
    parser.add_argument("--tau", type=float, default=8.0)
    parser.add_argument("--wmin", type=float, default=0.1)
    parser.add_argument("--wmax", type=float, default=0.9)
    parser.add_argument("--smooth-xy", type=float, default=1.0)
    parser.add_argument("--smooth-z", type=float, default=0.5)
    args = parser.parse_args()

    single_hr, aff_single, _ = load_nii(args.single_hr)
    t1g_hr, aff_t1g, _ = load_nii(args.t1guided_hr)
    lr_obs, _, _ = load_nii(args.lr_obs)
    ref, aff_ref, hdr_ref = load_nii(args.ref)

    if single_hr.shape != t1g_hr.shape:
        raise ValueError(f"Shape mismatch: single={single_hr.shape}, t1guided={t1g_hr.shape}")

    single_lr = simulate_lr(single_hr, aff_single, args.factor_z, args.sigma_z, args.mode)
    t1g_lr = simulate_lr(t1g_hr, aff_t1g, args.factor_z, args.sigma_z, args.mode)
    single_lr = maybe_resize_like(single_lr, lr_obs.shape)
    t1g_lr = maybe_resize_like(t1g_lr, lr_obs.shape)

    err_s = np.abs(single_lr - lr_obs).astype(np.float32)
    err_t = np.abs(t1g_lr - lr_obs).astype(np.float32)

    w_t_lr = softmax_weight(err_t, err_s, tau=args.tau)
    if (args.smooth_xy > 0 or args.smooth_z > 0) and HAVE_SCIPY:
        w_t_lr = gaussian_filter(
            w_t_lr,
            sigma=(max(args.smooth_xy, 0.0), max(args.smooth_xy, 0.0), max(args.smooth_z, 0.0)),
            mode="nearest",
        )
    w_t_lr = np.clip(w_t_lr, float(args.wmin), float(args.wmax)).astype(np.float32)

    w_t_hr = maybe_resize_like(w_t_lr, single_hr.shape)
    fused = w_t_hr * t1g_hr + (1.0 - w_t_hr) * single_hr
    fused = maybe_resize_like(fused, ref.shape)
    save_nii(args.out, fused.astype(np.float32), aff_ref, header=hdr_ref)


if __name__ == "__main__":
    main()
