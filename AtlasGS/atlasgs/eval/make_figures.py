import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atlasgs.ops.nifti_io import load_nii


def normalize_window(img, vmin, vmax):
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)


def get_z_indices(depth, mode, offset):
    center = depth // 2
    if mode == "center":
        return [center]
    z0 = max(0, center - offset)
    z1 = center
    z2 = min(depth - 1, center + offset)
    return sorted(set([z0, z1, z2]))


def masked_values(vol, mask):
    if mask is None:
        return vol.reshape(-1)
    return vol[mask > 0]


def main():
    parser = argparse.ArgumentParser(description="Make quick SR comparison figures.")
    parser.add_argument("--t1", required=True)
    parser.add_argument("--flair-gt", required=True)
    parser.add_argument("--flair-lr", required=False, default=None)
    parser.add_argument("--flair-interp", required=True)
    parser.add_argument("--flair-medgs-single", required=True)
    parser.add_argument("--flair-medgs-t1guided", required=True)
    parser.add_argument("--flair-medgs-t1guided-topology", default=None)
    parser.add_argument("--flair-medgs-t1guided-latent", default=None)
    parser.add_argument("--flair-medgs-our", default=None)
    parser.add_argument("--flair-alpine-a2", default=None)
    parser.add_argument("--flair-medgs-fused", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--slice-mode", choices=["center", "triple"], default="triple")
    parser.add_argument("--slice-offset", type=int, default=10)
    parser.add_argument("--pmin", type=float, default=1.0)
    parser.add_argument("--pmax", type=float, default=99.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t1, _, _ = load_nii(args.t1)
    flair_gt, _, _ = load_nii(args.flair_gt)
    flair_interp, _, _ = load_nii(args.flair_interp)
    flair_single, _, _ = load_nii(args.flair_medgs_single)
    flair_t1g, _, _ = load_nii(args.flair_medgs_t1guided)
    flair_t1g_topology = None
    if args.flair_medgs_t1guided_topology:
        flair_t1g_topology, _, _ = load_nii(args.flair_medgs_t1guided_topology)
    flair_t1g_latent = None
    if args.flair_medgs_t1guided_latent:
        flair_t1g_latent, _, _ = load_nii(args.flair_medgs_t1guided_latent)
    flair_our = None
    if args.flair_medgs_our:
        flair_our, _, _ = load_nii(args.flair_medgs_our)
    flair_alpine_a2 = None
    if args.flair_alpine_a2:
        flair_alpine_a2, _, _ = load_nii(args.flair_alpine_a2)
    flair_fused = None
    if args.flair_medgs_fused:
        flair_fused, _, _ = load_nii(args.flair_medgs_fused)

    mask = None
    if args.mask:
        mask, _, _ = load_nii(args.mask)
        if mask.shape != flair_gt.shape:
            mask = flair_gt > 0
        else:
            mask = mask > 0
    else:
        mask = (t1 > 0) if t1.shape == flair_gt.shape else (flair_gt > 0)

    z_indices = get_z_indices(flair_gt.shape[2], args.slice_mode, args.slice_offset)

    image_series = [
        ("GT FLAIR", flair_gt),
        ("Interp", flair_interp),
        ("MedGS Single", flair_single),
        ("MedGS T1-guided", flair_t1g),
    ]
    if flair_t1g_topology is not None:
        image_series.append(("T1-guided + Topology", flair_t1g_topology))
    if flair_t1g_latent is not None:
        image_series.append(("T1-guided + Latent", flair_t1g_latent))
    if flair_our is not None:
        image_series.append(("Our Method", flair_our))
    if flair_alpine_a2 is not None:
        image_series.append(("ALPINE A2", flair_alpine_a2))
    if flair_fused is not None:
        image_series.append(("T1-guided + LR-Fusion", flair_fused))

    err_series = [
        ("Interp |Err|", np.abs(flair_interp - flair_gt)),
        ("T1-guided |Err|", np.abs(flair_t1g - flair_gt)),
    ]
    if flair_t1g_topology is not None:
        err_series.append(("Topology |Err|", np.abs(flair_t1g_topology - flair_gt)))
    if flair_t1g_latent is not None:
        err_series.append(("Latent |Err|", np.abs(flair_t1g_latent - flair_gt)))
    if flair_our is not None:
        err_series.append(("Our Method |Err|", np.abs(flair_our - flair_gt)))
    if flair_alpine_a2 is not None:
        err_series.append(("ALPINE A2 |Err|", np.abs(flair_alpine_a2 - flair_gt)))
    if flair_fused is not None:
        err_series.append(("LR-Fusion |Err|", np.abs(flair_fused - flair_gt)))

    gt_vals = masked_values(flair_gt, mask)
    img_vmin = np.percentile(gt_vals, args.pmin)
    img_vmax = np.percentile(gt_vals, args.pmax)
    err_vals = np.concatenate([masked_values(err_vol, mask) for _, err_vol in err_series], axis=0)
    err_vmin = 0.0
    err_vmax = np.percentile(err_vals, args.pmax)
    if err_vmax <= 0:
        err_vmax = max(float(err_vals.max()), 1.0)

    nrows = len(z_indices)
    ncols = len(image_series) + len(err_series)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.8 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, z in enumerate(z_indices):
        col = 0
        for title, vol in image_series:
            ax = axes[row, col]
            ax.imshow(normalize_window(vol[:, :, z], img_vmin, img_vmax), cmap="gray")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"z={z}")
            ax.axis("off")
            col += 1
        for title, vol in err_series:
            ax = axes[row, col]
            ax.imshow(normalize_window(vol[:, :, z], err_vmin, err_vmax), cmap="hot")
            if row == 0:
                ax.set_title(title)
            ax.axis("off")
            col += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
