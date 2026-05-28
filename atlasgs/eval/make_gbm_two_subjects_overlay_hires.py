#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from atlasgs.ops.nifti_io import load_nii


def norm01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1.0
    y = (x - vmin) / (vmax - vmin + 1e-8)
    return np.clip(y, 0.0, 1.0)


def get_slice_index(tumor_seg: np.ndarray) -> int:
    zz = np.where(tumor_seg > 0)[2]
    if zz.size == 0:
        return tumor_seg.shape[2] // 2
    return int(np.round(np.mean(zz)))


def load_optional(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr, _, _ = load_nii(path)
    return arr


def extract_pred_slice(pred: np.ndarray, gt_shape: Tuple[int, int, int], z_gt: int) -> Optional[np.ndarray]:
    if pred.ndim != 3:
        return None
    if pred.shape[0] != gt_shape[0] or pred.shape[1] != gt_shape[1]:
        return None
    if pred.shape[2] <= 0 or gt_shape[2] <= 0:
        return None
    if pred.shape[2] == gt_shape[2]:
        z_pred = z_gt
    else:
        z_pred = int(round(z_gt * (pred.shape[2] - 1) / max(gt_shape[2] - 1, 1)))
    z_pred = int(np.clip(z_pred, 0, pred.shape[2] - 1))
    return pred[:, :, z_pred]


def choose_our_path(sub_out: Path, modality: str, factor_z: int) -> Optional[Path]:
    cand = [
        sub_out / f"{modality}_medgs_our_fused_lrcons_z{factor_z}.nii.gz",
        sub_out / f"{modality}_medgs_t1guided_our_z{factor_z}.nii.gz",
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def get_method_volumes(
    sid: str,
    modality: str,
    factor_z: int,
    data_root: Path,
    outputs_root: Path,
    sa_inr_root: Path,
) -> Dict[str, Optional[np.ndarray]]:
    sub_data = data_root / sid
    sub_out = outputs_root / sid
    gt = load_optional(sub_data / f"{modality}_gt.nii.gz")
    linear = load_optional(sub_out / f"{modality}_interp_z{factor_z}.nii.gz")
    sa_inr = load_optional(sa_inr_root / "predictions" / f"{sid}_ds{factor_z}_{modality}_pred.nii.gz")
    medgs = load_optional(sub_out / f"{modality}_medgs_single_z{factor_z}.nii.gz")
    our_p = choose_our_path(sub_out, modality, factor_z)
    our = load_optional(our_p) if our_p is not None else None
    mask = load_optional(sub_data / "mask_brain.nii.gz")
    tumor = load_optional(sub_data / "tumor_seg.nii.gz")

    return {
        "gt": gt,
        "linear": linear,
        "sa_inr": sa_inr,
        "medgs": medgs,
        "our_lrcons": our,
        "brain_mask": mask,
        "tumor_seg": tumor,
    }


def put_missing(ax, title: str) -> None:
    ax.imshow(np.zeros((32, 32), dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    ax.text(0.5, 0.5, "MISSING", color="red", fontsize=10, ha="center", va="center", transform=ax.transAxes)
    ax.text(
        0.02,
        0.98,
        title,
        color="white",
        fontsize=11,
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
    )
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="High-resolution GBM overlay figure for two subjects.")
    parser.add_argument("--subjects", type=str, required=True, help="Comma-separated subject ids (expect two).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--sa-inr-flair-root", type=Path, required=True)
    parser.add_argument("--sa-inr-t2-root", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if len(subjects) != 2:
        raise ValueError("Please provide exactly two subjects.")

    modalities = ["flair", "t2"]
    methods = [("linear", "Linear"), ("sa_inr", "SA-INR"), ("medgs", "MedGS"), ("our_lrcons", "MedGS-Our-LRCons")]

    # 2 subjects x 2 modalities x (recon+error) rows
    nrows = len(subjects) * len(modalities) * 2
    ncols = 5  # GT + 4 methods
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2.8 * nrows), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    pending_colorbars = []
    row = 0
    for sid in subjects:
        for modality in modalities:
            sa_root = args.sa_inr_flair_root if modality == "flair" else args.sa_inr_t2_root
            vols = get_method_volumes(
                sid=sid,
                modality=modality,
                factor_z=args.factor_z,
                data_root=args.data_root,
                outputs_root=args.outputs_root,
                sa_inr_root=sa_root,
            )
            gt = vols["gt"]
            tumor = vols["tumor_seg"]
            brain = vols["brain_mask"]
            if gt is None or tumor is None:
                raise FileNotFoundError(f"Missing GT/tumor for {sid} {modality}")
            if brain is None or brain.shape != gt.shape:
                brain = gt > 0
            else:
                brain = brain > 0

            z = get_slice_index(tumor)
            gt_vals = gt[brain > 0]
            p1 = float(np.percentile(gt_vals, 1.0)) if gt_vals.size > 0 else float(np.min(gt))
            p99 = float(np.percentile(gt_vals, 99.0)) if gt_vals.size > 0 else float(np.max(gt))
            gt_slice = gt[:, :, z]
            ts = (tumor[:, :, z] > 0).astype(np.float32)

            # Recon row
            ax = axes[row, 0]
            ax.set_facecolor("black")
            ax.imshow(norm01(gt_slice, p1, p99), cmap="gray", interpolation="none")
            if np.any(ts > 0):
                ax.contour(ts, levels=[0.5], colors="lime", linewidths=1.4)
            ax.text(
                0.02,
                0.98,
                f"{sid} | {modality.upper()} GT | z={z}",
                color="white",
                fontsize=11,
                ha="left",
                va="top",
                transform=ax.transAxes,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
            )
            ax.axis("off")

            err_slices = []
            for c, (key, name) in enumerate(methods, start=1):
                pred = vols[key]
                ax = axes[row, c]
                ax.set_facecolor("black")
                if pred is None:
                    put_missing(ax, name)
                    err_slices.append(None)
                    continue
                ps = extract_pred_slice(pred, gt.shape, z)
                if ps is None:
                    put_missing(ax, name)
                    err_slices.append(None)
                    continue
                ax.imshow(norm01(ps, p1, p99), cmap="gray", interpolation="none")
                if np.any(ts > 0):
                    ax.contour(ts, levels=[0.5], colors="lime", linewidths=1.4)
                ax.text(
                    0.02,
                    0.98,
                    name,
                    color="white",
                    fontsize=11,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
                )
                ax.axis("off")
                err_slices.append(np.abs(ps - gt_slice))

            # Error row with per-row fixed color scale and side bars
            row_err = row + 1
            ax = axes[row_err, 0]
            ax.set_facecolor("black")
            ax.imshow(norm01(gt_slice, p1, p99), cmap="gray", interpolation="none")
            if np.any(ts > 0):
                ax.contour(ts.astype(np.float32), levels=[0.5], colors="lime", linewidths=1.2)
            ax.text(
                0.02,
                0.98,
                "GT + tumor contour",
                color="white",
                fontsize=11,
                ha="left",
                va="top",
                transform=ax.transAxes,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
            )
            ax.axis("off")

            bz = (brain[:, :, z] > 0).astype(np.uint8)
            err_vals = (
                np.concatenate([e[bz > 0].reshape(-1) for e in err_slices if e is not None], axis=0)
                if any(e is not None for e in err_slices)
                else np.array([1.0], dtype=np.float32)
            )
            emax = float(np.percentile(err_vals, 99.0))
            emax = max(emax, 1e-4)

            row_err_axes = [axes[row_err, c] for c in range(ncols)]
            im_for_cb = None
            for c, ((key, name), err) in enumerate(zip(methods, err_slices), start=1):
                ax = axes[row_err, c]
                ax.set_facecolor("black")
                if err is None:
                    put_missing(ax, f"{name} |Err|")
                    continue
                im = ax.imshow(err, cmap="hot", vmin=0.0, vmax=emax, interpolation="none")
                if np.any(ts > 0):
                    ax.contour(ts.astype(np.float32), levels=[0.5], colors="lime", linewidths=1.2)
                ax.text(
                    0.02,
                    0.98,
                    f"{name} |Err|",
                    color="white",
                    fontsize=11,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
                )
                ax.axis("off")
                im_for_cb = im

            # Defer one colorbar per error row; place at far right after layout.
            if im_for_cb is not None:
                pending_colorbars.append((row_err, im_for_cb))

            row += 2

    fig.subplots_adjust(left=0.0, right=0.955, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    for row_err, im_for_cb in pending_colorbars:
        last_ax = axes[row_err, ncols - 1]
        pos = last_ax.get_position()
        cax_x = min(pos.x1 + 0.008, 0.992)
        cax_w = 0.010
        cax = fig.add_axes([cax_x, pos.y0, cax_w, pos.height])
        cb = fig.colorbar(im_for_cb, cax=cax)
        cb.ax.tick_params(labelsize=9, colors="white")
        cb.outline.set_edgecolor("white")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
