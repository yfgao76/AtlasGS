#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def normalize_window(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def pick_orthogonal_indices(mask: np.ndarray) -> Dict[str, int]:
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        x, y, z = mask.shape
        return {"sagittal": x // 2, "coronal": y // 2, "axial": z // 2}
    return {
        "sagittal": int(np.median(coords[0])),
        "coronal": int(np.median(coords[1])),
        "axial": int(np.median(coords[2])),
    }


def extract_plane(vol: np.ndarray, axis_name: str, index: int) -> np.ndarray:
    if axis_name == "axial":
        return vol[:, :, index]
    if axis_name == "coronal":
        return np.rot90(vol[:, index, :], 1)
    if axis_name == "sagittal":
        return np.rot90(vol[index, :, :], 1)
    raise ValueError(f"Unknown axis_name={axis_name}")


def center_crop_square(img2d: np.ndarray, side: int) -> np.ndarray:
    h, w = img2d.shape[:2]
    side = int(max(1, min(side, h, w)))
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img2d[y0 : y0 + side, x0 : x0 + side]


def plane_bbox(mask2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def plane_psnr(pred2d: np.ndarray, gt2d: np.ndarray, mask2d: np.ndarray) -> float:
    valid = mask2d > 0
    if np.count_nonzero(valid) < 16:
        return float("nan")
    diff = pred2d[valid] - gt2d[valid]
    mse = float(np.mean(diff * diff))
    gt_vals = gt2d[valid]
    data_range = float(np.percentile(gt_vals, 99.0) - np.percentile(gt_vals, 1.0))
    if data_range <= 0:
        data_range = float(gt_vals.max() - gt_vals.min())
    if data_range <= 0:
        data_range = 1.0
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def plane_ssim(pred2d: np.ndarray, gt2d: np.ndarray, mask2d: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")
    box = plane_bbox(mask2d)
    if box is None:
        return float("nan")
    y0, y1, x0, x1 = box
    pred_crop = pred2d[y0 : y1 + 1, x0 : x1 + 1]
    gt_crop = gt2d[y0 : y1 + 1, x0 : x1 + 1]
    gt_vals = gt_crop.reshape(-1)
    data_range = float(np.percentile(gt_vals, 99.0) - np.percentile(gt_vals, 1.0))
    if data_range <= 0:
        data_range = float(gt_vals.max() - gt_vals.min())
    if data_range <= 0:
        data_range = 1.0
    return float(skimage_ssim(gt_crop, pred_crop, data_range=data_range))


def align_to_gt(vol: np.ndarray, vol_aff: np.ndarray, gt_aff: np.ndarray, gt_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    if tuple(vol.shape) == tuple(gt_shape):
        return vol
    return resample_to_ref(vol, vol_aff, gt_aff, gt_shape, order=order)


def match_intensity_to_gt(pred: Optional[np.ndarray], gt: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if pred is None:
        return None
    valid = mask > 0
    if np.count_nonzero(valid) < 32:
        return pred
    pv = pred[valid]
    gv = gt[valid]
    p1 = float(np.percentile(pv, 1.0))
    p99 = float(np.percentile(pv, 99.0))
    g1 = float(np.percentile(gv, 1.0))
    g99 = float(np.percentile(gv, 99.0))
    pr = p99 - p1
    gr = g99 - g1
    if pr <= 1e-6 or gr <= 1e-6:
        return pred
    scale = gr / pr
    bias = g1 - scale * p1
    return pred * scale + bias


def maybe_match_intensity_to_gt(
    pred: Optional[np.ndarray],
    gt: np.ndarray,
    mask: np.ndarray,
    low_ratio: float = 0.25,
    high_ratio: float = 4.0,
) -> Optional[np.ndarray]:
    if pred is None:
        return None
    valid = mask > 0
    if np.count_nonzero(valid) < 32:
        return pred
    pv = pred[valid]
    gv = gt[valid]
    pr = float(np.percentile(pv, 99.0) - np.percentile(pv, 1.0))
    gr = float(np.percentile(gv, 99.0) - np.percentile(gv, 1.0))
    if pr <= 1e-6 or gr <= 1e-6:
        return pred
    ratio = pr / gr
    if (ratio < low_ratio) or (ratio > high_ratio):
        # Only apply affine intensity matching when scales are clearly inconsistent
        # (e.g., INR in [0,1] vs MRI in native intensity range).
        return match_intensity_to_gt(pred, gt, mask)
    return pred


def load_pred(path: Path, gt_aff: np.ndarray, gt_shape: Tuple[int, int, int], order: int = 1) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        vol, aff, _ = load_nii(path)
    except Exception:
        return None
    return align_to_gt(vol, aff, gt_aff, gt_shape, order=order)


def render_subject(
    subject_id: str,
    factor_z: int,
    data_root: Path,
    out_root_z: Path,
    inr_root: Path,
    sa_root: Path,
    overlay_root: Optional[Path],
    panel_width: float,
    panel_height: float,
    dpi: int,
    metric_fontsize: int,
    metric_color: str,
    square_crop: bool,
    overwrite: bool,
) -> Optional[Path]:
    sub_data = data_root / subject_id
    sub_out = out_root_z / subject_id
    out_sub = (overlay_root / subject_id) if overlay_root is not None else sub_out
    out_png = out_sub / f"overlays_orth_z{factor_z}.png"
    default_png = out_sub / f"overlays_z{factor_z}.png"
    if out_png.exists() and not overwrite:
        return out_png

    gt_path = sub_data / "flair_gt.nii.gz"
    t1_path = sub_data / "t1_gt.nii.gz"
    lr_path = sub_data / f"flair_lr_1x1x{factor_z}.nii.gz"
    linear_path = sub_out / f"flair_interp_z{factor_z}.nii.gz"
    medgs_path = sub_out / f"flair_medgs_single_z{factor_z}.nii.gz"
    ours_path = sub_out / f"flair_medgs_our_fused_lrcons_z{factor_z}.nii.gz"
    if not ours_path.exists():
        ours_path = sub_out / f"flair_medgs_our_z{factor_z}.nii.gz"
    inr_path = inr_root / "images" / subject_id / f"{subject_id}_ds{factor_z}_pred.nii.gz"
    sa_path = sa_root / "predictions" / f"{subject_id}_ds{factor_z}_pred.nii.gz"

    if not (gt_path.exists() and t1_path.exists() and lr_path.exists() and linear_path.exists()):
        return None

    try:
        gt, gt_aff, _ = load_nii(gt_path)
        t1, _, _ = load_nii(t1_path)
    except Exception:
        return None
    gt_shape = tuple(gt.shape)
    mask = (t1 > 0) if t1.shape == gt.shape else (gt > 0)

    try:
        lr, lr_aff, _ = load_nii(lr_path)
    except Exception:
        return None
    cubic = resample_to_ref(lr, lr_aff, gt_aff, gt_shape, order=3)

    linear = load_pred(linear_path, gt_aff, gt_shape, order=1)
    medgs = load_pred(medgs_path, gt_aff, gt_shape, order=1)
    ours = load_pred(ours_path, gt_aff, gt_shape, order=1)
    inr = load_pred(inr_path, gt_aff, gt_shape, order=1)
    sa = load_pred(sa_path, gt_aff, gt_shape, order=1)

    # External baselines are not always in the same intensity scale.
    # Keep native scales for methods already in MRI intensity domain.
    inr = maybe_match_intensity_to_gt(inr, gt, mask)
    sa = maybe_match_intensity_to_gt(sa, gt, mask)

    methods: List[Tuple[str, Optional[np.ndarray]]] = [
        ("GT", gt),
        ("Linear", linear),
        ("Cubic", cubic),
        ("INR", inr),
        ("SA-INR", sa),
        ("MedGS", medgs),
        ("Ours", ours),
    ]

    valid_vals = gt[mask > 0]
    vmin = float(np.percentile(valid_vals, 1.0))
    vmax = float(np.percentile(valid_vals, 99.0))

    idx = pick_orthogonal_indices(mask)
    square_side = int(min(gt.shape))
    planes = [
        ("Axial", "axial", idx["axial"]),
        ("Coronal", "coronal", idx["coronal"]),
        ("Sagittal", "sagittal", idx["sagittal"]),
    ]

    nrows = len(planes)
    ncols = len(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_width * ncols, panel_height * nrows))
    fig.patch.set_facecolor("black")

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (plane_title, plane_axis, plane_index) in enumerate(planes):
        gt_plane = extract_plane(gt, plane_axis, plane_index)
        mask_plane = extract_plane(mask.astype(np.uint8), plane_axis, plane_index)
        if square_crop:
            gt_plane = center_crop_square(gt_plane, square_side)
            mask_plane = center_crop_square(mask_plane, square_side)
        for c, (method_name, vol) in enumerate(methods):
            ax = axes[r, c]
            ax.set_facecolor("black")
            if vol is None:
                ax.imshow(np.zeros_like(gt_plane), cmap="gray", vmin=0.0, vmax=1.0)
                ax.text(
                    0.5,
                    0.5,
                    "MISSING",
                    color="red",
                    fontsize=14,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                metric_text = None
            else:
                pred_plane = extract_plane(vol, plane_axis, plane_index)
                if square_crop:
                    pred_plane = center_crop_square(pred_plane, square_side)
                disp = normalize_window(pred_plane, vmin, vmax)
                ax.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0)
                if method_name == "GT":
                    metric_text = None
                else:
                    ssim_val = plane_ssim(pred_plane, gt_plane, mask_plane)
                    psnr_val = plane_psnr(pred_plane, gt_plane, mask_plane)
                    if np.isfinite(ssim_val) and np.isfinite(psnr_val):
                        metric_text = f"SSIM {ssim_val:.3f}\nPSNR {psnr_val:.2f}"
                    elif np.isfinite(psnr_val):
                        metric_text = f"SSIM n/a\nPSNR {psnr_val:.2f}"
                    else:
                        metric_text = "SSIM n/a\nPSNR n/a"
            if metric_text:
                ax.text(
                    0.98,
                    0.02,
                    metric_text,
                    color=metric_color,
                    fontsize=metric_fontsize,
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
                )
            if r == 0:
                ax.text(
                    0.02,
                    0.98,
                    method_name,
                    color="white",
                    fontsize=16,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                )
            if c == 0:
                ax.text(
                    0.02,
                    0.02,
                    f"{plane_title}",
                    color="#88ffff",
                    fontsize=15,
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                )
            ax.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Keep compatibility with previous path conventions.
    if overlay_root is None:
        try:
            import shutil

            shutil.copy2(out_png, default_png)
        except Exception:
            pass
    return out_png


def discover_subjects(out_root_z: Path) -> List[str]:
    subs = []
    for p in sorted(out_root_z.iterdir()):
        if p.is_dir() and p.name.isdigit():
            subs.append(p.name)
    return subs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UKBB z7 overlays with orthogonal planes and per-plane SSIM/PSNR.")
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--inr-root", type=Path, default=Path("external_metrics/brain-gs/inr_ukbb_ds7"))
    parser.add_argument("--sa-root", type=Path, default=Path("external_metrics/brain-gs/sa_inr_ukbb_ds7"))
    parser.add_argument("--overlay-root", type=Path, default=None, help="Optional writable output root for overlays. Files will be saved under <overlay_root>/<subject_id>/")
    parser.add_argument("--panel-width", type=float, default=4.4)
    parser.add_argument("--panel-height", type=float, default=4.4)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--metric-fontsize", type=int, default=18)
    parser.add_argument("--metric-color", type=str, default="yellow")
    parser.add_argument("--no-square-crop", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    out_root_z = args.out_root / f"z{args.factor_z}"
    if not out_root_z.exists():
        raise FileNotFoundError(f"Missing folder: {out_root_z}")

    if args.subject_id:
        subjects = [str(args.subject_id)]
    else:
        subjects = discover_subjects(out_root_z)

    ok: List[str] = []
    miss: List[str] = []
    for sid in subjects:
        out = render_subject(
            subject_id=sid,
            factor_z=args.factor_z,
            data_root=args.data_root,
            out_root_z=out_root_z,
            inr_root=args.inr_root,
            sa_root=args.sa_root,
            overlay_root=args.overlay_root,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
            dpi=args.dpi,
            metric_fontsize=int(args.metric_fontsize),
            metric_color=str(args.metric_color),
            square_crop=not bool(args.no_square_crop),
            overwrite=args.overwrite,
        )
        if out is None:
            miss.append(sid)
        else:
            ok.append(sid)

    summary = {
        "factor_z": int(args.factor_z),
        "total_subjects_requested": len(subjects),
        "generated_count": len(ok),
        "missing_count": len(miss),
        "generated_subjects": ok,
        "missing_subjects": miss,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
