#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref

try:
    from scipy.stats import t as t_dist  # type: ignore
except Exception:
    t_dist = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


METHOD_KEYS = [
    "interp",
    "cubic",
    "mc_inr",
    "sa_inr",
    "alpine",
    "medgs",
    "ours",
]

METHOD_LABELS = {
    "interp": "Interp",
    "cubic": "Cubic",
    "mc_inr": "MC-INR",
    "sa_inr": "SA-INR",
    "alpine": "ALPINE",
    "medgs": "MedGS",
    "ours": "Ours",
}


def parse_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def t_critical(confidence: float, dof: int) -> float:
    if dof <= 0:
        return 0.0
    if t_dist is not None:
        return float(t_dist.ppf((1.0 + confidence) * 0.5, dof))
    return 1.959963984540054


def summarize_array(values: List[float], confidence: float) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(std / math.sqrt(n)) if n > 1 else 0.0
    half = float(t_critical(confidence, n - 1) * se) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half_width": half,
    }


def aggregate(per_subject: Dict[str, Dict[str, Dict[str, float]]], confidence: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for _, mdict in per_subject.items():
        for m, metrics in mdict.items():
            b = bucket.setdefault(m, {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    b.setdefault(k, []).append(float(v))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m, md in bucket.items():
        out[m] = {}
        for k, vals in md.items():
            s = summarize_array(vals, confidence)
            if s:
                out[m][k] = s
    return out


def bbox3d(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None
    return (
        int(coords[0].min()),
        int(coords[0].max()) + 1,
        int(coords[1].min()),
        int(coords[1].max()) + 1,
        int(coords[2].min()),
        int(coords[2].max()) + 1,
    )


def masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if np.count_nonzero(valid) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if np.count_nonzero(valid) == 0:
        return float("nan")
    diff = pred[valid] - gt[valid]
    mse = float(np.mean(diff * diff))
    gt_vals = gt[valid]
    data_range = float(gt_vals.max() - gt_vals.min())
    if data_range <= 0:
        data_range = 1.0
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def masked_ssim_3d(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")
    bb = bbox3d(mask)
    if bb is None:
        return float("nan")
    x0, x1, y0, y1, z0, z1 = bb
    m = (mask[x0:x1, y0:y1, z0:z1] > 0).astype(np.float32)
    g = gt[x0:x1, y0:y1, z0:z1].astype(np.float32) * m
    p = pred[x0:x1, y0:y1, z0:z1].astype(np.float32) * m
    gv = gt[mask > 0]
    dr = float(gv.max() - gv.min())
    if dr <= 0:
        dr = 1.0
    return float(skimage_ssim(g, p, data_range=dr))


def metric_triplet(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return {
        "mae": masked_mae(pred, gt, mask),
        "ssim": masked_ssim_3d(pred, gt, mask),
        "psnr": masked_psnr(pred, gt, mask),
    }


def load_and_align(path: Path, gt_aff: np.ndarray, gt_shape: Tuple[int, int, int], order: int = 1) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        vol, aff, _ = load_nii(path)
    except Exception:
        return None
    if tuple(vol.shape) != tuple(gt_shape):
        try:
            vol = resample_to_ref(vol, aff, gt_aff, gt_shape, order=order)
        except Exception:
            return None
    return vol.astype(np.float32, copy=False)


def maybe_scale_match(pred: Optional[np.ndarray], gt: np.ndarray, mask: np.ndarray, low_ratio: float = 0.2, high_ratio: float = 5.0) -> Optional[np.ndarray]:
    if pred is None:
        return None
    valid = mask > 0
    if np.count_nonzero(valid) < 64:
        return pred
    pv = pred[valid]
    gv = gt[valid]
    pr = float(np.percentile(pv, 99.0) - np.percentile(pv, 1.0))
    gr = float(np.percentile(gv, 99.0) - np.percentile(gv, 1.0))
    if pr <= 1e-6 or gr <= 1e-6:
        return pred
    ratio = pr / gr
    if (ratio >= low_ratio) and (ratio <= high_ratio):
        return pred
    p1 = float(np.percentile(pv, 1.0))
    p99 = float(np.percentile(pv, 99.0))
    g1 = float(np.percentile(gv, 1.0))
    g99 = float(np.percentile(gv, 99.0))
    if p99 <= p1:
        return pred
    scale = (g99 - g1) / (p99 - p1 + 1e-8)
    bias = g1 - scale * p1
    return (pred * scale + bias).astype(np.float32, copy=False)


def collect_subject_metrics_for_factor(
    factor_z: int,
    data_root: Path,
    out_root: Path,
    inr_base: Path,
    apply_scale_match: bool,
    num_workers: int = 1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out_z = out_root / f"z{factor_z}"
    if not out_z.exists():
        return {}

    per_subject: Dict[str, Dict[str, Dict[str, float]]] = {}
    subject_dirs = sorted([p for p in out_z.iterdir() if p.is_dir() and p.name.isdigit()])
    subject_ids = [p.name for p in subject_dirs]

    inr_root = inr_base / f"inr_ukbb_ds{factor_z}" / "images"
    sa_root = inr_base / f"sa_inr_ukbb_ds{factor_z}" / "predictions"

    def worker(sid: str) -> Tuple[str, Dict[str, Dict[str, float]]]:
        sub = out_z / sid
        gt_path = data_root / sid / "flair_gt.nii.gz"
        t1_path = data_root / sid / "t1_gt.nii.gz"
        lr_path = data_root / sid / f"flair_lr_1x1x{factor_z}.nii.gz"
        if not (gt_path.exists() and t1_path.exists() and lr_path.exists()):
            return sid, {}
        try:
            gt, gt_aff, _ = load_nii(gt_path)
            t1, _, _ = load_nii(t1_path)
        except Exception:
            return sid, {}
        mask = ((t1 > 0) if t1.shape == gt.shape else (gt > 0)).astype(np.uint8)
        gt_shape = tuple(gt.shape)

        try:
            lr, lr_aff, _ = load_nii(lr_path)
            cubic = resample_to_ref(lr, lr_aff, gt_aff, gt_shape, order=3).astype(np.float32)
        except Exception:
            cubic = None

        cubic_saved = load_and_align(sub / f"flair_interp_bspline3_z{factor_z}.nii.gz", gt_aff, gt_shape, order=3)
        cubic_pred = cubic_saved if cubic_saved is not None else cubic

        preds: Dict[str, Optional[np.ndarray]] = {
            "interp": load_and_align(sub / f"flair_interp_z{factor_z}.nii.gz", gt_aff, gt_shape, order=1),
            "cubic": cubic_pred,
            "mc_inr": load_and_align(inr_root / sid / f"{sid}_ds{factor_z}_pred.nii.gz", gt_aff, gt_shape, order=1),
            "sa_inr": load_and_align(sa_root / f"{sid}_ds{factor_z}_pred.nii.gz", gt_aff, gt_shape, order=1),
            "alpine": load_and_align(sub / f"flair_alpine_a2_z{factor_z}.nii.gz", gt_aff, gt_shape, order=1),
            "medgs": load_and_align(sub / f"flair_medgs_single_z{factor_z}.nii.gz", gt_aff, gt_shape, order=1),
            "ours": load_and_align(sub / f"flair_medgs_our_fused_lrcons_z{factor_z}.nii.gz", gt_aff, gt_shape, order=1),
        }
        if preds["ours"] is None:
            preds["ours"] = load_and_align(sub / f"flair_medgs_our_z{factor_z}.nii.gz", gt_aff, gt_shape, order=1)

        if apply_scale_match:
            # Normalize-scale handling for methods that may output normalized intensity.
            for k in ("mc_inr", "sa_inr"):
                preds[k] = maybe_scale_match(preds[k], gt, mask)

        md: Dict[str, Dict[str, float]] = {}
        for m in METHOD_KEYS:
            p = preds.get(m)
            if p is None:
                continue
            mm = metric_triplet(p, gt, mask)
            if all(k in mm and math.isfinite(mm[k]) for k in ("mae", "ssim", "psnr")):
                md[m] = mm
        return sid, md

    if num_workers <= 1:
        for sid in subject_ids:
            sid2, md = worker(sid)
            if md:
                per_subject[sid2] = md
    else:
        with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
            for sid2, md in ex.map(worker, subject_ids):
                if md:
                    per_subject[sid2] = md
    return per_subject


def build_matched_core(
    per_subject: Dict[str, Dict[str, Dict[str, float]]],
    required_methods: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out = {}
    for sid, md in per_subject.items():
        if all(m in md for m in required_methods):
            out[sid] = md
    return out


def write_csv(summary: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "scope", "method", "metric", "n", "mean", "std", "ci_low", "ci_high", "ci_half_width"])
        for factor_key, fdata in summary["summary_by_factor"].items():
            for scope in ("all_available", "matched_core"):
                sd = fdata.get(scope, {})
                for m, md in sd.items():
                    for metric, s in md.items():
                        w.writerow([
                            factor_key,
                            scope,
                            m,
                            metric,
                            s["n"],
                            f"{s['mean']:.10g}",
                            f"{s['std']:.10g}",
                            f"{s['ci_low']:.10g}",
                            f"{s['ci_high']:.10g}",
                            f"{s['ci_half_width']:.10g}",
                        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="UKBB metrics with brain mask + scale handling (z3/5/7).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--inr-base", type=Path, required=True)
    parser.add_argument("--factors", type=str, default="3,5,7")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--apply-scale-match", action="store_true")
    parser.add_argument("--required-methods", type=str, default="interp,cubic,mc_inr,sa_inr,alpine,medgs,ours")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    factors = parse_list(args.factors)
    required_methods = [x.strip() for x in args.required_methods.split(",") if x.strip()]

    payload = {
        "factors": factors,
        "confidence": args.confidence,
        "apply_scale_match": bool(args.apply_scale_match),
        "required_methods": required_methods,
        "summary_by_factor": {},
        "counts_by_factor": {},
    }

    for z in factors:
        key = f"z{z}"
        per_subject = collect_subject_metrics_for_factor(
            factor_z=z,
            data_root=args.data_root,
            out_root=args.out_root,
            inr_base=args.inr_base,
            apply_scale_match=bool(args.apply_scale_match),
            num_workers=max(1, int(args.num_workers)),
        )
        matched = build_matched_core(per_subject, required_methods)

        payload["summary_by_factor"][key] = {
            "all_available": aggregate(per_subject, args.confidence),
            "matched_core": aggregate(matched, args.confidence),
        }
        payload["counts_by_factor"][key] = {
            "subjects_all_available_pool": len(per_subject),
            "subjects_matched_core": len(matched),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_csv(payload, args.out_csv)
    print(json.dumps(payload["counts_by_factor"], indent=2))


if __name__ == "__main__":
    main()
