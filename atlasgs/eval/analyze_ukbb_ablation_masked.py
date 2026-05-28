#!/usr/bin/env python3
import argparse
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


METHODS = {
    "t1guided": "flair_medgs_t1guided_z{z}.nii.gz",
    "topology": "flair_medgs_t1guided_topology_z{z}.nii.gz",
    "latent": "flair_medgs_t1guided_latent_z{z}.nii.gz",
    "lrfusion": "flair_medgs_fused_lrcons_z{z}.nii.gz",
    "all": "flair_medgs_our_fused_lrcons_z{z}.nii.gz",
}


def parse_factors(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def t_critical(confidence: float, dof: int) -> float:
    if dof <= 0:
        return 0.0
    if t_dist is not None:
        return float(t_dist.ppf((1.0 + confidence) * 0.5, dof))
    return 1.959963984540054


def summarize(values: List[float], confidence: float) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(std / np.sqrt(n)) if n > 1 else 0.0
    half = float(t_critical(confidence, n - 1) * se) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half_width": half,
    }


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


def aggregate(per_subject: Dict[str, Dict[str, Dict[str, float]]], confidence: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in METHODS.keys():
        mvals = {"mae": [], "ssim": [], "psnr": []}
        for sid in per_subject:
            if method not in per_subject[sid]:
                continue
            mm = per_subject[sid][method]
            for k in mvals.keys():
                v = mm.get(k, float("nan"))
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    mvals[k].append(float(v))
        out[method] = {k: summarize(v, confidence) for k, v in mvals.items() if len(v) > 0}
    return out


def write_csv(payload: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "scope", "method", "metric", "n", "mean", "ci_half_width"])
        for z, zdata in payload["summary_by_factor"].items():
            for scope in ("all_available", "matched_core"):
                sd = zdata.get(scope, {})
                for m, md in sd.items():
                    for metric, s in md.items():
                        w.writerow([z, scope, m, metric, s["n"], s["mean"], s["ci_half_width"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="UKBB ablation metrics with masked + crop SSIM method.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--factors", type=str, default="3,5,7")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "factors": parse_factors(args.factors),
        "confidence": args.confidence,
        "summary_by_factor": {},
        "counts_by_factor": {},
    }

    for z in payload["factors"]:
        zkey = f"z{z}"
        out_z = args.out_root / zkey
        subs = sorted([p for p in out_z.iterdir() if p.is_dir() and p.name.isdigit()])

        per_subject: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sub in subs:
            sid = sub.name
            gt_path = args.data_root / sid / "flair_gt.nii.gz"
            t1_path = args.data_root / sid / "t1_gt.nii.gz"
            if not (gt_path.exists() and t1_path.exists()):
                continue
            try:
                gt, gt_aff, _ = load_nii(gt_path)
                t1, _, _ = load_nii(t1_path)
            except Exception:
                continue
            mask = ((t1 > 0) if t1.shape == gt.shape else (gt > 0)).astype(np.uint8)
            md: Dict[str, Dict[str, float]] = {}
            for m, pat in METHODS.items():
                pred = load_and_align(sub / pat.format(z=z), gt_aff, gt.shape, order=1)
                if pred is None and m == "all":
                    pred = load_and_align(sub / f"flair_medgs_our_z{z}.nii.gz", gt_aff, gt.shape, order=1)
                if pred is None:
                    continue
                mm = metric_triplet(pred, gt, mask)
                if all(math.isfinite(mm[k]) for k in ("mae", "ssim", "psnr")):
                    md[m] = mm
            if md:
                per_subject[sid] = md

        matched = {
            sid: md for sid, md in per_subject.items() if all(m in md for m in METHODS.keys())
        }

        payload["summary_by_factor"][zkey] = {
            "all_available": aggregate(per_subject, args.confidence),
            "matched_core": aggregate(matched, args.confidence),
        }
        payload["counts_by_factor"][zkey] = {
            "subjects_all_available_pool": len(per_subject),
            "subjects_matched_core": len(matched),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_csv(payload, args.out_csv)
    print(json.dumps(payload["counts_by_factor"], indent=2))


if __name__ == "__main__":
    main()
