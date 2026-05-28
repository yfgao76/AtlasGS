#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def read_subjects(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [str(r.get("subject_id", "")).strip() for r in rows if str(r.get("subject_id", "")).strip()]


def resolve_pred_path(sub_out: Path, modality: str, method: str, factor_z: int) -> Optional[Path]:
    if method == "medgs_single":
        p = sub_out / f"{modality}_medgs_single_z{factor_z}.nii.gz"
        return p if p.exists() else None
    if method == "ours":
        cand = [
            sub_out / f"{modality}_medgs_our_fused_lrcons_z{factor_z}.nii.gz",
            sub_out / f"{modality}_medgs_t1guided_our_z{factor_z}.nii.gz",
        ]
        for p in cand:
            if p.exists():
                return p
        return None
    return None


def masked_ssim_slicewise(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, min_voxels_per_slice: int = 32) -> float:
    if skimage_ssim is None:
        return float("nan")
    m = mask > 0
    vals = gt[m]
    if vals.size == 0:
        return float("nan")
    data_range = float(np.percentile(vals, 99.0) - np.percentile(vals, 1.0))
    if data_range <= 1e-8:
        data_range = float(vals.max() - vals.min())
    if data_range <= 1e-8:
        data_range = 1.0

    z_idx = np.where(np.any(m, axis=(0, 1)))[0]
    if z_idx.size == 0:
        return float("nan")

    score_sum = 0.0
    w_sum = 0.0
    for z in z_idx:
        mz = m[:, :, z]
        n = int(mz.sum())
        if n < int(min_voxels_per_slice):
            continue
        ys, xs = np.where(mz)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mc = mz[y0:y1, x0:x1].astype(np.float32)
        g = gt[y0:y1, x0:x1, z].astype(np.float32) * mc
        p = pred[y0:y1, x0:x1, z].astype(np.float32) * mc
        h, w = int(g.shape[0]), int(g.shape[1])
        win = min(7, h, w)
        if win < 3:
            continue
        if win % 2 == 0:
            win -= 1
        if win < 3:
            continue
        s = skimage_ssim(g, p, data_range=data_range, win_size=win)
        if np.isfinite(s):
            score_sum += float(s) * float(n)
            w_sum += float(n)
    if w_sum <= 0:
        return float("nan")
    return float(score_sum / w_sum)


def safe_mean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute HCP masked SSIM for MedGS and Ours on normal cases.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=3)
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=0.95,
        help="Keep subject-modality pairs where interp.ssim >= threshold.",
    )
    parser.add_argument("--exclude-subject-substrings", type=str, default="sub-06")
    parser.add_argument("--modalities", type=str, default="dwi,asl")
    parser.add_argument("--methods", type=str, default="medgs_single,ours")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    modalities = [x.strip().lower() for x in str(args.modalities).split(",") if x.strip()]
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    excluded_tokens = [x.strip() for x in str(args.exclude_subject_substrings).split(",") if x.strip()]

    subjects = read_subjects(args.test_csv)
    if excluded_tokens:
        subjects = [sid for sid in subjects if not any(tok in sid for tok in excluded_tokens)]

    per_subject: Dict[str, Dict[str, Dict[str, float]]] = {}
    issues: List[str] = []

    old_vals: Dict[str, Dict[str, List[float]]] = {mod: {m: [] for m in methods} for mod in modalities}
    new_vals: Dict[str, Dict[str, List[float]]] = {mod: {m: [] for m in methods} for mod in modalities}

    for sid in subjects:
        sub_data = args.data_root / sid
        sub_out = args.out_root / sid
        if not (sub_data.exists() and sub_out.exists()):
            issues.append(f"{sid}: missing data/out directory")
            continue

        per_subject.setdefault(sid, {})
        for mod in modalities:
            metrics_json_path = sub_out / f"metrics_{mod}_z{args.factor_z}.json"
            metrics_json = {}
            if metrics_json_path.exists():
                try:
                    metrics_json = json.loads(metrics_json_path.read_text())
                except Exception:
                    metrics_json = {}

            interp_ssim = float("nan")
            if "interp" in metrics_json and "ssim" in metrics_json["interp"]:
                try:
                    interp_ssim = float(metrics_json["interp"]["ssim"])
                except Exception:
                    interp_ssim = float("nan")
            if (not math.isfinite(interp_ssim)) or (interp_ssim < float(args.normal_threshold)):
                continue

            gt_path = sub_out / f"{mod}_gt_masked_z{args.factor_z}.nii.gz"
            gt_aff = None
            if gt_path.exists():
                gt, gt_aff, _ = load_nii(gt_path)
            else:
                gt_path = sub_data / f"{mod}_gt.nii.gz"
                if not gt_path.exists():
                    issues.append(f"{sid}/{mod}: missing GT")
                    continue
                gt, gt_aff, _ = load_nii(gt_path)

            mask_path = sub_out / f"{mod}_mask_brain_gt_z{args.factor_z}.nii.gz"
            if mask_path.exists():
                mask, _, _ = load_nii(mask_path)
            else:
                mb_path = sub_data / "mask_brain.nii.gz"
                if not mb_path.exists():
                    issues.append(f"{sid}/{mod}: missing mask")
                    continue
                mask, mask_aff, _ = load_nii(mb_path)
                if tuple(mask.shape) != tuple(gt.shape):
                    mask = resample_to_ref(mask, mask_aff, gt_aff, gt.shape, order=0)
            mask = (mask > 0).astype(np.uint8)
            if int(mask.sum()) == 0:
                issues.append(f"{sid}/{mod}: empty brain mask")
                continue

            per_subject[sid].setdefault(mod, {})
            for method in methods:
                pred_path = resolve_pred_path(sub_out, mod, method, int(args.factor_z))
                if pred_path is None:
                    issues.append(f"{sid}/{mod}/{method}: missing prediction")
                    continue
                pred, pred_aff, _ = load_nii(pred_path)
                if tuple(pred.shape) != tuple(gt.shape):
                    pred = resample_to_ref(pred, pred_aff, gt_aff, gt.shape, order=1)

                old_key = method
                if method == "ours":
                    old_key = "medgs_our"
                old_ssim = float("nan")
                if old_key in metrics_json and "ssim" in metrics_json[old_key]:
                    try:
                        old_ssim = float(metrics_json[old_key]["ssim"])
                    except Exception:
                        old_ssim = float("nan")

                new_ssim = masked_ssim_slicewise(pred, gt, mask)
                per_subject[sid][mod][method] = {
                    "old_ssim": old_ssim,
                    "new_masked_ssim": float(new_ssim),
                }
                if math.isfinite(old_ssim):
                    old_vals[mod][method].append(old_ssim)
                if math.isfinite(new_ssim):
                    new_vals[mod][method].append(float(new_ssim))

    summary: Dict[str, Dict] = {}
    for mod in modalities:
        summary[mod] = {}
        for method in methods:
            summary[mod][method] = {
                "n_old": int(len(old_vals[mod][method])),
                "n_new": int(len(new_vals[mod][method])),
                "old_ssim_mean": safe_mean(old_vals[mod][method]),
                "new_masked_ssim_mean": safe_mean(new_vals[mod][method]),
            }

    payload = {
        "subjects_evaluated": len(subjects),
        "modalities": modalities,
        "methods": methods,
        "normal_threshold": float(args.normal_threshold),
        "excluded_substring_tokens": excluded_tokens,
        "summary": summary,
        "issues_count": len(issues),
        "issues": issues[:200],
        "per_subject": per_subject,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modality", "method", "n_old", "old_ssim_mean", "n_new", "new_masked_ssim_mean"])
        for mod in modalities:
            for method in methods:
                s = summary[mod][method]
                w.writerow(
                    [
                        mod,
                        method,
                        s["n_old"],
                        s["old_ssim_mean"],
                        s["n_new"],
                        s["new_masked_ssim_mean"],
                    ]
                )

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    for mod in modalities:
        for method in methods:
            s = summary[mod][method]
            print(
                f"{mod}/{method}: old={s['old_ssim_mean']:.6f} (n={s['n_old']}), "
                f"new={s['new_masked_ssim_mean']:.6f} (n={s['n_new']})"
            )


if __name__ == "__main__":
    main()
