#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.ndimage import distance_transform_edt

from atlasgs.ops.metrics import masked_mae, masked_psnr, masked_ssim
from atlasgs.ops.nifti_io import load_nii


def t_critical(confidence: float, dof: int) -> float:
    if dof <= 0:
        return 0.0
    try:
        from scipy.stats import t as t_dist  # type: ignore

        return float(t_dist.ppf((1.0 + confidence) * 0.5, dof))
    except Exception:
        return 1.959963984540054


def summarize_arr(arr: np.ndarray, confidence: float) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
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


def summarize_metrics_lists(values: Dict[str, Dict[str, Dict[str, List[float]]]], confidence: float) -> Dict:
    out: Dict[str, Dict] = {}
    for modality, region_dict in values.items():
        out[modality] = {}
        for region, method_dict in region_dict.items():
            out[modality][region] = {}
            for method, metric_dict in method_dict.items():
                out[modality][region][method] = {}
                for metric_name, vals in metric_dict.items():
                    s = summarize_arr(np.asarray(vals, dtype=np.float64), confidence)
                    if s:
                        out[modality][region][method][metric_name] = s
    return out


def add_metrics(bucket: Dict, modality: str, region: str, method: str, metrics: Dict[str, float]) -> None:
    b1 = bucket.setdefault(modality, {})
    b2 = b1.setdefault(region, {})
    b3 = b2.setdefault(method, {})
    for k, v in metrics.items():
        fv = float(v)
        if math.isfinite(fv):
            b3.setdefault(k, []).append(fv)


def roi_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if int(mask.sum()) == 0:
        return {}
    out = {
        "mae": masked_mae(pred, gt, mask),
        "psnr": masked_psnr(pred, gt, mask),
    }
    ssim_val = masked_ssim(pred, gt, mask)
    if ssim_val is not None:
        out["ssim"] = float(ssim_val)
    return out


def candidate_preds(subject_out: Path, subject_id: str, modality: str, factor_z: int, inr_root: Path) -> Dict[str, Path]:
    pred = {
        "interp": subject_out / f"{modality}_interp_z{factor_z}.nii.gz",
        "medgs_single": subject_out / f"{modality}_medgs_single_z{factor_z}.nii.gz",
        "medgs_t1guided": subject_out / f"{modality}_medgs_t1guided_z{factor_z}.nii.gz",
        "medgs_our": subject_out / f"{modality}_medgs_t1guided_our_z{factor_z}.nii.gz",
        "medgs_our_fused_lrcons": subject_out / f"{modality}_medgs_our_fused_lrcons_z{factor_z}.nii.gz",
        "inr_external": inr_root / "images" / subject_id / f"{subject_id}_ds{factor_z}_{modality}_pred.nii.gz",
    }
    return {k: v for k, v in pred.items() if v.exists()}


def write_csv(summary: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "modality",
                "region",
                "method",
                "metric",
                "n",
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "ci_half_width",
            ]
        )
        for modality, region_dict in summary.items():
            for region, method_dict in region_dict.items():
                for method, metric_dict in method_dict.items():
                    for metric_name, s in metric_dict.items():
                        w.writerow(
                            [
                                modality,
                                region,
                                method,
                                metric_name,
                                int(s["n"]),
                                f"{s['mean']:.10g}",
                                f"{s['std']:.10g}",
                                f"{s['ci_low']:.10g}",
                                f"{s['ci_high']:.10g}",
                                f"{s['ci_half_width']:.10g}",
                            ]
                        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GBM ROI evaluation (tumor and peritumoral ring) with external INR comparisons."
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inr-flair-root", type=Path, required=True)
    parser.add_argument("--inr-t2-root", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--ring-inner-mm", type=float, default=5.0)
    parser.add_argument("--ring-outer-mm", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    data_root = args.data_root.resolve()
    inr_roots = {"flair": args.inr_flair_root.resolve(), "t2": args.inr_t2_root.resolve()}

    subjects = sorted([p.name for p in out_root.iterdir() if p.is_dir()])
    values: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    per_subject: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    issues: List[str] = []
    skipped_no_tumor = 0

    for idx, sid in enumerate(subjects, start=1):
        print(f"[{idx}/{len(subjects)}] {sid}", flush=True)
        d_out = out_root / sid
        d_data = data_root / sid
        if not d_data.exists():
            issues.append(f"{sid}: missing data directory")
            continue

        brain, _, _ = load_nii(d_data / "mask_brain.nii.gz")
        tumor, _, _ = load_nii(d_data / "tumor_seg.nii.gz")
        brain = (brain > 0)
        tumor = (tumor > 0) & brain
        if int(tumor.sum()) == 0:
            skipped_no_tumor += 1
            continue

        per_subject.setdefault(sid, {})
        for modality in ("flair", "t2"):
            gt_path = d_data / f"{modality}_gt.nii.gz"
            if not gt_path.exists():
                issues.append(f"{sid}/{modality}: missing GT")
                continue
            gt, _, gt_hdr = load_nii(gt_path)
            spacing = tuple(float(x) for x in gt_hdr.get_zooms()[:3]) if gt_hdr is not None else (1.0, 1.0, 1.0)

            dist = distance_transform_edt(~tumor, sampling=spacing)
            ring = (dist >= float(args.ring_inner_mm)) & (dist <= float(args.ring_outer_mm)) & brain & (~tumor)

            preds = candidate_preds(d_out, sid, modality, args.factor_z, inr_roots[modality])
            per_subject[sid].setdefault(modality, {})
            for method, pred_path in preds.items():
                pred, _, _ = load_nii(pred_path)
                if pred.shape != gt.shape:
                    issues.append(f"{sid}/{modality}/{method}: shape mismatch pred={pred.shape} gt={gt.shape}")
                    continue

                t_metrics = roi_metrics(pred, gt, tumor.astype(np.uint8))
                if t_metrics:
                    add_metrics(values, modality, "tumor", method, t_metrics)
                    per_subject[sid][modality].setdefault(method, {})
                    per_subject[sid][modality][method]["tumor"] = t_metrics

                r_metrics = roi_metrics(pred, gt, ring.astype(np.uint8))
                if r_metrics:
                    add_metrics(values, modality, "peritumor_ring_5_10mm", method, r_metrics)
                    per_subject[sid][modality].setdefault(method, {})
                    per_subject[sid][modality][method]["peritumor_ring_5_10mm"] = r_metrics

    summary = summarize_metrics_lists(values, args.confidence)

    out_json = args.out_json or (out_root / f"metrics_summary_with_inr_tumor_ring_z{args.factor_z}.json")
    out_csv = args.out_csv or (out_root / f"metrics_summary_with_inr_tumor_ring_z{args.factor_z}.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "out_root": str(out_root),
        "data_root": str(data_root),
        "inr_flair_root": str(inr_roots["flair"]),
        "inr_t2_root": str(inr_roots["t2"]),
        "factor_z": int(args.factor_z),
        "ring_inner_mm": float(args.ring_inner_mm),
        "ring_outer_mm": float(args.ring_outer_mm),
        "confidence": float(args.confidence),
        "num_subject_dirs": len(subjects),
        "num_subjects_skipped_no_tumor": skipped_no_tumor,
        "summary": summary,
        "issues": issues,
        "per_subject_metrics": per_subject,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_csv(summary, out_csv)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Subjects in out_root: {len(subjects)}")
    print(f"Skipped (no tumor voxels): {skipped_no_tumor}")
    print(f"Issues: {len(issues)}")


if __name__ == "__main__":
    main()
