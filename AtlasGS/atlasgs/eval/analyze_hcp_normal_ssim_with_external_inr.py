#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_sa_inr(ssim_json: Path) -> Dict[str, float]:
    if not ssim_json.exists():
        return {}
    try:
        data = json.loads(ssim_json.read_text())
    except Exception:
        return {}
    out: Dict[str, float] = {}
    per_subject = data.get("per_subject", [])
    if not isinstance(per_subject, list):
        return out
    for row in per_subject:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("subject_id", ""))
        ssim = row.get("ssim")
        if sid and isinstance(ssim, (int, float)) and math.isfinite(float(ssim)):
            out[sid] = float(ssim)
    return out


def bspline3_ssim_from_lr(sub_dir: Path, modality: str) -> float:
    gt_path = sub_dir / f"{modality}_gt_masked_z3.nii.gz"
    lr_path = sub_dir / f"{modality}_lr_1x1x3_masked.nii.gz"
    mask_path = sub_dir / f"{modality}_mask_brain_gt_z3.nii.gz"
    if not (gt_path.exists() and lr_path.exists() and mask_path.exists()):
        raise FileNotFoundError(f"missing files under {sub_dir} for {modality}")

    gt, gt_aff, _ = load_nii(gt_path)
    lr, lr_aff, _ = load_nii(lr_path)
    mask, _, _ = load_nii(mask_path)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        raise ValueError(f"empty brain mask in {mask_path}")

    pred = resample_to_ref(lr, lr_aff, gt_aff, gt.shape, order=3)
    gt_vals = gt[mask_bool]
    data_range = float(gt_vals.max() - gt_vals.min())
    if data_range <= 1e-8:
        data_range = 1.0
    # Match existing metrics implementation: full-volume SSIM with mask-based data range.
    return float(skimage_ssim(gt, pred, data_range=data_range))


def summarize_rows(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        mod = str(r["modality"])
        method = str(r["method"])
        val = float(r["ssim"])
        bucket.setdefault(mod, {}).setdefault(method, []).append(val)
    for mod, mvals in bucket.items():
        methods: Dict[str, Dict] = {}
        for method, vals in mvals.items():
            arr = np.asarray(vals, dtype=np.float64)
            methods[method] = {
                "n": int(arr.size),
                "mean_ssim": float(np.mean(arr)),
                "std_ssim": float(np.std(arr)),
                "median_ssim": float(np.median(arr)),
            }
        ranking = sorted(methods.items(), key=lambda kv: kv[1]["mean_ssim"], reverse=True)
        out[mod] = {
            "methods": methods,
            "ranking_by_mean_ssim": [k for k, _ in ranking],
        }
    return out


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "subject", "method", "ssim"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["modality"], x["subject"], x["method"])):
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HCP z3 normal-case SSIM with B-spline and SA-INR.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs_dataset/hcp_medgs_all_methods_z3"),
    )
    parser.add_argument("--modalities", type=str, default="asl,dwi")
    parser.add_argument("--normal-threshold", type=float, default=0.95)
    parser.add_argument(
        "--sa-asl-json",
        type=Path,
        default=Path("external_metrics/brain-gs/sa_inr_hcp_asl_ds3/eval_SA_INR_asl_ds3.json"),
    )
    parser.add_argument(
        "--sa-dwi-json",
        type=Path,
        default=Path("external_metrics/brain-gs/sa_inr_hcp_dwi_ds3/eval_SA_INR_dwi_ds3.json"),
    )
    parser.add_argument("--skip-bspline", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    modalities = parse_list(args.modalities)
    if not modalities:
        raise ValueError("no modalities provided")

    sa_map = {
        "asl": load_sa_inr(args.sa_asl_json),
        "dwi": load_sa_inr(args.sa_dwi_json),
    }
    rows: List[Dict] = []
    failures: List[Dict] = []
    normal_cases: Dict[str, List[str]] = {}

    for mod in modalities:
        metric_files = sorted(out_root.glob(f"sub-*/metrics_{mod}_z3.json"))
        keep: List[str] = []
        print(f"[{mod}] scanning {len(metric_files)} metric files")
        for mf in metric_files:
            sid = mf.parent.name
            try:
                data = json.loads(mf.read_text())
            except Exception as e:
                failures.append({"modality": mod, "subject": sid, "reason": f"bad_metrics_json:{e}"})
                continue
            interp = data.get("interp", {})
            interp_ssim = interp.get("ssim")
            if not isinstance(interp_ssim, (int, float)):
                failures.append({"modality": mod, "subject": sid, "reason": "missing_interp_ssim"})
                continue
            if float(interp_ssim) < args.normal_threshold:
                continue
            keep.append(sid)
            for method in ["interp", "medgs_single", "medgs_t1guided", "medgs_our"]:
                mdata = data.get(method, {})
                mssim = mdata.get("ssim")
                if isinstance(mssim, (int, float)):
                    rows.append({"modality": mod, "subject": sid, "method": method, "ssim": float(mssim)})
                else:
                    failures.append({"modality": mod, "subject": sid, "method": method, "reason": "missing_method_ssim"})
        normal_cases[mod] = keep
        print(f"[{mod}] normal cases kept={len(keep)} (threshold={args.normal_threshold})")

        # SA-INR SSIM from provided evaluation JSONs.
        for sid in keep:
            ssim_val = sa_map.get(mod, {}).get(sid)
            if ssim_val is None:
                failures.append({"modality": mod, "subject": sid, "method": "sa_inr", "reason": "missing_sa_ssim"})
            else:
                rows.append({"modality": mod, "subject": sid, "method": "sa_inr", "ssim": float(ssim_val)})

        # Cubic B-spline SSIM computed from LR.
        if not args.skip_bspline:
            total = len(keep)
            for i, sid in enumerate(keep, start=1):
                try:
                    val = bspline3_ssim_from_lr(out_root / sid, mod)
                    rows.append({"modality": mod, "subject": sid, "method": "interp_bspline3", "ssim": val})
                    print(f"[{mod}] bspline {i}/{total} {sid}: ssim={val:.6f}")
                except Exception as e:
                    failures.append(
                        {
                            "modality": mod,
                            "subject": sid,
                            "method": "interp_bspline3",
                            "reason": f"bspline_error:{e}",
                        }
                    )
                    print(f"[{mod}] bspline {i}/{total} {sid}: ERROR {e}")

    by_modality = summarize_rows(rows)
    overall = summarize_rows([{"modality": "overall", **r} for r in rows]).get("overall", {})

    payload = {
        "out_root": str(out_root),
        "modalities": modalities,
        "normal_threshold": args.normal_threshold,
        "normal_cases": normal_cases,
        "counts": {"rows": len(rows), "failures": len(failures)},
        "by_modality": by_modality,
        "overall": overall,
        "failures": failures,
        "notes": [
            "normal subset defined by interp.ssim >= normal_threshold from metrics_{mod}_z3.json",
            "interp/medgs_* SSIM values are read from existing per-subject metrics JSON",
            "sa_inr SSIM values are read from provided eval_SA_INR_*_ds3.json files",
            "interp_bspline3 SSIM is recomputed here via cubic B-spline upsampling from *_lr_1x1x3_masked.nii.gz",
            "SSIM definition for recomputed B-spline matches existing code: full-volume SSIM, data_range from GT brain-mask voxels",
        ],
    }

    out_json = args.out_json or (out_root / "ssim_normal_cases_plus_bspline_sainr_summary.json")
    out_csv = args.out_csv or (out_root / "ssim_normal_cases_plus_bspline_sainr.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    write_csv(rows, out_csv)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(rows)}  Failures: {len(failures)}")
    for mod, block in by_modality.items():
        rank = block.get("ranking_by_mean_ssim", [])
        print(f"{mod} ranking: {', '.join(rank)}")


if __name__ == "__main__":
    main()
