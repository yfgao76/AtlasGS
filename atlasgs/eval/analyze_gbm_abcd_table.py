#!/usr/bin/env python3
import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from atlasgs.ops.metrics import masked_mae, masked_psnr, masked_ssim
from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref

METHODS = ["interp", "cubic", "sa_inr", "medgs", "ours"]


def read_subjects(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: List[str] = []
    for r in rows:
        sid = str(r.get("subject_id", "")).strip()
        if sid:
            out.append(sid)
    return out


def robust_unit_scale(vol: np.ndarray, mask: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    vals = vol[mask > 0]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def topk_mask(score: np.ndarray, support_mask: np.ndarray, k: int) -> np.ndarray:
    support_flat = np.where(support_mask.reshape(-1) > 0)[0]
    pred_mask = np.zeros(score.size, dtype=bool)
    if support_flat.size == 0 or k <= 0:
        return pred_mask.reshape(score.shape)
    k = int(min(max(k, 0), support_flat.size))
    vals = score.reshape(-1)[support_flat]
    if k >= support_flat.size:
        pred_mask[support_flat] = True
        return pred_mask.reshape(score.shape)
    top_idx = np.argpartition(vals, -k)[-k:]
    pred_mask[support_flat[top_idx]] = True
    return pred_mask.reshape(score.shape)


def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    sa = int(a.sum())
    sb = int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    inter = int((a & b).sum())
    return float((2.0 * inter) / (sa + sb))


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


def _resolve_ours(subject_out: Path, modality: str, factor_z: int, gt_aff: np.ndarray, gt_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    cand = [
        subject_out / f"{modality}_medgs_our_fused_lrcons_z{factor_z}.nii.gz",
        subject_out / f"{modality}_medgs_t1guided_our_z{factor_z}.nii.gz",
    ]
    for p in cand:
        v = load_and_align(p, gt_aff, gt_shape, order=1)
        if v is not None:
            return v
    return None


def evaluate_gbm_subject_modality(
    sid: str,
    modality: str,
    factor_z: int,
    data_root: str,
    out_root: str,
    sa_flair_root: str,
    sa_t2_root: str,
) -> Dict:
    data_root_p = Path(data_root)
    out_root_p = Path(out_root)
    subject_data = data_root_p / sid
    subject_out = out_root_p / sid

    gt, gt_aff, _ = load_nii(subject_data / f"{modality}_gt.nii.gz")
    brain, _, _ = load_nii(subject_data / "mask_brain.nii.gz")
    brain = (brain > 0).astype(np.uint8)

    preds: Dict[str, Optional[np.ndarray]] = {}
    preds["interp"] = load_and_align(subject_out / f"{modality}_interp_z{factor_z}.nii.gz", gt_aff, gt.shape, order=1)

    lr, lr_aff, _ = load_nii(subject_data / f"{modality}_lr_1x1x{factor_z}.nii.gz")
    preds["cubic"] = resample_to_ref(lr, lr_aff, gt_aff, gt.shape, order=3).astype(np.float32)

    sa_root = Path(sa_flair_root) if modality == "flair" else Path(sa_t2_root)
    preds["sa_inr"] = load_and_align(sa_root / f"{sid}_ds{factor_z}_{modality}_pred.nii.gz", gt_aff, gt.shape, order=1)
    preds["medgs"] = load_and_align(subject_out / f"{modality}_medgs_single_z{factor_z}.nii.gz", gt_aff, gt.shape, order=1)
    preds["ours"] = _resolve_ours(subject_out, modality, factor_z, gt_aff, gt.shape)

    missing = [m for m in METHODS if preds[m] is None]
    if missing:
        return {"sid": sid, "modality": modality, "missing": missing}

    out: Dict[str, Dict[str, float]] = {}
    for m in METHODS:
        p = preds[m]
        ssim_v = masked_ssim(p, gt, brain)
        out[m] = {
            "mae": float(masked_mae(p, gt, brain)),
            "ssim": float(ssim_v) if ssim_v is not None else float("nan"),
            "psnr": float(masked_psnr(p, gt, brain)),
        }

    if modality == "flair":
        tumor, _, _ = load_nii(subject_data / "tumor_seg.nii.gz")
        tumor = ((tumor > 0) & (brain > 0)).astype(np.uint8)
        k = int(tumor.sum())
        if k > 0:
            for m in METHODS:
                p = preds[m]
                score = robust_unit_scale(p, brain)
                pred_tumor = topk_mask(score, brain, k)
                out[m]["dsc"] = float(dice_coef(pred_tumor, tumor))
    return {"sid": sid, "modality": modality, "metrics": out}


def evaluate_hcp_subject_modality(
    sid: str,
    modality: str,
    factor_z: int,
    data_root: str,
    out_root: str,
    sa_dwi_root: str,
    sa_asl_root: str,
) -> Dict:
    data_root_p = Path(data_root)
    out_root_p = Path(out_root)
    subject_data = data_root_p / sid
    subject_out = out_root_p / sid

    gt, gt_aff, _ = load_nii(subject_data / f"{modality}_gt.nii.gz")
    mask_path = subject_out / f"{modality}_mask_brain_gt_z{factor_z}.nii.gz"
    if mask_path.exists():
        brain, _, _ = load_nii(mask_path)
    else:
        brain, brain_aff, _ = load_nii(subject_data / "mask_brain.nii.gz")
        if tuple(brain.shape) != tuple(gt.shape):
            brain = resample_to_ref(brain, brain_aff, gt_aff, gt.shape, order=0)
    brain = (brain > 0).astype(np.uint8)

    preds: Dict[str, Optional[np.ndarray]] = {}
    preds["interp"] = load_and_align(subject_out / f"{modality}_interp_z{factor_z}.nii.gz", gt_aff, gt.shape, order=1)

    lr, lr_aff, _ = load_nii(subject_data / f"{modality}_lr_1x1x{factor_z}.nii.gz")
    preds["cubic"] = resample_to_ref(lr, lr_aff, gt_aff, gt.shape, order=3).astype(np.float32)

    sa_root = Path(sa_dwi_root) if modality == "dwi" else Path(sa_asl_root)
    preds["sa_inr"] = load_and_align(sa_root / f"{sid}_ds{factor_z}_{modality}_pred.nii.gz", gt_aff, gt.shape, order=1)
    preds["medgs"] = load_and_align(subject_out / f"{modality}_medgs_single_z{factor_z}.nii.gz", gt_aff, gt.shape, order=1)
    preds["ours"] = _resolve_ours(subject_out, modality, factor_z, gt_aff, gt.shape)

    missing = [m for m in METHODS if preds[m] is None]
    if missing:
        return {"sid": sid, "modality": modality, "missing": missing}

    out: Dict[str, Dict[str, float]] = {}
    for m in METHODS:
        p = preds[m]
        ssim_v = masked_ssim(p, gt, brain)
        out[m] = {
            "mae": float(masked_mae(p, gt, brain)),
            "ssim": float(ssim_v) if ssim_v is not None else float("nan"),
            "psnr": float(masked_psnr(p, gt, brain)),
        }
    return {"sid": sid, "modality": modality, "metrics": out}


def mean_of_metric(items: List[Dict[str, float]], key: str) -> float:
    vals = [float(x[key]) for x in items if key in x and math.isfinite(float(x[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_rows(results: List[Dict], modalities: List[str]) -> Dict:
    payload: Dict[str, Dict] = {}
    for mod in modalities:
        rows = [r for r in results if r.get("modality") == mod and "metrics" in r]
        payload[mod] = {"n_complete": len(rows), "methods": {}}
        for m in METHODS:
            metric_rows = [r["metrics"][m] for r in rows]
            payload[mod]["methods"][m] = {
                "mae": mean_of_metric(metric_rows, "mae"),
                "ssim": mean_of_metric(metric_rows, "ssim"),
                "psnr": mean_of_metric(metric_rows, "psnr"),
                "dsc": mean_of_metric(metric_rows, "dsc"),
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GBM+ABCD(HCP) table metrics for Interp/Cubic/SA-INR/MedGS/Ours.")
    parser.add_argument("--gbm-data-root", type=Path, required=True)
    parser.add_argument("--gbm-out-root", type=Path, required=True)
    parser.add_argument("--gbm-test-csv", type=Path, required=True)
    parser.add_argument("--gbm-sa-flair-root", type=Path, required=True)
    parser.add_argument("--gbm-sa-t2-root", type=Path, required=True)
    parser.add_argument("--gbm-factor-z", type=int, default=7)
    parser.add_argument("--hcp-data-root", type=Path, required=True)
    parser.add_argument("--hcp-out-root", type=Path, required=True)
    parser.add_argument("--hcp-test-csv", type=Path, required=True)
    parser.add_argument("--hcp-sa-dwi-root", type=Path, required=True)
    parser.add_argument("--hcp-sa-asl-root", type=Path, required=True)
    parser.add_argument("--hcp-factor-z", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--exclude-subject-substrings",
        type=str,
        default="",
        help="Comma-separated substrings; subjects containing any substring are excluded.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    gbm_subjects = read_subjects(args.gbm_test_csv)
    hcp_subjects = read_subjects(args.hcp_test_csv)
    exclude_substrings = [x.strip() for x in str(args.exclude_subject_substrings).split(",") if x.strip()]
    if exclude_substrings:
        def keep_sid(sid: str) -> bool:
            return not any(token in sid for token in exclude_substrings)
        gbm_subjects = [sid for sid in gbm_subjects if keep_sid(sid)]
        hcp_subjects = [sid for sid in hcp_subjects if keep_sid(sid)]

    gbm_jobs = [(sid, mod) for sid in gbm_subjects for mod in ("flair", "t2")]
    hcp_jobs = [(sid, mod) for sid in hcp_subjects for mod in ("dwi", "asl")]

    gbm_results: List[Dict] = []
    hcp_results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
        futs = []
        for sid, mod in gbm_jobs:
            futs.append(
                ex.submit(
                    evaluate_gbm_subject_modality,
                    sid,
                    mod,
                    int(args.gbm_factor_z),
                    str(args.gbm_data_root),
                    str(args.gbm_out_root),
                    str(args.gbm_sa_flair_root),
                    str(args.gbm_sa_t2_root),
                )
            )
        for sid, mod in hcp_jobs:
            futs.append(
                ex.submit(
                    evaluate_hcp_subject_modality,
                    sid,
                    mod,
                    int(args.hcp_factor_z),
                    str(args.hcp_data_root),
                    str(args.hcp_out_root),
                    str(args.hcp_sa_dwi_root),
                    str(args.hcp_sa_asl_root),
                )
            )
        for i, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            if res["modality"] in ("flair", "t2"):
                gbm_results.append(res)
            else:
                hcp_results.append(res)
            if i % 20 == 0:
                print(f"[{i}/{len(futs)}] done", flush=True)

    gbm_agg = aggregate_rows(gbm_results, ["t2", "flair"])
    hcp_agg = aggregate_rows(hcp_results, ["dwi", "asl"])

    gbm_missing = [r for r in gbm_results if "missing" in r]
    hcp_missing = [r for r in hcp_results if "missing" in r]

    payload = {
        "gbm": gbm_agg,
        "abcd_hcp": hcp_agg,
        "gbm_subjects_total": len(gbm_subjects),
        "hcp_subjects_total": len(hcp_subjects),
        "gbm_missing_count": len(gbm_missing),
        "hcp_missing_count": len(hcp_missing),
        "gbm_missing_examples": gbm_missing[:20],
        "hcp_missing_examples": hcp_missing[:20],
        "exclude_subject_substrings": exclude_substrings,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "modality", "method", "metric", "value"])
        for mod in ("t2", "flair"):
            for m in METHODS:
                for met in ("mae", "ssim", "psnr", "dsc"):
                    w.writerow(["gbm", mod, m, met, gbm_agg[mod]["methods"][m][met]])
        for mod in ("dwi", "asl"):
            for m in METHODS:
                for met in ("mae", "ssim", "psnr"):
                    w.writerow(["abcd_hcp", mod, m, met, hcp_agg[mod]["methods"][m][met]])

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print(f"GBM complete n: t2={gbm_agg['t2']['n_complete']} flair={gbm_agg['flair']['n_complete']}")
    print(f"ABCD/HCP complete n: dwi={hcp_agg['dwi']['n_complete']} asl={hcp_agg['asl']['n_complete']}")


if __name__ == "__main__":
    main()
