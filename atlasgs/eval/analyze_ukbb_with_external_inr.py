#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from atlasgs.ops.metrics import summarize_metrics
from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def t_critical(confidence: float, dof: int) -> float:
    if dof <= 0:
        return 0.0
    try:
        from scipy.stats import t as t_dist  # type: ignore

        return float(t_dist.ppf((1.0 + confidence) * 0.5, dof))
    except Exception:
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
    tcrit = t_critical(confidence, n - 1)
    half = float(tcrit * se) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half_width": half,
    }


def aggregate_subject_metrics(
    subject_metrics: Dict[str, Dict[str, Dict[str, float]]], confidence: float
) -> Dict[str, Dict[str, Dict[str, float]]]:
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for _, method_dict in subject_metrics.items():
        for method, metric_dict in method_dict.items():
            m_bucket = bucket.setdefault(method, {})
            for metric_name, val in metric_dict.items():
                fv = float(val)
                if math.isfinite(fv):
                    m_bucket.setdefault(metric_name, []).append(fv)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, metric_vals in bucket.items():
        out[method] = {}
        for metric_name, vals in metric_vals.items():
            s = summarize_array(vals, confidence)
            if s:
                out[method][metric_name] = s
    return out


def load_atlasgs_metrics(out_root_z: Path, factor_z: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    per_subject: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sub in sorted([p for p in out_root_z.iterdir() if p.is_dir()]):
        fp = sub / f"metrics_z{factor_z}.json"
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        sid = str(sub.name)
        per_subject.setdefault(sid, {})
        for method, metric_dict in data.items():
            if isinstance(metric_dict, dict):
                per_subject[sid][str(method)] = {
                    str(k): float(v)
                    for k, v in metric_dict.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
    return per_subject


def compute_bspline_subject_metrics(data_root: Path, subject_id: str, factor_z: int) -> Dict[str, float]:
    sub = data_root / subject_id
    lr_path = sub / f"flair_lr_1x1x{factor_z}.nii.gz"
    gt_path = sub / "flair_gt.nii.gz"
    t1_path = sub / "t1_gt.nii.gz"
    if not (lr_path.exists() and gt_path.exists() and t1_path.exists()):
        return {}

    lr, lr_aff, _ = load_nii(lr_path)
    gt, gt_aff, _ = load_nii(gt_path)
    t1, _, _ = load_nii(t1_path)

    pred = resample_to_ref(lr, lr_aff, gt_aff, gt.shape, order=3)
    mask = (t1 > 0).astype(np.uint8) if t1.shape == gt.shape else (gt > 0).astype(np.uint8)
    edge_ref = t1 if t1.shape == gt.shape else gt
    metrics = summarize_metrics(pred, gt, edge_ref, mask)
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}


def load_inr(inr_json: Path) -> Dict[str, Dict[str, float]]:
    if not inr_json.exists():
        return {}
    try:
        data = json.loads(inr_json.read_text())
    except Exception:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    per_sub = data.get("per_subject_metrics", {})
    if isinstance(per_sub, dict):
        for sid, metric_dict in per_sub.items():
            if isinstance(metric_dict, dict):
                out[str(sid)] = {
                    str(k): float(v)
                    for k, v in metric_dict.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
    return out


def load_sa_inr(sa_json: Path) -> Dict[str, Dict[str, float]]:
    if not sa_json.exists():
        return {}
    try:
        data = json.loads(sa_json.read_text())
    except Exception:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    per_sub = data.get("per_subject", [])
    if isinstance(per_sub, list):
        for row in per_sub:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("subject_id", ""))
            if not sid:
                continue
            vals: Dict[str, float] = {}
            for k in ("mae", "psnr", "ssim"):
                v = row.get(k)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals[k] = float(v)
            if vals:
                out[sid] = vals
    return out


def write_csv(summary_payload: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "factor_z",
                "scope",
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
        for factor_key, factor_data in summary_payload["summary_by_factor"].items():
            for scope in ("all_available", "matched_core"):
                scope_dict = factor_data.get(scope, {})
                for method, metric_dict in scope_dict.items():
                    for metric_name, s in metric_dict.items():
                        w.writerow(
                            [
                                factor_key,
                                scope,
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
    parser = argparse.ArgumentParser(description="Aggregate UKBB z3/z5/z7 metrics and add external INR / SA-INR results.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--factors", type=str, default="3,5,7")
    parser.add_argument("--inr-base", type=Path, default=Path("external_metrics/brain-gs"))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--add-bspline-from-lr", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/ukbb_medgs"))
    parser.add_argument(
        "--core-methods",
        type=str,
        default="interp,medgs_single,medgs_t1guided,medgs_our,inr_external,sa_inr_external",
        help="Methods required for matched-core subset.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    factors = [int(x) for x in parse_list(args.factors)]
    core_methods = parse_list(args.core_methods)

    payload = {
        "out_root": str(out_root),
        "factors": factors,
        "confidence": args.confidence,
        "core_methods": core_methods,
        "summary_by_factor": {},
        "counts_by_factor": {},
    }

    for z in factors:
        factor_key = f"z{z}"
        out_root_z = out_root / factor_key
        if not out_root_z.exists():
            continue

        per_subject = load_atlasgs_metrics(out_root_z, z)
        if args.add_bspline_from_lr:
            subject_dirs = sorted([p for p in out_root_z.iterdir() if p.is_dir() and p.name.isdigit()])
            for sub in subject_dirs:
                sid = str(sub.name)
                per_subject.setdefault(sid, {})
                bspline_metrics = compute_bspline_subject_metrics(args.data_root, sid, z)
                if bspline_metrics:
                    per_subject[sid]["interp_bspline3"] = bspline_metrics

        inr_json = args.inr_base / f"inr_ukbb_ds{z}" / f"eval_INR_ds{z}.json"
        sa_json = args.inr_base / f"sa_inr_ukbb_ds{z}" / f"eval_SA_INR_ds{z}.json"
        inr = load_inr(inr_json)
        sa = load_sa_inr(sa_json)

        for sid, m in inr.items():
            per_subject.setdefault(sid, {})
            per_subject[sid]["inr_external"] = m
        for sid, m in sa.items():
            per_subject.setdefault(sid, {})
            per_subject[sid]["sa_inr_external"] = m

        all_available_summary = aggregate_subject_metrics(per_subject, args.confidence)

        matched_subjects = []
        for sid, method_dict in per_subject.items():
            if all(method in method_dict for method in core_methods):
                matched_subjects.append(sid)
        matched_subjects = sorted(matched_subjects)
        matched_dict = {sid: per_subject[sid] for sid in matched_subjects}
        matched_summary = aggregate_subject_metrics(matched_dict, args.confidence)

        method_counts: Dict[str, int] = {}
        for sid, method_dict in per_subject.items():
            for m in method_dict.keys():
                method_counts[m] = method_counts.get(m, 0) + 1

        payload["summary_by_factor"][factor_key] = {
            "all_available": all_available_summary,
            "matched_core": matched_summary,
        }
        payload["counts_by_factor"][factor_key] = {
            "num_subject_dirs": len([p for p in out_root_z.iterdir() if p.is_dir()]),
            "num_subjects_any_method": len(per_subject),
            "num_subjects_matched_core": len(matched_subjects),
            "method_subject_counts": dict(sorted(method_counts.items(), key=lambda x: x[0])),
            "inr_json": str(inr_json),
            "sa_inr_json": str(sa_json),
        }

    out_json = args.out_json or (out_root / "ukbb_metrics_with_external_inr_summary.json")
    out_csv = args.out_csv or (out_root / "ukbb_metrics_with_external_inr_summary.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    write_csv(payload, out_csv)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    for factor_key, c in payload["counts_by_factor"].items():
        print(
            f"{factor_key}: any={c['num_subjects_any_method']} "
            f"matched_core={c['num_subjects_matched_core']}"
        )


if __name__ == "__main__":
    main()
