#!/usr/bin/env python3
"""Contribution analysis for retained T1-guided MedGS ablations.

This evaluates raw topology/latent outputs separately from LR-consistent fusion
and summarizes Gaussian counts/metadata. It is intentionally independent of the
pipeline metrics JSON because `medgs_our` may point to the fused output.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref
from atlasgs.eval.analyze_ukbb_ablation_masked import metric_triplet, summarize


METHOD_FILES = {
    "t1guided": "flair_medgs_t1guided_z{z}.nii.gz",
    "topology": "flair_medgs_t1guided_topology_z{z}.nii.gz",
    "latent": "flair_medgs_t1guided_latent_z{z}.nii.gz",
    "all_raw": "flair_medgs_t1guided_our_z{z}.nii.gz",
    "lr_fusion": "flair_medgs_fused_lrcons_z{z}.nii.gz",
    "all_fused": "flair_medgs_our_fused_lrcons_z{z}.nii.gz",
}

MODEL_DIRS = {
    "t1": "t1_model",
    "medgs_single": "flair_lr_model_z{z}",
    "t1guided": "flair_t1guided_model_z{z}",
    "topology": "flair_t1guided_topology_model_z{z}",
    "latent": "flair_t1guided_latent_model_z{z}",
    "all_raw": "flair_t1guided_our_model_z{z}",
}

METRICS = ("mae", "ssim", "psnr", "edge_ncc", "z_grad_energy")


def latest_ply_count(model_dir: Path) -> Optional[int]:
    point_root = model_dir / "point_cloud"
    if not point_root.exists():
        return None
    candidates = []
    for item in point_root.glob("iteration_*"):
        match = re.search(r"iteration_(\d+)$", item.name)
        if match:
            candidates.append((int(match.group(1)), item))
    if not candidates:
        return None
    ply = max(candidates)[1] / "point_cloud.ply"
    if not ply.exists():
        return None
    with ply.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def load_and_align(path: Path, gt_aff: np.ndarray, gt_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        vol, aff, _ = load_nii(path)
    except Exception:
        return None
    if tuple(vol.shape) != tuple(gt_shape):
        try:
            vol = resample_to_ref(vol, aff, gt_aff, gt_shape, order=1)
        except Exception:
            return None
    return vol.astype(np.float32, copy=False)


def finite_values(values: Iterable[float]) -> List[float]:
    out = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def summarize_values(values: List[float], confidence: float) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    return summarize(values, confidence)


def paired_summary(per_subject: Dict[str, Dict[str, Dict[str, float]]], base: str) -> Dict:
    out = {}
    directions = {"mae": -1.0, "ssim": 1.0, "psnr": 1.0, "edge_ncc": 1.0, "z_grad_energy": -1.0}
    for method in METHOD_FILES:
        if method == base:
            continue
        md = {}
        for metric, direction in directions.items():
            signed = []
            for subj, subj_data in per_subject.items():
                if base not in subj_data or method not in subj_data:
                    continue
                if metric not in subj_data[base] or metric not in subj_data[method]:
                    continue
                signed.append((float(subj_data[method][metric]) - float(subj_data[base][metric])) * direction)
            signed = finite_values(signed)
            md[metric] = {
                "n": len(signed),
                "wins": int(sum(v > 0 for v in signed)),
                "win_rate": float(sum(v > 0 for v in signed) / len(signed)) if signed else None,
                "mean_signed_improvement": float(mean(signed)) if signed else None,
                "median_signed_improvement": float(median(signed)) if signed else None,
            }
        out[method] = md
    return out


def collect_metrics(data_root: Path, out_root: Path, factor_z: int, confidence: float) -> Tuple[Dict, Dict]:
    per_subject: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sub_out in sorted(p for p in out_root.iterdir() if p.is_dir()):
        sid = sub_out.name
        gt_path = data_root / sid / "flair_gt.nii.gz"
        t1_path = data_root / sid / "t1_gt.nii.gz"
        mask_path = data_root / sid / "mask_brain.nii.gz"
        if not gt_path.exists() or not t1_path.exists():
            continue
        try:
            gt, gt_aff, _ = load_nii(gt_path)
            t1, _, _ = load_nii(t1_path)
            if mask_path.exists():
                mask, _, _ = load_nii(mask_path)
                if mask.shape != gt.shape:
                    mask = (gt > 0).astype(np.uint8)
                else:
                    mask = (mask > 0).astype(np.uint8)
            else:
                mask = ((t1 > 0) if t1.shape == gt.shape else (gt > 0)).astype(np.uint8)
        except Exception:
            continue
        edge_ref = t1 if t1.shape == gt.shape else gt
        subj_metrics = {}
        for method, pattern in METHOD_FILES.items():
            pred = load_and_align(sub_out / pattern.format(z=factor_z), gt_aff, gt.shape)
            if pred is None:
                continue
            subj_metrics[method] = metric_triplet(pred, gt, mask)
            # Keep edge/topology-sensitive terms from the project metric implementation.
            from atlasgs.ops.metrics import summarize_metrics

            subj_metrics[method].update(summarize_metrics(pred, gt, edge_ref, mask))
        if subj_metrics:
            per_subject[sid] = subj_metrics

    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in METHOD_FILES:
        aggregate[method] = {}
        for metric in METRICS:
            vals = finite_values(
                subj_data[method][metric]
                for subj_data in per_subject.values()
                if method in subj_data and metric in subj_data[method]
            )
            aggregate[method][metric] = summarize_values(vals, confidence)
    return per_subject, aggregate


def collect_counts_and_meta(out_root: Path, factor_z: int, confidence: float) -> Tuple[Dict, Dict]:
    counts: Dict[str, Dict[str, float]] = {}
    per_method_counts: Dict[str, List[float]] = {k: [] for k in MODEL_DIRS}
    meta_rows: Dict[str, List[Dict]] = {k: [] for k in ("t1guided", "topology", "latent", "all_raw")}
    meta_dir = {
        "t1guided": "flair_t1guided_model_z{z}",
        "topology": "flair_t1guided_topology_model_z{z}",
        "latent": "flair_t1guided_latent_model_z{z}",
        "all_raw": "flair_t1guided_our_model_z{z}",
    }
    for sub_out in sorted(p for p in out_root.iterdir() if p.is_dir()):
        for method, pattern in MODEL_DIRS.items():
            n = latest_ply_count(sub_out / pattern.format(z=factor_z))
            if n is not None:
                per_method_counts[method].append(float(n))
        for method, pattern in meta_dir.items():
            meta_path = sub_out / pattern.format(z=factor_z) / "t1guided_meta.json"
            if meta_path.exists():
                try:
                    meta_rows[method].append(json.loads(meta_path.read_text()))
                except Exception:
                    pass
    for method, vals in per_method_counts.items():
        counts[method] = summarize_values(vals, confidence)

    meta_summary = {}
    keys = [
        "lpvi_added",
        "persist_lambda",
        "persist_valid_slice_fraction",
        "coverage_ema_mean",
        "coverage_ema_lowfrac",
        "iterations",
        "appearance_latent_dim",
        "appearance_lambda_smooth",
        "appearance_lambda_theta_l2",
        "appearance_lambda_head_l2",
    ]
    for method, rows in meta_rows.items():
        cur = {"n": len(rows)}
        for key in keys:
            vals = finite_values(row.get(key) for row in rows if row.get(key) is not None)
            cur[key] = summarize_values(vals, confidence)
        cur["appearance_mode"] = {}
        cur["persist_has_topologylayer"] = {}
        for row in rows:
            mode = str(row.get("appearance_mode"))
            cur["appearance_mode"][mode] = cur["appearance_mode"].get(mode, 0) + 1
            topo = str(row.get("persist_has_topologylayer"))
            cur["persist_has_topologylayer"][topo] = cur["persist_has_topologylayer"].get(topo, 0) + 1
        meta_summary[method] = cur
    return counts, meta_summary


def write_csv(payload: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "method", "metric", "n", "mean", "ci_half_width", "wins_vs_t1guided", "win_rate_vs_t1guided"])
        for method, metrics in payload["aggregate_metrics"].items():
            for metric, stats in metrics.items():
                writer.writerow(["quality", method, metric, stats.get("n"), stats.get("mean"), stats.get("ci_half_width"), "", ""])
        for method, metrics in payload["paired_vs_t1guided"].items():
            for metric, stats in metrics.items():
                writer.writerow([
                    "paired_vs_t1guided",
                    method,
                    metric,
                    stats.get("n"),
                    stats.get("mean_signed_improvement"),
                    "",
                    stats.get("wins"),
                    stats.get("win_rate"),
                ])
        for method, stats in payload["point_counts"].items():
            writer.writerow(["point_count", method, "vertices", stats.get("n"), stats.get("mean"), stats.get("ci_half_width"), "", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw topology/latent/fusion contribution ablations.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    per_subject, aggregate = collect_metrics(args.data_root, args.out_root, args.factor_z, args.confidence)
    counts, meta = collect_counts_and_meta(args.out_root, args.factor_z, args.confidence)
    payload = {
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "factor_z": args.factor_z,
        "num_subjects_with_any_metrics": len(per_subject),
        "methods": list(METHOD_FILES),
        "aggregate_metrics": aggregate,
        "paired_vs_t1guided": paired_summary(per_subject, "t1guided"),
        "point_counts": counts,
        "metadata": meta,
        "per_subject": per_subject,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_csv(payload, args.out_csv)
    print(json.dumps({k: payload[k] for k in ("num_subjects_with_any_metrics", "methods")}, indent=2))


if __name__ == "__main__":
    main()
