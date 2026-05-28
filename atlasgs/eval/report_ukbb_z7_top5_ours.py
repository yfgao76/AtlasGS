#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from atlasgs.ops.metrics import summarize_metrics
from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def metric_triplet_from_pred(pred: np.ndarray, gt: np.ndarray, t1: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    m = summarize_metrics(pred, gt, t1, mask)
    out = {}
    for k in ("mae", "ssim", "psnr"):
        v = m.get(k, None)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def get_external_metric_map(json_path: Path) -> Dict[str, Dict[str, float]]:
    data = load_json(json_path)
    out: Dict[str, Dict[str, float]] = {}
    if "per_subject_metrics" in data and isinstance(data["per_subject_metrics"], dict):
        for sid, mm in data["per_subject_metrics"].items():
            if isinstance(mm, dict):
                vals = {}
                for k in ("mae", "ssim", "psnr"):
                    v = mm.get(k, None)
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        vals[k] = float(v)
                if vals:
                    out[str(sid)] = vals
    elif "per_subject" in data and isinstance(data["per_subject"], list):
        for row in data["per_subject"]:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("subject_id", ""))
            if not sid:
                continue
            vals = {}
            for k in ("mae", "ssim", "psnr"):
                v = row.get(k, None)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals[k] = float(v)
            if vals:
                out[sid] = vals
    return out


def compute_cubic_metrics(data_root: Path, subject_id: str, factor_z: int) -> Optional[Dict[str, float]]:
    sub = data_root / subject_id
    gt_path = sub / "flair_gt.nii.gz"
    t1_path = sub / "t1_gt.nii.gz"
    lr_path = sub / f"flair_lr_1x1x{factor_z}.nii.gz"
    if not (gt_path.exists() and t1_path.exists() and lr_path.exists()):
        return None
    gt, gt_aff, _ = load_nii(gt_path)
    t1, _, _ = load_nii(t1_path)
    lr, lr_aff, _ = load_nii(lr_path)
    pred = resample_to_ref(lr, lr_aff, gt_aff, gt.shape, order=3)
    mask = (t1 > 0) if t1.shape == gt.shape else (gt > 0)
    return metric_triplet_from_pred(pred, gt, t1 if t1.shape == gt.shape else gt, mask.astype(np.uint8))


def score_subject(ours: Dict[str, float], others: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not all(k in ours for k in ("mae", "ssim", "psnr")):
        return None
    o_mae = ours["mae"]
    o_ssim = ours["ssim"]
    o_psnr = ours["psnr"]

    comp_mae = min([m["mae"] for m in others if "mae" in m], default=None)
    comp_ssim = max([m["ssim"] for m in others if "ssim" in m], default=None)
    comp_psnr = max([m["psnr"] for m in others if "psnr" in m], default=None)
    if comp_mae is None or comp_ssim is None or comp_psnr is None:
        return None

    mae_margin = float(comp_mae - o_mae)  # positive = better
    ssim_margin = float(o_ssim - comp_ssim)  # positive = better
    psnr_margin = float(o_psnr - comp_psnr)  # positive = better

    win_mae = mae_margin > 0
    win_ssim = ssim_margin > 0
    win_psnr = psnr_margin > 0
    win_count = int(win_mae) + int(win_ssim) + int(win_psnr)

    # Composite score favors multi-metric wins and larger margins.
    composite = (
        100.0 * (mae_margin / (abs(comp_mae) + 1e-6))
        + 100.0 * ssim_margin
        + psnr_margin
        + 10.0 * win_count
    )
    return {
        "win_count": win_count,
        "mae_margin": mae_margin,
        "ssim_margin": ssim_margin,
        "psnr_margin": psnr_margin,
        "composite_score": composite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find UKBB z7 subjects where Ours wins most.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--inr-json", type=Path, default=Path("external_metrics/brain-gs/inr_ukbb_ds7/eval_INR_ds7.json"))
    parser.add_argument("--sa-json", type=Path, default=Path("external_metrics/brain-gs/sa_inr_ukbb_ds7/eval_SA_INR_ds7.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    out_root_z = args.out_root / f"z{args.factor_z}"
    subjects = sorted([p.name for p in out_root_z.iterdir() if p.is_dir() and p.name.isdigit()])

    inr_map = get_external_metric_map(args.inr_json)
    sa_map = get_external_metric_map(args.sa_json)

    rows = []
    for sid in subjects:
        metrics_path = out_root_z / sid / f"metrics_z{args.factor_z}.json"
        d = load_json(metrics_path)
        if not d:
            continue
        ours = d.get("medgs_our", {})
        if not isinstance(ours, dict):
            continue

        cubic = compute_cubic_metrics(args.data_root, sid, args.factor_z)
        if cubic is None:
            continue

        others = []
        for key in ("interp", "medgs_single"):
            mm = d.get(key, {})
            if isinstance(mm, dict):
                others.append(mm)
        if sid in inr_map:
            others.append(inr_map[sid])
        if sid in sa_map:
            others.append(sa_map[sid])
        others.append(cubic)

        scored = score_subject(ours, others)
        if scored is None:
            continue
        rows.append(
            {
                "subject_id": sid,
                "ours_mae": float(ours.get("mae", np.nan)),
                "ours_ssim": float(ours.get("ssim", np.nan)),
                "ours_psnr": float(ours.get("psnr", np.nan)),
                "cubic_mae": cubic.get("mae", np.nan),
                "cubic_ssim": cubic.get("ssim", np.nan),
                "cubic_psnr": cubic.get("psnr", np.nan),
                **scored,
            }
        )

    rows.sort(key=lambda x: (x["win_count"], x["composite_score"]), reverse=True)
    topk = rows[: max(1, args.top_k)]
    payload = {
        "factor_z": args.factor_z,
        "n_subjects_scored": len(rows),
        "top_subjects": topk,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
