#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from atlasgs.ops.metrics import summarize_metrics
from atlasgs.ops.nifti_io import load_nii
from atlasgs.ops.resample import resample_to_ref


def process_subject(subject_out: Path, data_root: Path, factor_z: int, overwrite: bool) -> Tuple[bool, str]:
    sid = subject_out.name
    if not sid.isdigit():
        return False, f"{sid}: skip non-subject dir"

    sub_data = data_root / sid
    lr_path = sub_data / f"flair_lr_1x1x{factor_z}.nii.gz"
    gt_path = sub_data / "flair_gt.nii.gz"
    t1_path = sub_data / "t1_gt.nii.gz"
    if not (lr_path.exists() and gt_path.exists() and t1_path.exists()):
        return False, f"{sid}: missing input(s)"

    bspline_out = subject_out / f"flair_interp_bspline3_z{factor_z}.nii.gz"
    metrics_path = subject_out / f"metrics_z{factor_z}.json"

    gt, gt_aff, gt_hdr = load_nii(gt_path)
    t1, _, _ = load_nii(t1_path)
    mask = (t1 > 0).astype(np.uint8) if t1.shape == gt.shape else (gt > 0).astype(np.uint8)
    edge_ref = t1 if t1.shape == gt.shape else gt

    if overwrite or not bspline_out.exists():
        lr, lr_aff, _ = load_nii(lr_path)
        bspline = resample_to_ref(lr, lr_aff, gt_aff, gt.shape, order=3)
        from atlasgs.ops.nifti_io import save_nii

        save_nii(bspline_out, bspline, gt_aff, header=gt_hdr)
    else:
        bspline, _, _ = load_nii(bspline_out)

    if bspline.shape != gt.shape:
        return False, f"{sid}: shape mismatch bspline={bspline.shape} gt={gt.shape}"

    m = summarize_metrics(bspline, gt, edge_ref, mask)
    payload = {}
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text())
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["interp_bspline3"] = {k: float(v) for k, v in m.items()}
    metrics_path.write_text(json.dumps(payload, indent=2))
    return True, f"{sid}: ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill UKBB cubic B-spline interpolation and metrics.")
    parser.add_argument("--out-root", type=Path, required=True, help=".../outputs_dataset/ukbb_medgs_all_methods")
    parser.add_argument("--data-root", type=Path, required=True, help=".../data/ukbb_medgs")
    parser.add_argument("--factors", type=str, default="3,5,7")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    factors = [int(x.strip()) for x in args.factors.split(",") if x.strip()]
    n_ok = 0
    n_fail = 0
    n_skip = 0

    for z in factors:
        out_z = args.out_root / f"z{z}"
        if not out_z.exists():
            print(f"[z{z}] missing out dir: {out_z}")
            continue
        subjects = sorted([p for p in out_z.iterdir() if p.is_dir()])
        print(f"[z{z}] subjects={len(subjects)}")
        for i, sub in enumerate(subjects, start=1):
            ok, msg = process_subject(sub, args.data_root, z, args.overwrite)
            if ok:
                n_ok += 1
            else:
                if "skip non-subject" in msg:
                    n_skip += 1
                else:
                    n_fail += 1
            if i % 20 == 0:
                print(f"[z{z}] {i}/{len(subjects)}")
        print(f"[z{z}] done")

    print(f"Done: ok={n_ok} fail={n_fail} skip={n_skip}")


if __name__ == "__main__":
    main()
