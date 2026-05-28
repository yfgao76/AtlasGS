#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from atlasgs.ops.nifti_io import load_nii, save_nii
from atlasgs.ops.resample import resample_to_ref


def run(cmd: List[str], env: Dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def nnunet_cmd(tool: str, args: List[str]) -> List[str]:
    """Use nnUNet console scripts when available, fallback to module entrypoints."""
    tool_to_script = {
        "plan_preprocess": "nnUNet_plan_and_preprocess",
        "train": "nnUNet_train",
        "predict": "nnUNet_predict",
    }
    tool_to_module = {
        "plan_preprocess": "nnunet.experiment_planning.nnUNet_plan_and_preprocess",
        "train": "nnunet.run.run_training",
        "predict": "nnunet.inference.predict_simple",
    }
    script = tool_to_script[tool]
    if shutil.which(script):
        return [script] + args
    module = tool_to_module[tool]
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"Missing nnUNet executable '{script}' and module '{module}'. "
            "Install nnUNet in this environment."
        )
    return [sys.executable, "-m", module] + args


def read_subjects(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        sid = str(r.get("subject_id", "")).strip()
        if sid:
            out.append(sid)
    return out


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, use_symlink: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def write_binary_label(src: Path, dst: Path) -> None:
    """Write tumor segmentation as binary {0,1} label map for nnUNet."""
    arr, affine, header = load_nii(src)
    bin_arr = (arr > 0).astype(np.uint8)
    save_nii(dst, bin_arr, affine, header)


def has_nonzero_voxels(path: Path) -> bool:
    """Guard nnUNet preprocessing: crop_to_nonzero crashes on all-zero inputs."""
    arr, _, _ = load_nii(path)
    return bool(np.any(arr != 0))


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    ps = int(p.sum())
    gs = int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    inter = int((p & g).sum())
    return float(2.0 * inter / (ps + gs))


def t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    try:
        from scipy.stats import t as t_dist  # type: ignore

        return float(t_dist.ppf(0.975, n - 1))
    except Exception:
        return 1.959963984540054


def summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(std / math.sqrt(n)) if n > 1 else 0.0
    half = float(t_critical_95(n) * se) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half_width": half,
    }


def prepare_nnunet_task(
    data_root: Path,
    train_subjects: List[str],
    test_subjects: List[str],
    modality: str,
    task_id: int,
    task_name_suffix: str,
    raw_base: Path,
    use_symlink: bool = True,
) -> Tuple[str, List[str], List[str]]:
    task_name = f"Task{task_id:03d}_{task_name_suffix}"
    task_dir = raw_base / "nnUNet_raw_data" / task_name
    images_tr = task_dir / "imagesTr"
    labels_tr = task_dir / "labelsTr"
    images_ts = task_dir / "imagesTs"
    ensure_clean_dir(images_tr)
    ensure_clean_dir(labels_tr)
    ensure_clean_dir(images_ts)

    tr_kept: List[str] = []
    for sid in train_subjects:
        sub = data_root / sid
        img = sub / f"{modality}_gt.nii.gz"
        seg = sub / "tumor_seg.nii.gz"
        if not (img.exists() and seg.exists()):
            continue
        if not has_nonzero_voxels(img):
            print(f"[SKIP][{modality}][train] all-zero image: {sid}", flush=True)
            continue
        link_or_copy(img, images_tr / f"{sid}_0000.nii.gz", use_symlink=use_symlink)
        write_binary_label(seg, labels_tr / f"{sid}.nii.gz")
        tr_kept.append(sid)

    ts_kept: List[str] = []
    for sid in test_subjects:
        sub = data_root / sid
        img = sub / f"{modality}_gt.nii.gz"
        if not img.exists():
            continue
        if not has_nonzero_voxels(img):
            print(f"[SKIP][{modality}][test] all-zero image: {sid}", flush=True)
            continue
        link_or_copy(img, images_ts / f"{sid}_0000.nii.gz", use_symlink=use_symlink)
        ts_kept.append(sid)

    dataset_json = {
        "name": f"GBM_{modality}_tumor",
        "description": "GBM tumor segmentation from reconstructed MRI",
        "tensorImageSize": "4D",
        "reference": "AtlasGS",
        "licence": "see repository license",
        "release": "1.0",
        "modality": {"0": modality},
        "labels": {"0": "background", "1": "tumor"},
        "numTraining": len(tr_kept),
        "numTest": len(ts_kept),
        "training": [
            {"image": f"./imagesTr/{sid}.nii.gz", "label": f"./labelsTr/{sid}.nii.gz"} for sid in tr_kept
        ],
        "test": [f"./imagesTs/{sid}.nii.gz" for sid in ts_kept],
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    return task_name, tr_kept, ts_kept


def method_paths(modality: str, factor_z: int) -> List[Tuple[str, List[str]]]:
    return [
        ("gt", [f"{modality}_gt.nii.gz"]),
        ("interp", [f"{modality}_interp_z{factor_z}.nii.gz"]),
        ("cubic", [f"{modality}_interp_bspline3_z{factor_z}.nii.gz"]),
        ("sa_inr", [f"{modality}_sa_inr_z{factor_z}.nii.gz"]),
        ("medgs_single", [f"{modality}_medgs_single_z{factor_z}.nii.gz"]),
        ("medgs_t1guided", [f"{modality}_medgs_t1guided_z{factor_z}.nii.gz"]),
        ("ours", [f"{modality}_medgs_our_fused_lrcons_z{factor_z}.nii.gz", f"{modality}_medgs_t1guided_our_z{factor_z}.nii.gz"]),
        ("alpine_a2", [f"{modality}_alpine_a2_z{factor_z}.nii.gz"]),
    ]


def build_inputs_for_method(
    method_name: str,
    candidates: List[str],
    modality: str,
    factor_z: int,
    test_subjects: List[str],
    outputs_root: Path,
    data_root: Path,
    input_dir: Path,
    sa_pred_root: Path = None,
    use_symlink: bool = True,
) -> List[str]:
    ensure_clean_dir(input_dir)
    kept: List[str] = []
    for sid in test_subjects:
        src = None
        dst = input_dir / f"{sid}_0000.nii.gz"
        if method_name == "gt":
            p = data_root / sid / f"{modality}_gt.nii.gz"
            if p.exists():
                src = p
        elif method_name == "cubic":
            # Prefer precomputed cubic interpolation if present under outputs.
            sub_out = outputs_root / sid
            for rel in candidates:
                p = sub_out / rel
                if p.exists():
                    src = p
                    break
            if src is not None:
                if not has_nonzero_voxels(src):
                    print(f"[SKIP][{modality}][{method_name}] all-zero input: {sid}", flush=True)
                    continue
                link_or_copy(src, dst, use_symlink=use_symlink)
                kept.append(sid)
                continue
            # Fallback: generate cubic interpolation on the fly from LR.
            lr = data_root / sid / f"{modality}_lr_1x1x{factor_z}.nii.gz"
            gt = data_root / sid / f"{modality}_gt.nii.gz"
            if not (lr.exists() and gt.exists()):
                continue
            if not has_nonzero_voxels(lr):
                print(f"[SKIP][{modality}][{method_name}] all-zero input: {sid}", flush=True)
                continue
            lr_arr, lr_aff, _ = load_nii(lr)
            gt_arr, gt_aff, gt_header = load_nii(gt)
            cubic = resample_to_ref(lr_arr, lr_aff, gt_aff, gt_arr.shape, order=3)
            save_nii(dst, cubic.astype(np.float32), gt_aff, gt_header)
            kept.append(sid)
            continue
        elif method_name == "sa_inr":
            if sa_pred_root is not None:
                p = sa_pred_root / f"{sid}_ds{factor_z}_{modality}_pred.nii.gz"
                if p.exists():
                    if not has_nonzero_voxels(p):
                        print(f"[SKIP][{modality}][{method_name}] all-zero input: {sid}", flush=True)
                        continue
                    gt = data_root / sid / f"{modality}_gt.nii.gz"
                    if not gt.exists():
                        continue
                    sa_arr, sa_aff, _ = load_nii(p)
                    gt_arr, gt_aff, gt_header = load_nii(gt)
                    # External SA-INR predictions can have truncated z-size; align to GT grid.
                    if sa_arr.shape != gt_arr.shape or not np.allclose(sa_aff, gt_aff):
                        sa_arr = resample_to_ref(sa_arr, sa_aff, gt_aff, gt_arr.shape, order=1)
                        save_nii(dst, sa_arr.astype(np.float32), gt_aff, gt_header)
                    else:
                        link_or_copy(p, dst, use_symlink=use_symlink)
                    kept.append(sid)
                    continue
            if src is None:
                sub_out = outputs_root / sid
                for rel in candidates:
                    p = sub_out / rel
                    if p.exists():
                        src = p
                        break
        else:
            sub_out = outputs_root / sid
            for rel in candidates:
                p = sub_out / rel
                if p.exists():
                    src = p
                    break
        if src is None:
            continue
        if not has_nonzero_voxels(src):
            print(f"[SKIP][{modality}][{method_name}] all-zero input: {sid}", flush=True)
            continue
        link_or_copy(src, dst, use_symlink=use_symlink)
        kept.append(sid)
    return kept


def evaluate_predictions(pred_dir: Path, data_root: Path, subjects: List[str]) -> Dict[str, Dict]:
    per_subject: Dict[str, float] = {}
    for sid in subjects:
        pred_path = pred_dir / f"{sid}.nii.gz"
        gt_path = data_root / sid / "tumor_seg.nii.gz"
        if not (pred_path.exists() and gt_path.exists()):
            continue
        pred, _, _ = load_nii(pred_path)
        gt, _, _ = load_nii(gt_path)
        if pred.shape != gt.shape:
            continue
        dsc = dice_score(pred > 0, gt > 0)
        per_subject[sid] = float(dsc)
    summary = summarize(list(per_subject.values()))
    return {"summary": summary, "per_subject_dsc": per_subject}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nnUNet on GBM GT and evaluate DSC on reconstructed methods.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True, help="e.g., .../outputs_dataset/gbm_medgs_all_methods_z7")
    parser.add_argument("--work-root", type=Path, required=True, help="where nnUNet raw/preprocessed/results/metrics are stored")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--modalities", type=str, default="flair,t2")
    parser.add_argument("--task-id-base", type=int, default=901)
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2")
    parser.add_argument("--model", type=str, default="2d")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, default="model_final_checkpoint")
    parser.add_argument("--sa-flair-root", type=Path, default=None)
    parser.add_argument("--sa-t2-root", type=Path, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="optional comma-separated method names to evaluate (default: all)",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plan", action="store_true")
    parser.add_argument("--copy-instead-of-symlink", action="store_true")
    args = parser.parse_args()

    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    train_subjects = read_subjects(args.train_csv)
    test_subjects = read_subjects(args.test_csv)
    use_symlink = not args.copy_instead_of_symlink

    raw_base = args.work_root / "nnunet_raw_base"
    preprocessed = args.work_root / "nnunet_preprocessed"
    results = args.work_root / "nnunet_results"
    predict_root = args.work_root / "predictions"
    metrics_root = args.work_root / "metrics"
    raw_base.mkdir(parents=True, exist_ok=True)
    preprocessed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    predict_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw_data_base"] = str(raw_base)
    env["nnUNet_preprocessed"] = str(preprocessed)
    env["RESULTS_FOLDER"] = str(results)

    all_metrics: Dict[str, Dict] = {}
    for idx, modality in enumerate(modalities):
        task_id = int(args.task_id_base) + idx
        task_suffix = f"GBM{modality.upper()}Tumor"
        task_name, tr_kept, ts_kept = prepare_nnunet_task(
            data_root=args.data_root,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            modality=modality,
            task_id=task_id,
            task_name_suffix=task_suffix,
            raw_base=raw_base,
            use_symlink=use_symlink,
        )
        print(f"[TASK] {task_name}: train={len(tr_kept)} test={len(ts_kept)}", flush=True)

        if not args.skip_plan:
            run(nnunet_cmd("plan_preprocess", ["-t", str(task_id), "--verify_dataset_integrity"]), env)
        if not args.skip_train:
            run(nnunet_cmd("train", [args.model, args.trainer, str(task_id), str(args.fold)]), env)

        modality_out = {}
        sa_root = args.sa_flair_root if modality == "flair" else args.sa_t2_root
        method_specs = method_paths(modality, args.factor_z)
        if args.methods.strip():
            requested = [m.strip() for m in args.methods.split(",") if m.strip()]
            spec_map = {name: cands for name, cands in method_specs}
            missing = [m for m in requested if m not in spec_map]
            if missing:
                raise ValueError(f"Unknown method(s) requested: {missing}")
            method_specs = [(m, spec_map[m]) for m in requested]

        for mname, candidates in method_specs:
            inp_dir = predict_root / f"{task_name}_{mname}_input"
            pred_dir = predict_root / f"{task_name}_{mname}_pred"
            kept = build_inputs_for_method(
                method_name=mname,
                candidates=candidates,
                modality=modality,
                factor_z=args.factor_z,
                test_subjects=ts_kept,
                outputs_root=args.outputs_root,
                data_root=args.data_root,
                input_dir=inp_dir,
                sa_pred_root=sa_root,
                use_symlink=use_symlink,
            )
            if len(kept) == 0:
                modality_out[mname] = {"summary": {}, "per_subject_dsc": {}, "num_subjects": 0}
                continue
            ensure_clean_dir(pred_dir)
            run(
                nnunet_cmd(
                    "predict",
                    [
                    "-i",
                    str(inp_dir),
                    "-o",
                    str(pred_dir),
                    "-t",
                    str(task_id),
                    "-m",
                    args.model,
                    "-f",
                    str(args.fold),
                    "-chk",
                    args.checkpoint,
                    ],
                ),
                env,
            )
            eval_res = evaluate_predictions(pred_dir=pred_dir, data_root=args.data_root, subjects=kept)
            eval_res["num_subjects"] = len(kept)
            modality_out[mname] = eval_res
            print(
                f"[{modality}][{mname}] n={eval_res['summary'].get('n', 0)} "
                f"dsc={eval_res['summary'].get('mean', float('nan')):.4f}",
                flush=True,
            )

        all_metrics[modality] = {
            "task_id": task_id,
            "task_name": task_name,
            "train_subjects": len(tr_kept),
            "test_subjects": len(ts_kept),
            "methods": modality_out,
        }

    out_json = metrics_root / f"nnunet_dsc_summary_z{args.factor_z}.json"
    out_json.write_text(json.dumps(all_metrics, indent=2))
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
