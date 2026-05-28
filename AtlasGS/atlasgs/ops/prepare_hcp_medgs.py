import argparse
import csv
import os
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, map_coordinates
from scipy.optimize import minimize

from .degrade import simulate_aniso
from .nifti_io import load_nii, save_json, save_nii


MODS = ("T1w", "T2w", "ASL", "DWI_b0", "DWI_b900", "DWI_b2000")


def parse_run_idx(name: str):
    match = re.search(r"_run-(\d+)_", name)
    return int(match.group(1)) if match else 999


def select_first_run(paths):
    return sorted(paths, key=lambda path: (parse_run_idx(path.name), path.name))[0]


def detect_modality(name: str):
    if "_T1w" in name:
        return "T1w"
    if "_T2w" in name:
        return "T2w"
    if "_asl" in name:
        return "ASL"
    if "_dwi_" in name and "bval0" in name:
        return "DWI_b0"
    if "_dwi_" in name and "bval900" in name:
        return "DWI_b900"
    if "_dwi_" in name and "bval2000" in name:
        return "DWI_b2000"
    return None


def gather_cases(src_root: Path):
    cases = defaultdict(lambda: defaultdict(list))
    for path in src_root.rglob("*.nii.gz"):
        sub_match = re.search(r"(sub-\d+)", str(path))
        ses_match = re.search(r"(ses-\d+)", str(path))
        if not sub_match or not ses_match:
            continue
        modality = detect_modality(path.name)
        if modality is None:
            continue
        key = (sub_match.group(1), ses_match.group(1))
        cases[key][modality].append(path)
    return cases


def write_split_csv(path: Path, subject_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id"])
        for subject_id in subject_ids:
            writer.writerow([subject_id])


def base_subject_id(case_id: str) -> str:
    if "_ses-" in case_id:
        return case_id.split("_ses-")[0]
    return case_id


def split_cases_grouped_by_subject(case_ids, train_ratio, seed):
    grouped = defaultdict(list)
    for case_id in case_ids:
        grouped[base_subject_id(case_id)].append(case_id)

    subjects = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    for subject in subjects:
        grouped[subject] = sorted(grouped[subject])

    n_subjects = len(subjects)
    n_cases = len(case_ids)
    target_subjects = int(round(n_subjects * train_ratio))
    target_subjects = max(0, min(n_subjects, target_subjects))
    target_cases = int(round(n_cases * train_ratio))
    target_cases = max(0, min(n_cases, target_cases))

    # DP on (num_subjects, num_cases) to keep a balanced split while preventing subject leakage.
    states = {(0, 0): 0}
    for idx, subject in enumerate(subjects):
        bit = 1 << idx
        weight = len(grouped[subject])
        updated = dict(states)
        for (count_subjects, count_cases), mask in states.items():
            nxt = (count_subjects + 1, count_cases + weight)
            if nxt not in updated:
                updated[nxt] = mask | bit
        states = updated

    best_count_subjects = 0
    best_count_cases = 0
    best_mask = 0
    best_key = None
    for (count_subjects, count_cases), mask in states.items():
        key = (
            abs(count_subjects - target_subjects),
            abs(count_cases - target_cases),
            count_subjects,
            count_cases,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_count_subjects = count_subjects
            best_count_cases = count_cases
            best_mask = mask

    train_subjects = {subjects[idx] for idx in range(n_subjects) if (best_mask >> idx) & 1}
    train_cases = []
    test_cases = []
    for subject in sorted(subjects):
        target = train_cases if subject in train_subjects else test_cases
        target.extend(grouped[subject])

    stats = {
        "train_subjects": int(best_count_subjects),
        "test_subjects": int(n_subjects - best_count_subjects),
        "train_cases": int(best_count_cases),
        "test_cases": int(n_cases - best_count_cases),
    }
    return train_cases, test_cases, stats


def world_center(shape, affine):
    voxel_center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    return nib.affines.apply_affine(affine, voxel_center)


def center_align_affine(moving_shape, moving_affine, fixed_shape, fixed_affine):
    fixed_center = world_center(fixed_shape, fixed_affine)
    moving_center = world_center(moving_shape, moving_affine)
    delta = fixed_center - moving_center
    aligned = moving_affine.copy()
    aligned[:3, 3] += delta
    return aligned, delta.astype(np.float32)


def rotate_ccw_xy_with_affine(data, affine):
    rot = np.rot90(data, k=1, axes=(0, 1)).astype(np.float32)
    n0 = int(data.shape[0])
    transform = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, float(n0 - 1)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_affine = affine @ transform
    return rot, rotated_affine


def euler_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def rigid_world_matrix(params, pivot_world):
    tx, ty, tz, rx, ry, rz = params
    rot = euler_matrix(rx, ry, rz)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

    pivot_plus = np.eye(4, dtype=np.float64)
    pivot_plus[:3, 3] = pivot_world
    pivot_minus = np.eye(4, dtype=np.float64)
    pivot_minus[:3, 3] = -pivot_world
    return pivot_plus @ transform @ pivot_minus


def sample_world(volume, affine, world_points, order=1):
    inv = np.linalg.inv(affine)
    vox = nib.affines.apply_affine(inv, world_points)
    coords = [vox[:, 0], vox[:, 1], vox[:, 2]]
    return map_coordinates(volume, coords, order=order, mode="nearest")


def resample_to_ref_chunked(moving, moving_affine, ref_affine, ref_shape, order=1, chunk_z=8):
    inv = np.linalg.inv(moving_affine)
    out = np.zeros(ref_shape, dtype=np.float32)

    ii = np.arange(ref_shape[0], dtype=np.float32)
    jj = np.arange(ref_shape[1], dtype=np.float32)

    for z0 in range(0, ref_shape[2], chunk_z):
        z1 = min(ref_shape[2], z0 + chunk_z)
        kk = np.arange(z0, z1, dtype=np.float32)
        grid = np.stack(np.meshgrid(ii, jj, kk, indexing="ij"), axis=-1).reshape(-1, 3)
        world = nib.affines.apply_affine(ref_affine, grid)
        vox = nib.affines.apply_affine(inv, world)
        coords = [vox[:, 0], vox[:, 1], vox[:, 2]]
        sampled = map_coordinates(moving, coords, order=order, mode="nearest")
        out[:, :, z0:z1] = sampled.reshape(ref_shape[0], ref_shape[1], z1 - z0)

    return out


def build_fixed_samples(fixed_volume, fixed_mask, fixed_affine, step=4, max_points=120000, seed=0):
    shape = fixed_volume.shape
    ii = np.arange(0, shape[0], step, dtype=np.int32)
    jj = np.arange(0, shape[1], step, dtype=np.int32)
    kk = np.arange(0, shape[2], step, dtype=np.int32)
    grid = np.stack(np.meshgrid(ii, jj, kk, indexing="ij"), axis=-1).reshape(-1, 3)
    in_mask = fixed_mask[grid[:, 0], grid[:, 1], grid[:, 2]] > 0
    grid = grid[in_mask]
    if len(grid) == 0:
        return None, None
    if len(grid) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(grid), size=max_points, replace=False)
        grid = grid[idx]
    world = nib.affines.apply_affine(fixed_affine, grid.astype(np.float64))
    fixed_vals = fixed_volume[grid[:, 0], grid[:, 1], grid[:, 2]]
    return world.astype(np.float32), fixed_vals.astype(np.float32)


def ncc_loss(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean
    denom = np.sqrt((a_centered * a_centered).mean()) * np.sqrt((b_centered * b_centered).mean())
    if denom < 1e-8:
        return 1.0
    ncc = (a_centered * b_centered).mean() / denom
    return float(-ncc)


def rigid_register_affine(
    moving,
    moving_affine,
    fixed,
    fixed_mask,
    fixed_affine,
    step=4,
    max_iter=70,
    seed=0,
):
    centered_affine, center_delta = center_align_affine(moving.shape, moving_affine, fixed.shape, fixed_affine)
    fixed_grad = gaussian_gradient_magnitude(fixed, sigma=1.0).astype(np.float32)
    moving_grad = gaussian_gradient_magnitude(moving, sigma=1.0).astype(np.float32)

    world_points, fixed_vals = build_fixed_samples(
        fixed_grad,
        fixed_mask,
        fixed_affine,
        step=step,
        max_points=120000,
        seed=seed,
    )
    if world_points is None or fixed_vals is None:
        return centered_affine, {"ok": False, "reason": "no fixed samples", "center_delta_mm": center_delta.tolist()}

    pivot = world_center(fixed.shape, fixed_affine)

    def objective(params):
        tx, ty, tz, rx, ry, rz = params
        if (
            abs(tx) > 25
            or abs(ty) > 25
            or abs(tz) > 25
            or abs(rx) > np.deg2rad(15)
            or abs(ry) > np.deg2rad(15)
            or abs(rz) > np.deg2rad(15)
        ):
            return 10.0
        rigid_world = rigid_world_matrix(params, pivot)
        moved_affine = rigid_world @ centered_affine
        moving_vals = sample_world(moving_grad, moved_affine, world_points, order=1)
        return ncc_loss(fixed_vals, moving_vals)

    result = minimize(
        objective,
        x0=np.zeros(6, dtype=np.float64),
        method="Powell",
        options={"maxiter": max_iter, "xtol": 1e-2, "ftol": 1e-4},
    )
    best_params = result.x if result.success else np.zeros(6, dtype=np.float64)
    rigid_world = rigid_world_matrix(best_params, pivot)
    out_affine = rigid_world @ centered_affine
    summary = {
        "ok": bool(result.success),
        "message": str(result.message),
        "fun": float(result.fun),
        "iters": int(getattr(result, "nit", -1)),
        "center_delta_mm": center_delta.tolist(),
        "rigid_translation_mm": [float(x) for x in best_params[:3]],
        "rigid_rotation_deg": [float(np.rad2deg(x)) for x in best_params[3:]],
    }
    return out_affine, summary


def run_synthstrip(apptainer_sif: Path, t1_in: Path, t1_out: Path, mask_out: Path, gpu: bool):
    case_dir = str(t1_in.parent.resolve())
    bind_paths = [f"{case_dir}:{case_dir}"]
    extra_bind = os.environ.get("APPTAINER_BINDPATH", "").strip()
    if extra_bind:
        bind_paths.append(extra_bind)

    cmd = [
        "apptainer",
        "exec",
        "--bind",
        ",".join(bind_paths),
        str(apptainer_sif),
        "mri_synthstrip",
        "-i",
        str(t1_in),
        "-o",
        str(t1_out),
        "-m",
        str(mask_out),
    ]
    if gpu:
        cmd.append("-g")
    subprocess.run(cmd, check=True)


def parse_factors(text):
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return sorted(set(values))


def parse_dwi_pref(text):
    order = []
    for token in text.split(","):
        token = token.strip()
        if token == "0":
            order.append("DWI_b0")
        elif token == "900":
            order.append("DWI_b900")
        elif token == "2000":
            order.append("DWI_b2000")
    if not order:
        order = ["DWI_b900", "DWI_b2000", "DWI_b0"]
    return order


def main():
    parser = argparse.ArgumentParser(description="Prepare HCP/FOMO PT008 dataset into atlasgs-compatible per-subject layout.")
    parser.add_argument("--src", required=True, help="Source root, e.g. PT008_AdolescentBrainDevelopment.")
    parser.add_argument("--out", required=True, help="Output root, e.g. data/hcp_medgs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument(
        "--align-method",
        choices=("center", "rigid"),
        default="rigid",
        help="Cross-modality alignment method into T1 space.",
    )
    parser.add_argument("--dwi-pref", type=str, default="900,2000,0", help="Preferred DWI b-values order.")
    parser.add_argument("--degrade-factors", type=str, default="3,5,7", help="Comma-separated z degradation factors.")
    parser.add_argument("--degrade-modalities", type=str, default="t2,dwi,asl")
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--degrade-mode", type=str, default="avgpool")
    parser.add_argument(
        "--rotate-ccw-90",
        dest="rotate_ccw_90",
        action="store_true",
        help="Rotate each modality 90 degrees CCW in the XY plane before skull stripping and registration.",
    )
    parser.add_argument(
        "--no-rotate-ccw-90",
        dest="rotate_ccw_90",
        action="store_false",
        help="Disable XY-plane 90 degree CCW rotation.",
    )
    parser.set_defaults(rotate_ccw_90=True)
    parser.add_argument(
        "--allow-missing-asl",
        action="store_true",
        help="Allow sessions without ASL (default: strict, skip missing ASL).",
    )
    parser.add_argument(
        "--allow-missing-dwi",
        action="store_true",
        help="Allow sessions without preferred DWI (default: strict, skip missing DWI).",
    )
    parser.add_argument(
        "--preserve-native-grids",
        action="store_true",
        help=(
            "Do not resample target modalities to T1 grid. "
            "Keep native voxel grid and write aligned affine only."
        ),
    )
    parser.add_argument("--no-skullstrip", action="store_true", help="Skip SynthStrip and use nonzero T1 as mask.")
    parser.add_argument(
        "--apptainer-sif",
        type=str,
        default=None,
        help="FreeSurfer container path with mri_synthstrip.",
    )
    parser.add_argument("--synthstrip-gpu", action="store_true", help="Run SynthStrip with -g.")
    parser.add_argument("--max-cases", type=int, default=0, help="If > 0, process only first N sorted sub-ses.")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = gather_cases(src_root)
    keys = sorted(cases.keys())
    if args.max_cases > 0:
        keys = keys[: args.max_cases]

    if not keys:
        raise RuntimeError(f"No HCP cases found under {src_root}")

    dwi_pref = parse_dwi_pref(args.dwi_pref)
    factors = parse_factors(args.degrade_factors)
    degrade_modalities = {token.strip().lower() for token in args.degrade_modalities.split(",") if token.strip()}

    processed = []
    skipped = []

    for sub, ses in keys:
        mods = cases[(sub, ses)]
        if "T1w" not in mods:
            skipped.append((sub, ses, "missing T1w"))
            continue

        picked = {"T1w": select_first_run(mods["T1w"])}
        if "T2w" in mods:
            picked["T2w"] = select_first_run(mods["T2w"])
        if "ASL" in mods:
            picked["ASL"] = select_first_run(mods["ASL"])
        elif not args.allow_missing_asl:
            skipped.append((sub, ses, "missing ASL"))
            continue

        chosen_dwi_mod = None
        for key in dwi_pref:
            if key in mods:
                chosen_dwi_mod = key
                picked["DWI"] = select_first_run(mods[key])
                break
        if chosen_dwi_mod is None and not args.allow_missing_dwi:
            skipped.append((sub, ses, "missing DWI in preferred set"))
            continue

        case_id = f"{sub}_{ses}"
        case_out = out_root / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        t1_data, t1_aff, t1_header = load_nii(picked["T1w"])
        if args.rotate_ccw_90:
            t1_data, t1_aff = rotate_ccw_xy_with_affine(t1_data, t1_aff)

        t1_native_path = case_out / "t1_native.nii.gz"
        t1_brain_path = case_out / "t1_gt.nii.gz"
        mask_path = case_out / "mask_brain.nii.gz"
        save_nii(t1_native_path, t1_data, t1_aff, header=t1_header)

        if args.no_skullstrip:
            mask = (t1_data > 0).astype(np.uint8)
            save_nii(mask_path, mask, t1_aff, header=t1_header)
            save_nii(t1_brain_path, (t1_data * mask).astype(np.float32), t1_aff, header=t1_header)
        else:
            if not args.apptainer_sif:
                raise ValueError("Provide --apptainer-sif or use --no-skullstrip.")
            run_synthstrip(
                Path(args.apptainer_sif),
                t1_native_path,
                t1_brain_path,
                mask_path,
                args.synthstrip_gpu,
            )

        t1_brain, fixed_aff, fixed_header = load_nii(t1_brain_path)
        fixed_mask, _, _ = load_nii(mask_path)
        fixed_mask = (fixed_mask > 0.5).astype(np.float32)

        reg_info = {}

        modality_map = [
            ("T2w", "t2_gt.nii.gz", "t2"),
            ("ASL", "asl_gt.nii.gz", "asl"),
            ("DWI", "dwi_gt.nii.gz", "dwi"),
        ]

        for key, out_name, short_name in modality_map:
            if key not in picked:
                continue
            moving, moving_aff, moving_header = load_nii(picked[key])
            if args.rotate_ccw_90:
                moving, moving_aff = rotate_ccw_xy_with_affine(moving, moving_aff)

            if args.align_method == "center":
                aligned_aff, delta = center_align_affine(moving.shape, moving_aff, t1_brain.shape, fixed_aff)
                reg_info[short_name] = {
                    "method": "center",
                    "center_delta_mm": delta.tolist(),
                }
            else:
                aligned_aff, summary = rigid_register_affine(
                    moving=moving,
                    moving_affine=moving_aff,
                    fixed=t1_brain,
                    fixed_mask=fixed_mask,
                    fixed_affine=fixed_aff,
                    step=4,
                    max_iter=70,
                    seed=args.seed,
                )
                summary["method"] = "rigid"
                reg_info[short_name] = summary

            if args.preserve_native_grids:
                # Keep native sampling and write the aligned affine only.
                # This avoids active intensity resampling during preprocessing.
                moving_native = moving.astype(np.float32, copy=False)
                save_nii(case_out / out_name, moving_native, aligned_aff, header=moving_header)
                gt_for_degrade = moving_native
                aff_for_degrade = aligned_aff
                hdr_for_degrade = moving_header
            else:
                resampled = resample_to_ref_chunked(
                    moving,
                    aligned_aff,
                    fixed_aff,
                    t1_brain.shape,
                    order=1,
                    chunk_z=8,
                )
                masked = (resampled * fixed_mask).astype(np.float32)
                save_nii(case_out / out_name, masked, fixed_aff, header=fixed_header)
                gt_for_degrade = masked
                aff_for_degrade = fixed_aff
                hdr_for_degrade = fixed_header

            if short_name in degrade_modalities and factors:
                for factor in factors:
                    lr, lr_aff = simulate_aniso(
                        gt_for_degrade,
                        aff_for_degrade,
                        factor_z=factor,
                        sigma_z=args.sigma_z,
                        mode=args.degrade_mode,
                    )
                    lr_path = case_out / f"{short_name}_lr_1x1x{factor}.nii.gz"
                    save_nii(lr_path, lr, lr_aff, header=hdr_for_degrade)
                    save_json(
                        lr_path.with_suffix(".json"),
                        {
                            "input": str(case_out / out_name),
                            "output": str(lr_path),
                            "factor_z": int(factor),
                            "sigma_z": float(args.sigma_z),
                            "mode": args.degrade_mode,
                            "align_method": args.align_method,
                        },
                    )

        meta = {
            "case_id": case_id,
            "source_paths": {k: str(v) for k, v in picked.items()},
            "align_method": args.align_method,
            "dwi_selected": chosen_dwi_mod,
            "modalities_present": sorted([k for k in picked if k != "T1w"]),
            "registration": reg_info,
            "rotate_ccw_90": bool(args.rotate_ccw_90),
            "allow_missing_asl": bool(args.allow_missing_asl),
            "allow_missing_dwi": bool(args.allow_missing_dwi),
            "skullstrip": "synthstrip" if not args.no_skullstrip else "t1_nonzero",
            "degrade_modalities": sorted(list(degrade_modalities)),
            "degrade_factors": factors,
            "preserve_native_grids": bool(args.preserve_native_grids),
        }
        save_json(case_out / "meta.json", meta)
        processed.append(case_id)

    train_subjects, test_subjects, split_stats = split_cases_grouped_by_subject(
        processed,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    ordered_all = sorted(processed)

    write_split_csv(out_root / "all.csv", ordered_all)
    write_split_csv(out_root / "train.csv", train_subjects)
    write_split_csv(out_root / "test.csv", test_subjects)

    if skipped:
        save_json(
            out_root / "skipped.json",
            [{"sub": sub, "ses": ses, "reason": reason} for sub, ses, reason in skipped],
        )

    print(f"Source cases considered: {len(keys)}")
    print(f"Processed cases: {len(processed)}")
    print(f"Skipped cases: {len(skipped)}")
    print(f"Train/Test cases: {split_stats['train_cases']}/{split_stats['test_cases']}")
    print(f"Train/Test subjects: {split_stats['train_subjects']}/{split_stats['test_subjects']}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
