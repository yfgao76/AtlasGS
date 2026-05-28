import argparse
import csv
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

from atlasgs.ops.nifti_io import save_nii


MODALITY_DIRS = {
    "t1": "T1w",
    "t2": "T2w",
    "flair": "Flair",
    "mask": "mask",
    "seg": "seg",
    "tumor_seg": "tumor_seg",
    "ventricle_mask": "ventricle_mask",
}


def sorted_slice_files(folder: Path):
    files = []
    for p in folder.glob("*.npy"):
        try:
            files.append((int(p.stem), p))
        except ValueError:
            continue
    files.sort(key=lambda item: item[0])
    return files


def discover_subject_indices(subject_dir: Path):
    z_values = set()
    slice_shape = None
    modality_files = {}
    for key, rel in MODALITY_DIRS.items():
        folder = subject_dir / rel
        if not folder.is_dir():
            modality_files[key] = []
            continue
        files = sorted_slice_files(folder)
        modality_files[key] = files
        for z_idx, _ in files:
            z_values.add(z_idx)
        if files and slice_shape is None:
            sample = np.load(files[0][1], mmap_mode="r")
            slice_shape = tuple(sample.shape)
    return modality_files, z_values, slice_shape


def build_volume(files, shape_hw, depth, rotate_ccw_90=True):
    volume = np.zeros((shape_hw[0], shape_hw[1], depth), dtype=np.float32)
    for z_idx, file_path in files:
        if z_idx < 0 or z_idx >= depth:
            continue
        arr = np.load(file_path)
        if arr.shape != shape_hw:
            continue
        if rotate_ccw_90:
            arr = np.rot90(arr, k=1)
        volume[:, :, z_idx] = arr.astype(np.float32, copy=False)
    return volume


def write_split_csv(path: Path, subject_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id"])
        for subject_id in subject_ids:
            writer.writerow([subject_id])


def process_one_subject(subject_path_str, out_root_str, min_depth, rotate_ccw_90):
    subject_path = Path(subject_path_str)
    out_root = Path(out_root_str)
    affine = np.eye(4, dtype=np.float32)
    modality_files, z_values, slice_shape = discover_subject_indices(subject_path)
    if slice_shape is None or not z_values:
        return False, subject_path.name, "no slices"

    depth = max(z_values) + 1
    if depth < min_depth:
        return False, subject_path.name, f"depth<{min_depth}"

    subject_out = out_root / subject_path.name
    subject_out.mkdir(parents=True, exist_ok=True)

    volumes = {}
    for key in MODALITY_DIRS:
        files = modality_files.get(key, [])
        volumes[key] = build_volume(files, slice_shape, depth, rotate_ccw_90=rotate_ccw_90)

    mask_bin = (volumes["mask"] > 0.5).astype(np.float32)
    t1 = volumes["t1"] * mask_bin
    t2 = volumes["t2"] * mask_bin
    flair = volumes["flair"] * mask_bin

    save_nii(subject_out / "t1_gt.nii.gz", t1, affine)
    save_nii(subject_out / "t2_gt.nii.gz", t2, affine)
    save_nii(subject_out / "flair_gt.nii.gz", flair, affine)
    save_nii(subject_out / "mask_brain.nii.gz", mask_bin.astype(np.uint8), affine)
    save_nii(subject_out / "seg.nii.gz", volumes["seg"], affine)
    save_nii(subject_out / "tumor_seg.nii.gz", volumes["tumor_seg"], affine)
    save_nii(subject_out / "ventricle_mask.nii.gz", volumes["ventricle_mask"], affine)
    return True, subject_path.name, "ok"


def main():
    parser = argparse.ArgumentParser(description="Convert GBM npy slices into UKBB-like atlasgs NIfTI layout.")
    parser.add_argument("--src", required=True, help="Source GBM_Dataset root.")
    parser.add_argument("--out", required=True, help="Output root, e.g., data/gbm_medgs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--no-rotate-ccw-90", action="store_true", help="Disable default 90deg CCW slice rotation.")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("UPENN-GBM-")])
    processed = []
    skipped = []

    rotate_ccw_90 = not args.no_rotate_ccw_90
    worker_args = [
        (str(subject), str(out_root), int(args.min_depth), rotate_ccw_90)
        for subject in subjects
    ]
    total = len(worker_args)
    done = 0

    with mp.Pool(processes=max(1, int(args.workers))) as pool:
        for ok, name, reason in pool.starmap(process_one_subject, worker_args):
            done += 1
            if ok:
                processed.append(name)
            else:
                skipped.append((name, reason))
            if done % 20 == 0 or done == total:
                print(f"Processed {done}/{total}")

    random.Random(args.seed).shuffle(processed)
    n_train = int(round(len(processed) * args.train_ratio))
    n_train = max(0, min(len(processed), n_train))
    train_subjects = processed[:n_train]
    test_subjects = processed[n_train:]

    write_split_csv(out_root / "all.csv", processed)
    write_split_csv(out_root / "train.csv", train_subjects)
    write_split_csv(out_root / "test.csv", test_subjects)

    print(f"Source subjects: {len(subjects)}")
    print(f"Processed subjects: {len(processed)}")
    print(f"Skipped subjects: {len(skipped)}")
    if skipped:
        print("First skipped:", skipped[:5])
    print(f"Train/Test: {len(train_subjects)}/{len(test_subjects)}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
