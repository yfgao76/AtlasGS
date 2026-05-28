import argparse
import csv
import random
import shutil
from pathlib import Path


def list_subjects_with_suffix(root, suffix):
    subjects = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.endswith(suffix):
            continue
        sid = name[: -len(suffix)]
        if sid:
            subjects[sid] = name
    return subjects


def resolve_pair(t1_dir, flair_dir, t1_rel, flair_rel, t1_name, flair_name):
    candidates = [
        (t1_dir / t1_rel, flair_dir / flair_rel),
        (t1_dir / t1_name, flair_dir / flair_name),
    ]
    for t1, flair in candidates:
        if t1.exists() and flair.exists():
            return t1, flair
    return None


def copy_subject(out_root, subject_id, t1_path, flair_path):
    sub_dir = out_root / subject_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(t1_path, sub_dir / "t1_gt.nii.gz")
    shutil.copy2(flair_path, sub_dir / "flair_gt.nii.gz")


def write_split_csv(path, subjects):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id"])
        for sid in subjects:
            writer.writerow([sid])


def main():
    parser = argparse.ArgumentParser(description="Prepare UKBiobank T1/FLAIR pairs for MedGS.")
    parser.add_argument("--src", required=True, help="UKBiobank/orig root.")
    parser.add_argument("--out", required=True, help="Output data folder.")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t1-rel", type=str, default="T1/T1_brain.nii.gz")
    parser.add_argument("--flair-rel", type=str, default="T2_FLAIR/T2_FLAIR_brain.nii.gz")
    parser.add_argument("--t1-name", type=str, default="T1_brain.nii.gz")
    parser.add_argument("--flair-name", type=str, default="T2_FLAIR_brain.nii.gz")
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    args = parser.parse_args()

    src = Path(args.src)
    t1_root = src / "T1w"
    flair_root = src / "T2flair"
    if not t1_root.is_dir() or not flair_root.is_dir():
        raise FileNotFoundError("Expected T1w and T2flair subfolders under --src.")

    t1_suffix = "_20252_2_0"
    flair_suffix = "_20253_2_0"
    t1_subjects = list_subjects_with_suffix(t1_root, t1_suffix)
    flair_subjects = list_subjects_with_suffix(flair_root, flair_suffix)
    common = sorted(set(t1_subjects.keys()) & set(flair_subjects.keys()))

    valid = []
    for sid in common:
        t1_dir = t1_root / t1_subjects[sid]
        flair_dir = flair_root / flair_subjects[sid]
        pair = resolve_pair(
            t1_dir,
            flair_dir,
            args.t1_rel,
            args.flair_rel,
            args.t1_name,
            args.flair_name,
        )
        if pair is None:
            continue
        valid.append((sid, pair[0], pair[1]))

    if len(valid) < args.num:
        raise ValueError(f"Only {len(valid)} valid subjects found, need {args.num}.")

    random.seed(args.seed)
    random.shuffle(valid)
    selected = valid[:args.num]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for sid, t1_path, flair_path in selected:
        copy_subject(out_root, sid, t1_path, flair_path)

    subjects = [sid for sid, _, _ in selected]
    if args.train_count + args.test_count > len(subjects):
        raise ValueError("train_count + test_count exceeds selected subjects.")
    train = subjects[: args.train_count]
    test = subjects[args.train_count : args.train_count + args.test_count]

    write_split_csv(out_root / "train.csv", train)
    write_split_csv(out_root / "test.csv", test)

    print(f"Selected {len(subjects)} subjects.")
    print(f"Train: {len(train)} Test: {len(test)}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
