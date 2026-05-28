import argparse
from pathlib import Path

from .degrade import simulate_aniso
from .nifti_io import load_nii, save_nii, save_json


def main():
    parser = argparse.ArgumentParser(
        description="Batch degrade multiple GT modalities (e.g., t2/flair) to 1x1xF anisotropy."
    )
    parser.add_argument("--data-root", required=True, help="Root with per-subject folders.")
    parser.add_argument(
        "--modalities",
        type=str,
        default="t2,flair",
        help="Comma-separated modalities among {t1,t2,flair}.",
    )
    parser.add_argument("--factors", type=str, default="3,5,7", help="Comma-separated z factors.")
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="avgpool")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    factors = [int(x) for x in args.factors.split(",") if x.strip()]
    if not factors:
        raise ValueError("No valid factors provided.")

    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    valid_modalities = {"t1", "t2", "flair"}
    bad = [m for m in modalities if m not in valid_modalities]
    if bad:
        raise ValueError(f"Invalid modalities: {bad}. Valid: {sorted(valid_modalities)}")

    subjects = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not subjects:
        raise FileNotFoundError(f"No subject folders under {root}")

    processed = 0
    skipped = 0

    for sub in subjects:
        for modality in modalities:
            gt_path = sub / f"{modality}_gt.nii.gz"
            if not gt_path.exists():
                skipped += 1
                continue
            gt_vol, affine, header = load_nii(gt_path)
            for factor_z in factors:
                out_path = sub / f"{modality}_lr_1x1x{factor_z}.nii.gz"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                lr_vol, new_affine = simulate_aniso(
                    gt_vol,
                    affine,
                    factor_z=factor_z,
                    sigma_z=args.sigma_z,
                    mode=args.mode,
                )
                save_nii(out_path, lr_vol, new_affine, header=header)
                save_json(
                    out_path.with_suffix(".json"),
                    {
                        "input": str(gt_path),
                        "output": str(out_path),
                        "modality": modality,
                        "factor_z": factor_z,
                        "sigma_z": args.sigma_z,
                        "mode": args.mode,
                    },
                )
                processed += 1

    print(f"Subjects: {len(subjects)}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
