import argparse
from pathlib import Path

from .degrade import simulate_aniso
from .nifti_io import load_nii, save_nii, save_json


def main():
    parser = argparse.ArgumentParser(description="Batch degrade FLAIR to 1x1xF anisotropy.")
    parser.add_argument("--data-root", required=True, help="Root with per-subject folders.")
    parser.add_argument("--flair-name", default="flair_gt.nii.gz")
    parser.add_argument("--out-name", default="flair_lr_1x1x{factor}.nii.gz")
    parser.add_argument("--factor-z", type=int, default=3)
    parser.add_argument("--factors", type=str, default="", help="Comma-separated list, e.g. 3,5,7")
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="avgpool")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    subjects = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not subjects:
        raise FileNotFoundError(f"No subject folders under {root}")

    processed = 0
    skipped = 0
    if args.factors:
        factors = [int(x) for x in args.factors.split(",") if x.strip()]
    else:
        factors = [args.factor_z]

    for sub in subjects:
        flair_path = sub / args.flair_name
        if not flair_path.exists():
            skipped += 1
            continue

        flair_gt, affine, header = load_nii(flair_path)
        for fz in factors:
            out_name = args.out_name.format(factor=fz)
            out_path = sub / out_name
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            flair_lr, new_affine = simulate_aniso(
                flair_gt,
                affine,
                factor_z=fz,
                sigma_z=args.sigma_z,
                mode=args.mode,
            )
            save_nii(out_path, flair_lr, new_affine, header=header)
            save_json(
                out_path.with_suffix(".json"),
                {
                    "input": str(flair_path),
                    "output": str(out_path),
                    "factor_z": fz,
                    "sigma_z": args.sigma_z,
                    "mode": args.mode,
                },
            )
            processed += 1

    print(f"Processed: {processed} Skipped: {skipped} Total: {len(subjects)}")


if __name__ == "__main__":
    main()
