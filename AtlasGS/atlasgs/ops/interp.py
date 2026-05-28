import argparse

from .nifti_io import load_nii, save_nii
from .resample import resample_to_ref


def main():
    parser = argparse.ArgumentParser(description="Upsample to reference grid.")
    parser.add_argument("--lr", required=True, help="Low-res NIfTI.")
    parser.add_argument("--ref", required=True, help="Reference NIfTI for grid.")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--method",
        choices=["linear", "bspline3"],
        default="linear",
        help="Interpolation kernel: linear (order=1) or cubic B-spline (order=3).",
    )
    args = parser.parse_args()

    lr, lr_aff, _ = load_nii(args.lr)
    ref, ref_aff, ref_header = load_nii(args.ref)
    order = 3 if args.method == "bspline3" else 1
    up = resample_to_ref(lr, lr_aff, ref_aff, ref.shape, order=order)
    save_nii(args.out, up, ref_aff, header=ref_header)


if __name__ == "__main__":
    main()
