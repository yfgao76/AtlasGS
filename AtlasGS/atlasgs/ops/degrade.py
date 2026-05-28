from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .nifti_io import load_nii, save_nii, save_json


def simulate_aniso(flair_gt, affine, factor_z=3, sigma_z=1.0, mode="avgpool"):
    """Simulate anisotropic 1x1x(factor_z) from isotropic FLAIR.

    mode: "avgpool" or "stride"
    """
    if sigma_z > 0:
        blurred = gaussian_filter1d(flair_gt, sigma=sigma_z, axis=2)
    else:
        blurred = flair_gt

    if mode == "avgpool":
        z = blurred.shape[2]
        z_trim = z - (z % factor_z)
        cropped = blurred[:, :, :z_trim]
        pooled = cropped.reshape(
            cropped.shape[0],
            cropped.shape[1],
            z_trim // factor_z,
            factor_z,
        ).mean(axis=3)
        lr = pooled
    elif mode == "stride":
        lr = blurred[:, :, ::factor_z]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    new_affine = affine.copy()
    new_affine[:3, 2] *= factor_z
    return lr.astype(np.float32), new_affine


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--flair_gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor_z", type=int, default=3)
    parser.add_argument("--sigma_z", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="avgpool")
    parser.add_argument("--params_out", default=None)
    args = parser.parse_args()

    flair_gt, affine, header = load_nii(args.flair_gt)
    flair_lr, new_affine = simulate_aniso(
        flair_gt,
        affine,
        factor_z=args.factor_z,
        sigma_z=args.sigma_z,
        mode=args.mode,
    )
    save_nii(args.out, flair_lr, new_affine, header=header)

    params = {
        "factor_z": args.factor_z,
        "sigma_z": args.sigma_z,
        "mode": args.mode,
        "input": str(args.flair_gt),
        "output": str(args.out),
    }
    if args.params_out:
        save_json(args.params_out, params)
    else:
        params_path = Path(args.out).with_suffix(".json")
        save_json(params_path, params)


if __name__ == "__main__":
    main()
