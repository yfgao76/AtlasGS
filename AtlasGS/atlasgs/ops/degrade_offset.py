import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .nifti_io import load_nii, save_json, save_nii


def simulate_aniso_offset(vol, affine, factor_z=3, sigma_z=1.0, mode="avgpool", offset=0):
    factor_z = int(max(1, factor_z))
    offset = int(max(0, offset))

    if sigma_z > 0:
        blurred = gaussian_filter1d(vol, sigma=float(sigma_z), axis=2)
    else:
        blurred = vol

    if mode == "avgpool":
        z = blurred.shape[2]
        if offset >= z:
            raise ValueError(f"offset={offset} out of range for depth={z}")
        usable = z - offset
        z_trim = usable - (usable % factor_z)
        cropped = blurred[:, :, offset : offset + z_trim]
        lr = cropped.reshape(
            cropped.shape[0],
            cropped.shape[1],
            z_trim // factor_z,
            factor_z,
        ).mean(axis=3)
    elif mode == "stride":
        lr = blurred[:, :, offset::factor_z]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + float(offset) * affine[:3, 2]
    new_affine[:3, 2] = affine[:3, 2] * float(factor_z)
    return lr.astype(np.float32), new_affine


def main():
    parser = argparse.ArgumentParser(
        description="Simulate anisotropic LR with controllable z-window offset."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor-z", type=int, default=3)
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="avgpool")
    parser.add_argument("--offset", type=int, default=0, help="Start offset along z before pooling/stride.")
    parser.add_argument("--params-out", default=None)
    args = parser.parse_args()

    vol, affine, header = load_nii(args.input)
    lr, lr_affine = simulate_aniso_offset(
        vol,
        affine,
        factor_z=args.factor_z,
        sigma_z=args.sigma_z,
        mode=args.mode,
        offset=args.offset,
    )
    save_nii(args.out, lr, lr_affine, header=header)

    params = {
        "input": str(args.input),
        "output": str(args.out),
        "factor_z": int(args.factor_z),
        "sigma_z": float(args.sigma_z),
        "mode": str(args.mode),
        "offset": int(args.offset),
    }
    if args.params_out:
        save_json(args.params_out, params)
    else:
        params_path = Path(args.out).with_suffix(".json")
        save_json(params_path, params)


if __name__ == "__main__":
    main()
