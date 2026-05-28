import numpy as np
from scipy.ndimage import map_coordinates

from .nifti_io import load_nii, save_nii, world_grid_from_affine


def resample_to_ref(moving, moving_affine, ref_affine, ref_shape, order=1):
    """Resample moving volume to ref grid using affine mapping."""
    ref_xyz = world_grid_from_affine(ref_shape, ref_affine)
    # world -> moving voxel
    inv = np.linalg.inv(moving_affine)
    ijk = np.dot(np.c_[ref_xyz, np.ones(len(ref_xyz))], inv.T)[:, :3]
    coords = [ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    sampled = map_coordinates(moving, coords, order=order, mode="nearest")
    return sampled.reshape(ref_shape).astype(np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--order", type=int, default=1)
    args = parser.parse_args()

    moving, moving_affine, _ = load_nii(args.moving)
    ref, ref_affine, ref_header = load_nii(args.ref)
    res = resample_to_ref(moving, moving_affine, ref_affine, ref.shape, order=args.order)
    save_nii(args.out, res, ref_affine, header=ref_header)


if __name__ == "__main__":
    main()
