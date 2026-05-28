import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nii(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_nii(path, data, affine, header=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        img = nib.Nifti1Image(data, affine)
    else:
        img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, str(path))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def world_grid_from_affine(shape, affine):
    """Return world coords for each voxel in shape, as (N, 3) float32."""
    ijk = np.stack(np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk)
    return xyz.astype(np.float32)


def normalize_world_coords(xyz, mask=None):
    """Normalize coords to [-1, 1] using bounding box in world space."""
    if mask is not None:
        xyz_use = xyz[mask.reshape(-1) > 0]
    else:
        xyz_use = xyz
    mins = xyz_use.min(axis=0)
    maxs = xyz_use.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins) / 2.0
    scale[scale == 0] = 1.0
    xyz_n = (xyz - center) / scale
    return xyz_n.astype(np.float32), center, scale
