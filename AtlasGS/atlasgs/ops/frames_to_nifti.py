import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .nifti_io import load_nii, save_nii


def frame_sort_key(path):
    stem = path.stem
    if "_" in stem:
        view_token, interp_token = stem.rsplit("_", 1)
        if view_token.isdigit() and interp_token.isdigit():
            return (0, int(view_token), int(interp_token), path.name)
    if stem.isdigit():
        return (1, int(stem), 0, path.name)
    return (2, 0, 0, path.name)


def load_frames(frames_dir):
    frames_dir = Path(frames_dir)
    if (frames_dir / "original").is_dir():
        frames_dir = frames_dir / "original"
    files = sorted(frames_dir.glob("*.png"), key=frame_sort_key)
    if not files:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    slices = []
    for path in files:
        img = Image.open(path).convert("L")
        slices.append(np.array(img, dtype=np.float32))
    return slices


def stack_slices(slices, axis):
    if axis == 0:
        return np.stack(slices, axis=0)
    if axis == 1:
        return np.stack(slices, axis=1)
    return np.stack(slices, axis=2)


def apply_normalization(vol, stats):
    vol = vol / 255.0
    if stats is None:
        return vol.astype(np.float32)
    lo = float(stats.get("lo", stats.get("p1", 0.0)))
    hi = float(stats.get("hi", stats.get("p99", 1.0)))
    return (vol * (hi - lo) + lo).astype(np.float32)


def maybe_resample(vol, target_shape):
    if vol.shape == target_shape:
        return vol
    zooms = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, zooms, order=1).astype(np.float32)


def match_z_by_index(vol, target_shape):
    if vol.shape[0] != target_shape[0] or vol.shape[1] != target_shape[1]:
        raise ValueError(
            f"XY shape mismatch for z-index matching: got {vol.shape[:2]}, "
            f"target {target_shape[:2]}"
        )
    src_z = int(vol.shape[2])
    tgt_z = int(target_shape[2])
    if src_z == tgt_z:
        return vol.astype(np.float32)
    if src_z <= 1:
        return np.repeat(vol, tgt_z, axis=2).astype(np.float32)
    idx = np.linspace(0, src_z - 1, num=tgt_z)
    idx = np.clip(np.round(idx).astype(np.int64), 0, src_z - 1)
    return vol[:, :, idx].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert MedGS frames to NIfTI volume.")
    parser.add_argument("--frames", required=True, help="Frames directory or its original/ subdir.")
    parser.add_argument("--ref", required=True, help="Reference NIfTI for affine and target shape.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--norm-json", default=None, help="normalize.json from nifti_to_frames.")
    parser.add_argument("--allow-resample", action="store_true")
    parser.add_argument(
        "--match-z-only",
        action="store_true",
        help="When shape mismatches, only align z by index selection; forbid XY resampling.",
    )
    args = parser.parse_args()

    ref_data, ref_affine, ref_header = load_nii(args.ref)
    slices = load_frames(args.frames)
    vol = stack_slices(slices, args.axis)

    stats = None
    if args.norm_json:
        with open(args.norm_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

    vol = apply_normalization(vol, stats)

    if vol.shape != ref_data.shape:
        if not args.allow_resample:
            raise ValueError(f"Frame volume shape {vol.shape} != ref shape {ref_data.shape}")
        if args.match_z_only:
            vol = match_z_by_index(vol, ref_data.shape)
        else:
            vol = maybe_resample(vol, ref_data.shape)

    save_nii(args.out, vol, ref_affine, header=ref_header)


if __name__ == "__main__":
    main()
