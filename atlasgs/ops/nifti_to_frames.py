import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .nifti_io import load_nii


def robust_minmax(data, pmin=1.0, pmax=99.0):
    flat = data[np.isfinite(data)]
    if flat.size == 0:
        return 0.0, 1.0
    nonzero = flat[flat != 0]
    if nonzero.size >= 10:
        flat = nonzero
    lo = float(np.percentile(flat, pmin))
    hi = float(np.percentile(flat, pmax))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalize_to_uint8(data, lo, hi):
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    data = (data * 255.0).round().astype(np.uint8)
    return data


def slice_volume(data, axis):
    if axis == 0:
        return [data[i, :, :] for i in range(data.shape[0])]
    if axis == 1:
        return [data[:, i, :] for i in range(data.shape[1])]
    return [data[:, :, i] for i in range(data.shape[2])]


def save_frames(slices, out_dir, mirror_mode="flip", skip_empty=True, threshold=0.01):
    out_dir = Path(out_dir)
    orig_dir = out_dir / "original"
    mir_dir = out_dir / "mirror"
    orig_dir.mkdir(parents=True, exist_ok=True)
    mir_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    for slc in slices:
        if skip_empty:
            non_zero_ratio = np.count_nonzero(slc) / slc.size
            if non_zero_ratio < threshold:
                continue
        kept.append(slc)

    for idx, slc in enumerate(kept):
        img = Image.fromarray(slc).convert("RGB")
        img.save(orig_dir / f"{idx:04d}.png")

    if mirror_mode == "reverse":
        for idx, slc in enumerate(reversed(kept)):
            img = Image.fromarray(slc).convert("RGB")
            img.save(mir_dir / f"{idx:04d}.png")
    else:
        for idx, slc in enumerate(kept):
            img = Image.fromarray(slc).convert("RGB")
            if mirror_mode == "flip":
                img = ImageOps.mirror(img)
            img.save(mir_dir / f"{idx:04d}.png")

    return len(kept)


def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI volume to MedGS frames.")
    parser.add_argument("--input", required=True, help="Path to input NIfTI.")
    parser.add_argument("--output", required=True, help="Output directory for frames.")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--pmin", type=float, default=1.0)
    parser.add_argument("--pmax", type=float, default=99.0)
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument(
        "--mirror-mode",
        type=str,
        default="flip",
        choices=["flip", "copy", "reverse"],
        help="How to build mirror/ frames.",
    )
    parser.add_argument("--copy-ref", action="store_true")
    args = parser.parse_args()

    data, affine, header = load_nii(args.input)
    lo, hi = robust_minmax(data, pmin=args.pmin, pmax=args.pmax)
    data_u8 = normalize_to_uint8(data, lo, hi)
    slices = slice_volume(data_u8, args.axis)

    out_dir = Path(args.output)
    count = save_frames(
        slices,
        out_dir,
        mirror_mode=args.mirror_mode,
        skip_empty=args.skip_empty,
        threshold=args.threshold,
    )

    stats = {
        "input": str(args.input),
        "axis": args.axis,
        "pmin": args.pmin,
        "pmax": args.pmax,
        "lo": lo,
        "hi": hi,
        "shape": list(data.shape),
        "frames": count,
    }
    with open(out_dir / "normalize.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if args.copy_ref:
        ref_path = out_dir / "ref.nii.gz"
        if not ref_path.exists():
            import shutil
            shutil.copy(args.input, ref_path)


if __name__ == "__main__":
    main()
