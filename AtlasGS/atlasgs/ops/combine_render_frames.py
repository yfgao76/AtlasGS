import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def list_pngs(path):
    path = Path(path)
    if (path / "original").is_dir():
        path = path / "original"
    files = sorted(path.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {path}")
    return files


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def save_gray(path, arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(description="Compose two rendered frame folders by sum or weighted sum.")
    parser.add_argument("--base", required=True, help="Base rendered frames folder.")
    parser.add_argument("--residual", required=True, help="Residual rendered frames folder.")
    parser.add_argument("--out", required=True, help="Output composed frames folder.")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    args = parser.parse_args()

    base_files = list_pngs(args.base)
    res_files = list_pngs(args.residual)
    if len(base_files) != len(res_files):
        raise ValueError(f"Frame count mismatch: base={len(base_files)} residual={len(res_files)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (b_file, r_file) in enumerate(zip(base_files, res_files)):
        if b_file.name != r_file.name:
            raise ValueError(f"Frame name mismatch at index {idx}: {b_file.name} vs {r_file.name}")
        base = load_gray(b_file)
        residual = load_gray(r_file)
        composed = args.base_weight * base + args.residual_weight * residual
        save_gray(out_dir / b_file.name, composed)


if __name__ == "__main__":
    main()
