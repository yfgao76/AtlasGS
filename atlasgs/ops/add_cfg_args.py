import argparse
from pathlib import Path
import argparse as _argparse


def write_cfg_args(model_path, source_path):
    model_path = Path(model_path)
    cfg_path = model_path / "cfg_args"
    if cfg_path.exists():
        return False
    cfg = _argparse.Namespace(
        sh_degree=0,
        source_path=str(Path(source_path).resolve()),
        model_path=str(model_path.resolve()),
        images="images",
        depths="",
        resolution=-1,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=False,
        gs_type="gs",
        camera="mirror",
        distance=1.0,
        num_pts=100_000,
        poly_degree=7,
    )
    model_path.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(str(cfg), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add cfg_args to MedGS model folders.")
    parser.add_argument("--out-root", required=True, help="Root output folder with subject subdirs.")
    parser.add_argument("--data-root", required=True, help="Data root for locating frames.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    data_root = Path(args.data_root)
    if not out_root.is_dir():
        raise FileNotFoundError(out_root)

    updated = 0
    scanned = 0
    for sub in sorted(out_root.iterdir()):
        if not sub.is_dir():
            continue
        subject_id = sub.name
        flair_lr_frames = data_root / subject_id / "flair_lr_frames"
        if not flair_lr_frames.exists():
            flair_lr_frames = sub / "flair_lr_frames"

        for model_name in ["flair_lr_model", "t1_model", "flair_t1guided_model"]:
            model_path = sub / model_name
            if not model_path.exists():
                continue
            scanned += 1
            if write_cfg_args(model_path, flair_lr_frames):
                updated += 1

    print(f"Scanned {scanned} models, added cfg_args to {updated}.")


if __name__ == "__main__":
    main()
