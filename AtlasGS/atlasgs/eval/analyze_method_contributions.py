import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median


METHODS = [
    "medgs_t1guided",
    "medgs_t1guided_topology",
    "medgs_t1guided_latent",
    "medgs_our",
    "medgs_fused_lrcons",
]

MODEL_DIRS = {
    "t1": "t1_model",
    "medgs_single": "flair_lr_model_z7",
    "medgs_t1guided": "flair_t1guided_model_z7",
    "medgs_t1guided_topology": "flair_t1guided_topology_model_z7",
    "medgs_t1guided_latent": "flair_t1guided_latent_model_z7",
    "medgs_our": "flair_t1guided_our_model_z7",
}

METRIC_DIRECTIONS = {
    "mae": -1,
    "ssim": 1,
    "psnr": 1,
    "edge_ncc": 1,
    "band_mae": -1,
    "z_grad_energy": -1,
}


def latest_ply_count(model_dir):
    point_root = Path(model_dir) / "point_cloud"
    if not point_root.exists():
        return None
    candidates = []
    for item in point_root.glob("iteration_*"):
        match = re.search(r"iteration_(\d+)$", item.name)
        if match:
            candidates.append((int(match.group(1)), item))
    if not candidates:
        return None
    ply = max(candidates)[1] / "point_cloud.ply"
    if not ply.exists():
        return None
    with open(ply, "rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def summarize(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def load_metrics(root, metrics_name):
    rows = []
    for path in sorted(Path(root).glob(f"*/{metrics_name}")):
        with open(path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append((path.parent.name, metrics))
    return rows


def quality_summary(rows, methods, base_method):
    out = {"means": {}, "paired_vs_base": {}}
    for method in methods:
        out["means"][method] = {}
        for metric in METRIC_DIRECTIONS:
            vals = [
                float(data[method][metric])
                for _, data in rows
                if method in data and metric in data[method] and data[method][metric] is not None
            ]
            out["means"][method][metric] = summarize(vals)

    paired_rows = [(subj, data) for subj, data in rows if base_method in data]
    for method in methods:
        if method == base_method:
            continue
        method_out = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            signed = []
            wins = 0
            for _, data in paired_rows:
                if method not in data:
                    continue
                if metric not in data[base_method] or metric not in data[method]:
                    continue
                delta = (float(data[method][metric]) - float(data[base_method][metric])) * direction
                signed.append(delta)
                wins += int(delta > 0)
            method_out[metric] = {
                "n": len(signed),
                "wins": wins,
                "win_rate": wins / len(signed) if signed else None,
                "mean_signed_improvement": mean(signed) if signed else None,
                "median_signed_improvement": median(signed) if signed else None,
            }
        out["paired_vs_base"][method] = method_out
    return out


def point_count_summary(root):
    out = {}
    root = Path(root)
    for method, model_dir_name in MODEL_DIRS.items():
        vals = []
        for subject_dir in root.iterdir():
            if not subject_dir.is_dir():
                continue
            count = latest_ply_count(subject_dir / model_dir_name)
            if count is not None:
                vals.append(count)
        out[method] = summarize(vals)
    return out


def meta_summary(root):
    out = {}
    root = Path(root)
    for branch in [
        "flair_t1guided_model_z7",
        "flair_t1guided_topology_model_z7",
        "flair_t1guided_latent_model_z7",
        "flair_t1guided_our_model_z7",
    ]:
        rows = []
        for path in root.glob(f"*/{branch}/t1guided_meta.json"):
            with open(path, "r", encoding="utf-8") as handle:
                rows.append(json.load(handle))
        branch_out = {"n": len(rows)}
        for key in [
            "lpvi_added",
            "persist_lambda",
            "persist_valid_slice_fraction",
            "coverage_ema_mean",
            "coverage_ema_lowfrac",
            "iterations",
        ]:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            branch_out[key] = summarize(vals)
        branch_out["persist_has_topologylayer"] = {
            str(val): sum(1 for row in rows if row.get("persist_has_topologylayer") == val)
            for val in [False, True]
        }
        branch_out["appearance_mode"] = {}
        for row in rows:
            mode = str(row.get("appearance_mode"))
            branch_out["appearance_mode"][mode] = branch_out["appearance_mode"].get(mode, 0) + 1
        out[branch] = branch_out
    return out


def write_csv(summary, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "method_or_branch", "metric", "n", "mean", "median", "min", "max"])
        for method, metrics in summary["quality"]["means"].items():
            for metric, vals in metrics.items():
                writer.writerow(["quality_mean", method, metric, vals["n"], vals["mean"], vals["median"], vals["min"], vals["max"]])
        for method, vals in summary["point_counts"].items():
            writer.writerow(["point_count", method, "vertices", vals["n"], vals["mean"], vals["median"], vals["min"], vals["max"]])
        for branch, branch_data in summary["metadata"].items():
            for metric, vals in branch_data.items():
                if isinstance(vals, dict) and {"n", "mean", "median", "min", "max"}.issubset(vals):
                    writer.writerow(["metadata", branch, metric, vals["n"], vals["mean"], vals["median"], vals["min"], vals["max"]])


def main():
    parser = argparse.ArgumentParser(description="Summarize method contribution metrics, Gaussian counts, and metadata.")
    parser.add_argument("--root", required=True, help="Output root containing one directory per subject.")
    parser.add_argument("--metrics-name", default="metrics_z7.json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--base-method", default="medgs_t1guided")
    args = parser.parse_args()

    rows = load_metrics(args.root, args.metrics_name)
    summary = {
        "root": str(Path(args.root)),
        "metrics_name": args.metrics_name,
        "num_metric_files": len(rows),
        "methods": METHODS,
        "base_method": args.base_method,
        "quality": quality_summary(rows, METHODS, args.base_method),
        "point_counts": point_count_summary(args.root),
        "metadata": meta_summary(args.root),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.out_csv:
        write_csv(summary, args.out_csv)


if __name__ == "__main__":
    main()
