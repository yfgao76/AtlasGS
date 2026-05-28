#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _t_critical(confidence: float, dof: int) -> float:
    if dof <= 0:
        return 0.0
    try:
        from scipy.stats import t as t_dist  # type: ignore

        return float(t_dist.ppf((1.0 + confidence) * 0.5, dof))
    except Exception:
        # Normal approximation fallback if scipy is unavailable.
        if abs(confidence - 0.95) < 1e-9:
            return 1.959963984540054
        return 1.959963984540054


def collect_metrics(metrics_files: List[Path]) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    values: Dict[str, Dict[str, List[float]]] = {}
    valid_files = 0
    for fp in metrics_files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        valid_files += 1
        for method, metric_dict in data.items():
            if not isinstance(metric_dict, dict):
                continue
            method_bucket = values.setdefault(method, {})
            for metric_name, v in metric_dict.items():
                if not _is_number(v):
                    continue
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                method_bucket.setdefault(metric_name, []).append(fv)
    return values, valid_files


def summarize(values: Dict[str, Dict[str, List[float]]], confidence: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in sorted(values.keys()):
        out[method] = {}
        for metric in sorted(values[method].keys()):
            arr = np.asarray(values[method][metric], dtype=np.float64)
            n = int(arr.size)
            if n == 0:
                continue
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            se = float(std / math.sqrt(n)) if n > 0 else float("nan")
            tcrit = _t_critical(confidence, n - 1)
            half = float(tcrit * se) if n > 1 else 0.0
            out[method][metric] = {
                "n": n,
                "mean": mean,
                "std": std,
                "se": se,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "ci_half_width": half,
            }
    return out


def write_csv(summary: Dict[str, Dict[str, Dict[str, float]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "n", "mean", "std", "se", "ci_low", "ci_high", "ci_half_width"])
        for method in sorted(summary.keys()):
            for metric in sorted(summary[method].keys()):
                s = summary[method][metric]
                writer.writerow(
                    [
                        method,
                        metric,
                        int(s["n"]),
                        f"{s['mean']:.10g}",
                        f"{s['std']:.10g}",
                        f"{s['se']:.10g}",
                        f"{s['ci_low']:.10g}",
                        f"{s['ci_high']:.10g}",
                        f"{s['ci_half_width']:.10g}",
                    ]
                )


def print_table(summary: Dict[str, Dict[str, Dict[str, float]]], confidence: float) -> None:
    ci_label = f"{int(round(confidence * 100))}%CI"
    print(f"{'Method':<28} {'Metric':<16} {'n':>4} {'Mean':>12} {ci_label:>22}")
    print("-" * 90)
    for method in sorted(summary.keys()):
        for metric in sorted(summary[method].keys()):
            s = summary[method][metric]
            print(
                f"{method:<28} {metric:<16} {int(s['n']):>4d} "
                f"{s['mean']:>12.6f} "
                f"[{s['ci_low']:>10.6f}, {s['ci_high']:>10.6f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-subject metrics into mean and confidence intervals.")
    parser.add_argument("--root", required=True, type=Path, help="Directory containing subject folders with metrics json files.")
    parser.add_argument(
        "--glob",
        default="*/metrics*.json",
        help="Glob relative to --root used to find metrics files (default: */metrics*.json).",
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI (default: 0.95).")
    parser.add_argument("--out-json", type=Path, default=None, help="Output json summary path.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output csv summary path.")
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(root.glob(args.glob))
    if not files:
        raise SystemExit(f"No metrics files found under {root} with glob '{args.glob}'.")

    values, n_valid = collect_metrics(files)
    summary = summarize(values, args.confidence)

    out_json = args.out_json or (root / "metrics_mean_ci95.json")
    out_csv = args.out_csv or (root / "metrics_mean_ci95.csv")

    payload = {
        "root": str(root),
        "glob": args.glob,
        "confidence": args.confidence,
        "num_files_found": len(files),
        "num_files_loaded": n_valid,
        "summary": summary,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    write_csv(summary, out_csv)

    print(f"Loaded {n_valid}/{len(files)} metrics files from {root}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print()
    print_table(summary, args.confidence)


if __name__ == "__main__":
    main()
