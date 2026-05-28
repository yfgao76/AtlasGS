import argparse
import json
from pathlib import Path

import numpy as np

from atlasgs.ops.metrics import summarize_metrics
from atlasgs.ops.nifti_io import load_nii, save_json


def load_mask(mask_path, t1, flair_gt):
    if mask_path:
        mask, _, _ = load_nii(mask_path)
        if mask.shape == flair_gt.shape:
            return (mask > 0).astype(np.uint8)
    base = t1 if (t1 is not None and t1.shape == flair_gt.shape) else flair_gt
    return (base > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR FLAIR outputs.")
    parser.add_argument("--t1", required=True)
    parser.add_argument("--flair-gt", required=True)
    parser.add_argument("--flair-interp", required=True)
    parser.add_argument("--flair-medgs-single", required=True)
    parser.add_argument("--flair-medgs-t1guided", required=True)
    parser.add_argument("--flair-medgs-t1guided-topology", default=None)
    parser.add_argument("--flair-medgs-t1guided-latent", default=None)
    parser.add_argument("--flair-medgs-our", default=None)
    parser.add_argument("--flair-alpine-a2", default=None)
    parser.add_argument("--flair-medgs-fused", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t1, _, _ = load_nii(args.t1)
    flair_gt, _, _ = load_nii(args.flair_gt)
    flair_interp, _, _ = load_nii(args.flair_interp)
    flair_single, _, _ = load_nii(args.flair_medgs_single)
    flair_t1g, _, _ = load_nii(args.flair_medgs_t1guided)
    flair_t1g_topology = None
    if args.flair_medgs_t1guided_topology:
        flair_t1g_topology, _, _ = load_nii(args.flair_medgs_t1guided_topology)
    flair_t1g_latent = None
    if args.flair_medgs_t1guided_latent:
        flair_t1g_latent, _, _ = load_nii(args.flair_medgs_t1guided_latent)
    flair_our = None
    if args.flair_medgs_our:
        flair_our, _, _ = load_nii(args.flair_medgs_our)
    flair_alpine_a2 = None
    if args.flair_alpine_a2:
        flair_alpine_a2, _, _ = load_nii(args.flair_alpine_a2)
    flair_fused = None
    if args.flair_medgs_fused:
        flair_fused, _, _ = load_nii(args.flair_medgs_fused)

    gt_shape = flair_gt.shape
    if flair_interp.shape != gt_shape:
        raise ValueError(f"Interp shape {flair_interp.shape} != GT shape {gt_shape}")
    if flair_single.shape != gt_shape:
        raise ValueError(f"Single model shape {flair_single.shape} != GT shape {gt_shape}")
    if flair_t1g.shape != gt_shape:
        raise ValueError(f"T1-guided shape {flair_t1g.shape} != GT shape {gt_shape}")
    if flair_t1g_topology is not None and flair_t1g_topology.shape != gt_shape:
        raise ValueError(f"T1-guided topology shape {flair_t1g_topology.shape} != GT shape {gt_shape}")
    if flair_t1g_latent is not None and flair_t1g_latent.shape != gt_shape:
        raise ValueError(f"T1-guided latent shape {flair_t1g_latent.shape} != GT shape {gt_shape}")
    if flair_our is not None and flair_our.shape != gt_shape:
        raise ValueError(f"Our method shape {flair_our.shape} != GT shape {gt_shape}")
    if flair_alpine_a2 is not None and flair_alpine_a2.shape != gt_shape:
        raise ValueError(f"ALPINE A2 shape {flair_alpine_a2.shape} != GT shape {gt_shape}")
    if flair_fused is not None and flair_fused.shape != gt_shape:
        raise ValueError(f"Fused shape {flair_fused.shape} != GT shape {gt_shape}")

    mask = load_mask(args.mask, t1, flair_gt)

    edge_ref = t1 if t1.shape == gt_shape else flair_gt
    out = {
        "interp": summarize_metrics(flair_interp, flair_gt, edge_ref, mask),
        "medgs_single": summarize_metrics(flair_single, flair_gt, edge_ref, mask),
        "medgs_t1guided": summarize_metrics(flair_t1g, flair_gt, edge_ref, mask),
    }
    if flair_t1g_topology is not None:
        out["medgs_t1guided_topology"] = summarize_metrics(flair_t1g_topology, flair_gt, edge_ref, mask)
    if flair_t1g_latent is not None:
        out["medgs_t1guided_latent"] = summarize_metrics(flair_t1g_latent, flair_gt, edge_ref, mask)
    if flair_our is not None:
        out["medgs_our"] = summarize_metrics(flair_our, flair_gt, edge_ref, mask)
    if flair_alpine_a2 is not None:
        out["alpine_a2"] = summarize_metrics(flair_alpine_a2, flair_gt, edge_ref, mask)
    if flair_fused is not None:
        out["medgs_fused_lrcons"] = summarize_metrics(flair_fused, flair_gt, edge_ref, mask)
    save_json(Path(args.out), out)


if __name__ == "__main__":
    main()
