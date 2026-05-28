import argparse
import subprocess
from pathlib import Path

import numpy as np


def run(cmd, cwd=None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_cfg_args(model_path, source_path):
    cfg_path = Path(model_path) / "cfg_args"
    if cfg_path.exists():
        return
    cfg = argparse.Namespace(
        sh_degree=0,
        source_path=str(Path(source_path).resolve()),
        model_path=str(Path(model_path).resolve()),
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
    cfg_path.write_text(str(cfg), encoding="utf-8")


def ensure_lr(gt_path, lr_path, factor_z, sigma_z, mode):
    if lr_path.exists():
        return
    run(
        [
            "python",
            "-m",
            "atlasgs.ops.degrade",
            "--flair_gt",
            str(gt_path),
            "--out",
            str(lr_path),
            "--factor_z",
            str(factor_z),
            "--sigma_z",
            str(sigma_z),
            "--mode",
            mode,
        ]
    )


def count_frames(frames_dir):
    frames_dir = Path(frames_dir)
    if (frames_dir / "original").is_dir():
        frames_dir = frames_dir / "original"
    return len(list(frames_dir.glob("*.png")))


def z_tag(factor_z):
    return f"z{int(factor_z)}"


def verify_render_series(render_dir, num_views, interp, model_name):
    render_dir = Path(render_dir)
    files = sorted(render_dir.glob("*.png"))
    expected = int(num_views) * int(interp)
    if len(files) != expected:
        raise RuntimeError(
            f"{model_name}: rendered {len(files)} frames, expected {expected} "
            f"(views={num_views}, interp={interp}) in {render_dir}"
        )

    by_view = {}
    for frame in files:
        stem = frame.stem
        if "_" not in stem:
            raise RuntimeError(f"{model_name}: frame missing interp suffix: {frame.name}")
        view_token, interp_token = stem.rsplit("_", 1)
        try:
            view_idx = int(view_token)
            interp_idx = int(interp_token)
        except ValueError as exc:
            raise RuntimeError(f"{model_name}: cannot parse frame name: {frame.name}") from exc
        by_view.setdefault(view_idx, set()).add(interp_idx)

    if len(by_view) != int(num_views):
        raise RuntimeError(f"{model_name}: got {len(by_view)} views, expected {num_views}")

    expected_interp = set(range(int(interp)))
    for view_idx in range(int(num_views)):
        got_interp = by_view.get(view_idx, set())
        if got_interp != expected_interp:
            raise RuntimeError(
                f"{model_name}: view {view_idx} interp mismatch, "
                f"got {sorted(got_interp)} expected {sorted(expected_interp)}"
            )


def parse_target_modalities(text):
    targets = []
    seen = set()
    for token in text.split(","):
        raw = token.strip().lower()
        if not raw:
            continue
        if raw in {"flair", "t2flair"}:
            name = "flair"
        elif raw in {"t2", "t2w"}:
            name = "t2"
        elif raw in {"asl"}:
            name = "asl"
        elif raw in {"dwi", "dwi_b0", "dwi_b900", "dwi_b2000"}:
            name = "dwi"
        else:
            raise ValueError(f"Unknown target modality: {raw}. Use one of: flair,t2,asl,dwi")
        if name not in seen:
            targets.append(name)
            seen.add(name)
    if not targets:
        raise ValueError("No valid target modalities provided.")
    return targets


MODALITY_FILE_STEMS = {
    "t1": ["t1", "t1w"],
    "t2": ["t2", "t2w"],
    "flair": ["flair", "t2flair"],
    "asl": ["asl"],
    "dwi": ["dwi", "dwi_b900", "dwi_b0", "dwi_b2000"],
}


def resolve_modality_gt(subject_dir, modality):
    stems = MODALITY_FILE_STEMS.get(modality, [modality])
    for stem in stems:
        p = subject_dir / f"{stem}_gt.nii.gz"
        if p.exists():
            return p
    return None


def resolve_modality_gt_t1ref(subject_dir, modality):
    stems = MODALITY_FILE_STEMS.get(modality, [modality])
    for stem in stems:
        p = subject_dir / f"{stem}_gt_t1ref.nii.gz"
        if p.exists():
            return p
    return None


def resolve_modality_lr(subject_dir, modality, factor_z):
    stems = MODALITY_FILE_STEMS.get(modality, [modality])
    factor = int(factor_z)
    for stem in stems:
        p = subject_dir / f"{stem}_lr_1x1x{factor}.nii.gz"
        if p.exists():
            return p
    return None


def frames_to_nifti_cmd(frames_dir, ref_path, norm_json, out_path, target_name, allow_resample=True):
    cmd = [
        "python",
        "-m",
        "atlasgs.ops.frames_to_nifti",
        "--frames",
        str(frames_dir),
        "--ref",
        str(ref_path),
        "--norm-json",
        str(norm_json),
        "--out",
        str(out_path),
    ]
    if allow_resample:
        cmd.append("--allow-resample")
    if target_name in {"dwi", "asl"}:
        cmd.append("--match-z-only")
    return cmd


def extend_t1guided_common_flags(cmd, args, time_init_scale=None, time_init_shift=None):
    cmd.extend(["--factor-z", str(args.factor_z)])
    if args.t1guided_time_lr > 0:
        cmd.extend(["--time-lr", str(args.t1guided_time_lr)])
    if time_init_scale is not None:
        cmd.append(f"--time-init-scale={float(time_init_scale)}")
    if time_init_shift is not None:
        cmd.append(f"--time-init-shift={float(time_init_shift)}")

    if args.t1guided_no_slab_forward:
        cmd.append("--no-slab-forward")
    if args.t1guided_slab_samples > 0:
        cmd.extend(["--slab-samples", str(args.t1guided_slab_samples)])

    if args.t1guided_use_interp_loss:
        cmd.append("--use-interp-loss")
    if args.t1guided_interp_weight != 0.5:
        cmd.extend(["--interp-weight", str(args.t1guided_interp_weight)])

    if args.t1guided_views_per_iter != 1:
        cmd.extend(["--views-per-iter", str(args.t1guided_views_per_iter)])
    if args.t1guided_coverage_ema_decay != 0.995:
        cmd.extend(["--coverage-ema-decay", str(args.t1guided_coverage_ema_decay)])
    if args.t1guided_coverage_low_thresh != 0.02:
        cmd.extend(["--coverage-low-thresh", str(args.t1guided_coverage_low_thresh)])
    if args.t1guided_coverage_warmup_iters != 1000:
        cmd.extend(["--coverage-warmup-iters", str(args.t1guided_coverage_warmup_iters)])

    if args.t1guided_lambda_coverage_opacity > 0:
        cmd.extend(["--lambda-coverage-opacity", str(args.t1guided_lambda_coverage_opacity)])
    if args.t1guided_lambda_coverage_feature > 0:
        cmd.extend(["--lambda-coverage-feature", str(args.t1guided_lambda_coverage_feature)])
    if args.t1guided_lambda_coverage_xyz_anchor > 0:
        cmd.extend(["--lambda-coverage-xyz-anchor", str(args.t1guided_lambda_coverage_xyz_anchor)])
    if args.t1guided_lambda_coverage_cov_anchor > 0:
        cmd.extend(["--lambda-coverage-cov-anchor", str(args.t1guided_lambda_coverage_cov_anchor)])


def compute_temporal_init_linear_map(source_nii, target_nii):
    import nibabel as nib

    def geom(path):
        img = nib.load(str(path))
        shape = np.asarray(img.shape[:3], dtype=np.float64)
        aff = np.asarray(img.affine, dtype=np.float64)
        cx = (shape[0] - 1.0) * 0.5
        cy = (shape[1] - 1.0) * 0.5
        p0 = nib.affines.apply_affine(aff, np.asarray([cx, cy, 0.0], dtype=np.float64))
        p1 = nib.affines.apply_affine(aff, np.asarray([cx, cy, max(shape[2] - 1.0, 0.0)], dtype=np.float64))
        vec = p1 - p0
        span = float(np.linalg.norm(vec))
        if span < 1e-6:
            zooms = img.header.get_zooms()
            z_mm = float(zooms[2]) if len(zooms) >= 3 else 1.0
            span = max(z_mm * max(float(shape[2] - 1.0), 1.0), 1e-6)
            direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = vec / span
        center = 0.5 * (p0 + p1)
        return span, direction, center

    src_span, src_dir, src_center = geom(source_nii)
    tgt_span, tgt_dir, tgt_center = geom(target_nii)
    align = float(np.dot(src_dir, tgt_dir))
    if abs(align) < 1e-4:
        align = 1.0
    scale = (src_span * align) / max(tgt_span, 1e-6)
    shift = float(np.dot(src_center - tgt_center, tgt_dir)) / max(tgt_span, 1e-6)
    shift = shift + 0.5 * (1.0 - scale)
    return float(scale), float(shift), {
        "src_span_mm": float(src_span),
        "tgt_span_mm": float(tgt_span),
        "axis_align": float(align),
    }


def prepare_target_on_t1_grid(target, target_gt, t1_gt, prep_root, factor_z, sigma_z, mode, overwrite=False, resample_order=1):
    prep_root.mkdir(parents=True, exist_ok=True)
    target_gt_t1 = prep_root / f"{target}_gt_t1grid.nii.gz"
    target_lr_t1 = prep_root / f"{target}_lr_1x1x{int(factor_z)}.nii.gz"

    if overwrite or not target_gt_t1.exists():
        run(
            [
                "python",
                "-m",
                "atlasgs.ops.resample",
                "--moving",
                str(target_gt),
                "--ref",
                str(t1_gt),
                "--out",
                str(target_gt_t1),
                "--order",
                str(int(resample_order)),
            ]
        )

    if overwrite and target_lr_t1.exists():
        target_lr_t1.unlink()
    ensure_lr(target_gt_t1, target_lr_t1, factor_z, sigma_z, mode)
    return target_gt_t1, target_lr_t1


def apply_binary_mask(src_path, mask_path, out_path):
    from atlasgs.ops.nifti_io import load_nii, save_nii

    vol, aff, hdr = load_nii(src_path)
    mask, _, _ = load_nii(mask_path)
    if vol.shape != mask.shape:
        raise ValueError(
            f"Mask/volume shape mismatch: vol={vol.shape}, mask={mask.shape}, "
            f"vol_path={src_path}, mask_path={mask_path}"
        )
    masked = (vol * (mask > 0.5).astype(np.float32)).astype(np.float32)
    save_nii(out_path, masked, aff, header=hdr)


def add_single_prior_flags(cmd, args, single_model):
    if args.t1guided_lambda_single_prior <= 0:
        return
    cmd.extend(
        [
            "--single-model",
            str(single_model),
            "--single-iter",
            "latest",
            "--lambda-single-prior",
            str(args.t1guided_lambda_single_prior),
            "--single-prior-tau",
            str(args.t1guided_single_prior_tau),
            "--single-prior-wmin",
            str(args.t1guided_single_prior_wmin),
            "--single-prior-wmax",
            str(args.t1guided_single_prior_wmax),
            "--single-prior-start-iter",
            str(args.t1guided_single_prior_start_iter),
        ]
    )


def add_t1guided_regularization_flags(cmd, args, force_no_freeze_time):
    if not args.no_t1guided_train_opacity:
        cmd.append("--train-opacity")
    if args.t1guided_feature_l2 > 0:
        cmd.extend(["--feature-l2", str(args.t1guided_feature_l2)])
    if args.t1guided_lambda_boundary > 0:
        cmd.extend(["--lambda-boundary", str(args.t1guided_lambda_boundary)])
    if args.t1guided_lambda_gradalign > 0:
        cmd.extend(["--lambda-gradalign", str(args.t1guided_lambda_gradalign)])
    if args.t1guided_lambda_normal > 0:
        cmd.extend(["--lambda-normal", str(args.t1guided_lambda_normal)])
    if args.t1guided_normal_k != 8:
        cmd.extend(["--normal-k", str(args.t1guided_normal_k)])
    if args.t1guided_normal_sigma != 2.0:
        cmd.extend(["--normal-sigma", str(args.t1guided_normal_sigma)])
    if args.t1guided_normal_sample != 2000:
        cmd.extend(["--normal-sample", str(args.t1guided_normal_sample)])
    if args.t1guided_cov_lr > 0:
        cmd.extend(["--cov-lr", str(args.t1guided_cov_lr)])
    if args.t1guided_opacity_init is not None:
        cmd.extend(["--opacity-init", str(args.t1guided_opacity_init)])
    if args.t1guided_opacity_scale is not None:
        cmd.extend(["--opacity-scale", str(args.t1guided_opacity_scale)])
    if args.t1guided_reset_appearance:
        cmd.append("--reset-appearance")
    if args.t1guided_boundary_p != 95.0:
        cmd.extend(["--boundary-p", str(args.t1guided_boundary_p)])
    if args.t1guided_boundary_gamma != 2.0:
        cmd.extend(["--boundary-gamma", str(args.t1guided_boundary_gamma)])
    if args.t1guided_no_freeze_time or force_no_freeze_time:
        cmd.append("--no-freeze-time")


def add_topology_flags(cmd, args):
    cmd.extend(
        [
            "--lpvi-enable",
            "--lpvi-anchor-samples",
            str(args.t1guided_topology_lpvi_anchor_samples),
            "--lpvi-k-max",
            str(args.t1guided_topology_lpvi_k_max),
            "--lpvi-k-min",
            str(args.t1guided_topology_lpvi_k_min),
            "--lpvi-threshold",
            str(args.t1guided_topology_lpvi_threshold),
            "--lpvi-max-new-points",
            str(args.t1guided_topology_lpvi_max_new_points),
            "--lpvi-min-opacity",
            str(args.t1guided_topology_lpvi_min_opacity),
            "--lpvi-max-vertices-per-anchor",
            str(args.t1guided_topology_lpvi_max_vertices_per_anchor),
            "--lpvi-scale-shrink",
            str(args.t1guided_topology_lpvi_scale_shrink),
            "--lpvi-dist-model",
            str(args.t1guided_topology_lpvi_dist_model),
            "--lpvi-complex-max-edge",
            str(args.t1guided_topology_lpvi_complex_max_edge),
            "--persist-lambda",
            str(args.t1guided_topology_persist_lambda),
            "--persist-dims",
            str(args.t1guided_topology_persist_dims),
            "--persist-ks",
            str(args.t1guided_topology_persist_ks),
            "--persist-downsample",
            str(args.t1guided_topology_persist_downsample),
            "--persist-spatial-weight",
            str(args.t1guided_topology_persist_spatial_weight),
            "--persist-intensity-weight",
            str(args.t1guided_topology_persist_intensity_weight),
            "--persist-threshold-count",
            str(args.t1guided_topology_persist_threshold_count),
            "--persist-sigmoid-tau",
            str(args.t1guided_topology_persist_sigmoid_tau),
            "--persist-fg-eps",
            str(args.t1guided_topology_persist_fg_eps),
            "--persist-min-fg-ratio",
            str(args.t1guided_topology_persist_min_fg_ratio),
            "--persist-min-fg-points",
            str(args.t1guided_topology_persist_min_fg_points),
            "--persist-min-std",
            str(args.t1guided_topology_persist_min_std),
        ]
    )
    if args.t1guided_topology_preprune_keep_ratio < 1.0:
        cmd.extend(["--preprune-keep-ratio", str(args.t1guided_topology_preprune_keep_ratio)])
    if args.t1guided_topology_preprune_min_opacity > 0:
        cmd.extend(["--preprune-min-opacity", str(args.t1guided_topology_preprune_min_opacity)])
    if args.t1guided_topology_preprune_min_keep != 50000:
        cmd.extend(["--preprune-min-keep", str(args.t1guided_topology_preprune_min_keep)])
    if args.t1guided_topology_no_freeze_xyz:
        cmd.append("--no-freeze-xyz")
    if args.t1guided_topology_no_freeze_cov:
        cmd.append("--no-freeze-cov")
    if args.t1guided_topology_xyz_lr > 0:
        cmd.extend(["--xyz-lr", str(args.t1guided_topology_xyz_lr)])
    if args.t1guided_topology_cov_lr > 0:
        cmd.extend(["--cov-lr", str(args.t1guided_topology_cov_lr)])
    if args.t1guided_topology_lambda_xyz_anchor > 0:
        cmd.extend(["--lambda-xyz-anchor", str(args.t1guided_topology_lambda_xyz_anchor)])
    if args.t1guided_topology_lambda_xyz_zero_mean > 0:
        cmd.extend(["--lambda-xyz-zero-mean", str(args.t1guided_topology_lambda_xyz_zero_mean)])
    if args.t1guided_topology_lambda_xyz_max > 0:
        cmd.extend(["--lambda-xyz-max", str(args.t1guided_topology_lambda_xyz_max)])
    if args.t1guided_topology_xyz_max_mm > 0:
        cmd.extend(["--xyz-max-mm", str(args.t1guided_topology_xyz_max_mm)])
    if args.t1guided_topology_xyz_delta_clamp_mm > 0:
        cmd.extend(["--xyz-delta-clamp-mm", str(args.t1guided_topology_xyz_delta_clamp_mm)])
    if args.t1guided_topology_grad_clip > 0:
        cmd.extend(["--grad-clip", str(args.t1guided_topology_grad_clip)])
    if args.t1guided_topology_persist_skip_lowinfo:
        cmd.append("--persist-skip-lowinfo")
    else:
        cmd.append("--persist-keep-lowinfo")
    if args.t1guided_topology_lpvi_no_topology_check:
        cmd.append("--lpvi-no-topology-check")


def train_t1guided_branch(
    args,
    t1_guidance_model,
    t1_gt,
    target_lr_frames,
    out_model,
    iterations,
    lambda_dssim,
    seed,
    single_model=None,
    force_no_freeze_time=False,
    time_init_scale=None,
    time_init_shift=None,
    appearance_mode=None,
    add_topology=False,
    interp_lambda_dssim=None,
):
    cmd = [
        "python",
        "-m",
        "atlasgs.train.train_flair_t1guided_medgs",
        "--medgs-root",
        str(args.medgs_root),
        "--t1-model",
        str(t1_guidance_model),
        "--t1-iter",
        "latest",
        "--t1",
        str(t1_gt),
        "--flair-dataset",
        str(target_lr_frames),
        "--out",
        str(out_model),
        "--iterations",
        str(iterations),
        "--poly-degree",
        str(args.poly_degree),
        "--sh-degree",
        str(args.sh_degree),
        "--seed",
        str(seed),
        "--lambda-dssim",
        str(lambda_dssim),
    ]

    if appearance_mode == "latent":
        cmd.extend(
            [
                "--appearance-mode",
                "latent",
                "--appearance-latent-dim",
                str(args.t1guided_latent_dim),
                "--appearance-latent-hidden",
                str(args.t1guided_latent_hidden),
                "--appearance-lambda-smooth",
                str(args.t1guided_latent_lambda_smooth),
                "--appearance-lambda-theta-l2",
                str(args.t1guided_latent_lambda_theta_l2),
                "--appearance-lambda-head-l2",
                str(args.t1guided_latent_lambda_head_l2),
                "--appearance-smooth-k",
                str(args.t1guided_latent_smooth_k),
                "--appearance-smooth-sigma",
                str(args.t1guided_latent_smooth_sigma),
                "--appearance-smooth-sample",
                str(args.t1guided_latent_smooth_sample),
            ]
        )

    if add_topology:
        add_topology_flags(cmd, args)
        if interp_lambda_dssim is not None:
            cmd.extend(["--interp-lambda-dssim", str(interp_lambda_dssim)])

    if single_model is not None:
        add_single_prior_flags(cmd, args, single_model)

    add_t1guided_regularization_flags(cmd, args, force_no_freeze_time=force_no_freeze_time)
    extend_t1guided_common_flags(cmd, args, time_init_scale=time_init_scale, time_init_shift=time_init_shift)
    run(cmd)


def render_model(args, model_dir, source_frames):
    render_dir = Path(model_dir) / "render"
    if args.overwrite or not render_dir.exists():
        ensure_cfg_args(model_dir, source_frames)
        run(
            [
                "python",
                str(Path(args.medgs_root) / "render.py"),
                "--model_path",
                str(model_dir),
                "--interp",
                str(args.interp_factor),
                "--pipeline",
                "img",
            ]
        )
    return render_dir


def convert_render_to_nifti(args, render_dir, target_gt, norm_json, out_path, target_name):
    if args.overwrite or not out_path.exists():
        run(
            frames_to_nifti_cmd(
                frames_dir=render_dir,
                ref_path=target_gt,
                norm_json=norm_json,
                out_path=out_path,
                target_name=target_name,
                allow_resample=True,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Run MedGS experiments for one subject (clean methods only).")
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--medgs-root", required=True)

    parser.add_argument("--interp-factor", type=int, default=3)
    parser.add_argument("--factor-z", type=int, default=3)
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="avgpool")

    parser.add_argument("--apply-target-brain-mask", action="store_true")
    parser.add_argument("--target-modalities", type=str, default="flair", help="Comma-separated among {flair,t2,asl,dwi}.")

    parser.add_argument("--train-iterations", type=int, default=30000)
    parser.add_argument("--t1guided-iterations", type=int, default=10000)

    parser.add_argument("--resample-targets-to-t1", action="store_true")
    parser.add_argument("--resample-order", type=int, default=1)

    parser.add_argument("--poly-degree", type=int, default=7)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--t1guided-no-freeze-time", action="store_true")
    parser.add_argument("--retry-t1guided", action="store_true")
    parser.add_argument("--t1guided-feature-l2", type=float, default=0.0)
    parser.add_argument("--t1guided-lambda-dssim", type=float, default=0.2)
    parser.add_argument("--no-t1guided-train-opacity", action="store_true")
    parser.add_argument("--t1guided-lambda-boundary", type=float, default=0.0)
    parser.add_argument("--t1guided-lambda-gradalign", type=float, default=0.0)
    parser.add_argument("--t1guided-lambda-normal", type=float, default=0.0)
    parser.add_argument("--t1guided-normal-k", type=int, default=8)
    parser.add_argument("--t1guided-normal-sigma", type=float, default=2.0)
    parser.add_argument("--t1guided-normal-sample", type=int, default=2000)
    parser.add_argument("--t1guided-cov-lr", type=float, default=0.0)
    parser.add_argument("--t1guided-time-lr", type=float, default=5e-4)
    parser.add_argument("--t1guided-time-init-auto", action="store_true")
    parser.add_argument("--t1guided-opacity-init", type=float, default=None)
    parser.add_argument("--t1guided-opacity-scale", type=float, default=None)
    parser.add_argument("--t1guided-reset-appearance", action="store_true")
    parser.add_argument("--t1guided-boundary-p", type=float, default=95.0)
    parser.add_argument("--t1guided-boundary-gamma", type=float, default=2.0)

    parser.add_argument("--t1guided-lambda-single-prior", type=float, default=0.0)
    parser.add_argument("--t1guided-single-prior-tau", type=float, default=8.0)
    parser.add_argument("--t1guided-single-prior-wmin", type=float, default=0.1)
    parser.add_argument("--t1guided-single-prior-wmax", type=float, default=0.9)
    parser.add_argument("--t1guided-single-prior-start-iter", type=int, default=0)

    parser.add_argument("--t1guided-views-per-iter", type=int, default=1)
    parser.add_argument("--t1guided-no-slab-forward", action="store_true")
    parser.add_argument("--t1guided-slab-samples", type=int, default=0)
    parser.add_argument("--t1guided-use-interp-loss", action="store_true")
    parser.add_argument("--t1guided-interp-weight", type=float, default=0.5)

    parser.add_argument("--t1guided-coverage-ema-decay", type=float, default=0.995)
    parser.add_argument("--t1guided-coverage-low-thresh", type=float, default=0.02)
    parser.add_argument("--t1guided-coverage-warmup-iters", type=int, default=1000)
    parser.add_argument("--t1guided-lambda-coverage-opacity", type=float, default=0.0)
    parser.add_argument("--t1guided-lambda-coverage-feature", type=float, default=0.0)
    parser.add_argument("--t1guided-lambda-coverage-xyz-anchor", type=float, default=0.0)
    parser.add_argument("--t1guided-lambda-coverage-cov-anchor", type=float, default=0.0)

    parser.add_argument("--run-t1guided-topology", action="store_true")
    parser.add_argument("--t1guided-topology-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-topology-lambda-dssim", type=float, default=0.2)
    parser.add_argument("--t1guided-topology-interp-lambda-dssim", type=float, default=-1.0)
    parser.add_argument("--t1guided-topology-persist-lambda", type=float, default=0.1)
    parser.add_argument("--t1guided-topology-persist-dims", type=str, default="0,1")
    parser.add_argument("--t1guided-topology-persist-ks", type=str, default="64,32")
    parser.add_argument("--t1guided-topology-persist-downsample", type=float, default=0.125)
    parser.add_argument("--t1guided-topology-persist-spatial-weight", type=float, default=0.35)
    parser.add_argument("--t1guided-topology-persist-intensity-weight", type=float, default=1.0)
    parser.add_argument("--t1guided-topology-persist-threshold-count", type=int, default=8)
    parser.add_argument("--t1guided-topology-persist-sigmoid-tau", type=float, default=0.05)
    parser.add_argument("--t1guided-topology-persist-fg-eps", type=float, default=0.02)
    parser.add_argument("--t1guided-topology-persist-min-fg-ratio", type=float, default=0.01)
    parser.add_argument("--t1guided-topology-persist-min-fg-points", type=int, default=8)
    parser.add_argument("--t1guided-topology-persist-min-std", type=float, default=0.01)
    parser.add_argument("--t1guided-topology-persist-skip-lowinfo", dest="t1guided_topology_persist_skip_lowinfo", action="store_true")
    parser.add_argument("--t1guided-topology-persist-keep-lowinfo", dest="t1guided_topology_persist_skip_lowinfo", action="store_false")
    parser.set_defaults(t1guided_topology_persist_skip_lowinfo=True)
    parser.add_argument("--t1guided-topology-lpvi-anchor-samples", type=int, default=2048)
    parser.add_argument("--t1guided-topology-lpvi-k-max", type=int, default=8)
    parser.add_argument("--t1guided-topology-lpvi-k-min", type=int, default=4)
    parser.add_argument("--t1guided-topology-lpvi-threshold", type=float, default=0.25)
    parser.add_argument("--t1guided-topology-lpvi-max-new-points", type=int, default=120000)
    parser.add_argument("--t1guided-topology-lpvi-min-opacity", type=float, default=0.01)
    parser.add_argument("--t1guided-topology-lpvi-max-vertices-per-anchor", type=int, default=4)
    parser.add_argument("--t1guided-topology-lpvi-scale-shrink", type=float, default=0.9)
    parser.add_argument("--t1guided-topology-lpvi-dist-model", type=str, default="W", choices=["W", "B"])
    parser.add_argument("--t1guided-topology-lpvi-complex-max-edge", type=float, default=2.0)
    parser.add_argument("--t1guided-topology-lpvi-no-topology-check", action="store_true")
    parser.add_argument("--t1guided-topology-preprune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--t1guided-topology-preprune-min-opacity", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-preprune-min-keep", type=int, default=50000)
    parser.add_argument("--t1guided-topology-no-freeze-xyz", action="store_true")
    parser.add_argument("--t1guided-topology-no-freeze-cov", action="store_true")
    parser.add_argument("--t1guided-topology-xyz-lr", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-cov-lr", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-lambda-xyz-anchor", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-lambda-xyz-zero-mean", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-lambda-xyz-max", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-xyz-max-mm", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-xyz-delta-clamp-mm", type=float, default=0.0)
    parser.add_argument("--t1guided-topology-grad-clip", type=float, default=0.0)

    parser.add_argument("--run-t1guided-latent", action="store_true")
    parser.add_argument("--t1guided-latent-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-latent-dim", type=int, default=4)
    parser.add_argument("--t1guided-latent-hidden", type=int, default=32)
    parser.add_argument("--t1guided-latent-lambda-smooth", type=float, default=0.02)
    parser.add_argument("--t1guided-latent-lambda-theta-l2", type=float, default=0.001)
    parser.add_argument("--t1guided-latent-lambda-head-l2", type=float, default=0.0001)
    parser.add_argument("--t1guided-latent-smooth-k", type=int, default=8)
    parser.add_argument("--t1guided-latent-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--t1guided-latent-smooth-sample", type=int, default=2000)

    parser.add_argument("--run-t1guided-our", action="store_true")
    parser.add_argument("--t1guided-our-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-our-lambda-dssim", type=float, default=0.2)
    parser.add_argument("--t1guided-refine-t1", action="store_true")
    parser.add_argument("--t1guided-refine-t1-iterations", type=int, default=10000)
    parser.add_argument("--run-t1guided-our-fusion", dest="run_t1guided_our_fusion", action="store_true")
    parser.add_argument("--no-run-t1guided-our-fusion", dest="run_t1guided_our_fusion", action="store_false")
    parser.set_defaults(run_t1guided_our_fusion=True)

    parser.add_argument("--run-lr-fusion", action="store_true")
    parser.add_argument("--fusion-tau", type=float, default=8.0)
    parser.add_argument("--fusion-wmin", type=float, default=0.1)
    parser.add_argument("--fusion-wmax", type=float, default=0.9)
    parser.add_argument("--fusion-smooth-xy", type=float, default=1.0)
    parser.add_argument("--fusion-smooth-z", type=float, default=0.5)

    parser.add_argument("--alpine-root", type=str, default=None)
    parser.add_argument("--run-alpine-a2", dest="run_alpine_a2", action="store_true")
    parser.add_argument("--no-run-alpine-a2", dest="run_alpine_a2", action="store_false")
    parser.set_defaults(run_alpine_a2=True)
    parser.add_argument("--alpine-a2-pretrain-iters", type=int, default=2000)
    parser.add_argument("--alpine-a2-fit-iters", type=int, default=4000)
    parser.add_argument("--alpine-a2-batch-size", type=int, default=32768)
    parser.add_argument("--alpine-a2-feature-dim", type=int, default=64)
    parser.add_argument("--alpine-a2-hidden-features", type=int, default=128)
    parser.add_argument("--alpine-a2-hidden-layers", type=int, default=5)
    parser.add_argument("--alpine-a2-trunk-lr", type=float, default=1e-4)
    parser.add_argument("--alpine-a2-head-lr", type=float, default=1e-3)
    parser.add_argument("--alpine-a2-lambda-pseudo", type=float, default=0.1)
    parser.add_argument("--alpine-a2-lambda-alpha-reg", type=float, default=1e-4)
    parser.add_argument("--alpine-a2-chunk-size", type=int, default=262144)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    tag = z_tag(args.factor_z)
    targets = parse_target_modalities(args.target_modalities)

    sub = data_root / args.subject_id
    if not sub.is_dir():
        raise FileNotFoundError(f"Subject folder not found: {sub}")

    t1_gt = resolve_modality_gt(sub, "t1")
    mask = sub / "mask_brain.nii.gz"
    if t1_gt is None:
        stems = ",".join(MODALITY_FILE_STEMS["t1"])
        raise FileNotFoundError(f"Missing T1 GT under {sub}. Tried stems: {stems} (expected *_gt.nii.gz)")

    sub_out = out_root / args.subject_id
    sub_out.mkdir(parents=True, exist_ok=True)
    alpine_root = Path(args.alpine_root) if args.alpine_root else (Path(args.medgs_root).resolve().parent / "alpine")

    t1_frames = sub_out / "t1_frames"
    if args.overwrite or not (t1_frames / "original").exists():
        run(
            [
                "python",
                "-m",
                "atlasgs.ops.nifti_to_frames",
                "--input",
                str(t1_gt),
                "--output",
                str(t1_frames),
                "--axis",
                "2",
                "--pmin",
                "1",
                "--pmax",
                "99",
                "--copy-ref",
            ]
        )
    t1_frame_count = count_frames(t1_frames)

    t1_model = sub_out / "t1_model"
    if args.overwrite or not (t1_model / "point_cloud").exists():
        run(
            [
                "python",
                str(Path(args.medgs_root) / "train.py"),
                "-s",
                str(t1_frames),
                "-m",
                str(t1_model),
                "--iterations",
                str(args.train_iterations),
                "--poly_degree",
                str(args.poly_degree),
                "--sh_degree",
                str(args.sh_degree),
            ]
        )

    t1_guidance_model = t1_model
    if args.t1guided_refine_t1:
        t1_guidance_model = sub_out / f"t1_refined_latent_topology_{tag}"
        if args.overwrite or not (t1_guidance_model / "point_cloud").exists():
            cmd = [
                "python",
                "-m",
                "atlasgs.train.train_flair_t1guided_medgs",
                "--medgs-root",
                str(args.medgs_root),
                "--t1-model",
                str(t1_model),
                "--t1-iter",
                "latest",
                "--t1",
                str(t1_gt),
                "--flair-dataset",
                str(t1_frames),
                "--out",
                str(t1_guidance_model),
                "--iterations",
                str(args.t1guided_refine_t1_iterations),
                "--poly-degree",
                str(args.poly_degree),
                "--sh-degree",
                str(args.sh_degree),
                "--seed",
                str(args.seed),
                "--lambda-dssim",
                str(args.t1guided_our_lambda_dssim),
                "--appearance-mode",
                "latent",
                "--appearance-latent-dim",
                str(args.t1guided_latent_dim),
                "--appearance-latent-hidden",
                str(args.t1guided_latent_hidden),
                "--appearance-lambda-smooth",
                str(args.t1guided_latent_lambda_smooth),
                "--appearance-lambda-theta-l2",
                str(args.t1guided_latent_lambda_theta_l2),
                "--appearance-lambda-head-l2",
                str(args.t1guided_latent_lambda_head_l2),
                "--appearance-smooth-k",
                str(args.t1guided_latent_smooth_k),
                "--appearance-smooth-sigma",
                str(args.t1guided_latent_smooth_sigma),
                "--appearance-smooth-sample",
                str(args.t1guided_latent_smooth_sample),
                "--interp-lambda-dssim",
                str(args.t1guided_topology_interp_lambda_dssim),
            ]
            add_topology_flags(cmd, args)
            if not args.no_t1guided_train_opacity:
                cmd.append("--train-opacity")
            if args.t1guided_feature_l2 > 0:
                cmd.extend(["--feature-l2", str(args.t1guided_feature_l2)])
            if args.t1guided_no_freeze_time:
                cmd.append("--no-freeze-time")
            if args.t1guided_reset_appearance:
                cmd.append("--reset-appearance")
            extend_t1guided_common_flags(cmd, args)
            run(cmd)

    processed_targets = 0
    for target in targets:
        alpine_a2_ready = False

        target_gt = resolve_modality_gt(sub, target)
        target_gt_t1ref = resolve_modality_gt_t1ref(sub, target)
        if target_gt is None:
            stems = ",".join(MODALITY_FILE_STEMS.get(target, [target]))
            print(f"[skip] Missing {target} GT for {args.subject_id}; tried stems: {stems}")
            continue

        if args.resample_targets_to_t1:
            prep_root = sub_out / "_t1grid_prepared"
            target_gt, target_lr = prepare_target_on_t1_grid(
                target=target,
                target_gt=target_gt,
                t1_gt=t1_gt,
                prep_root=prep_root,
                factor_z=args.factor_z,
                sigma_z=args.sigma_z,
                mode=args.mode,
                overwrite=args.overwrite,
                resample_order=args.resample_order,
            )
            target_gt_t1ref = target_gt
        else:
            target_lr = resolve_modality_lr(sub, target, args.factor_z)
            if target_lr is None:
                target_lr = sub / f"{target}_lr_1x1x{int(args.factor_z)}.nii.gz"
                ensure_lr(target_gt, target_lr, args.factor_z, args.sigma_z, args.mode)

        if args.apply_target_brain_mask and mask.exists():
            target_mask_gt = sub_out / f"{target}_mask_brain_gt_{tag}.nii.gz"
            target_mask_lr = sub_out / f"{target}_mask_brain_lr_{tag}.nii.gz"
            target_gt_masked = sub_out / f"{target}_gt_masked_{tag}.nii.gz"
            target_lr_masked = sub_out / f"{target}_lr_1x1x{int(args.factor_z)}_masked.nii.gz"

            if args.overwrite or not target_mask_gt.exists():
                run(
                    [
                        "python",
                        "-m",
                        "atlasgs.ops.resample",
                        "--moving",
                        str(mask),
                        "--ref",
                        str(target_gt),
                        "--out",
                        str(target_mask_gt),
                        "--order",
                        "0",
                    ]
                )
            if args.overwrite or not target_mask_lr.exists():
                run(
                    [
                        "python",
                        "-m",
                        "atlasgs.ops.resample",
                        "--moving",
                        str(mask),
                        "--ref",
                        str(target_lr),
                        "--out",
                        str(target_mask_lr),
                        "--order",
                        "0",
                    ]
                )
            if args.overwrite or not target_gt_masked.exists():
                apply_binary_mask(target_gt, target_mask_gt, target_gt_masked)
            if args.overwrite or not target_lr_masked.exists():
                apply_binary_mask(target_lr, target_mask_lr, target_lr_masked)
            target_gt = target_gt_masked
            target_lr = target_lr_masked

        if target_gt_t1ref is None and target in {"dwi", "asl"}:
            target_gt_t1ref = sub_out / f"{target}_gt_t1ref_{tag}.nii.gz"
            if args.overwrite or not target_gt_t1ref.exists():
                run(
                    [
                        "python",
                        "-m",
                        "atlasgs.ops.resample",
                        "--moving",
                        str(target_gt),
                        "--ref",
                        str(t1_gt),
                        "--out",
                        str(target_gt_t1ref),
                        "--order",
                        str(int(args.resample_order)),
                    ]
                )

        interp_out = sub_out / f"{target}_interp_{tag}.nii.gz"
        if args.overwrite or not interp_out.exists():
            run(
                [
                    "python",
                    "-m",
                    "atlasgs.ops.interp",
                    "--lr",
                    str(target_lr),
                    "--ref",
                    str(target_gt),
                    "--out",
                    str(interp_out),
                ]
            )

        target_lr_frames = sub_out / f"{target}_lr_frames_{tag}"
        if args.overwrite or not (target_lr_frames / "original").exists():
            run(
                [
                    "python",
                    "-m",
                    "atlasgs.ops.nifti_to_frames",
                    "--input",
                    str(target_lr),
                    "--output",
                    str(target_lr_frames),
                    "--axis",
                    "2",
                    "--pmin",
                    "1",
                    "--pmax",
                    "99",
                    "--copy-ref",
                ]
            )

        target_frame_count = count_frames(target_lr_frames)
        force_no_freeze_time = False
        if t1_frame_count != target_frame_count:
            print(f"Frame count mismatch ({target}): t1={t1_frame_count} lr={target_frame_count}")
            force_no_freeze_time = True

        time_init_scale = None
        time_init_shift = None
        if args.t1guided_time_init_auto:
            try:
                time_init_scale, time_init_shift, ti_dbg = compute_temporal_init_linear_map(t1_gt, target_lr)
                print(
                    f"[time-init:{target}] scale={time_init_scale:.6f} shift={time_init_shift:.6f} "
                    f"src_span_mm={ti_dbg['src_span_mm']:.3f} tgt_span_mm={ti_dbg['tgt_span_mm']:.3f} "
                    f"axis_align={ti_dbg['axis_align']:.5f}"
                )
            except Exception as exc:
                print(f"[time-init:{target}] failed, fallback to model default: {exc}")

        single_model = sub_out / f"{target}_lr_model_{tag}"
        if args.overwrite or not (single_model / "point_cloud").exists():
            run(
                [
                    "python",
                    str(Path(args.medgs_root) / "train.py"),
                    "-s",
                    str(target_lr_frames),
                    "-m",
                    str(single_model),
                    "--iterations",
                    str(args.train_iterations),
                    "--poly_degree",
                    str(args.poly_degree),
                    "--sh_degree",
                    str(args.sh_degree),
                ]
            )

        t1guided_model = sub_out / f"{target}_t1guided_model_{tag}"
        t1guided_topology_model = sub_out / f"{target}_t1guided_topology_model_{tag}"
        t1guided_latent_model = sub_out / f"{target}_t1guided_latent_model_{tag}"
        t1guided_our_model = sub_out / f"{target}_t1guided_our_model_{tag}"

        if args.overwrite or not (t1guided_model / "point_cloud").exists():
            cmd = [
                "python",
                "-m",
                "atlasgs.train.train_flair_t1guided_medgs",
                "--medgs-root",
                str(args.medgs_root),
                "--t1-model",
                str(t1_guidance_model),
                "--t1-iter",
                "latest",
                "--t1",
                str(t1_gt),
                "--flair-dataset",
                str(target_lr_frames),
                "--out",
                str(t1guided_model),
                "--iterations",
                str(args.t1guided_iterations),
                "--poly-degree",
                str(args.poly_degree),
                "--sh-degree",
                str(args.sh_degree),
                "--seed",
                str(args.seed),
                "--lambda-dssim",
                str(args.t1guided_lambda_dssim),
            ]
            add_single_prior_flags(cmd, args, single_model)
            add_t1guided_regularization_flags(cmd, args, force_no_freeze_time=force_no_freeze_time)
            extend_t1guided_common_flags(cmd, args, time_init_scale=time_init_scale, time_init_shift=time_init_shift)
            try:
                run(cmd)
            except subprocess.CalledProcessError as exc:
                if not args.retry_t1guided:
                    raise
                print(f"T1-guided training failed ({target}) with {exc}. Retrying after regenerating frames.")
                run(
                    [
                        "python",
                        "-m",
                        "atlasgs.ops.nifti_to_frames",
                        "--input",
                        str(t1_gt),
                        "--output",
                        str(t1_frames),
                        "--axis",
                        "2",
                        "--pmin",
                        "1",
                        "--pmax",
                        "99",
                        "--copy-ref",
                    ]
                )
                run(
                    [
                        "python",
                        "-m",
                        "atlasgs.ops.nifti_to_frames",
                        "--input",
                        str(target_lr),
                        "--output",
                        str(target_lr_frames),
                        "--axis",
                        "2",
                        "--pmin",
                        "1",
                        "--pmax",
                        "99",
                        "--copy-ref",
                    ]
                )
                run(cmd)

        if args.run_t1guided_topology and (args.overwrite or not (t1guided_topology_model / "point_cloud").exists()):
            train_t1guided_branch(
                args=args,
                t1_guidance_model=t1_guidance_model,
                t1_gt=t1_gt,
                target_lr_frames=target_lr_frames,
                out_model=t1guided_topology_model,
                iterations=args.t1guided_topology_iterations,
                lambda_dssim=args.t1guided_topology_lambda_dssim,
                seed=args.seed,
                single_model=single_model,
                force_no_freeze_time=force_no_freeze_time,
                time_init_scale=time_init_scale,
                time_init_shift=time_init_shift,
                appearance_mode=None,
                add_topology=True,
                interp_lambda_dssim=args.t1guided_topology_interp_lambda_dssim,
            )

        if args.run_t1guided_latent and (args.overwrite or not (t1guided_latent_model / "point_cloud").exists()):
            train_t1guided_branch(
                args=args,
                t1_guidance_model=t1_guidance_model,
                t1_gt=t1_gt,
                target_lr_frames=target_lr_frames,
                out_model=t1guided_latent_model,
                iterations=args.t1guided_latent_iterations,
                lambda_dssim=args.t1guided_lambda_dssim,
                seed=args.seed,
                single_model=single_model,
                force_no_freeze_time=force_no_freeze_time,
                time_init_scale=time_init_scale,
                time_init_shift=time_init_shift,
                appearance_mode="latent",
                add_topology=False,
                interp_lambda_dssim=None,
            )

        if args.run_t1guided_our and (args.overwrite or not (t1guided_our_model / "point_cloud").exists()):
            train_t1guided_branch(
                args=args,
                t1_guidance_model=t1_guidance_model,
                t1_gt=t1_gt,
                target_lr_frames=target_lr_frames,
                out_model=t1guided_our_model,
                iterations=args.t1guided_our_iterations,
                lambda_dssim=args.t1guided_our_lambda_dssim,
                seed=args.seed,
                single_model=single_model,
                force_no_freeze_time=force_no_freeze_time,
                time_init_scale=time_init_scale,
                time_init_shift=time_init_shift,
                appearance_mode="latent",
                add_topology=True,
                interp_lambda_dssim=args.t1guided_topology_interp_lambda_dssim,
            )

        single_render = render_model(args, single_model, target_lr_frames)
        t1guided_render = render_model(args, t1guided_model, target_lr_frames)
        if args.run_t1guided_topology:
            t1guided_topology_render = render_model(args, t1guided_topology_model, target_lr_frames)
        else:
            t1guided_topology_render = None
        if args.run_t1guided_latent:
            t1guided_latent_render = render_model(args, t1guided_latent_model, target_lr_frames)
        else:
            t1guided_latent_render = None
        if args.run_t1guided_our:
            t1guided_our_render = render_model(args, t1guided_our_model, target_lr_frames)
        else:
            t1guided_our_render = None

        verify_render_series(single_render, target_frame_count, args.interp_factor, f"MedGS single ({target})")
        verify_render_series(t1guided_render, target_frame_count, args.interp_factor, f"MedGS T1-guided ({target})")
        if t1guided_topology_render is not None:
            verify_render_series(t1guided_topology_render, target_frame_count, args.interp_factor, f"MedGS T1-guided topology ({target})")
        if t1guided_latent_render is not None:
            verify_render_series(t1guided_latent_render, target_frame_count, args.interp_factor, f"MedGS T1-guided latent ({target})")
        if t1guided_our_render is not None:
            verify_render_series(t1guided_our_render, target_frame_count, args.interp_factor, f"MedGS our ({target})")

        norm_json = target_lr_frames / "normalize.json"
        target_single_out = sub_out / f"{target}_medgs_single_{tag}.nii.gz"
        target_t1g_out = sub_out / f"{target}_medgs_t1guided_{tag}.nii.gz"
        target_t1g_topology_out = sub_out / f"{target}_medgs_t1guided_topology_{tag}.nii.gz"
        target_t1g_latent_out = sub_out / f"{target}_medgs_t1guided_latent_{tag}.nii.gz"
        target_t1g_our_out = sub_out / f"{target}_medgs_t1guided_our_{tag}.nii.gz"
        target_alpine_a2_out = sub_out / f"{target}_alpine_a2_{tag}.nii.gz"
        target_fused_out = sub_out / f"{target}_medgs_fused_lrcons_{tag}.nii.gz"
        target_our_fused_out = sub_out / f"{target}_medgs_our_fused_lrcons_{tag}.nii.gz"

        convert_render_to_nifti(args, single_render, target_gt, norm_json, target_single_out, target)
        convert_render_to_nifti(args, t1guided_render, target_gt, norm_json, target_t1g_out, target)
        if t1guided_topology_render is not None:
            convert_render_to_nifti(args, t1guided_topology_render, target_gt, norm_json, target_t1g_topology_out, target)
        if t1guided_latent_render is not None:
            convert_render_to_nifti(args, t1guided_latent_render, target_gt, norm_json, target_t1g_latent_out, target)
        if t1guided_our_render is not None:
            convert_render_to_nifti(args, t1guided_our_render, target_gt, norm_json, target_t1g_our_out, target)

        if args.run_alpine_a2 and (args.overwrite or not target_alpine_a2_out.exists()):
            alpine_out_dir = sub_out / f"{target}_alpine_a2_model_{tag}"
            cmd = [
                "python",
                "-m",
                "atlasgs.train.train_alpine_t1_to_target_a2",
                "--alpine-root",
                str(alpine_root),
                "--t1",
                str(t1_gt),
                "--target-gt",
                str(target_gt),
                "--target-lr",
                str(target_lr),
                "--target-pseudo-hr",
                str(interp_out),
                "--out-nifti",
                str(target_alpine_a2_out),
                "--out-dir",
                str(alpine_out_dir),
                "--factor-z",
                str(args.factor_z),
                "--pretrain-iters",
                str(args.alpine_a2_pretrain_iters),
                "--fit-iters",
                str(args.alpine_a2_fit_iters),
                "--batch-size",
                str(args.alpine_a2_batch_size),
                "--feature-dim",
                str(args.alpine_a2_feature_dim),
                "--hidden-features",
                str(args.alpine_a2_hidden_features),
                "--hidden-layers",
                str(args.alpine_a2_hidden_layers),
                "--trunk-lr",
                str(args.alpine_a2_trunk_lr),
                "--head-lr",
                str(args.alpine_a2_head_lr),
                "--lambda-pseudo",
                str(args.alpine_a2_lambda_pseudo),
                "--lambda-alpha-reg",
                str(args.alpine_a2_lambda_alpha_reg),
                "--chunk-size",
                str(args.alpine_a2_chunk_size),
                "--seed",
                str(args.seed),
            ]
            if mask.exists():
                cmd.extend(["--mask", str(mask)])
            try:
                run(cmd)
                alpine_a2_ready = target_alpine_a2_out.exists()
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] ALPINE A2 failed for target={target}; continuing without ALPINE. error={exc}")
                alpine_a2_ready = False
        elif args.run_alpine_a2:
            alpine_a2_ready = target_alpine_a2_out.exists()

        if args.run_lr_fusion and (args.overwrite or not target_fused_out.exists()):
            run(
                [
                    "python",
                    "-m",
                    "atlasgs.ops.fuse_lr_consistency",
                    "--single-hr",
                    str(target_single_out),
                    "--t1guided-hr",
                    str(target_t1g_out),
                    "--lr-obs",
                    str(target_lr),
                    "--ref",
                    str(target_gt),
                    "--out",
                    str(target_fused_out),
                    "--factor-z",
                    str(args.factor_z),
                    "--sigma-z",
                    str(args.sigma_z),
                    "--mode",
                    str(args.mode),
                    "--tau",
                    str(args.fusion_tau),
                    "--wmin",
                    str(args.fusion_wmin),
                    "--wmax",
                    str(args.fusion_wmax),
                    "--smooth-xy",
                    str(args.fusion_smooth_xy),
                    "--smooth-z",
                    str(args.fusion_smooth_z),
                ]
            )

        if args.run_t1guided_our and args.run_t1guided_our_fusion and (args.overwrite or not target_our_fused_out.exists()):
            run(
                [
                    "python",
                    "-m",
                    "atlasgs.ops.fuse_lr_consistency",
                    "--single-hr",
                    str(target_single_out),
                    "--t1guided-hr",
                    str(target_t1g_our_out),
                    "--lr-obs",
                    str(target_lr),
                    "--ref",
                    str(target_gt),
                    "--out",
                    str(target_our_fused_out),
                    "--factor-z",
                    str(args.factor_z),
                    "--sigma-z",
                    str(args.sigma_z),
                    "--mode",
                    str(args.mode),
                    "--tau",
                    str(args.fusion_tau),
                    "--wmin",
                    str(args.fusion_wmin),
                    "--wmax",
                    str(args.fusion_wmax),
                    "--smooth-xy",
                    str(args.fusion_smooth_xy),
                    "--smooth-z",
                    str(args.fusion_smooth_z),
                ]
            )

        our_eval_out = None
        if args.run_t1guided_our:
            if args.run_t1guided_our_fusion and target_our_fused_out.exists():
                our_eval_out = target_our_fused_out
            else:
                our_eval_out = target_t1g_our_out

        if target_gt_t1ref is not None and target in {"dwi", "asl"}:
            t1ref_targets = [
                interp_out,
                target_single_out,
                target_t1g_out,
                target_fused_out if args.run_lr_fusion else None,
                target_alpine_a2_out if alpine_a2_ready else None,
                target_t1g_topology_out if args.run_t1guided_topology else None,
                target_t1g_latent_out if args.run_t1guided_latent else None,
                target_t1g_our_out if args.run_t1guided_our else None,
                target_our_fused_out if args.run_t1guided_our and args.run_t1guided_our_fusion else None,
            ]
            for src_path in t1ref_targets:
                if src_path is None or not src_path.exists():
                    continue
                out_t1ref = src_path.with_name(f"{src_path.stem}_t1ref.nii.gz")
                if args.overwrite or not out_t1ref.exists():
                    run(
                        [
                            "python",
                            "-m",
                            "atlasgs.ops.resample",
                            "--moving",
                            str(src_path),
                            "--ref",
                            str(t1_gt),
                            "--out",
                            str(out_t1ref),
                            "--order",
                            str(int(args.resample_order)),
                        ]
                    )

        if len(targets) == 1 and target == "flair":
            metrics_out = sub_out / f"metrics_{tag}.json"
            fig_out = sub_out / f"overlays_{tag}.png"
        else:
            metrics_out = sub_out / f"metrics_{target}_{tag}.json"
            fig_out = sub_out / f"overlays_{target}_{tag}.png"

        if args.overwrite or not metrics_out.exists():
            cmd = [
                "python",
                "-m",
                "atlasgs.eval.eval_sr_flair",
                "--t1",
                str(t1_gt),
                "--flair-gt",
                str(target_gt),
                "--flair-interp",
                str(interp_out),
                "--flair-medgs-single",
                str(target_single_out),
                "--flair-medgs-t1guided",
                str(target_t1g_out),
                "--out",
                str(metrics_out),
            ]
            if args.run_lr_fusion:
                cmd.extend(["--flair-medgs-fused", str(target_fused_out)])
            if alpine_a2_ready:
                cmd.extend(["--flair-alpine-a2", str(target_alpine_a2_out)])
            if args.run_t1guided_topology:
                cmd.extend(["--flair-medgs-t1guided-topology", str(target_t1g_topology_out)])
            if args.run_t1guided_latent:
                cmd.extend(["--flair-medgs-t1guided-latent", str(target_t1g_latent_out)])
            if our_eval_out is not None:
                cmd.extend(["--flair-medgs-our", str(our_eval_out)])
            if mask.exists():
                cmd.extend(["--mask", str(mask)])
            run(cmd)

        if args.overwrite or not fig_out.exists():
            cmd = [
                "python",
                "-m",
                "atlasgs.eval.make_figures",
                "--t1",
                str(t1_gt),
                "--flair-gt",
                str(target_gt),
                "--flair-interp",
                str(interp_out),
                "--flair-medgs-single",
                str(target_single_out),
                "--flair-medgs-t1guided",
                str(target_t1g_out),
                "--out",
                str(fig_out),
            ]
            if args.run_lr_fusion:
                cmd.extend(["--flair-medgs-fused", str(target_fused_out)])
            if alpine_a2_ready:
                cmd.extend(["--flair-alpine-a2", str(target_alpine_a2_out)])
            if args.run_t1guided_topology:
                cmd.extend(["--flair-medgs-t1guided-topology", str(target_t1g_topology_out)])
            if args.run_t1guided_latent:
                cmd.extend(["--flair-medgs-t1guided-latent", str(target_t1g_latent_out)])
            if our_eval_out is not None:
                cmd.extend(["--flair-medgs-our", str(our_eval_out)])
            if mask.exists():
                cmd.extend(["--mask", str(mask)])
            run(cmd)

        processed_targets += 1

    if processed_targets == 0:
        raise FileNotFoundError(
            f"No target modalities were processed for subject {args.subject_id}. "
            f"Requested={targets}, available files under {sub}"
        )


if __name__ == "__main__":
    main()
