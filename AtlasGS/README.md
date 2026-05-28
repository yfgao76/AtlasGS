# AtlasGS

AtlasGS is a research codebase for multi-modal MRI super-resolution with Gaussian splatting. It provides dataset preparation, model training, baseline comparison, ablation, and evaluation scripts for the experiments reported in the AtlasGS paper.

The code focuses on through-plane super-resolution for brain MRI, including T1-guided FLAIR, ASL, DWI, and tumor MRI settings.

## Highlights

- Multi-modal Gaussian-splatting super-resolution for anisotropic MRI volumes.
- T1-guided target reconstruction with geometry transfer and appearance adaptation.
- Support for UK Biobank FLAIR, HCP/FOMO ASL-DWI, and GBM T2/FLAIR experiments.
- Reproducible experiment entrypoints for interpolation, INR, MedGS, AtlasGS variants, LR-consistency fusion, and ablation studies.
- Evaluation utilities for masked PSNR/SSIM/MAE, confidence intervals, visual overlays, and summary tables.

## Installation

Create a Python environment with a CUDA-enabled PyTorch build that matches your system, then install this repository in editable mode.

```bash
git clone https://github.com/yfgao76/AtlasGS.git
cd AtlasGS
python -m pip install -e .
```

Core Python dependencies are listed in `requirements.txt`. Training requires PyTorch and a working MedGS installation. AtlasGS calls MedGS training and rendering code directly; pass its path with `--medgs-root` or set `MEDGS_ROOT` in the provided experiment scripts.

## Repository Structure

```text
atlasgs/
  ops/           data preparation, degradation, NIfTI/frame conversion, resampling
  train/         AtlasGS and baseline training modules
  experiments/   single-subject, dataset-level, and SLURM experiment entrypoints
  eval/          metric aggregation, ablation analysis, overlays, and figures
  models/        lightweight INR and regularization modules
examples/        minimal input templates
```

Generated data, checkpoints, renders, metrics, logs, and external baselines are intentionally kept out of version control. The default ignored output locations are `data/`, `outputs/`, `outputs_dataset/`, `outputs_contribution/`, `external/`, and `external_metrics/`.

## Data Layout

AtlasGS expects each prepared dataset to contain one directory per subject and a split CSV with a `subject_id` column.

```text
data/ukbb_medgs/
  train.csv
  test.csv
  <subject_id>/
    t1_gt.nii.gz
    flair_gt.nii.gz
    flair_lr_1x1x7.nii.gz
    mask_brain.nii.gz
```

Exact modality names vary by dataset. The experiment runner resolves supported target names from the prepared subject directory.

## Data Preparation

Prepare UK Biobank T1/FLAIR pairs:

```bash
python -m atlasgs.ops.prepare_ukbb_medgs \
  --src /path/to/ukbb/source \
  --out data/ukbb_medgs
```

Prepare HCP/FOMO ASL-DWI data:

```bash
SRC_ROOT=/path/to/PT008_AdolescentBrainDevelopment \
SYNTHSTRIP_SIF=/path/to/freesurfer.sif \
sbatch atlasgs/experiments/prepare_hcp.sbatch
```

Prepare GBM T2/FLAIR data:

```bash
SRC_ROOT=/path/to/GBM_Dataset \
sbatch atlasgs/experiments/prepare_gbm.sbatch
```

`SYNTHSTRIP_SIF` is required only when HCP preparation uses SynthStrip skull stripping. Use `NO_SKULLSTRIP=1` to skip that step.

## Running Experiments

All SLURM scripts are written to run from the repository root. Override data, output, MedGS, and external-metric paths with environment variables when needed.

Run UK Biobank FLAIR experiments at z-axis factors 3, 5, and 7:

```bash
MEDGS_ROOT=/path/to/MedGS \
sbatch atlasgs/experiments/run_ukbb_all_methods_z7.sbatch
```

Run HCP/FOMO ASL-DWI experiments at factor 3:

```bash
MEDGS_ROOT=/path/to/MedGS \
sbatch atlasgs/experiments/run_hcp_all_methods_z3.sbatch
```

Run GBM T2/FLAIR experiments at factor 7:

```bash
MEDGS_ROOT=/path/to/MedGS \
sbatch atlasgs/experiments/run_gbm_all_methods_z7.sbatch
```

Run one subject directly:

```bash
python -m atlasgs.experiments.run_subject \
  --subject-id <subject_id> \
  --data-root data/ukbb_medgs \
  --out-root outputs_dataset/ukbb_medgs_all_methods/z7 \
  --medgs-root /path/to/MedGS \
  --target-modalities flair \
  --factor-z 7 \
  --interp-factor 7 \
  --run-t1guided-latent \
  --run-t1guided-topology \
  --run-t1guided-our \
  --run-lr-fusion
```

Dataset-level execution is handled by:

```bash
python -m atlasgs.experiments.run_all_methods_dataset \
  --data-root data/ukbb_medgs \
  --out-root outputs_dataset/ukbb_medgs_all_methods/z7 \
  --medgs-root /path/to/MedGS \
  --csv data/ukbb_medgs/test.csv \
  --target-modalities flair \
  --factor-z 7 \
  --interp-factor 7
```

## Evaluation

Aggregate UK Biobank metrics and external INR/SA-INR baselines:

```bash
INR_BASE=/path/to/external_metrics/brain-gs \
sbatch atlasgs/experiments/run_analyze_ukbb_with_external_inr.sbatch
```

Compute GBM and HCP table metrics:

```bash
sbatch atlasgs/experiments/run_analyze_gbm_abcd_table.sbatch
```

Generate UK Biobank orthogonal-plane overlays:

```bash
INR_ROOT=/path/to/inr_ukbb_ds7 \
SA_ROOT=/path/to/sa_inr_ukbb_ds7 \
sbatch atlasgs/experiments/run_ukbb_overlays_orth_z7_all.sbatch
```

Additional ablation and contribution analyses are available in `atlasgs/experiments/run_ukbb_contribution_*.sbatch` and `atlasgs/experiments/run_analyze_ukbb_ablation_masked.sbatch`.

## Outputs

Experiment outputs are written under `outputs_dataset/` by default. A typical subject output includes reconstructed NIfTI volumes, rendered frames, per-subject metric JSON files, and optional visual figures. Aggregated metrics are written as JSON and CSV files by the evaluation scripts.

## Citation

If you use AtlasGS, please cite the AtlasGS paper. BibTeX information will be added after publication.
