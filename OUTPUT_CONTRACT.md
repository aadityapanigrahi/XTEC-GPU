# Output Contract

This document defines the expected output artifacts for current XTEC CLI/workflow
commands.

## Clustering Commands

Commands:
- `xtec-gpu xtec-d ...`
- `xtec-gpu xtec-s ...`
- `xtec-gpu label-smooth ...` (when enabled)

Example:
```bash
xtec-gpu xtec-d data.nxs -o results_d -n 4
```

Expected artifacts in output directory:
- `results.h5`
- `qmap.png`
- `trajectories.png`
- `avg_intensities.png`

`results.h5` datasets:
- `cluster_assignments`
- `pixel_assignments`
- `data_indices`
- `data_thresholded`
- `cluster_means`
- `cluster_covariances`

### Shape contract

- `data_indices` has shape `(N_pixels, D)`. `N_pixels` is the number of
  surviving voxels after thresholding.
- `pixel_assignments` has shape `(N_pixels,)`. It is always aligned with
  `data_indices` and is the right label array for any spatial computation
  (e.g. building a Q-map, computing per-pixel neighborhood agreement).
- `cluster_assignments` has shape `(N_pixels,)` for `xtec-d` and
  `label-smooth`, and is identical to `pixel_assignments` for those modes.
  For `xtec-s` it has shape `(N_peaks,)` — peaks, not pixels — because the
  GMM is fit on peak-averaged trajectories. Downstream code that joins
  cluster labels with spatial coordinates must use `pixel_assignments`.
- `cluster_means` has shape `(K, T)`, `cluster_covariances` has shape
  `(K, T)` (diagonal covariances stored per temperature).

## BIC Sweep Commands

Commands:
- `xtec-gpu bic-d ...`
- `xtec-gpu bic-s ...`

Example:
```bash
xtec-gpu bic-d data.nxs -o bic_d --min-nc 2 --max-nc 14
```

Expected artifacts:
- `bic_xtec_d.h5` or `bic_xtec_s.h5`
- `bic_xtec_d.png` or `bic_xtec_s.png`

HDF5 datasets:
- `n_clusters`
- `bic_scores`

## Agentic Workflow

Command:
- `python scripts/xtec_agentic_workflow.py ...`

Example:
```bash
python scripts/xtec_agentic_workflow.py data.nxs -o workflow_runs/run1 --device cuda:1
```

Expected top-level artifacts:
- `workflow_report.json`
- `bic_sweeps/`
- optionally `sweep_artifacts/`
- optionally `final_run/`

`workflow_report.json` required keys:
- `input`
- `output_root`
- `settings`
- `bic_results`
- `recommendation`
- `final_command`

## Full Sweep Workflow

Command:
- `xtec-gpu full-sweep ...`

Example:
```bash
xtec-gpu full-sweep data.nxs -o sweep_runs/run1 --device cuda:1 --min-nc 2 --max-nc 8
```

Expected top-level artifacts:
- `manifest.json`
- `sweep_summary.json`
- `recommendation.json`
- `bic_curves/<mode>_<init_slug>/{bic_xtec_<mode>.h5, bic_xtec_<mode>.png}`
- `runs/<mode>_k<kk>_<init_slug>/` (one directory per (mode, k, init))
- `final_run/` (relative symlink to the winning combination)

Per-run files inside each `runs/<mode>_k<kk>_<init_slug>/`:
- `results.h5` (same datasets as `xtec-d`/`xtec-s`)
- `qmap.png`
- `trajectories.png`
- `avg_intensities.png`
- `metrics.json`
- `timing.json`

`manifest.json` required keys:
- `input`
- `output_root`
- `config`
- `device`
- `torch_version`
- `git_sha`
- `host`
- `plan`
- `started_at`
- `finished_at`

`sweep_summary.json` required keys:
- `candidates` (list of per-run records)
- `manifest_ref`

`recommendation.json` required keys:
- `winner`
- `top_n`
- `reasoning`
- `weights`
- `bic_argmin`
- `bic_argmin_overridden`

Per-run `metrics.json` keys:
- `bic`, `n_points`, `cluster_sizes`
- `min_cluster_frac`, `degeneracy_flag`
- `mean_pairwise_sep`, `mean_trajectory_smoothness`, `spatial_coherence`
- `bic_norm` (filled in by the judge across same-mode candidates)
