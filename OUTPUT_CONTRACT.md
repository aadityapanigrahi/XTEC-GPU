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
