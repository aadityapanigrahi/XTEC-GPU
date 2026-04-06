# XTEC-GPU Performance Comparison (Fresh Current-Code Benchmark)

Date: 2026-04-06

## Benchmark Setup

- Input: `/data/XTEC_GPU/test_dataset/srn0_XTEC_20gb_24x72x1201x1201.nxs`
- Shape: `(24, 72, 1201, 1201)`
- ROI slicing: disabled (full unsliced input)
- CPU workflow: `--device cpu`
- GPU workflow: `--device cuda:1 --streamed-preprocess --streamed-chunk-voxels 0 --streamed-exact-log-limit 50000000`
- Candidate modes: `d,s`; BIC sweep `k=2..3` (`--min-nc 2 --max-nc 4`)
- Execution backend: `inprocess`

## Apples-to-Apples Workflow Timings (Fresh)

| Metric | CPU (s) | CUDA (s) | CPU/CUDA |
|---|---:|---:|---:|
| Workflow wall time | 569.459 | 169.777 | 3.354x |
| Data load total | 0.006 | 0.011 | 0.489x |
| Preprocess mask zeros | 73.226 | 0.000 | inf |
| Preprocess threshold | 236.393 | 0.000 | inf |
| Preprocess peak averaging | 2.905 | 0.000 | inf |
| Preprocess rescale | 3.219 | 0.011 | 281.957x |
| Preprocess total | 315.742 | 0.011 | 27656.862x |
| GMM RunEM total | 221.899 | 2.137 | 103.856x |
| I/O total | 8.604 | 8.952 | 0.961x |
| bic_d command total | 258.791 | 153.691 | 1.684x |
| bic_s command total | 157.302 | 3.285 | 47.888x |
| xtec_d command total | 149.472 | 9.740 | 15.346x |
| xtec_s command total | 3.641 | 3.024 | 1.204x |

## Process-by-Process Breakdown

### `bic_d`

- CPU command total: `258.791 s`
- CUDA command total: `153.691 s`

| Stage | CPU (s) | CUDA (s) | CPU/CUDA | CPU share | CUDA share |
|---|---:|---:|---:|---:|---:|
| `preprocess:threshold_background` | 121.072 | 0.000 | inf | 46.78% | 0.00% |
| `compute:gmm_runem` | 83.307 | 0.681 | 122.368x | 32.19% | 0.44% |
| `preprocess:mask_zeros` | 36.408 | 0.000 | inf | 14.07% | 0.00% |
| `preprocess:rescale` | 0.786 | 0.008 | 101.660x | 0.30% | 0.01% |
| `load_data` | 0.006 | 0.011 | 0.489x | 0.00% | 0.01% |

### `bic_s`

- CPU command total: `157.302 s`
- CUDA command total: `3.285 s`

| Stage | CPU (s) | CUDA (s) | CPU/CUDA | CPU share | CUDA share |
|---|---:|---:|---:|---:|---:|
| `preprocess:threshold_background` | 115.320 | 0.000 | inf | 73.31% | 0.00% |
| `preprocess:mask_zeros` | 36.818 | 0.000 | inf | 23.41% | 0.00% |
| `preprocess:peak_averaging` | 2.905 | 0.000 | inf | 1.85% | 0.00% |
| `compute:gmm_runem` | 0.891 | 0.289 | 3.088x | 0.57% | 8.79% |
| `preprocess:rescale` | 0.007 | 0.000 | 19.676x | 0.00% | 0.01% |

### `xtec_d`

- CPU command total: `149.472 s`
- CUDA command total: `9.740 s`

| Stage | CPU (s) | CUDA (s) | CPU/CUDA | CPU share | CUDA share |
|---|---:|---:|---:|---:|---:|
| `compute:run_direct_gmm_total` | 143.289 | 2.949 | 48.591x | 95.86% | 30.28% |
| `compute:gmm_runem` | 136.706 | 0.855 | 159.940x | 91.46% | 8.78% |
| `io:save_results` | 1.383 | 2.598 | 0.532x | 0.93% | 26.67% |
| `preprocess:rescale` | 2.414 | 0.003 | 956.741x | 1.61% | 0.03% |
| `io:plot_qmap` | 2.112 | 1.882 | 1.122x | 1.41% | 19.32% |
| `io:plot_avg_intensities` | 1.547 | 1.500 | 1.032x | 1.04% | 15.40% |
| `io:plot_trajectories` | 1.074 | 0.697 | 1.540x | 0.72% | 7.16% |

### `xtec_s`

- CPU command total: `3.641 s`
- CUDA command total: `3.024 s`

| Stage | CPU (s) | CUDA (s) | CPU/CUDA | CPU share | CUDA share |
|---|---:|---:|---:|---:|---:|
| `io:plot_qmap` | 1.255 | 1.288 | 0.975x | 34.48% | 42.58% |
| `compute:gmm_runem` | 0.994 | 0.312 | 3.183x | 27.30% | 10.33% |
| `io:plot_avg_intensities` | 0.674 | 0.437 | 1.545x | 18.52% | 14.44% |
| `io:plot_trajectories` | 0.444 | 0.438 | 1.015x | 12.21% | 14.48% |
| `io:save_results` | 0.113 | 0.113 | 1.007x | 3.11% | 3.72% |
| `preprocess:rescale` | 0.012 | 0.001 | 14.544x | 0.33% | 0.03% |

## Cold-Start Standalone Timings (Fresh)

| Command | CPU Standard (s) | GPU Streaming (s) | CPU/CUDA |
|---|---:|---:|---:|
| `bic-d` | 264.903 | 65.412 | 4.050x |
| `bic-s` | 235.177 | 68.231 | 3.447x |
| `bic-d + bic-s` | 500.080 | 133.643 | 3.742x |

## Parity Summary (Fresh)

- Recommendation CPU: `{'mode': 'd', 'n_clusters': 3, 'init_strategy_mode': 'kmeans++'}`
- Recommendation CUDA: `{'mode': 'd', 'n_clusters': 3, 'init_strategy_mode': 'kmeans++'}`
- Recommendation match: `True`

| Mode | K-grid equal | Max abs BIC diff | Mean abs BIC diff |
|---|---|---:|---:|
| `d` | `True` | 8.000000 | 8.000000 |
| `s` | `True` | 2.812500 | 1.484375 |

### Final Run Parity (Recommended Mode)

- Mode CPU: `d`
- Mode CUDA: `d`
- Cluster assignment exact equal: `False`
- Cluster assignment match ratio: `0.999985603`
- Cluster means MAE: `1.680006e-05`
- Cluster means L-inf: `1.101494e-04`
- Cluster covariances MAE: `2.056723e-05`
- Cluster covariances L-inf: `3.347397e-04`

### Threshold Cutoff Parity (CPU Standard vs GPU Streaming)

- CPU cutoff: `2.529293105740551`
- CUDA cutoff: `2.529293105740550`
- Cutoff abs diff: `4.440892098500626162e-16`
- CPU thresholded count: `1736459`
- CUDA thresholded count: `1736459`
- Thresholded count diff: `0`
- Thresholded voxel set equal: `True`
- Thresholded voxel order equal: `True`
- Stream mode: `exact-kl`
- Cutoff compute device: `cuda:1`
- Chunk shape: `[72, 279, 278]`
- Chunk voxels: `5592405`
- Chunks per axis: `[1, 5, 5]`
- Number of streaming chunks: `25`
- Estimated chunk bytes: `1072217088`
- Valid log means: `40793977`
- IQR relative error: `0.0`

## Raw Artifacts (Fresh)

- `/tmp/wf_present_cpu_std_20260406/timing_breakdown.json`
- `/tmp/wf_present_gpu_stream_ds_20260406/timing_breakdown.json`
- `/tmp/wf_present_cpu_std_20260406/workflow_report.json`
- `/tmp/wf_present_gpu_stream_ds_20260406/workflow_report.json`
- `/tmp/wf_present_gpu_stream_ds_20260406/parity_vs_cpu.json`
- `/tmp/wf_present_gpu_stream_ds_20260406/threshold_parity_vs_cpu.json`
- `/tmp/wf_present_standalone_20260406/standalone_timings.json`

## Notes

- This report uses only freshly generated artifacts from the current code state.
- Ratios are shown as `inf` when CPU time is positive and CUDA stage time is exactly zero.
