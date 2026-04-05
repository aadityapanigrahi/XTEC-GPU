# XTEC-GPU Performance Comparison (CPU vs CUDA)

Date: 2026-04-05

## Benchmark Setup

- Source input: `/data/XTEC_GPU/test_dataset/srn0_XTEC.nxs`
- Benchmark input (derived ~20GB working set): `/data/XTEC_GPU/test_dataset/srn0_XTEC_20gb_24x72x1201x1201.nxs`
- Runtime slicing: `none` (unsliced input for both runs)
- Loaded shape: `(24, 72, 1201, 1201)`
- Workflow config:
  - `candidate_modes=d`
  - `min_nc=2`, `max_nc=4`
  - `execution_backend=inprocess`
  - sweep artifacts enabled
  - final run enabled
- CPU run: `--device cpu`
- GPU run: `--device cuda:1 --streamed-preprocess --streamed-exact-log-limit 50000000`
- Python: `/data/XTEC_GPU/XTEC-GPU/.venv/bin/python3`

## Apples-to-Apples Timing Results

| Metric | CPU (s) | CUDA (s) | CPU/CUDA |
|---|---:|---:|---:|
| Workflow wall time | 366.001 | 408.002 | 0.897x |
| Data load total (instrumented) | 0.005 | 0.013 | 0.341x |
| Preprocess mask zeros (instrumented) | 32.756 | 0.000 | N/A |
| Preprocess threshold (instrumented) | 103.086 | 0.000 | N/A |
| Preprocess peak averaging (instrumented) | 0.000 | 0.000 | N/A |
| Preprocess rescale (instrumented) | 3.058 | 0.050 | 61.687x |
| Preprocess total (instrumented) | 138.900 | 0.050 | 2801.518x |
| GMM RunEM total | 197.522 | 2.360 | 83.685x |
| I/O total | 6.548 | 18.338 | 0.357x |
| bic_d command total | 228.682 | 386.794 | 0.591x |
| xtec_d command total | 137.238 | 21.204 | 6.472x |

## Observed Bottleneck Shares

- CPU:
  - Load: `0.00%`
  - Preprocess: `37.95%`
  - RunEM: `53.97%`
  - I/O: `1.79%`
- CUDA:
  - Load: `0.00%`
  - Preprocess: `0.01%`
  - RunEM: `0.58%`
  - I/O: `4.49%`

## Consistency Check

- CPU recommendation: `{"mode": "d", "n_clusters": 3, "init_strategy_mode": "kmeans++"}`
- CUDA recommendation: `{"mode": "d", "n_clusters": 3, "init_strategy_mode": "kmeans++"}`
- Recommendation equal: `True`
- BIC n-cluster grid equal: `True`
- BIC score max abs diff: `208.0`
- BIC score mean abs diff: `114.0`
- Final cluster assignment match ratio (raw order): `0.566454`
- Final cluster means MAE: `1.128376e-03`
- Final cluster covariances MAE: `9.301571e-04`
- Thresholded index set equal (aligned): `True`
- Final cluster assignment match ratio (aligned by `data_indices`): `0.999453`
- Final ARI (aligned by `data_indices`): `0.998920`

## Threshold Parity (CPU Standard vs GPU Streaming Exact)

- `cutoff_abs_diff`: `4.441e-16`
- `n_thresholded_diff`: `0`
- thresholded voxel set parity: `true`
- streamed cutoff mode: `exact-kl`
- cutoff compute device: `cuda:1`
- CPU cutoff: `2.529293105740551`
- GPU streamed cutoff: `2.529293105740550`
- Streaming chunks: `25` (`chunks_per_axis=[1, 5, 5]`, `chunk_shape=[72, 279, 278]`)
- Estimated chunk bytes: `1072217088` (~1.00 GiB)
- Valid log means used for cutoff: `40793977`

## Raw Timing Artifacts

- CPU JSON: `/tmp/wf_cmp20gb_cpu_std_d_20260405/timing_breakdown.json`
- CUDA JSON: `/tmp/wf_cmp20gb_gpu_stream_d_20260405/timing_breakdown.json`
- Workflow parity JSON: `/tmp/wf_cmp20gb_gpu_stream_d_20260405/parity_vs_cpu.json`
- Threshold parity JSON: `/tmp/wf_cmp20gb_gpu_stream_d_20260405/threshold_parity_vs_cpu.json`

## Notes

- The streamed preprocessing path bypasses several function hooks used by the profiler (`Mask_Zeros`, `Threshold_Background`, `Peak_averaging`), so those instrumented rows read near-zero for CUDA and should not be interpreted as literal total preprocessing cost.
- For strict CPU-standard vs GPU-streaming parity on this ~20GB working set, `candidate_modes=d` was used. The `s` path is currently non-streamed and can exceed GPU memory at this scale.
