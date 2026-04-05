# XTEC-GPU Performance Comparison (CPU vs CUDA)

Date: 2026-04-05

## Benchmark Setup

- Input: `/data/XTEC_GPU/test_dataset/srn0_XTEC.nxs`
- Slice (largest feasible in current workflow path): `:,0.0:1.0,-30.0:30.0,-30.0:30.0`
- Loaded shape: `(24, 21, 1201, 1201)`
- Workflow config:
  - `candidate_modes=d,s`
  - `min_nc=2`, `max_nc=4`
  - `execution_backend=inprocess`
  - sweep artifacts enabled
  - final run enabled
- CPU run: `--device cpu`
- GPU run: `--device cuda:1`
- Python: `/home/ap2563/miniconda3/envs/torchgpu/bin/python`

## Apples-to-Apples Timing Results

| Metric | CPU (s) | CUDA (s) | CPU/CUDA |
|---|---:|---:|---:|
| Workflow wall time | 897.000 | 531.743 | 1.687x |
| Data load total | 531.376 | 485.998 | 1.093x |
| Preprocess mask zeros | 72.944 | 19.996 | 3.648x |
| Preprocess threshold | 183.079 | 1.852 | 98.862x |
| Preprocess peak averaging | 3.629 | 10.992 | 0.330x |
| Preprocess rescale | 0.708 | 0.002 | 332.371x |
| Preprocess total | 260.360 | 32.843 | 7.927x |
| GMM RunEM total | 92.428 | 2.296 | 40.254x |
| I/O total | 6.881 | 6.714 | 1.025x |
| xtec_d command total | 382.081 | 222.157 | 1.720x |
| xtec_s command total | 267.709 | 158.883 | 1.685x |
| bic_d command total | 139.566 | 74.266 | 1.879x |
| bic_s command total | 107.638 | 76.431 | 1.408x |

## Observed Bottleneck Shares

- CPU:
  - Load: `59.24%`
  - Preprocess: `29.03%`
  - RunEM: `10.30%`
  - I/O: `0.77%`
- CUDA:
  - Load: `91.40%`
  - Preprocess: `6.18%`
  - RunEM: `0.43%`
  - I/O: `1.26%`

## Consistency Check

- CPU recommendation: `{"mode": "d", "n_clusters": 3, "init_strategy_mode": "kmeans++"}`
- CUDA recommendation: `{"mode": "d", "n_clusters": 3, "init_strategy_mode": "kmeans++"}`

## Raw Timing Artifacts

- CPU JSON: `/tmp/wf_timing_maxslice_cpu/timing_breakdown.json`
- CUDA JSON: `/tmp/wf_timing_maxslice_cuda/timing_breakdown.json`
