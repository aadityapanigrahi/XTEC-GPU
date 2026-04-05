# XTEC-GPU

**GPU-accelerated X-ray TEmperature Clustering** using PyTorch.

> Venderley *et al.* ([10.1073/pnas.2109665119](https://doi.org/10.1073/pnas.2109665119))

## Overview

XTEC is an unsupervised machine learning algorithm that clusters
temperature-dependent x-ray diffraction data `I(q, T)` to identify
order parameters and their fluctuations.

**XTEC-GPU** rewrites the entire backend in PyTorch so that preprocessing,
GMM clustering, and label smoothing all run on GPU:

| Feature | XTEC (original) | XTEC-GPU |
|---------|-----------------|----------|
| GMM backend | NumPy EM / sklearn | **torchgmm** + KMeans |
| Preprocessing | NumPy | **PyTorch** tensors |
| Label smoothing | scipy sparse (CPU) | **torch.sparse** (GPU) |
| GPU support | ❌ | ✅ CUDA / MPS |
| CPU fallback | ✅ | ✅ automatic |
| CLI | ❌ | ✅ `xtec-gpu` |

## Installation

```bash
git clone https://github.com/KimGroup/XTEC-GPU
cd XTEC-GPU
pip install -e .
```

## File Structure

```
XTEC-GPU/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── src/
│   └── xtec_gpu/
│       ├── __init__.py
│       ├── GMM.py               # GMM + GMM_kernels + label smoothing
│       ├── Preprocessing.py     # Mask_Zeros, Threshold, Peak_averaging
│       ├── xtec_cli.py          # Command-line interface
│       └── plugins/
│           ├── __init__.py      # NeXpy plugin entry point
│           └── cluster_data.py  # NeXpy GUI plugin
└── tutorials/
    ├── Tutorial_XTEC_GPU-d.ipynb
    └── Tutorial_XTEC_GPU-s_with_peak_averaging.ipynb
```

---

## Python API

```python
import torch
from xtec_gpu.Preprocessing import Mask_Zeros, Threshold_Background, Peak_averaging
from xtec_gpu.GMM import GMM, GMM_kernels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### XTEC-d  (direct clustering)

```python
masked    = Mask_Zeros(I, device=device)
threshold = Threshold_Background(masked, device=device)
Rescaled  = threshold.Rescale_mean(threshold.data_thresholded)

gmm = GMM(Rescaled.T, cluster_num=4)
gmm.RunEM()
print(gmm.num_per_cluster)
```

### XTEC-s  (peak-averaged clustering)

```python
masked    = Mask_Zeros(I, device=device)
threshold = Threshold_Background(masked, device=device)
peak_avg  = Peak_averaging(I, threshold, device=device)
Rescaled  = threshold.Rescale_mean(peak_avg.peak_avg_data)

gmm = GMM(Rescaled.T, cluster_num=4)
gmm.RunEM()
gmm.Get_pixel_labels(peak_avg)
```

### Label smoothing

```python
Data_ind = threshold.ind_thresholded
Markov   = GMM_kernels.Build_Markov_Matrix(
    Data_ind, L_scale=0.05, kernel_type="local", device=device
)

gmm = GMM(Rescaled.T, cluster_num=4)
gmm.RunEM(label_smoothing_flag=True, Markov_matrix=Markov)
```

---

## Command-Line Interface

```bash
# Hardware & Environment Sanity Check
xtec-gpu test

# XTEC-d
xtec-gpu xtec-d data.nxs -o results/ -n 4 --rescale mean

# XTEC-s (peak averaging)
xtec-gpu xtec-s data.nxs -o results/ -n 4

# Label smoothing
xtec-gpu label-smooth data.nxs -o results/ -n 4 --L-scale 0.05

# BIC sweep
xtec-gpu bic-d data.nxs -o results/ --min-nc 2 --max-nc 14
xtec-gpu bic-s data.nxs -o results/ --min-nc 2 --max-nc 14
```

## Agentic Workflow + MCP

For automatic mode (`d` vs `s`) and cluster-count selection, use:

```bash
python scripts/xtec_agentic_workflow.py data.nxs -o workflow_runs/run1 --device cuda:1
```

If the recommended mode is `d`, the workflow uses `sklearn-kmeans` init by default.
By default, the workflow also materializes per-`k` sweep artifacts for manual oversight.

Docs:

- `running_instructions.md`
- `agent.md`
- `claude.md`
- `gemini.md`

MCP server:

```bash
python scripts/xtec_workflow_mcp.py
```

### Workflow script options (`scripts/xtec_agentic_workflow.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Path to `.nxs` input file |
| `-o`, `--output-root` | — | Workflow output directory |
| `--entry` | `entry/data` | HDF5 dataset path in input file |
| `--slices` | `None` | Slice string, e.g. `":,0.0:1.0,-10.0:10.0,-15.0:15.0"` |
| `--threshold` / `--no-threshold` | on | Enable/disable KL background thresholding |
| `--rescale` | `mean` | `mean`, `z-score`, `log-mean`, or `None` |
| `--device` | `auto` | Compute device (`auto`, `cuda`, `cuda:1`, `mps`, `cpu`) |
| `--min-nc` | `2` | Minimum cluster count in BIC sweep |
| `--max-nc` | `14` | Maximum cluster count in BIC sweep |
| `--candidate-modes` | `d,s` | Comma-separated candidate modes to evaluate |
| `--random-state` | `0` | Random seed used for final `xtec-d` run |
| `--no-run-final` | off | Skip final recommended command execution (recommendation-only mode) |
| `--no-save-sweep-artifacts` | off | Skip per-`k` sweep artifact runs |

### Workflow sweep artifacts

When sweep artifacts are enabled, the workflow writes:

- `workflow_runs/<run>/sweep_artifacts/d/d_kXX/` (for each `k` in `bic-d`)
- `workflow_runs/<run>/sweep_artifacts/s/s_kXX/` (for each `k` in `bic-s`)

Each per-`k` directory includes:

- `results.h5` (cluster assignments, means/covariances, indices, and `data_thresholded`)
- `trajectories.png`
- `qmap.png`
- `avg_intensities.png`

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Path to `.nxs` file |
| `-o` | — | Output directory |
| `-n` | `4` | Number of clusters |
| `--rescale` | `mean` | `mean` / `z-score` / `log-mean` / `None` |
| `--threshold` / `--no-threshold` | on | KL background thresholding |
| `--device` | `auto` | Specify compute device (`auto`, `cuda`, `cuda:1`, `mps`, `cpu`) |
| `--entry` | `entry/data` | HDF5 path in `.nxs` file |
| `--slices` | `None` | Slice string, e.g. `":,0:1,-10:10"` |

### Label-smooth options

| Flag | Default | Description |
|------|---------|-------------|
| `--L-scale` | `0.05` | Smoothing length (pixel units) |
| `--smooth-type` | `local` | `local` or `periodic` |

### Output files

The clustering commands (`xtec-d`, `xtec-s`, `label-smooth`) write the
following files into the `-o` output directory:

**Data file:**

| File | Format | Description |
|------|--------|-------------|
| `results.h5` | HDF5 | All clustering results in a single file |

The `results.h5` file contains five datasets:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `cluster_assignments` | `(N,)` | Integer cluster label (0 to K−1) for each data point |
| `pixel_assignments` | `(N,)` | Per-pixel cluster labels (same as above for `xtec-d`/`label-smooth`; maps peaks → pixels for `xtec-s`) |
| `data_indices` | `(N, D)` | Spatial coordinates (H, K, L) of each data point |
| `cluster_means` | `(K, T)` | Mean trajectory per cluster |
| `cluster_covariances` | `(K, T)` | Diagonal covariance per cluster |

**Plots:**

| File | Description |
|------|-------------|
| `qmap.png` | **Cluster Q-map** — 2D reciprocal-space image color-coded by cluster assignment |
| `trajectories.png` | **Cluster trajectories** — Mean intensity vs. temperature with ±1σ envelope |
| `avg_intensities.png` | **Average intensities** — Raw intensity averaged within each cluster vs. temperature |

**BIC commands** (`bic-d`, `bic-s`) produce:

| File | Description |
|------|-------------|
| `bic_xtec_d.h5` / `bic_xtec_s.h5` | HDF5 with `n_clusters` and `bic_scores` datasets |
| `bic_xtec_d.png` / `bic_xtec_s.png` | BIC score plot — the minimum indicates optimal cluster count |

**Loading results in Python:**

```python
import h5py

with h5py.File("results/results.h5", "r") as f:
    labels = f["cluster_assignments"][:]      # (N,)
    means  = f["cluster_means"][:]            # (K, T)
    covs   = f["cluster_covariances"][:]      # (K, T)
    inds   = f["data_indices"][:]             # (N, D)
```

---

## API Reference

### `GMM(data, cluster_num, cov_type="diag", device=None, alpha=0.7, ...)`

GPU Gaussian Mixture Model using torchgmm with KMeans seeding.

**Key methods:**

| Method | Description |
|--------|-------------|
| `RunEM(label_smoothing_flag, Markov_matrix, ...)` | Fit GMM, optionally with E→Smooth→M loop |
| `Get_pixel_labels(Peak_avg)` | Map peak labels to pixel labels |
| `Plot_Cluster_Results_traj(x_train)` | Plot cluster trajectories |
| `Plot_Cluster_kspace_2D_slice(threshold, ...)` | Plot 2D Q-map |

**Attributes after `RunEM()`:**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `cluster_assignments` | `(N,)` | Hard labels |
| `cluster_probs` | `(K, N)` | Soft responsibilities |
| `means` | `(K, T)` | Cluster means |
| `covs` | `(K, T)` | Cluster covariances |
| `mixing_weights` | `(K,)` | Mixture weights |
| `num_per_cluster` | list | Samples per cluster |

### `GMM_kernels.Build_Markov_Matrix(data_inds, L_scale, kernel_type, ...)`

Static method. Builds a GPU sparse Markov matrix for label smoothing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_inds` | — | Spatial indices `(N, D)` |
| `L_scale` | `1` | Kernel bandwidth |
| `kernel_type` | `"local"` | `"local"` or `"periodic"` |
| `device` | `None` | Target device |
| `chunk_size` | `4096` | Batch size for memory management |

Returns a `torch.sparse_csr_tensor` of shape `(N, N)`.

### `Mask_Zeros(data, device=None)`

Removes zero-intensity data points. Access `data_masked` and `mask`.

### `Threshold_Background(masked, threshold_type="KL", device=None)`

KL-divergence thresholding. Access `data_thresholded`, `ind_thresholded`.
Methods: `Rescale_mean()`, `Rescale_zscore()`.

### `Peak_averaging(data, threshold, device=None)`

Averages intensities within connected Bragg peaks.
Access `peak_avg_data`, `peak_avg_ind_list`.

---

## Tutorials

| Notebook | Description |
|----------|-------------|
| `Tutorial_XTEC_GPU-d` | XTEC-d on Sr₃Rh₄Sn₁₃ XRD data |
| `Tutorial_XTEC_GPU-s_with_peak_averaging` | XTEC-s with peak averaging |

**Data**: Download `srn0_XTEC.nxs` (~32 GB) from
https://dx.doi.org/10.18126/iidy-30e7


## Attribution

XTEC-GPU: **Yanjun Liu** and **Aaditya Panigrahi**.
Original XTEC: **Jordan Venderley**, with modifications by **Krishnanand Mallayya**.

## Contact

Aaditya Panigrahi — ap2563@cornell.edu

## License

[MIT](LICENSE)
