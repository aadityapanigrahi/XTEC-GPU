# Running Instructions: Agentic XTEC Workflow

This workflow helps you:
1. choose `d` vs `s` mode,
2. choose a cluster count (`n_clusters`) via BIC sweeps,
3. save per-`k` sweep artifacts for manual oversight,
4. run the recommended final command.

## 1) Environment

Use your GPU env and source tree:

```bash
cd /data/XTEC_GPU/XTEC-GPU-Baseline
export PYTHONPATH=/data/XTEC_GPU/XTEC-GPU-Baseline/src
```

If RAPIDS (`cuml`) is installed in a custom target (for example `/data/XTEC_GPU/.pylibs`), add:

```bash
export PYTHONPATH=/data/XTEC_GPU/.pylibs:$PYTHONPATH
export LD_LIBRARY_PATH="/data/XTEC_GPU/.pylibs/libcuml/lib64:/data/XTEC_GPU/.pylibs/libcudf/lib64:/data/XTEC_GPU/.pylibs/libraft/lib64:/data/XTEC_GPU/.pylibs/librmm/lib64:/data/XTEC_GPU/.pylibs/libkvikio/lib64:/data/XTEC_GPU/.pylibs/lib64:/data/XTEC_GPU/.pylibs/nvidia/cublas/lib:/data/XTEC_GPU/.pylibs/nvidia/cusolver/lib:/data/XTEC_GPU/.pylibs/nvidia/cusparse/lib:/data/XTEC_GPU/.pylibs/nvidia/cufft/lib:/data/XTEC_GPU/.pylibs/nvidia/curand/lib:/data/XTEC_GPU/.pylibs/nvidia/nccl/lib:/data/XTEC_GPU/.pylibs/nvidia/cuda_runtime/lib:/data/XTEC_GPU/.pylibs/nvidia/cuda_nvrtc/lib:/data/XTEC_GPU/.pylibs/nvidia/nvjitlink/lib:/data/XTEC_GPU/.pylibs/nvidia/libnvcomp/lib64:${LD_LIBRARY_PATH}"
```

## 2) Run the Agentic Workflow (CLI)

```bash
python scripts/xtec_agentic_workflow.py \
  /data/XTEC_GPU/test_dataset/srn0_XTEC.nxs \
  -o /data/XTEC_GPU/XTEC-GPU-Baseline/workflow_runs/srn0_auto \
  --device cuda:1 \
  --rescale mean \
  --slices ":,0.0:1.0,-10.0:10.0,-15.0:15.0" \
  --candidate-modes d,s \
  --min-nc 2 --max-nc 14
```

## 3) Outputs (Agentic Workflow)

Main report:

- `/data/XTEC_GPU/XTEC-GPU-Baseline/workflow_runs/srn0_auto/workflow_report.json`

Includes:

- BIC sweeps for `d` and `s`
- per-`k` sweep artifact runs (`results.h5`, `trajectories.png`, `qmap.png`, `avg_intensities.png`)
- best `k` per mode
- recommended mode + `n_clusters`
- for `xtec-d`, init strategy defaults to `kmeans++`
- final command executed

For faithful replication against legacy/tutorial behavior, pass:

```bash
--init-strategy-mode sklearn-kmeans
```

To skip generating sweep artifacts, add:

```bash
--no-save-sweep-artifacts
```

## 4) Run the Full Sweep Workflow (CLI)

The full sweep runs BIC across `k ∈ [min_nc, max_nc)` for every requested mode
and initialization strategy, saves **all artifacts** for every combination, and
then scores each candidate on multiple criteria (BIC, cluster degeneracy,
trajectory separation, trajectory smoothness, spatial coherence) to emit an
agent-driven recommendation. Designed for exhaustive analysis when memory is
not a constraint.

```bash
xtec-gpu full-sweep \
  /data/XTEC_GPU/test_dataset/srn0_XTEC.nxs \
  -o /data/XTEC_GPU/XTEC-GPU/sweep_runs/srn0_full \
  --device cuda:1 \
  --rescale mean \
  --slices ":,0.0:1.0,-10.0:10.0,-15.0:15.0" \
  --min-nc 2 --max-nc 8 \
  --modes d,s \
  --inits kmeans++
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--min-nc` / `--max-nc` | `2` / `8` | Cluster-count range. Upper bound is exclusive (matches `np.arange`). |
| `--modes` | `d,s` | Comma-separated modes to sweep. |
| `--inits` | `kmeans++` | GPU-resident init strategies to sweep. |
| `--include-cpu-inits` | off | Also sweep `sklearn-kmeans` and `xtec` (slow, CPU-bound). |
| `--plots` | `all` | `all` / `primary` / `none`. Default keeps every PNG for later inspection. |
| `--stream` | `auto` | `auto` enables streaming for inputs above `--stream-threshold-gb`. |
| `--stream-threshold-gb` | `2.0` | File-size cutoff for `--stream auto`. |
| `--resume` | off | Skip combos whose `results.h5` already matches the manifest. |
| `--dry-run` | off | Print the execution plan without running anything. |
| `--judge-weights` | None | Path to JSON with weight overrides for the scoring rule. |
| `--judge-top-n` | `5` | Number of ranked candidates to include in `recommendation.json`. |

Output layout (under `sweep_runs/srn0_full/`):

```
manifest.json                       full config, git sha, host, device, input hash
sweep_summary.json                  one row per (mode, k, init) with bic + metrics + score
recommendation.json                 winner + ranked candidates + reasoning trace
bic_curves/<mode>_<init_slug>/      bic_xtec_<mode>.h5 and .png
runs/<mode>_k<kk>_<init_slug>/      results.h5, qmap.png, trajectories.png,
                                    avg_intensities.png, metrics.json, timing.json
final_run/                          relative symlink to the winner's runs/... directory
```

To re-score an existing sweep with different metric weights (no re-clustering):

```bash
xtec-gpu full-sweep-judge sweep_runs/srn0_full --judge-weights my_weights.json
```

To inspect the top-N candidates:

```bash
xtec-gpu full-sweep-inspect sweep_runs/srn0_full --top 5
```

To regenerate plots in-place from existing `results.h5` files (e.g. after
plot code changes) without re-clustering:

```bash
xtec-gpu full-sweep --replot -o sweep_runs/srn0_full \
    /data/XTEC_GPU/test_dataset/srn0_XTEC_20gb_24x72x1201x1201.nxs
```

The replot walks every `runs/<id>/` and redraws `qmap.png`, `trajectories.png`,
and `avg_intensities.png`. The qmap uses the same discrete color per cluster
as the trajectories (so cluster 1 has the same color in all three plots) on
a white background; unassigned voxels are white.

## 5) MCP Server Option

Run the MCP server (stdio):

```bash
python scripts/xtec_workflow_mcp.py
```

This exposes tool `recommend_xtec_workflow(...)`.
