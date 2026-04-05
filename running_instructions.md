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

## 3) Outputs

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

## 4) MCP Server Option

Run the MCP server (stdio):

```bash
python scripts/xtec_workflow_mcp.py
```

This exposes tool `recommend_xtec_workflow(...)`.
