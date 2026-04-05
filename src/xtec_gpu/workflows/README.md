# Workflows Package

This package contains orchestration logic for workflow execution.

Current modules:

- `agentic.py`: mode/cluster recommendation workflow
- `comparison.py`: CPU vs GPU tutorial comparison workflow

Scripts in `scripts/` provide command entry points for these workflows.

## Execution Backend

`agentic.py` supports two execution backends:

- `inprocess` (default): directly calls `xtec_gpu.xtec_cli` functions in the
  same Python process (lowest orchestration overhead).
- `subprocess`: spawns child processes for each command (strong process
  isolation, higher overhead).

Use `--execution-backend inprocess|subprocess` with
`scripts/xtec_agentic_workflow.py`.

## Benchmark Notes

On `/data/XTEC_GPU/test_dataset/srn0_XTEC.nxs` (GPU: `cuda:1`) with:

- `--slices ':,0.0:1.0,-4.0:4.0,-4.0:4.0'`
- `--min-nc 2 --max-nc 4`
- `--candidate-modes d,s`

observed end-to-end workflow runtime:

- `subprocess`: `82.55 s`
- `inprocess`: `30.87 s`

Speedup: about `2.67x` for the workflow orchestration path, with identical
final recommendation and final `results.h5` for this benchmark run.

## CPU reference vs GPU workflow benchmark

Using `comparison.py` against the original KimGroup CPU implementation
(`/tmp/KimGroup_XTEC_ref`) on the same sliced input
(`:,0.0:1.0,-4.0:4.0,-4.0:4.0`), measured runtime:

- CPU reference tutorial workflow: `34.95 s`
- GPU CLI tutorial workflow: `16.31 s`

Observed speedup: about `2.14x`.

Note: the default tutorial second-pass selection uses auto-discovery and can
choose different first-pass clusters across CPU/GPU, so second-pass sample
counts may differ unless you pin selection with `--good-cluster` or
`--bad-clusters`.
