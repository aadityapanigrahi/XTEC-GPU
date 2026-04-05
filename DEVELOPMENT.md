# Development Guide

## Quickstart

```bash
cd /data/XTEC_GPU/XTEC-GPU
source .venv/bin/activate
```

## Where To Edit

- Core algorithms: `src/xtec_gpu/GMM.py`, `src/xtec_gpu/Preprocessing.py`
- CLI behavior/options: `src/xtec_gpu/xtec_cli.py`
- Agentic workflow logic: `src/xtec_gpu/workflows/agentic.py`
- CPU vs GPU comparison workflow: `src/xtec_gpu/workflows/comparison.py`
- Shared workflow helpers: `src/xtec_gpu/workflows/shared.py`

## Running Checks

```bash
python -m unittest -q tests/test_refactor_regressions.py
python scripts/xtec_agentic_workflow.py --help
python scripts/generate_cpu_gpu_tutorial_outputs.py --help
```

## Output Expectations

See `OUTPUT_CONTRACT.md` for required files and report keys.
