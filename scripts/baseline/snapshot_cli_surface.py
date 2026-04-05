#!/usr/bin/env python3
"""Capture a lightweight baseline snapshot of CLI/workflow surfaces.

This script intentionally avoids changing any runtime behavior. It records:
- `xtec-gpu --help` output
- script `--help` output for workflow entry points
- current git metadata

Usage:
    python scripts/baseline/snapshot_cli_surface.py -o baseline_snapshots/latest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Mapping


def _run_text(cmd: List[str], cwd: Path, env: Mapping[str, str]) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), env=dict(env), capture_output=True, text=True)
    header = f"$ {' '.join(cmd)}\nexit_code={proc.returncode}\n"
    return header + "\n[stdout]\n" + proc.stdout + "\n[stderr]\n" + proc.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture CLI/workflow baseline help outputs.")
    parser.add_argument("-o", "--output", default="baseline_snapshots/latest",
                        help="Output directory for snapshot artifacts")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

    # Use active environment python when available to keep snapshot reproducible.
    python_bin = str((repo_root / ".venv" / "bin" / "python").resolve())
    if not Path(python_bin).exists():
        python_bin = "python3"

    captures = {
        "xtec_gpu_help.txt": [python_bin, "-m", "xtec_gpu.xtec_cli", "--help"],
        "agentic_workflow_help.txt": [python_bin, "scripts/xtec_agentic_workflow.py", "--help"],
        "mcp_workflow_help.txt": [python_bin, "scripts/xtec_workflow_mcp.py", "--help"],
        "comparison_workflow_help.txt": [python_bin, "scripts/generate_cpu_gpu_tutorial_outputs.py", "--help"],
    }

    for name, cmd in captures.items():
        (out_dir / name).write_text(_run_text(cmd, repo_root, env))

    git_meta = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=str(repo_root), text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=str(repo_root), text=True),
    }
    (out_dir / "git_metadata.json").write_text(json.dumps(git_meta, indent=2))


if __name__ == "__main__":
    main()
