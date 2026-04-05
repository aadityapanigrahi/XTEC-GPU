"""Shared helpers used by workflow orchestration modules."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional

from xtec_gpu.config.run_config import AgenticWorkflowConfig

WORKFLOW_REPORT_REQUIRED_KEYS = (
    "input",
    "output_root",
    "settings",
    "bic_results",
    "recommendation",
    "final_command",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_checked(cmd: List[str], env: Optional[Mapping[str, str]] = None) -> None:
    """Run a command and raise on non-zero exit status."""
    subprocess.run(list(cmd), check=True, env=dict(env) if env is not None else None)


def _append_common_args(cmd: List[str], cfg: AgenticWorkflowConfig) -> None:
    cmd.extend([
        "--entry",
        cfg.entry,
        "--rescale",
        cfg.rescale,
        "--device",
        cfg.device,
    ])
    if cfg.slices:
        cmd.extend(["--slices", cfg.slices])
    if not cfg.threshold:
        cmd.append("--no-threshold")
    if bool(cfg.streamed_preprocess):
        cmd.append("--streamed-preprocess")
        cmd.extend([
            "--streamed-chunk-voxels",
            str(int(cfg.streamed_chunk_voxels)),
            "--streamed-reservoir-size",
            str(int(cfg.streamed_reservoir_size)),
            "--streamed-max-bins",
            str(int(cfg.streamed_max_bins)),
            "--streamed-exact-log-limit",
            str(int(cfg.streamed_exact_log_limit)),
            "--streamed-seed",
            str(int(cfg.streamed_seed)),
        ])


def build_bic_command(mode: str, input_path: str, output_dir: Path, cfg: AgenticWorkflowConfig) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        f"bic-{mode}",
        input_path,
        "-o",
        str(output_dir),
    ]
    _append_common_args(cmd, cfg)
    cmd.extend(["--min-nc", str(cfg.min_nc), "--max-nc", str(cfg.max_nc)])
    return cmd


def build_xtec_d_command(
    input_path: str,
    output_dir: Path,
    cfg: AgenticWorkflowConfig,
    n_clusters: int,
    init_strategy: str,
    random_state: int,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        "xtec-d",
        input_path,
        "-o",
        str(output_dir),
    ]
    _append_common_args(cmd, cfg)
    cmd.extend([
        "-n",
        str(n_clusters),
        "--init-strategy-mode",
        init_strategy,
        "--solver-mode",
        "torchgmm",
        "--post-stepwise-epochs",
        "0",
        "--random-state",
        str(random_state),
    ])
    return cmd


def build_xtec_s_command(
    input_path: str,
    output_dir: Path,
    cfg: AgenticWorkflowConfig,
    n_clusters: int,
    init_strategy: str,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        "xtec-s",
        input_path,
        "-o",
        str(output_dir),
    ]
    _append_common_args(cmd, cfg)
    cmd.extend(["-n", str(n_clusters), "--init-strategy-mode", init_strategy])
    return cmd
