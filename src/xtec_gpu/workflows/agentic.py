"""Agentic workflow orchestration for XTEC mode/cluster recommendation.

Owns:
- BIC sweep orchestration
- workflow recommendation assembly
- workflow report generation

Does not own:
- core clustering math (see `xtec_gpu.GMM` / `xtec_gpu.Preprocessing`)
- CLI command implementations (see `xtec_gpu.xtec_cli`)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from xtec_gpu import xtec_cli
from xtec_gpu.config.run_config import AgenticWorkflowConfig
from xtec_gpu.workflows.logging_utils import get_logger
from xtec_gpu.workflows.shared import WORKFLOW_REPORT_REQUIRED_KEYS
from xtec_gpu.workflows.shared import write_json
from xtec_gpu.workflows.shared import (
    build_bic_command,
    build_xtec_d_command,
    build_xtec_s_command,
)
from xtec_gpu.workflows.shared import run_checked
from xtec_gpu.workflows.types import WorkflowReport


logger = get_logger(__name__)


def _base_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    if extra_env:
        env.update(extra_env)
    return env


def _load_bic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        ks = f["n_clusters"][...].astype(int)
        bics = f["bic_scores"][...].astype(float)
    return ks, bics


def _run_bic(
    mode: str,
    cfg: AgenticWorkflowConfig,
    output_dir: Path,
    env: Dict[str, str],
    runtime_cache: Optional[Dict[object, object]] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_bic_command(mode=mode, input_path=cfg.input_path, output_dir=output_dir, cfg=cfg)
    if cfg.execution_backend == "inprocess":
        bic_args = argparse.Namespace(
            input=cfg.input_path,
            output=str(output_dir),
            entry=cfg.entry,
            slices=cfg.slices,
            threshold=bool(cfg.threshold),
            rescale=cfg.rescale,
            device=cfg.device,
            streamed_preprocess=bool(cfg.streamed_preprocess),
            streamed_chunk_voxels=int(cfg.streamed_chunk_voxels),
            streamed_reservoir_size=int(cfg.streamed_reservoir_size),
            streamed_max_bins=int(cfg.streamed_max_bins),
            streamed_exact_log_limit=int(cfg.streamed_exact_log_limit),
            streamed_seed=int(cfg.streamed_seed),
            min_nc=int(cfg.min_nc),
            max_nc=int(cfg.max_nc),
            runtime_cache=runtime_cache,
        )
        if mode == "d":
            xtec_cli.run_bic_d(bic_args)
        else:
            xtec_cli.run_bic_s(bic_args)
    else:
        run_checked(cmd, env=env)
    h5_name = "bic_xtec_d.h5" if mode == "d" else "bic_xtec_s.h5"
    ks, bics = _load_bic(output_dir / h5_name)
    best_idx = int(np.argmin(bics))
    return {
        "mode": mode,
        "command": cmd,
        "execution_backend": cfg.execution_backend,
        "n_clusters": ks.tolist(),
        "bic_scores": [float(x) for x in bics],
        "best_k": int(ks[best_idx]),
        "best_bic": float(bics[best_idx]),
    }


def _run_xtec_d_with_init(
    cfg: AgenticWorkflowConfig,
    n_clusters: int,
    init_strategy: str,
    out_dir: Path,
    env: Dict[str, str],
    runtime_cache: Optional[Dict[object, object]] = None,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_xtec_d_command(
        input_path=cfg.input_path,
        output_dir=out_dir,
        cfg=cfg,
        n_clusters=n_clusters,
        init_strategy=init_strategy,
        random_state=cfg.random_state,
    )
    if cfg.execution_backend == "inprocess":
        run_args = argparse.Namespace(
            input=cfg.input_path,
            output=str(out_dir),
            entry=cfg.entry,
            slices=cfg.slices,
            threshold=bool(cfg.threshold),
            rescale=cfg.rescale,
            device=cfg.device,
            streamed_preprocess=bool(cfg.streamed_preprocess),
            streamed_chunk_voxels=int(cfg.streamed_chunk_voxels),
            streamed_reservoir_size=int(cfg.streamed_reservoir_size),
            streamed_max_bins=int(cfg.streamed_max_bins),
            streamed_exact_log_limit=int(cfg.streamed_exact_log_limit),
            streamed_seed=int(cfg.streamed_seed),
            n_clusters=int(n_clusters),
            random_state=int(cfg.random_state),
            solver_mode="torchgmm",
            init_strategy_mode=str(init_strategy),
            post_stepwise_epochs=0,
            post_stepwise_tol=None,
            batch_num=1,
            max_batch_epoch=50,
            max_full_epoch=500,
            reorder_clusters=True,
            runtime_cache=runtime_cache,
        )
        xtec_cli.run_xtec_d(run_args)
    else:
        run_checked(cmd, env=env)
    return cmd


def _run_xtec_s(
    cfg: AgenticWorkflowConfig,
    n_clusters: int,
    init_strategy: str,
    out_dir: Path,
    env: Dict[str, str],
    runtime_cache: Optional[Dict[object, object]] = None,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_xtec_s_command(
        input_path=cfg.input_path,
        output_dir=out_dir,
        cfg=cfg,
        n_clusters=n_clusters,
        init_strategy=init_strategy,
    )
    if cfg.execution_backend == "inprocess":
        run_args = argparse.Namespace(
            input=cfg.input_path,
            output=str(out_dir),
            entry=cfg.entry,
            slices=cfg.slices,
            threshold=bool(cfg.threshold),
            rescale=cfg.rescale,
            device=cfg.device,
            streamed_preprocess=bool(cfg.streamed_preprocess),
            streamed_chunk_voxels=int(cfg.streamed_chunk_voxels),
            streamed_reservoir_size=int(cfg.streamed_reservoir_size),
            streamed_max_bins=int(cfg.streamed_max_bins),
            streamed_exact_log_limit=int(cfg.streamed_exact_log_limit),
            streamed_seed=int(cfg.streamed_seed),
            n_clusters=int(n_clusters),
            random_state=None,
            solver_mode="torchgmm",
            init_strategy_mode=str(init_strategy),
            post_stepwise_epochs=0,
            post_stepwise_tol=None,
            batch_num=1,
            max_batch_epoch=50,
            max_full_epoch=500,
            reorder_clusters=True,
            runtime_cache=runtime_cache,
        )
        xtec_cli.run_xtec_s(run_args)
    else:
        run_checked(cmd, env=env)
    return cmd


def _run_sweep_artifacts_for_mode(
    mode: str,
    cfg: AgenticWorkflowConfig,
    ks: Sequence[int],
    sweep_root: Path,
    env: Dict[str, str],
    runtime_cache: Optional[Dict[object, object]] = None,
) -> List[Dict[str, object]]:
    artifacts: List[Dict[str, object]] = []
    for k in ks:
        run_dir = sweep_root / f"{mode}_k{int(k):02d}"
        if mode == "d":
            cmd = _run_xtec_d_with_init(
                cfg=cfg,
                n_clusters=int(k),
                init_strategy=cfg.init_strategy_mode,
                out_dir=run_dir,
                env=env,
                runtime_cache=runtime_cache,
            )
        else:
            cmd = _run_xtec_s(
                cfg=cfg,
                n_clusters=int(k),
                init_strategy=cfg.init_strategy_mode,
                out_dir=run_dir,
                env=env,
                runtime_cache=runtime_cache,
            )
        artifacts.append(
            {
                "k": int(k),
                "output_dir": str(run_dir),
                "command": cmd,
                "expected_files": [
                    "results.h5",
                    "qmap.png",
                    "trajectories.png",
                    "avg_intensities.png",
                ],
            }
        )
    return artifacts


def _recommend_mode(bic_results: Dict[str, Dict[str, object]]) -> str:
    # Note: absolute BIC across d/s is only a heuristic because sample spaces differ.
    d_bic = bic_results.get("d", {}).get("best_bic")
    s_bic = bic_results.get("s", {}).get("best_bic")
    if d_bic is None and s_bic is None:
        raise RuntimeError("No BIC results available.")
    if d_bic is None:
        return "s"
    if s_bic is None:
        return "d"
    return "d" if float(d_bic) <= float(s_bic) else "s"


def recommend_workflow(
    cfg: AgenticWorkflowConfig, extra_env: Optional[Dict[str, str]] = None
) -> WorkflowReport:
    logger.info("Running agentic workflow for input=%s", cfg.input_path)
    env = _base_env(extra_env)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    bic_root = cfg.output_root / "bic_sweeps"
    sweep_artifacts_root = cfg.output_root / "sweep_artifacts"
    final_root = cfg.output_root / "final_run"
    bic_root.mkdir(parents=True, exist_ok=True)
    runtime_cache: Optional[Dict[object, object]] = {} if cfg.execution_backend == "inprocess" else None

    bic_results: Dict[str, Dict[str, object]] = {}
    sweep_artifacts: Dict[str, List[Dict[str, object]]] = {}
    for mode in cfg.candidate_modes:
        mode = mode.strip().lower()
        if mode not in {"d", "s"}:
            continue
        mode_bic = _run_bic(mode, cfg, bic_root / f"bic_{mode}", env, runtime_cache=runtime_cache)
        bic_results[mode] = mode_bic
        if cfg.save_sweep_artifacts:
            sweep_artifacts[mode] = _run_sweep_artifacts_for_mode(
                mode=mode,
                cfg=cfg,
                ks=[int(x) for x in mode_bic["n_clusters"]],
                sweep_root=sweep_artifacts_root / mode,
                env=env,
                runtime_cache=runtime_cache,
            )

    recommended_mode = _recommend_mode(bic_results)
    recommended_init = cfg.init_strategy_mode if "d" in bic_results else None

    final_cmd: Optional[List[str]] = None
    if cfg.run_final:
        final_root.mkdir(parents=True, exist_ok=True)
        if recommended_mode == "d":
            final_cmd = _run_xtec_d_with_init(
                cfg=cfg,
                n_clusters=int(bic_results["d"]["best_k"]),
                init_strategy=recommended_init or cfg.init_strategy_mode,
                out_dir=final_root / "xtec_d",
                env=env,
                runtime_cache=runtime_cache,
            )
        else:
            cmd = _run_xtec_s(
                cfg=cfg,
                n_clusters=int(bic_results["s"]["best_k"]),
                init_strategy=cfg.init_strategy_mode,
                out_dir=final_root / "xtec_s",
                env=env,
                runtime_cache=runtime_cache,
            )
            final_cmd = cmd

    report: WorkflowReport = {
        "input": cfg.input_path,
        "output_root": str(cfg.output_root),
        "settings": {
            "entry": cfg.entry,
            "slices": cfg.slices,
            "threshold": cfg.threshold,
            "rescale": cfg.rescale,
            "device": cfg.device,
            "min_nc": cfg.min_nc,
            "max_nc": cfg.max_nc,
            "candidate_modes": list(cfg.candidate_modes),
            "init_strategy_mode": cfg.init_strategy_mode,
            "random_state": cfg.random_state,
            "execution_backend": cfg.execution_backend,
            "streamed_preprocess": cfg.streamed_preprocess,
            "streamed_chunk_voxels": cfg.streamed_chunk_voxels,
            "streamed_reservoir_size": cfg.streamed_reservoir_size,
            "streamed_max_bins": cfg.streamed_max_bins,
            "streamed_exact_log_limit": cfg.streamed_exact_log_limit,
            "streamed_seed": cfg.streamed_seed,
        },
        "bic_results": bic_results,
        "sweep_artifacts": sweep_artifacts if cfg.save_sweep_artifacts else None,
        "recommendation": {
            "mode": recommended_mode,
            "n_clusters": int(bic_results[recommended_mode]["best_k"]),
            "init_strategy_mode": recommended_init if recommended_mode == "d" else None,
        },
        "final_command": final_cmd,
    }

    for key in WORKFLOW_REPORT_REQUIRED_KEYS:
        if key not in report:
            raise RuntimeError(f"Workflow report missing required key: {key}")

    write_json(cfg.output_root / "workflow_report.json", report)
    logger.info("Workflow report written to %s", cfg.output_root / "workflow_report.json")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic XTEC workflow runner.")
    p.add_argument("input", help="Path to input .nxs")
    p.add_argument("-o", "--output-root", required=True, help="Workflow output directory")
    p.add_argument("--entry", default="entry/data")
    p.add_argument("--slices", default=None)
    p.add_argument("--threshold", action="store_true", default=True)
    p.add_argument("--no-threshold", dest="threshold", action="store_false")
    p.add_argument("--rescale", choices=["mean", "z-score", "log-mean", "None"], default="mean")
    p.add_argument("--device", default="auto")
    p.add_argument("--streamed-preprocess", action="store_true", default=False)
    p.add_argument("--streamed-chunk-voxels", type=int, default=0)
    p.add_argument("--streamed-reservoir-size", type=int, default=500000)
    p.add_argument("--streamed-max-bins", type=int, default=4096)
    p.add_argument("--streamed-exact-log-limit", type=int, default=20000000)
    p.add_argument("--streamed-seed", type=int, default=0)
    p.add_argument("--min-nc", type=int, default=2)
    p.add_argument("--max-nc", type=int, default=14)
    p.add_argument("--candidate-modes", default="d,s",
                   help="Comma-separated modes to evaluate: d,s")
    p.add_argument(
        "--init-strategy-mode",
        choices=["kmeans++", "xtec", "sklearn-kmeans", "cuml-kmeans"],
        default="kmeans++",
        help="Initialization strategy passed to xtec-d/xtec-s runs (default: kmeans++).",
    )
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--no-run-final", dest="run_final", action="store_false", default=True,
                   help="Only compute recommendation, do not execute final command")
    p.add_argument(
        "--no-save-sweep-artifacts",
        dest="save_sweep_artifacts",
        action="store_false",
        default=True,
        help="Skip per-k sweep artifact runs (results.h5 + trajectories/qmap/avg_intensities).",
    )
    p.add_argument(
        "--execution-backend",
        choices=["inprocess", "subprocess"],
        default="inprocess",
        help="Run workflow commands in-process for lower overhead, or via subprocess for strict process isolation (default: inprocess).",
    )
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> AgenticWorkflowConfig:
    return AgenticWorkflowConfig(
        input_path=args.input,
        output_root=Path(args.output_root),
        entry=args.entry,
        slices=args.slices,
        threshold=bool(args.threshold),
        rescale=args.rescale,
        device=args.device,
        streamed_preprocess=bool(args.streamed_preprocess),
        streamed_chunk_voxels=int(args.streamed_chunk_voxels),
        streamed_reservoir_size=int(args.streamed_reservoir_size),
        streamed_max_bins=int(args.streamed_max_bins),
        streamed_exact_log_limit=int(args.streamed_exact_log_limit),
        streamed_seed=int(args.streamed_seed),
        min_nc=int(args.min_nc),
        max_nc=int(args.max_nc),
        candidate_modes=[x.strip() for x in args.candidate_modes.split(",") if x.strip()],
        random_state=int(args.random_state),
        run_final=bool(args.run_final),
        save_sweep_artifacts=bool(args.save_sweep_artifacts),
        init_strategy_mode=str(args.init_strategy_mode),
        execution_backend=str(args.execution_backend),
    )


def main() -> None:
    args = _parse_args()
    cfg = config_from_args(args)
    report = recommend_workflow(cfg)
    print(json.dumps(report["recommendation"], indent=2))
    print(f"Report: {cfg.output_root / 'workflow_report.json'}")


__all__ = ["AgenticWorkflowConfig", "recommend_workflow", "config_from_args", "main"]
