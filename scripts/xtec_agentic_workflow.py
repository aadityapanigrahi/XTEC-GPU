#!/usr/bin/env python3
"""
Agentic workflow for selecting XTEC mode/cluster count.

This script can:
1) run BIC sweeps for XTEC-d and/or XTEC-s
2) pick the best cluster count per mode
3) recommend a final command and execute it
4) save a JSON report for agent/tooling consumption
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass
class RunConfig:
    input_path: str
    output_root: Path
    entry: str
    slices: Optional[str]
    threshold: bool
    rescale: str
    device: str
    min_nc: int
    max_nc: int
    candidate_modes: Sequence[str]
    random_state: int
    run_final: bool
    save_sweep_artifacts: bool


def _base_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    if extra_env:
        env.update(extra_env)
    return env


def _run(cmd: Sequence[str], env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(list(cmd), check=True, env=env)


def _load_bic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        ks = f["n_clusters"][...].astype(int)
        bics = f["bic_scores"][...].astype(float)
    return ks, bics


def _run_bic(mode: str, cfg: RunConfig, output_dir: Path, env: Dict[str, str]) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        f"bic-{mode}",
        cfg.input_path,
        "-o",
        str(output_dir),
        "--entry",
        cfg.entry,
        "--rescale",
        cfg.rescale,
        "--device",
        cfg.device,
        "--min-nc",
        str(cfg.min_nc),
        "--max-nc",
        str(cfg.max_nc),
    ]
    if cfg.slices:
        cmd.extend(["--slices", cfg.slices])
    if not cfg.threshold:
        cmd.append("--no-threshold")

    _run(cmd, env=env)
    h5_name = "bic_xtec_d.h5" if mode == "d" else "bic_xtec_s.h5"
    ks, bics = _load_bic(output_dir / h5_name)
    best_idx = int(np.argmin(bics))
    return {
        "mode": mode,
        "command": cmd,
        "n_clusters": ks.tolist(),
        "bic_scores": [float(x) for x in bics],
        "best_k": int(ks[best_idx]),
        "best_bic": float(bics[best_idx]),
    }


def _run_xtec_d_with_init(
    cfg: RunConfig,
    n_clusters: int,
    init_strategy: str,
    out_dir: Path,
    env: Dict[str, str],
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        "xtec-d",
        cfg.input_path,
        "-o",
        str(out_dir),
        "--entry",
        cfg.entry,
        "--rescale",
        cfg.rescale,
        "--device",
        cfg.device,
        "-n",
        str(n_clusters),
        "--init-strategy-mode",
        init_strategy,
        "--solver-mode",
        "torchgmm",
        "--post-stepwise-epochs",
        "0",
        "--random-state",
        str(cfg.random_state),
    ]
    if cfg.slices:
        cmd.extend(["--slices", cfg.slices])
    if not cfg.threshold:
        cmd.append("--no-threshold")
    _run(cmd, env=env)
    return cmd


def _run_xtec_s(
    cfg: RunConfig,
    n_clusters: int,
    out_dir: Path,
    env: Dict[str, str],
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        "xtec-s",
        cfg.input_path,
        "-o",
        str(out_dir),
        "--entry",
        cfg.entry,
        "--rescale",
        cfg.rescale,
        "--device",
        cfg.device,
        "-n",
        str(n_clusters),
    ]
    if cfg.slices:
        cmd.extend(["--slices", cfg.slices])
    if not cfg.threshold:
        cmd.append("--no-threshold")
    _run(cmd, env=env)
    return cmd


def _run_sweep_artifacts_for_mode(
    mode: str,
    cfg: RunConfig,
    ks: Sequence[int],
    sweep_root: Path,
    env: Dict[str, str],
) -> List[Dict[str, object]]:
    artifacts: List[Dict[str, object]] = []
    for k in ks:
        run_dir = sweep_root / f"{mode}_k{int(k):02d}"
        if mode == "d":
            cmd = _run_xtec_d_with_init(
                cfg=cfg,
                n_clusters=int(k),
                init_strategy="sklearn-kmeans",
                out_dir=run_dir,
                env=env,
            )
        else:
            cmd = _run_xtec_s(
                cfg=cfg,
                n_clusters=int(k),
                out_dir=run_dir,
                env=env,
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


def recommend_workflow(cfg: RunConfig, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    env = _base_env(extra_env)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    bic_root = cfg.output_root / "bic_sweeps"
    sweep_artifacts_root = cfg.output_root / "sweep_artifacts"
    final_root = cfg.output_root / "final_run"
    bic_root.mkdir(parents=True, exist_ok=True)

    bic_results: Dict[str, Dict[str, object]] = {}
    sweep_artifacts: Dict[str, List[Dict[str, object]]] = {}
    for mode in cfg.candidate_modes:
        mode = mode.strip().lower()
        if mode not in {"d", "s"}:
            continue
        mode_bic = _run_bic(mode, cfg, bic_root / f"bic_{mode}", env)
        bic_results[mode] = mode_bic
        if cfg.save_sweep_artifacts:
            sweep_artifacts[mode] = _run_sweep_artifacts_for_mode(
                mode=mode,
                cfg=cfg,
                ks=[int(x) for x in mode_bic["n_clusters"]],
                sweep_root=sweep_artifacts_root / mode,
                env=env,
            )

    recommended_mode = _recommend_mode(bic_results)
    recommended_init = "sklearn-kmeans" if "d" in bic_results else None

    final_cmd: Optional[List[str]] = None
    if cfg.run_final:
        final_root.mkdir(parents=True, exist_ok=True)
        if recommended_mode == "d":
            final_cmd = _run_xtec_d_with_init(
                cfg=cfg,
                n_clusters=int(bic_results["d"]["best_k"]),
                init_strategy=recommended_init or "sklearn-kmeans",
                out_dir=final_root / "xtec_d",
                env=env,
            )
        else:
            cmd = [
                sys.executable,
                "-m",
                "xtec_gpu.xtec_cli",
                "xtec-s",
                cfg.input_path,
                "-o",
                str(final_root / "xtec_s"),
                "--entry",
                cfg.entry,
                "--rescale",
                cfg.rescale,
                "--device",
                cfg.device,
                "-n",
                str(int(bic_results["s"]["best_k"])),
            ]
            if cfg.slices:
                cmd.extend(["--slices", cfg.slices])
            if not cfg.threshold:
                cmd.append("--no-threshold")
            _run(cmd, env=env)
            final_cmd = cmd

    report = {
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
            "random_state": cfg.random_state,
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
    report_path = cfg.output_root / "workflow_report.json"
    report_path.write_text(json.dumps(report, indent=2))
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
    p.add_argument("--min-nc", type=int, default=2)
    p.add_argument("--max-nc", type=int, default=14)
    p.add_argument("--candidate-modes", default="d,s",
                   help="Comma-separated modes to evaluate: d,s")
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = RunConfig(
        input_path=args.input,
        output_root=Path(args.output_root),
        entry=args.entry,
        slices=args.slices,
        threshold=bool(args.threshold),
        rescale=args.rescale,
        device=args.device,
        min_nc=int(args.min_nc),
        max_nc=int(args.max_nc),
        candidate_modes=[x.strip() for x in args.candidate_modes.split(",") if x.strip()],
        random_state=int(args.random_state),
        run_final=bool(args.run_final),
        save_sweep_artifacts=bool(args.save_sweep_artifacts),
    )
    report = recommend_workflow(cfg)
    print(json.dumps(report["recommendation"], indent=2))
    print(f"Report: {cfg.output_root / 'workflow_report.json'}")


if __name__ == "__main__":
    main()
