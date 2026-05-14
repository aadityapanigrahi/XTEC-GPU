"""Full-sweep orchestrator: exhaustive (mode, k, init) sweep + judge.

Design rationale
----------------

1. **BIC-then-artifact dedup**: the existing ``run_bic_d`` / ``run_bic_s``
   already fit a full GMM at every ``k``. Standalone ``xtec-d``/``xtec-s``
   commands then re-fit the same GMM solely to dump ``results.h5`` + plots.
   This orchestrator passes a ``save_artifacts_inline`` payload through
   ``argparse.Namespace`` so the BIC loop saves artifacts on the already-fitted
   model. One GMM fit per ``(mode, k, init)`` — half the clustering work of
   the naive sweep.

2. **One shared ``runtime_cache``**: a single dict is threaded through every
   sub-run via ``args.runtime_cache``. Data loading, masking, thresholding,
   and peak-averaging are computed once. Mode ``d`` runs first so the threshold
   tensor it builds is reused by all ``s`` runs.

3. **Inprocess only**: no subprocess option exposed — subprocess execution
   forfeits the cache and the dedup, so it's not offered here.

4. **All artifacts stored**: per the user's stated preference (memory not a
   constraint), every combination's ``results.h5`` + plots + metrics + timing
   are kept.

5. **Offline judge**: scoring is delegated to ``xtec_gpu.workflows.judge``,
   which reads ``results.h5`` files and writes ``recommendation.json``. Re-run
   the judge separately via ``full-sweep-judge`` to iterate on metric weights
   without re-clustering.

Public entry points
-------------------
- ``run_full_sweep(cfg) -> dict``    — programmatic
- ``main()``                          — CLI entry (registered as ``full-sweep``)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from xtec_gpu.workflows.logging_utils import get_logger
from xtec_gpu.workflows.sweep_types import (
    CPU_INITS,
    GPU_INITS,
    FullSweepConfig,
    MANIFEST_REQUIRED_KEYS,
    SWEEP_SUMMARY_REQUIRED_KEYS,
    init_slug,
    run_id,
)


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _input_hash(path: str, max_bytes: int = 10 * 1024 ** 3) -> Dict[str, Any]:
    """Return a small fingerprint of the input file.

    Hashes the file content if the size is <= ``max_bytes``; otherwise records
    size + mtime only. We only need to detect "did the input change between
    a run and a --resume" — not cryptographic integrity.
    """
    p = Path(path)
    size = p.stat().st_size if p.exists() else -1
    mtime = p.stat().st_mtime if p.exists() else 0.0
    out: Dict[str, Any] = {"size": int(size), "mtime": float(mtime)}
    if 0 <= size <= max_bytes and p.exists():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out


def _serialize_config(cfg: FullSweepConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d["output_root"] = str(cfg.output_root)
    return d


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def _resolve_inits(cfg: FullSweepConfig) -> List[str]:
    inits = list(cfg.inits) if cfg.inits else list(GPU_INITS)
    if cfg.include_cpu_inits:
        for ci in CPU_INITS:
            if ci not in inits:
                inits.append(ci)
    return inits


def _resolve_stream_mode(cfg: FullSweepConfig) -> bool:
    if cfg.stream_mode == "on":
        return True
    if cfg.stream_mode == "off":
        return False
    # auto
    try:
        size_gb = Path(cfg.input_path).stat().st_size / 1e9
    except OSError:
        size_gb = 0.0
    return bool(size_gb >= cfg.stream_threshold_gb)


def _check_gpu_memory(device: torch.device, min_free_gb: float = 1.0) -> None:
    if device.type != "cuda":
        return
    try:
        free, _total = torch.cuda.mem_get_info(device.index or 0)
    except Exception:
        return
    if free < min_free_gb * (1024 ** 3):
        raise RuntimeError(
            f"Only {free / 1e9:.2f} GB free on {device}. Need at least "
            f"{min_free_gb:.1f} GB. Try --slices to reduce ROI, run on a "
            f"less-loaded GPU, or use --device cpu."
        )


def _build_plan(cfg: FullSweepConfig, inits: List[str]) -> List[Dict[str, Any]]:
    """Enumerate (mode, init) groups, ordered d-before-s for cache reuse."""
    plan: List[Dict[str, Any]] = []
    mode_order = ["d", "s"]
    for mode in mode_order:
        if mode not in cfg.modes:
            continue
        for init in inits:
            plan.append(
                {
                    "mode": mode,
                    "init": init,
                    "ks": list(range(int(cfg.min_nc), int(cfg.max_nc))),
                }
            )
    return plan


# ---------------------------------------------------------------------------
# Sub-run dispatch
# ---------------------------------------------------------------------------

def _make_ns_for_bic(
    cfg: FullSweepConfig,
    mode: str,
    init: str,
    bic_output_dir: Path,
    runs_root: Path,
    streamed: bool,
    runtime_cache: Dict[Any, Any],
) -> argparse.Namespace:
    """Construct the argparse.Namespace consumed by run_bic_d / run_bic_s."""
    save_inline = {
        "runs_root": runs_root,
        "mode": mode,
        "init": init,
        "plots_level": cfg.plots_level,
        "reorder_clusters": bool(cfg.reorder_clusters),
        "random_state": int(cfg.random_state),
        "rescale": cfg.rescale,
    }
    return argparse.Namespace(
        input=cfg.input_path,
        output=str(bic_output_dir),
        entry=cfg.entry,
        slices=cfg.slices,
        threshold=bool(cfg.threshold),
        rescale=cfg.rescale,
        device=cfg.device,
        streamed_preprocess=bool(streamed),
        streamed_chunk_voxels=int(cfg.streamed_chunk_voxels),
        streamed_reservoir_size=int(cfg.streamed_reservoir_size),
        streamed_max_bins=int(cfg.streamed_max_bins),
        streamed_exact_log_limit=int(cfg.streamed_exact_log_limit),
        streamed_seed=int(cfg.streamed_seed),
        min_nc=int(cfg.min_nc),
        max_nc=int(cfg.max_nc),
        runtime_cache=runtime_cache,
        save_artifacts_inline=save_inline,
    )


def _resume_filter(
    plan_entry: Dict[str, Any],
    runs_root: Path,
) -> List[int]:
    """Return the ks in this plan entry that still need to run under --resume."""
    todo: List[int] = []
    for k in plan_entry["ks"]:
        rid = run_id(plan_entry["mode"], k, plan_entry["init"])
        results_path = runs_root / rid / "results.h5"
        if not results_path.exists():
            todo.append(k)
    return todo


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_full_sweep(cfg: FullSweepConfig) -> Dict[str, Any]:
    """Run the full sweep and return the recommendation payload."""
    # Local import to avoid circular import at module load time.
    from xtec_gpu import xtec_cli
    from xtec_gpu.workflows import judge

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    bic_root = output_root / "bic_curves"
    bic_root.mkdir(parents=True, exist_ok=True)

    # 1. Resolve plan
    inits = _resolve_inits(cfg)
    streamed = _resolve_stream_mode(cfg)
    plan = _build_plan(cfg, inits)

    # 2. Device + memory pre-flight
    device = xtec_cli._get_device(cfg.device)
    _check_gpu_memory(device)

    # 3. Manifest
    started_at = time.time()
    manifest: Dict[str, Any] = {
        "input": cfg.input_path,
        "output_root": str(output_root),
        "config": _serialize_config(cfg),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "plan": plan,
        "resolved": {
            "inits": inits,
            "streamed_preprocess": streamed,
        },
        "input_fingerprint": _input_hash(cfg.input_path),
        "started_at": started_at,
        "finished_at": None,
    }
    for key in MANIFEST_REQUIRED_KEYS:
        if key not in manifest:
            raise RuntimeError(f"Manifest missing required key: {key}")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if cfg.dry_run:
        logger.info("Dry run — plan written, no clustering performed.")
        total = sum(len(p["ks"]) for p in plan)
        print(json.dumps(
            {
                "dry_run": True,
                "total_combos": total,
                "plan": plan,
                "resolved_inits": inits,
                "streamed_preprocess": streamed,
                "output_root": str(output_root),
            },
            indent=2,
        ))
        return {"dry_run": True, "manifest": manifest}

    # 4. Execute plan, one shared runtime_cache
    runtime_cache: Dict[Any, Any] = {}
    candidates: List[Dict[str, Any]] = []

    for entry in plan:
        mode = entry["mode"]
        init = entry["init"]
        ks = entry["ks"]
        if cfg.resume:
            ks = _resume_filter(entry, runs_root)
            if not ks:
                logger.info(
                    "Resume: all k for mode=%s init=%s already complete, skipping",
                    mode, init,
                )
                # Still need to record the existing candidates for the judge.
                for k in entry["ks"]:
                    rid = run_id(mode, k, init)
                    candidates.append(
                        _candidate_from_existing(rid, mode, k, init, runs_root)
                    )
                continue

        bic_dir = bic_root / f"{mode}_{init_slug(init)}"
        bic_dir.mkdir(parents=True, exist_ok=True)

        ns = _make_ns_for_bic(
            cfg=cfg,
            mode=mode,
            init=init,
            bic_output_dir=bic_dir,
            runs_root=runs_root,
            streamed=streamed,
            runtime_cache=runtime_cache,
        )
        # Override the k-range only if --resume narrowed it.
        if cfg.resume and ks != entry["ks"]:
            ns.min_nc = int(min(ks))
            ns.max_nc = int(max(ks)) + 1

        logger.info(
            "Running BIC sweep: mode=%s init=%s ks=%s", mode, init, ks,
        )
        if mode == "d":
            result = xtec_cli.run_bic_d(ns)
        else:
            result = xtec_cli.run_bic_s(ns)

        # When save_artifacts_inline is set, the patched BIC functions return
        # {"ks": [...], "bics": [...], "timings": {k: {...}}}
        if isinstance(result, dict) and "ks" in result:
            for k in result["ks"]:
                rid = run_id(mode, int(k), init)
                timing = result["timings"].get(int(k), {})
                candidates.append(
                    {
                        "id": rid,
                        "mode": mode,
                        "k": int(k),
                        "init": init,
                        "output_dir": str(Path("runs") / rid),
                        "bic": float(timing.get("bic")) if timing.get("bic") is not None else None,
                        "wall_s": float(timing.get("wall_s", 0.0)),
                        "cluster_sizes": timing.get("cluster_sizes", []),
                    }
                )
        # If --resume skipped some ks, also harvest the existing ones we
        # didn't re-run.
        if cfg.resume:
            done_now = set(int(k) for k in (result.get("ks", []) if isinstance(result, dict) else []))
            for k in entry["ks"]:
                if int(k) in done_now:
                    continue
                rid = run_id(mode, int(k), init)
                candidates.append(_candidate_from_existing(rid, mode, int(k), init, runs_root))

    # 5. Sweep summary
    summary = {
        "candidates": candidates,
        "manifest_ref": "manifest.json",
    }
    for key in SWEEP_SUMMARY_REQUIRED_KEYS:
        if key not in summary:
            raise RuntimeError(f"sweep_summary.json missing required key: {key}")
    (output_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # 6. Judge
    recommendation = judge.score_and_recommend(
        output_root=output_root,
        weights=cfg.judge_weights,
        top_n=int(cfg.judge_top_n),
    )

    # 7. (final_run symlink is now created by judge.score_and_recommend so
    #     standalone re-judges keep the directory consistent.)

    # 8. Finish manifest
    manifest["finished_at"] = time.time()
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Sweep complete. Winner: %s", recommendation["winner"]["id"])
    return recommendation


def replot_existing_sweep(
    output_root: Path,
    plots_level: str = "all",
) -> int:
    """Regenerate plots for every run in an existing sweep, from ``results.h5``.

    No GMM is run. Used when plot code changes (e.g. the discrete-color qmap)
    and you want to refresh artifacts without re-clustering.

    Parameters
    ----------
    output_root : Path
        Sweep directory containing ``manifest.json`` and ``runs/<id>/results.h5``.
    plots_level : {"all", "primary", "none"}
        ``"none"`` is a no-op; ``"all"`` and ``"primary"`` both regenerate the
        standard three plots (qmap, trajectories, avg_intensities) — the
        distinction is mainly meaningful at sweep time, not at replot time.

    Returns
    -------
    int
        Number of run directories successfully replotted.
    """
    # Local imports to keep the workflow module light at import time.
    import h5py
    from xtec_gpu.xtec_cli import (
        _load_data,
        _plot_avg_intensities,
        _plot_qmap,
        _plot_trajectories,
    )

    output_root = Path(output_root)
    if plots_level == "none":
        logger.info("plots=none: nothing to do")
        return 0

    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}; cannot replot."
        )
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest.get("config", {})

    logger.info(
        "Replot: loading source NXdata from %s (entry=%r, slices=%r)",
        cfg.get("input_path"), cfg.get("entry"), cfg.get("slices"),
    )
    data = _load_data(
        cfg["input_path"], cfg.get("entry", "entry/data"), cfg.get("slices"),
    )
    rescale = cfg.get("rescale", "mean")

    runs_root = output_root / "runs"
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    replotted = 0
    for rd in run_dirs:
        results_path = rd / "results.h5"
        if not results_path.exists():
            logger.warning("skip %s: no results.h5", rd.name)
            continue
        with h5py.File(results_path, "r") as f:
            pixel_assigns = f["pixel_assignments"][...]
            data_indices = f["data_indices"][...]
            data_thresh = f["data_thresholded"][...]
            cluster_means = f["cluster_means"][...]
            cluster_covs = f["cluster_covariances"][...]
            cluster_assigns = f["cluster_assignments"][...]
        nc = int(cluster_means.shape[0])

        # If this run was recorded by a sweep that captured the chosen slice
        # in timing.json["qmap"], honor that choice so a replot reproduces
        # the original view exactly. Otherwise let _plot_qmap pick the
        # densest slice.
        slice_idx = None
        slice_ax = 0
        timing_path = rd / "timing.json"
        if timing_path.exists():
            try:
                qmap_info = json.loads(timing_path.read_text()).get("qmap") or {}
                slice_idx = qmap_info.get("slice_index")
                if qmap_info.get("slice_axis") is not None:
                    slice_ax = int(qmap_info["slice_axis"])
            except Exception:
                pass

        new_qmap_info = _plot_qmap(
            data, data_indices, pixel_assigns, nc, str(rd),
            slice_index=slice_idx, slice_axis=slice_ax,
        )
        # Refresh the qmap entry in timing.json (counts may differ if a
        # newer _plot_qmap reports more fields).
        if timing_path.exists() and isinstance(new_qmap_info, dict):
            try:
                payload = json.loads(timing_path.read_text())
                payload["qmap"] = new_qmap_info
                timing_path.write_text(json.dumps(payload, indent=2))
            except Exception:
                pass
        _plot_trajectories(
            data, cluster_means, cluster_covs, nc, rescale, str(rd),
        )
        # avg_intensities needs per-pixel labels aligned with data_thresh
        # columns. d-mode/label-smooth: both arrays equal. s-mode:
        # cluster_assignments is per-peak; pixel_assignments is per-pixel.
        labels = (
            pixel_assigns
            if pixel_assigns.shape[0] == data_thresh.shape[1]
            else cluster_assigns
        )
        _plot_avg_intensities(data, data_thresh, labels, nc, str(rd))
        replotted += 1
        print(f"  replotted {rd.name}")

    return replotted


def _candidate_from_existing(
    rid: str, mode: str, k: int, init: str, runs_root: Path,
) -> Dict[str, Any]:
    """Reconstruct a candidate dict from an already-complete run directory."""
    timing_path = runs_root / rid / "timing.json"
    bic = None
    wall_s = 0.0
    cluster_sizes: List[int] = []
    if timing_path.exists():
        try:
            t = json.loads(timing_path.read_text())
            bic = t.get("bic")
            wall_s = float(t.get("wall_s", 0.0))
            cluster_sizes = t.get("cluster_sizes", [])
        except Exception:
            pass
    return {
        "id": rid,
        "mode": mode,
        "k": int(k),
        "init": init,
        "output_dir": str(Path("runs") / rid),
        "bic": float(bic) if bic is not None else None,
        "wall_s": wall_s,
        "cluster_sizes": cluster_sizes,
    }


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def add_full_sweep_arguments(p: argparse.ArgumentParser) -> None:
    """Register the full-sweep CLI flags on an existing parser.

    Kept as a separate function so ``xtec_cli.build_parser`` can call it for
    the ``full-sweep`` subcommand without dragging the rest of the module into
    its argparse code.
    """
    p.add_argument("input", help="Path to the input .nxs file")
    p.add_argument("-o", "--output-root", required=True, dest="output_root",
                   help="Output directory for the sweep")
    p.add_argument("--entry", default="entry/data",
                   help="HDF5 path inside the input file (default: entry/data)")
    p.add_argument("--slices", default=None,
                   help="Slice string, e.g. ':,0.0:1.0,-10:10,-15:15'")
    p.add_argument("--threshold", action="store_true", default=True,
                   help="KL background thresholding (default: on)")
    p.add_argument("--no-threshold", dest="threshold", action="store_false")
    p.add_argument("--rescale",
                   choices=["mean", "z-score", "log-mean", "None"],
                   default="mean")
    p.add_argument("--device", default="auto", type=str)

    p.add_argument("--min-nc", type=int, default=2)
    p.add_argument("--max-nc", type=int, default=8,
                   help="Exclusive upper bound (matches np.arange)")
    p.add_argument("--modes", default="d,s",
                   help="Comma-separated modes to sweep (default: d,s)")
    p.add_argument("--inits", default="kmeans++",
                   help="Comma-separated GPU-resident inits (default: kmeans++)")
    p.add_argument("--include-cpu-inits", action="store_true", default=False,
                   help="Also sweep sklearn-kmeans and xtec (slow, CPU)")

    p.add_argument("--plots", choices=["all", "primary", "none"], default="all",
                   dest="plots_level")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--reorder-clusters", dest="reorder_clusters",
                   action="store_true", default=True)
    p.add_argument("--no-reorder-clusters", dest="reorder_clusters",
                   action="store_false")

    p.add_argument("--stream", choices=["auto", "on", "off"], default="auto",
                   dest="stream_mode")
    p.add_argument("--stream-threshold-gb", type=float, default=2.0)
    p.add_argument("--streamed-chunk-voxels", type=int, default=0)
    p.add_argument("--streamed-reservoir-size", type=int, default=500000)
    p.add_argument("--streamed-max-bins", type=int, default=4096)
    p.add_argument("--streamed-exact-log-limit", type=int, default=50000000)
    p.add_argument("--streamed-seed", type=int, default=0)

    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument(
        "--replot",
        action="store_true",
        default=False,
        help=("Regenerate plots in-place for every run under --output-root "
              "from existing results.h5 files, without re-clustering. "
              "Useful after plot code changes."),
    )

    p.add_argument("--judge-weights", default=None,
                   help="Path to JSON file with weight overrides")
    p.add_argument("--judge-top-n", type=int, default=5)


def config_from_args(args: argparse.Namespace) -> FullSweepConfig:
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    inits = [i.strip() for i in args.inits.split(",") if i.strip()]
    weights: Optional[Dict[str, float]] = None
    if getattr(args, "judge_weights", None):
        weights = json.loads(Path(args.judge_weights).read_text())
    return FullSweepConfig(
        input_path=args.input,
        output_root=Path(args.output_root),
        entry=args.entry,
        slices=args.slices,
        threshold=bool(args.threshold),
        rescale=args.rescale,
        device=args.device,
        min_nc=int(args.min_nc),
        max_nc=int(args.max_nc),
        modes=modes,
        inits=inits,
        include_cpu_inits=bool(args.include_cpu_inits),
        plots_level=str(args.plots_level),
        random_state=int(args.random_state),
        reorder_clusters=bool(args.reorder_clusters),
        stream_mode=str(args.stream_mode),
        stream_threshold_gb=float(args.stream_threshold_gb),
        streamed_chunk_voxels=int(args.streamed_chunk_voxels),
        streamed_reservoir_size=int(args.streamed_reservoir_size),
        streamed_max_bins=int(args.streamed_max_bins),
        streamed_exact_log_limit=int(args.streamed_exact_log_limit),
        streamed_seed=int(args.streamed_seed),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        judge_weights=weights,
        judge_top_n=int(args.judge_top_n),
    )


def run_full_sweep_cli(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    if getattr(args, "replot", False):
        n = replot_existing_sweep(cfg.output_root, plots_level=cfg.plots_level)
        print(f"Replotted {n} runs under {cfg.output_root}")
        return
    result = run_full_sweep(cfg)
    if cfg.dry_run:
        return
    print(json.dumps(result["winner"], indent=2))
    print(f"Report: {cfg.output_root / 'recommendation.json'}")


def run_judge_cli(args: argparse.Namespace) -> None:
    from xtec_gpu.workflows import judge
    weights: Optional[Dict[str, float]] = None
    if getattr(args, "judge_weights", None):
        weights = json.loads(Path(args.judge_weights).read_text())
    payload = judge.score_and_recommend(
        output_root=Path(args.output_root),
        weights=weights,
        top_n=int(args.judge_top_n),
    )
    print(json.dumps(payload["winner"], indent=2))


def run_inspect_cli(args: argparse.Namespace) -> None:
    from xtec_gpu.workflows import judge
    judge.inspect(Path(args.output_root), top=int(args.top))


def main() -> None:
    """Convenience entry point for `python -m xtec_gpu.workflows.sweep`."""
    parser = argparse.ArgumentParser(
        prog="xtec-gpu full-sweep",
        description="Exhaustive (mode, k, init) sweep with agent-judged recommendation.",
    )
    add_full_sweep_arguments(parser)
    args = parser.parse_args()
    run_full_sweep_cli(args)


__all__ = [
    "run_full_sweep",
    "replot_existing_sweep",
    "config_from_args",
    "add_full_sweep_arguments",
    "run_full_sweep_cli",
    "run_judge_cli",
    "run_inspect_cli",
    "main",
]
