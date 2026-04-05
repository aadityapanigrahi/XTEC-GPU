#!/usr/bin/env python3
"""Profile agentic workflow with stage-level timing breakdown.

This script instruments xtec_cli + workflow entry points at runtime and writes:
- workflow outputs under --output-root
- timing JSON: <output-root>/timing_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from xtec_gpu import xtec_cli
from xtec_gpu.config.run_config import AgenticWorkflowConfig
from xtec_gpu.workflows import agentic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--entry", default="entry/data")
    p.add_argument("--slices", default=":,0.0:1.0,-10.0:10.0,-15.0:15.0")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--min-nc", type=int, default=2)
    p.add_argument("--max-nc", type=int, default=4)
    p.add_argument("--candidate-modes", default="d,s")
    p.add_argument("--init-strategy-mode", default="kmeans++")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--streamed-preprocess", action="store_true", default=False)
    p.add_argument("--streamed-chunk-voxels", type=int, default=200000)
    p.add_argument("--streamed-reservoir-size", type=int, default=500000)
    p.add_argument("--streamed-max-bins", type=int, default=4096)
    p.add_argument("--streamed-exact-log-limit", type=int, default=2000000)
    p.add_argument("--streamed-seed", type=int, default=0)
    p.add_argument("--execution-backend", choices=["inprocess", "subprocess"], default="inprocess")
    p.add_argument("--no-save-sweep-artifacts", action="store_true")
    p.add_argument("--no-run-final", action="store_true")
    args = p.parse_args()

    records = []
    ctx = ["workflow"]

    def record(name: str, dt: float) -> None:
        records.append({"name": name, "seconds": float(dt), "context": ctx[0]})

    def wrap_function(module, fn_name: str, stage_name: str) -> None:
        orig = getattr(module, fn_name)

        def wrapped(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                record(stage_name, time.perf_counter() - t0)

        setattr(module, fn_name, wrapped)

    def wrap_command(fn_name: str, command_name: str) -> None:
        orig = getattr(xtec_cli, fn_name)

        def wrapped(*a, **k):
            prev = ctx[0]
            ctx[0] = command_name
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                record(f"command_total:{command_name}", time.perf_counter() - t0)
                ctx[0] = prev

        setattr(xtec_cli, fn_name, wrapped)

    def wrap_workflow_step(fn_name: str, step_name: str) -> None:
        orig = getattr(agentic, fn_name)

        def wrapped(*a, **k):
            prev = ctx[0]
            ctx[0] = step_name
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                record(f"workflow_step_total:{step_name}", time.perf_counter() - t0)
                ctx[0] = prev

        setattr(agentic, fn_name, wrapped)

    wrap_function(xtec_cli, "_load_data", "load_data")
    wrap_function(xtec_cli, "Mask_Zeros", "preprocess:mask_zeros")
    wrap_function(xtec_cli, "Threshold_Background", "preprocess:threshold_background")
    wrap_function(xtec_cli, "Peak_averaging", "preprocess:peak_averaging")
    wrap_function(xtec_cli, "_rescale", "preprocess:rescale")
    wrap_function(xtec_cli, "_save_results", "io:save_results")
    wrap_function(xtec_cli, "_plot_qmap", "io:plot_qmap")
    wrap_function(xtec_cli, "_plot_trajectories", "io:plot_trajectories")
    wrap_function(xtec_cli, "_plot_avg_intensities", "io:plot_avg_intensities")
    wrap_function(xtec_cli, "_run_direct_gmm", "compute:run_direct_gmm_total")

    orig_run_em = xtec_cli.GMM.RunEM

    def run_em_wrapped(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig_run_em(self, *a, **k)
        finally:
            record("compute:gmm_runem", time.perf_counter() - t0)

    xtec_cli.GMM.RunEM = run_em_wrapped

    wrap_command("run_bic_d", "bic_d")
    wrap_command("run_bic_s", "bic_s")
    wrap_command("run_xtec_d", "xtec_d")
    wrap_command("run_xtec_s", "xtec_s")

    wrap_workflow_step("_run_bic", "workflow_run_bic")
    wrap_workflow_step("_run_sweep_artifacts_for_mode", "workflow_sweep_artifacts")
    wrap_workflow_step("_run_xtec_d_with_init", "workflow_final_or_sweep_xtec_d")
    wrap_workflow_step("_run_xtec_s", "workflow_final_or_sweep_xtec_s")

    cfg = AgenticWorkflowConfig(
        input_path=args.input,
        output_root=Path(args.output_root),
        entry=args.entry,
        slices=args.slices,
        threshold=True,
        rescale="mean",
        device=args.device,
        min_nc=int(args.min_nc),
        max_nc=int(args.max_nc),
        candidate_modes=[x.strip() for x in args.candidate_modes.split(",") if x.strip()],
        random_state=int(args.random_state),
        streamed_preprocess=bool(args.streamed_preprocess),
        streamed_chunk_voxels=int(args.streamed_chunk_voxels),
        streamed_reservoir_size=int(args.streamed_reservoir_size),
        streamed_max_bins=int(args.streamed_max_bins),
        streamed_exact_log_limit=int(args.streamed_exact_log_limit),
        streamed_seed=int(args.streamed_seed),
        run_final=not bool(args.no_run_final),
        save_sweep_artifacts=not bool(args.no_save_sweep_artifacts),
        init_strategy_mode=args.init_strategy_mode,
        execution_backend=args.execution_backend,
    )

    t0 = time.perf_counter()
    report = agentic.recommend_workflow(cfg)
    wall = time.perf_counter() - t0

    totals = defaultdict(float)
    counts = defaultdict(int)
    by_context = defaultdict(lambda: defaultdict(float))
    for rec in records:
        n = rec["name"]
        c = rec["context"]
        dt = float(rec["seconds"])
        totals[n] += dt
        counts[n] += 1
        by_context[c][n] += dt

    out = {
        "wall_seconds": wall,
        "settings": {
            "input": args.input,
            "output_root": args.output_root,
            "slices": args.slices,
            "device": args.device,
            "min_nc": args.min_nc,
            "max_nc": args.max_nc,
            "candidate_modes": [x.strip() for x in args.candidate_modes.split(",") if x.strip()],
            "execution_backend": args.execution_backend,
            "streamed_preprocess": bool(args.streamed_preprocess),
            "streamed_chunk_voxels": int(args.streamed_chunk_voxels),
            "streamed_reservoir_size": int(args.streamed_reservoir_size),
            "streamed_max_bins": int(args.streamed_max_bins),
            "streamed_exact_log_limit": int(args.streamed_exact_log_limit),
            "streamed_seed": int(args.streamed_seed),
            "save_sweep_artifacts": not bool(args.no_save_sweep_artifacts),
            "run_final": not bool(args.no_run_final),
        },
        "recommendation": report.get("recommendation"),
        "totals_by_stage": dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True)),
        "counts_by_stage": dict(sorted(counts.items())),
        "totals_by_context": {
            k: dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True))
            for k, v in by_context.items()
        },
        "crosscut_summary": {
            "preprocess_seconds": sum(v for k, v in totals.items() if k.startswith("preprocess:")),
            "io_seconds": sum(v for k, v in totals.items() if k.startswith("io:")),
            "compute_seconds": sum(v for k, v in totals.items() if k.startswith("compute:")),
            "load_seconds": totals.get("load_data", 0.0),
        },
        "records": records,
    }

    out_path = Path(args.output_root) / "timing_breakdown.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
