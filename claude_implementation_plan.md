# Claude Implementation Plan — Full Sweep Workflow

Date initialized: 2026-05-13
Primary goal: exhaustive `(mode, k, init)` sweep with all artifacts stored, plus a multi-metric agent-driven recommendation. Designed for maximum speed + accuracy on GPU with the existing `runtime_cache` reuse pattern.

---

## 1. Scope and non-goals

### In scope
- A new `xtec-gpu full-sweep` subcommand that:
  - Runs BIC sweep across `k ∈ [min_nc, max_nc]` for modes `d` and `s`.
  - At every `k` and every init strategy, saves **all artifacts** (`results.h5`, `qmap.png`, `trajectories.png`, `avg_intensities.png`, per-run `metrics.json`, per-run `timing.json`).
  - Computes a multi-metric score per candidate.
  - Emits a `recommendation.json` with the agent's chosen `(mode, k, init)` plus a reasoning trace.
- A `full-sweep-judge` subcommand that re-scores an existing sweep directory without re-clustering.
- A `full-sweep-inspect` subcommand that prints the top-N candidates with artifact paths.
- A `--dry-run` flag that prints the execution plan without running.
- A `--resume` mode that skips combos whose `results.h5` already exists with a matching manifest hash.
- An inline-artifact patch to the existing `run_bic_d` / `run_bic_s` BIC sweep loops in `xtec_cli.py` to eliminate the BIC-then-artifact double GMM fit.

### Out of scope
- Refactoring the existing `agentic.py` orchestrator. It remains usable for the lightweight BIC-only flow.
- CPU-init sweeping by default (`sklearn-kmeans`, `xtec` legacy). Available only via `--include-cpu-inits` for tutorial parity.
- Subprocess execution backend. Sweep always runs in-process for cache reuse.
- New CLI dispatcher or `pyproject.toml` console-script entry. New command lives as a subcommand of `xtec-gpu`.

---

## 2. Architecture and data flow

```
xtec-gpu full-sweep INPUT.nxs -o sweep_runs/<name> [options]
        |
        v
scripts/xtec_full_sweep.py     (thin entry point, ~30 LOC)
        |
        v
src/xtec_gpu/workflows/sweep.py  (orchestrator)
   - enumerate (mode, k, init) combos
   - build manifest + check resume
   - construct ONE runtime_cache dict
   - run mode d sweeps -> writes runs/d_kXX_<init>/...
   - run mode s sweeps -> writes runs/s_kXX_<init>/...
   - dispatch to judge.py
        |
        v
src/xtec_gpu/xtec_cli.py  (patched)
   - run_bic_d / run_bic_s accept save_artifacts_inline kwarg
   - inside the BIC k-loop: if flag, also save artifacts for each k using the already-fitted GMM
        |
        v
src/xtec_gpu/workflows/judge.py  (scorer)
   - read every runs/*/results.h5
   - compute per-candidate metrics
   - score and rank
   - emit recommendation.json
```

The orchestrator never directly calls `GMM` or preprocessing functions — it only calls the existing `xtec_cli` command entry points (`run_bic_d`, `run_bic_s`) with the new inline-artifact flag. This keeps the core math path single-source.

---

## 3. Files to create or modify

| Path | Action | LOC estimate |
|---|---|---|
| `src/xtec_gpu/xtec_cli.py` | Edit — add `save_artifacts_inline` path in `run_bic_d`/`run_bic_s`; register `full-sweep`, `full-sweep-judge`, `full-sweep-inspect` subcommands | ~150 |
| `src/xtec_gpu/workflows/sweep.py` | New — orchestrator | ~250 |
| `src/xtec_gpu/workflows/judge.py` | New — metric computation + scoring + recommendation | ~250 |
| `src/xtec_gpu/workflows/sweep_types.py` | New — shared typed dicts and constants for sweep | ~80 |
| `scripts/xtec_full_sweep.py` | New — entry-point wrapper matching `xtec_agentic_workflow.py` convention | ~30 |
| `running_instructions.md` | Edit — add "Full Sweep Workflow" section | ~80 |
| `OUTPUT_CONTRACT.md` | Edit — add "Full Sweep" artifacts section | ~40 |
| `README.md` | Edit — add CLI row, code-map entry | ~15 |

No changes to: `agentic.py`, `comparison.py`, `Preprocessing.py`, `GMM.py`, `streamed_preprocessing.py`, `claude.md`, `agent.md`, `gemini.md`, `implementation_plan.md`.

---

## 4. Detailed change list

### 4.1. `xtec_cli.py` — inline-artifact patch

**Where**: `run_bic_d` (currently at lines 1209–1275) and `run_bic_s` (currently at lines 1278–1325).

**Why**: Today both BIC functions fit a full `GMM` per `k`, compute BIC, then throw the fitted model away. The orchestrator then calls `xtec-d -n k` separately to dump artifacts, which re-fits the exact same GMM. This is the Win #1 dedup: ~2x reduction in clustering wall time across the sweep.

**Change**: Add an optional argument `save_artifacts_inline` (default `False`, preserving current behavior). When set:

```python
def run_bic_d(args):
    common_cfg = _common_config_from_args(args)
    data = _get_or_load_data(args, common_cfg.entry, common_cfg.slices)
    device = _get_device(common_cfg.device)
    ...
    save_inline = getattr(args, "save_artifacts_inline", None)  # None | dict
    # save_inline schema: {
    #   "runs_root": Path, "mode": "d", "init": "kmeans++",
    #   "plots_level": "all" | "primary" | "none",
    #   "reorder_clusters": True, "random_state": 0,
    # }

    ks = np.arange(args.min_nc, args.max_nc)
    bics = []
    timings_per_k = {}
    for k in ks:
        t0 = time.time()
        clusterGMM = GMM(Data_for_GMM, int(k), cov_type="diag",
                         init_strategy_mode=save_inline["init"] if save_inline else "kmeans++",
                         random_state=save_inline["random_state"] if save_inline else 0)
        clusterGMM.RunEM()
        bics.append(_bic_from_loglikelihood(...))

        if save_inline is not None:
            run_dir = save_inline["runs_root"] / f"d_k{int(k):02d}_{_init_slug(save_inline['init'])}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cluster_assigns = _to_numpy(clusterGMM.cluster_assignments)
            cluster_means = _to_numpy(clusterGMM.means)
            cluster_covs = [_to_numpy(clusterGMM.cluster[i].cov) for i in range(int(k))]
            Data_thresh_np = _to_numpy(Data_thresh)
            Data_ind_np = _to_numpy(threshold.ind_thresholded)

            if save_inline["reorder_clusters"]:
                temp_values = data.nxaxes[0].nxvalue
                cluster_assigns, _, cluster_means, cluster_covs = \
                    _reorder_clusters(cluster_assigns, cluster_assigns,
                                      cluster_means, cluster_covs,
                                      Data_thresh_np, int(k), temp_values)
                _sync_cluster_model(clusterGMM, cluster_assigns, cluster_means, cluster_covs)

            _save_results(str(run_dir), cluster_assigns, cluster_assigns,
                          Data_ind_np, Data_thresh_np, cluster_means, cluster_covs)
            if save_inline["plots_level"] in ("all", "primary"):
                _plot_qmap(data, Data_ind_np, cluster_assigns, int(k), str(run_dir))
                _plot_trajectories(data, cluster_means, cluster_covs, int(k),
                                   common_cfg.rescale, str(run_dir))
                _plot_avg_intensities(data, Data_thresh_np, cluster_assigns,
                                      int(k), str(run_dir))

            timings_per_k[int(k)] = {
                "wall_s": time.time() - t0,
                "bic": float(bics[-1]),
                "cluster_sizes": [int(np.sum(cluster_assigns == c)) for c in range(int(k))],
            }
        else:
            print(f"  k={k}: BIC={bics[-1]:.2f}")

    # Existing BIC h5 + png writes remain unchanged
    ...
    if save_inline is not None:
        return {"ks": ks.tolist(), "bics": bics, "timings": timings_per_k}
```

**Symmetric change** for `run_bic_s` — same pattern, but the inner loop uses the cached `_get_or_build_s_preprocessed` output and `Peak_avg` for pixel-label mapping. Pseudocode is identical aside from the s-mode `_save_results` call needing the pixel-label expansion (call `clusterGMM.Get_pixel_labels(peak_avg)` first, then save).

**Helper**: `_init_slug(init: str) -> str` — returns filename-safe slug. `"kmeans++"` → `"kmeanspp"`, `"sklearn-kmeans"` → `"sklearnkmeans"`, `"cuml-kmeans"` → `"cumlkmeans"`, `"xtec"` → `"xtec"`.

**Backward compat**: When `save_artifacts_inline` is `None`, behavior is byte-identical to current. All existing callers (`agentic.py`, direct CLI) pass nothing and get the old path.

### 4.2. `workflows/sweep.py` — orchestrator (new)

**Module docstring** (top of file) describes:
1. The BIC-then-artifact dedup rationale.
2. The d→s ordering for threshold cache reuse.
3. The `runtime_cache` lifetime (one dict, persistent for the whole sweep).
4. The inputs/outputs contract.

**Key signature**:

```python
@dataclass
class FullSweepConfig:
    input_path: str
    output_root: Path
    entry: str = "entry/data"
    slices: Optional[str] = None
    threshold: bool = True
    rescale: str = "mean"
    device: str = "auto"
    min_nc: int = 2
    max_nc: int = 8  # exclusive upper bound, matches np.arange semantics in BIC funcs
    modes: List[str] = ("d", "s")
    inits: List[str] = ("kmeans++",)  # GPU-resident; CPU inits added via include_cpu_inits
    include_cpu_inits: bool = False
    plots_level: str = "all"  # "all" | "primary" | "none"
    random_state: int = 0
    reorder_clusters: bool = True
    stream_mode: str = "auto"  # "auto" | "on" | "off"; auto -> on if input > 2 GB
    stream_threshold_gb: float = 2.0
    resume: bool = False
    dry_run: bool = False
    judge_weights: Optional[Dict[str, float]] = None  # passed through to judge.py
    judge_top_n: int = 5

def run_full_sweep(cfg: FullSweepConfig) -> Dict[str, Any]:
    """Returns the loaded recommendation dict; writes everything under cfg.output_root."""
```

**Internal stages**:

1. **Pre-flight**
   - Resolve device. If CUDA, check free memory with `torch.cuda.mem_get_info`; abort with a useful error if free memory < 1 GB.
   - Resolve `stream_mode == "auto"` by stat'ing the input file size.
   - Resolve init list. If `include_cpu_inits`, extend with `["sklearn-kmeans", "xtec"]`. (No cuml auto-detect — per your decision, drop it.)
   - Build the full plan list `[(mode, init), ...]`.
   - If `dry_run`: print plan + estimated combos (`len(modes) × len(inits) × (max_nc - min_nc)`), exit.

2. **Manifest write**
   - Write `manifest.json` at the start with: full config, git sha (best-effort via `git rev-parse HEAD` in the package directory; fall back to `"unknown"`), hostname, device name, torch + CUDA versions, input file path + size + sha256 (computed only if file ≤ 10 GB, else just `mtime`), timestamps, resolved plan.
   - If `resume`: read existing manifest at `output_root/manifest.json`. Compare config fields. If mismatch on any compute-affecting field, abort with a clear error.

3. **One runtime_cache**
   - Construct `runtime_cache: Dict[Any, Any] = {}`.
   - This single dict is threaded through every sub-run via the `argparse.Namespace.runtime_cache` attribute that `_runtime_cache_from_args` reads ([xtec_cli.py:165](src/xtec_gpu/xtec_cli.py#L165)).

4. **Mode d combos first** (cache warming order)
   - For each `init in resolved_inits`:
     - Build a `Namespace` with `save_artifacts_inline = {"runs_root": output_root / "runs", "mode": "d", "init": init, "plots_level": plots_level, "reorder_clusters": reorder_clusters, "random_state": random_state}`.
     - If `resume`: pre-check which `k` values in the range already have `results.h5` at the expected path. If all exist and manifest hash matches, skip this combo entirely. Otherwise reduce `min_nc`/`max_nc` to the missing-only contiguous range (or run all if non-contiguous — simpler).
     - Call `xtec_cli.run_bic_d(ns)`. The patched function returns `{"ks": ..., "bics": ..., "timings": ...}` and also writes the standard `bic_xtec_d.h5` + `.png`.
     - Move the standard BIC outputs to `output_root / "bic_curves" / f"bic_d_{init_slug}"` to disambiguate per-init.

5. **Mode s combos** (reuses threshold cache from d)
   - Same pattern with `run_bic_s`.

6. **Per-run timing capture**
   - The inline-artifact path writes `timing.json` and `metrics.json` per run. `metrics.json` at this stage holds *only* `bic` and `cluster_sizes`; the full metrics are computed by `judge.py`.

7. **Sweep summary write**
   - Aggregate every combo's `timings` into `sweep_summary.json`. Schema:
     ```
     {
       "candidates": [
         {
           "id": "d_k03_kmeanspp",
           "mode": "d", "k": 3, "init": "kmeans++",
           "output_dir": "runs/d_k03_kmeanspp",
           "bic": 12345.6,
           "wall_s": 4.21,
           "cluster_sizes": [...]
         },
         ...
       ],
       "manifest_ref": "manifest.json"
     }
     ```

8. **Dispatch to judge**
   - Call `judge.score_and_recommend(output_root, weights=cfg.judge_weights, top_n=cfg.judge_top_n)`.
   - This writes `recommendation.json` and updates `sweep_summary.json` in place by merging the per-candidate metrics + score.

9. **Final symlink**
   - `output_root / "final_run"` → `output_root / "runs" / <winner_id>` (use `os.symlink` with relative target).

### 4.3. `workflows/judge.py` — scoring (new)

**Reads** only `results.h5` files; never invokes `GMM`. This makes it cheap to re-run with different weights.

**Per-candidate metrics** computed from `results.h5` datasets `cluster_assignments`, `cluster_means`, `cluster_covariances`, `data_indices`:

| Metric | Formula | Range |
|---|---|---|
| `min_cluster_frac` | `min(cluster_sizes) / N` | `[0, 1/K]` |
| `degeneracy_flag` | `bool(min_cluster_frac < 0.005)` | bool |
| `mean_pairwise_sep` | mean over `i<j` of `‖mean_i − mean_j‖₂ / (σ_pooled_ij)` where `σ_pooled = 0.5*(mean(cov_i) + mean(cov_j))` | `[0, ∞)` |
| `mean_trajectory_smoothness` | mean over clusters of `‖Δ²(mean_k)‖₂ / ‖mean_k‖₂` (second discrete difference along T) | `[0, ∞)` — *higher = noisier* |
| `spatial_coherence` | mean over points of (fraction of k-nearest-neighbors in `data_indices` with same cluster label, k=8) | `[0, 1]` |
| `bic_norm` | `(bic - min_bic_in_mode) / (max_bic_in_mode - min_bic_in_mode)` (computed across this mode's candidates) | `[0, 1]` |

**Cross-mode metric** computed once per `(k_d, k_s)` pair with `k_d == k_s`:
- `d_vs_s_consistency[k]`: Hungarian-match the `cluster_means` of best-d and best-s at this `k` (best by BIC within each mode), report mean Pearson correlation of matched trajectories.

**Score**:
```
score = (
    + w_sep   * mean_pairwise_sep
    + w_coh   * spatial_coherence
    + w_bic   * (1.0 - bic_norm)            # bic_norm ∈ [0,1], lower bic → higher score contribution
    - w_smooth* mean_trajectory_smoothness
    - w_deg   * float(degeneracy_flag)
)
```

**Default weights**:
```python
DEFAULT_WEIGHTS = {
    "w_sep":    1.0,
    "w_coh":    0.5,
    "w_bic":    0.2,
    "w_smooth": 0.3,
    "w_deg":    2.0,
}
```

**Recommendation logic**:
1. Compute score per candidate.
2. Pick winner = argmax(score) across all candidates of all modes.
3. If two candidates tie within 1% on score, prefer (a) lower `k`, (b) mode `s` if peak count is healthy (≥ 50 peaks), (c) `kmeans++` init.

**`recommendation.json` schema**:
```json
{
  "winner": {
    "id": "d_k03_kmeanspp",
    "mode": "d", "k": 3, "init": "kmeans++",
    "output_dir": "runs/d_k03_kmeanspp",
    "score": 2.91,
    "metrics": { ... },
    "bic": 12345.6
  },
  "top_n": [ ... ranked candidates, descending score ... ],
  "reasoning": [
    "BIC argmin was d_k04_kmeanspp (BIC=12340.1) but its smallest cluster is 0.3% (degenerate).",
    "d_k03_kmeanspp has well-separated trajectories (sep=3.42) and coherent qmap (coh=0.78).",
    "s-mode best at k=3 trajectory-matched d_k03 with mean correlation 0.94 (high cross-mode agreement)."
  ],
  "weights": { ... actual weights used ... },
  "bic_argmin": { "mode": "d", "k": 4, "id": "d_k04_kmeanspp" },
  "bic_argmin_overridden": true
}
```

**Reasoning trace generator**: a small function that pattern-matches on the winner vs. `bic_argmin`, vs. the second-place candidate, vs. degeneracy flags, and emits 2-4 human-readable sentences. Deterministic, not LLM-generated.

### 4.4. `workflows/sweep_types.py` — shared types and constants (new)

- `FullSweepConfig` dataclass.
- `INIT_SLUGS: Dict[str, str]` mapping init strategy strings to filename-safe slugs.
- `GPU_INITS = ("kmeans++",)` and `CPU_INITS = ("sklearn-kmeans", "xtec")`.
- `SWEEP_SUMMARY_REQUIRED_KEYS` for validation.
- `RECOMMENDATION_REQUIRED_KEYS`.
- `DEFAULT_JUDGE_WEIGHTS`.

### 4.5. `scripts/xtec_full_sweep.py` — entry-point wrapper (new)

Mirror `scripts/xtec_agentic_workflow.py` exactly:

```python
"""Convenience wrapper around xtec_gpu.workflows.sweep.main."""
from xtec_gpu.workflows.sweep import main

if __name__ == "__main__":
    main()
```

### 4.6. `xtec_cli.py` — subcommand registration

Add three subparsers under the existing `xtec-gpu` argparser:

| Subcommand | Function dispatched to | Notes |
|---|---|---|
| `full-sweep` | `xtec_gpu.workflows.sweep.run_full_sweep` (via `main`) | Primary command |
| `full-sweep-judge` | `xtec_gpu.workflows.judge.score_and_recommend` | Reads existing sweep dir, no clustering |
| `full-sweep-inspect` | `xtec_gpu.workflows.judge.inspect` | Prints top-N candidates with artifact paths |

The CLI flags for `full-sweep` exactly mirror `FullSweepConfig` fields, with hyphens (e.g., `--max-nc`, `--include-cpu-inits`, `--plots`, `--judge-top-n`, `--judge-weights`, `--stream`, `--stream-threshold-gb`).

---

## 5. Output contract

### 5.1. Storage layout

```
sweep_runs/<run-name>/
├── manifest.json                       # full config, git sha, host, device, input hash, timestamps, resolved plan
├── sweep_summary.json                  # one row per (mode, k, init) with bic + metrics + score + path + wall_s
├── recommendation.json                 # winner + ranked candidates + reasoning trace
├── bic_curves/
│   ├── bic_d_kmeanspp/{bic_xtec_d.h5, bic_xtec_d.png}
│   ├── bic_d_cumlkmeans/{...}          # only if user passed cuml-kmeans explicitly
│   └── bic_s_kmeanspp/{...}
├── runs/                               # ALL artifacts, one dir per combo
│   ├── d_k02_kmeanspp/
│   │   ├── results.h5
│   │   ├── qmap.png
│   │   ├── trajectories.png
│   │   ├── avg_intensities.png
│   │   ├── metrics.json                # written by judge.py
│   │   └── timing.json                 # written by inline-artifact patch
│   ├── d_k03_kmeanspp/...
│   ├── ...
│   ├── d_k07_kmeanspp/...
│   ├── s_k02_kmeanspp/...
│   ├── ...
│   └── s_k07_kmeanspp/...
└── final_run -> runs/<winner_id>/      # relative symlink
```

### 5.2. `results.h5` schema (per run)

Unchanged from current `OUTPUT_CONTRACT.md`:
- `cluster_assignments` `(N,)`
- `pixel_assignments` `(N,)` (same as `cluster_assignments` for d-mode; peak→pixel mapping for s-mode)
- `data_indices` `(N, D)`
- `data_thresholded` `(N, T)`
- `cluster_means` `(K, T)`
- `cluster_covariances` `(K, T)`

### 5.3. `metrics.json` schema (per run)

```json
{
  "bic": 12345.6,
  "n_points": 1736459,
  "cluster_sizes": [482, 1031, 223],
  "min_cluster_frac": 0.128,
  "degeneracy_flag": false,
  "mean_pairwise_sep": 3.42,
  "mean_trajectory_smoothness": 0.011,
  "spatial_coherence": 0.78,
  "bic_norm": 0.0
}
```

### 5.4. `timing.json` schema (per run)

```json
{
  "wall_s": 4.21,
  "stages": {
    "gmm_fit_s": 0.34,
    "save_results_s": 0.12,
    "plot_qmap_s": 1.88,
    "plot_trajectories_s": 0.70,
    "plot_avg_intensities_s": 1.50
  }
}
```

---

## 6. CLI surface

### 6.1. Primary command

```bash
xtec-gpu full-sweep INPUT.nxs -o sweep_runs/run_name \
    --device cuda:1 \
    --slices ":,0.0:1.0,-10.0:10.0,-15.0:15.0" \
    --rescale mean \
    --min-nc 2 --max-nc 8 \
    --modes d,s \
    --inits kmeans++ \
    --plots all \
    --random-state 0 \
    --stream auto \
    --judge-top-n 5
```

| Flag | Default | Description |
|---|---|---|
| `input` | — | Path to `.nxs` input file (positional) |
| `-o`, `--output-root` | — | Output directory for the sweep |
| `--entry` | `entry/data` | HDF5 dataset path in input file |
| `--slices` | `None` | Slice string |
| `--threshold` / `--no-threshold` | on | KL background thresholding |
| `--rescale` | `mean` | `mean` / `z-score` / `log-mean` / `None` |
| `--device` | `auto` | Compute device |
| `--min-nc` | `2` | Min cluster count |
| `--max-nc` | `8` | Max cluster count (exclusive, matches `np.arange`) |
| `--modes` | `d,s` | Comma-separated modes |
| `--inits` | `kmeans++` | Comma-separated GPU-resident inits |
| `--include-cpu-inits` | off | Also sweep `sklearn-kmeans` and `xtec` (slow, CPU) |
| `--plots` | `all` | `all` / `primary` / `none` |
| `--random-state` | `0` | Random seed |
| `--reorder-clusters` / `--no-reorder-clusters` | on | Deterministic low-T cluster ordering |
| `--stream` | `auto` | `auto` / `on` / `off`; auto = on if input > `--stream-threshold-gb` |
| `--stream-threshold-gb` | `2.0` | Threshold for `--stream auto` |
| `--resume` | off | Skip combos with valid existing `results.h5` |
| `--dry-run` | off | Print plan, do not execute |
| `--judge-weights` | None | Path to JSON with weight overrides |
| `--judge-top-n` | `5` | Top-N candidates in `recommendation.json` |

### 6.2. Re-judge

```bash
xtec-gpu full-sweep-judge sweep_runs/run_name [--judge-weights weights.json] [--judge-top-n 5]
```

Reads existing `runs/*/results.h5`, recomputes metrics + scores, rewrites `recommendation.json` and the `metrics.json` files. Does **not** touch `results.h5` or any plots.

### 6.3. Inspect

```bash
xtec-gpu full-sweep-inspect sweep_runs/run_name [--top 5]
```

Prints (to stdout):
- Winner + score + paths to its 3 PNGs.
- Top-N candidates ranked with one-line per candidate showing `id`, `score`, `bic`, `min_cluster_frac`, paths.

---

## 7. Efficiency claims (verifying against prior conversation)

| # | Efficiency | Implemented in | Status |
|---|---|---|---|
| 1 | BIC-then-artifact dedup (single `RunEM` per `k`) | `xtec_cli.py` `save_artifacts_inline` | ✅ |
| 2 | Shared `runtime_cache` across all runs | `sweep.py` constructs one dict | ✅ (uses existing infrastructure) |
| 3 | `d → s` ordering for threshold cache reuse | `sweep.py` orchestration | ✅ |
| 4 | GPU-resident inits only by default | `sweep_types.GPU_INITS` | ✅ |
| 5 | Streaming auto-on at >2 GB | `sweep.py` pre-flight | ✅ |
| 6 | Plot tier (`all` / `primary` / `none`) | Inline-artifact `plots_level` | ✅ |
| 7 | Resume / skip-completed | `sweep.py` pre-flight | ✅ |
| 8 | `--dry-run` | `sweep.py` pre-flight | ✅ |
| 9 | Judge re-runs offline without re-clustering | `judge.py` operates only on `results.h5` | ✅ |
| 10 | Multi-metric decision rule | `judge.py` scoring | ✅ |
| 11 | `inprocess` execution backend always | `sweep.py` (no subprocess option exposed) | ✅ |
| 12 | Rescaled-data caching across inits at same mode | Extension of `runtime_cache` keyed on `(mode, rescale_text)` | ✅ |
| 13 | Fail-fast GPU memory check (no cache-clearing) | `sweep.py` pre-flight | ✅ |

---

## 8. Test plan

### 8.1. Existing test that must still pass
```
python -m unittest -q tests/test_refactor_regressions.py
```

### 8.2. New unit tests (in `tests/test_full_sweep.py`)

| Test | What it validates |
|---|---|
| `test_inline_artifact_parity_d` | Run `bic-d` with `save_artifacts_inline` set, then separately `xtec-d` at the same `k`. Compare `cluster_assignments`, `cluster_means`, `cluster_covariances` allclose. Same `random_state`. |
| `test_inline_artifact_parity_s` | Same as above for s-mode. |
| `test_inline_artifact_no_flag_byte_identical` | Run `bic-d` without the flag, compare `bic_xtec_d.h5` byte-for-byte against pre-patch baseline. |
| `test_dry_run_writes_no_artifacts` | After `--dry-run`, only stdout output exists; no files in `output_root`. |
| `test_resume_skips_completed` | Run partial sweep, kill, re-run with `--resume`, verify skipped combos are not re-fit (check `mtime` of `results.h5`). |
| `test_judge_deterministic` | Same `results.h5` set → identical `recommendation.json`. |
| `test_judge_offline` | Run sweep once, delete `metrics.json` and `recommendation.json`, re-run `full-sweep-judge`, verify regeneration. |
| `test_judge_weight_override` | Custom weights → different winner when scenario constructed to flip it. |
| `test_degeneracy_guard_triggers` | Synthetic data with one tiny cluster → `degeneracy_flag=true`, winner is NOT the degenerate candidate even if BIC favors it. |
| `test_storage_layout_complete` | After a full sweep, every combo has all 6 files (results.h5, 3 PNGs, metrics.json, timing.json). `final_run` symlink exists and resolves. |

### 8.3. Benchmark (manual, recorded in `performance_comparision.md`)

Configuration:
- Input: `/data/XTEC_GPU/test_dataset/srn0_XTEC.nxs`
- Slices: `:,0.0:1.0,-10.0:10.0,-15.0:15.0`
- `--device cuda:1 --min-nc 2 --max-nc 8 --modes d,s --inits kmeans++ --plots all`

Acceptance:
- Total wall time ≤ (current `xtec_agentic_workflow.py` wall time on equivalent settings) − 30%. The 30% margin is conservative; the dedup alone should give ~50% on the clustering phase.
- `cluster_assignments` for the winner equal a reference `xtec-d -n k_winner` run with `match_ratio ≥ 0.9999`.

---

## 9. Documentation updates

### 9.1. `running_instructions.md` — add new section

Insert after the existing "2) Run the Agentic Workflow (CLI)" section:

```
## 3) Run the Full Sweep Workflow (CLI)

The full sweep runs the BIC sweep and saves all artifacts for every (mode, k, init)
combination, then scores each candidate on multiple criteria and writes a recommendation.

    xtec-gpu full-sweep \
      /data/XTEC_GPU/test_dataset/srn0_XTEC.nxs \
      -o /data/XTEC_GPU/XTEC-GPU/sweep_runs/srn0_full \
      --device cuda:1 \
      --rescale mean \
      --slices ":,0.0:1.0,-10.0:10.0,-15.0:15.0" \
      --min-nc 2 --max-nc 8 \
      --modes d,s

Outputs:
- manifest.json, sweep_summary.json, recommendation.json
- bic_curves/ per (mode, init)
- runs/<mode>_k<kk>_<init>/ for every combo, with results.h5 + 3 plots + metrics.json + timing.json
- final_run/ symlink to the winning combo

To re-score with different weights without re-clustering:
    xtec-gpu full-sweep-judge sweep_runs/srn0_full --judge-weights my_weights.json

To inspect:
    xtec-gpu full-sweep-inspect sweep_runs/srn0_full --top 5
```

Also expand "3) Outputs" → renumber to "4) Outputs" with a subsection for full-sweep outputs.

### 9.2. `OUTPUT_CONTRACT.md` — add new section

Add at the end:

```
## Full Sweep Workflow

Command:
- xtec-gpu full-sweep ...

Expected top-level artifacts:
- manifest.json
- sweep_summary.json
- recommendation.json
- bic_curves/
- runs/<mode>_k<kk>_<init>/
- final_run/ (symlink)

Per-run files inside each runs/<mode>_k<kk>_<init>/:
- results.h5 (same datasets as xtec-d/xtec-s)
- qmap.png
- trajectories.png
- avg_intensities.png
- metrics.json
- timing.json

recommendation.json required keys:
- winner
- top_n
- reasoning
- weights
- bic_argmin
- bic_argmin_overridden
```

### 9.3. `README.md` — minor edits

In the CLI section, add:

```bash
# Full sweep (exhaustive + agent-judged)
xtec-gpu full-sweep data.nxs -o sweep_runs/run1 --device cuda:1
```

In "Start Here (Code Map)", add:

```
- Full sweep + judge:
  - src/xtec_gpu/workflows/sweep.py
  - src/xtec_gpu/workflows/judge.py
- Script entry point:
  - scripts/xtec_full_sweep.py
```

### 9.4. `claude.md`, `agent.md`, `gemini.md`
**No changes.** They already redirect to `running_instructions.md`, which will pick up the new section automatically.

---

## 10. Implementation order

Each step has a clear stop point so review can happen before more code is written.

| Step | Deliverable | Stop point for review |
|---|---|---|
| 1 | `running_instructions.md` + `OUTPUT_CONTRACT.md` + `README.md` edits | Confirm the user surface (CLI flags, outputs) before any code |
| 2 | `sweep_types.py` + module docstrings + function signatures in `sweep.py` and `judge.py` (no implementation) | Confirm module boundaries and types |
| 3 | `xtec_cli.py` inline-artifact patch + unit test `test_inline_artifact_parity_d` | Confirm Win #1 works and produces parity |
| 4 | `xtec_cli.py` same for s-mode + `test_inline_artifact_parity_s` | Same |
| 5 | `workflows/judge.py` full implementation + tests | Confirm scoring is sensible |
| 6 | `workflows/sweep.py` full implementation + tests | End-to-end |
| 7 | `xtec_cli.py` subcommand registration + `scripts/xtec_full_sweep.py` | Wire up |
| 8 | Benchmark vs. current agentic workflow, record in `performance_comparision.md` | Verify speedup target |

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Inline-artifact path subtly diverges from standalone `xtec-d` output | Strict parity unit tests at every k with same seed; CI runs them on every change |
| `runtime_cache` grows large on huge inputs | Fail-fast pre-flight memory check; user is told to slice. No silent cache-clear. |
| `_save_results` for s-mode requires `peak_avg` for pixel-label expansion; inline path must call `Get_pixel_labels` first | Test `test_inline_artifact_parity_s` specifically checks `pixel_assignments` parity |
| Judge weights chosen here may not generalize across datasets | `full-sweep-judge` can re-score offline with new weights; weights logged in `recommendation.json` for auditability |
| Spatial coherence metric (k-NN over `data_indices`) may be slow for very large N | Subsample to max 50k points for the metric; record actual N sampled in `metrics.json` |
| Resume logic miscompares manifests due to floating-point or path differences | Compare a normalized subset of fields (input path resolved, slices string, all numeric configs) — not raw dict equality |

---

## 12. Progress ledger

- [ ] Step 1: doc edits (user surface)
- [ ] Step 2: types + signatures
- [ ] Step 3: inline-artifact patch for d-mode + parity test
- [ ] Step 4: inline-artifact patch for s-mode + parity test
- [ ] Step 5: `judge.py` + tests
- [ ] Step 6: `sweep.py` + tests
- [ ] Step 7: subcommand registration + entry-point script
- [ ] Step 8: benchmark, update `performance_comparision.md`
