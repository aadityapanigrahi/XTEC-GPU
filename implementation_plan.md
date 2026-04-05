# Workflow Optimization Implementation Plan

Date initialized: 2026-04-05  
Branch: `streamed_data`  
Primary target: reduce agentic workflow wall time by reusing shared preprocessing inputs while preserving behavior.

## Objective

The bottleneck profile in `performance_comparision.md` shows repeated data loading and preprocessing across:

- `bic-d`
- `bic-s`
- sweep artifact runs (`xtec-d`/`xtec-s` over multiple `k`)
- final run

All these paths generally use the same `(input, entry, slices, threshold, rescale, device)` settings, so we will cache and reuse intermediate results in-process.

## Guardrails

- One variable changed per phase.
- Each phase gets:
  - parity checks (old baseline vs new)
  - perf check
  - keep/revert decision
- One commit per phase on `streamed_data`.
- No changes to other branches.
- Subprocess backend behavior remains unchanged.

## Baseline Matrix

Use fixed settings for parity/perf before and after each phase:

- Input: `/data/XTEC_GPU/test_dataset/srn0_XTEC.nxs`
- Entry: `entry/data`
- Candidate modes: `d,s`
- `min_nc=2`, `max_nc=4`
- `init_strategy_mode=kmeans++`
- `random_state=0`
- `execution_backend=inprocess`

Primary slice for iteration speed:

- `:,0.0:1.0,-10.0:10.0,-15.0:15.0`

Optional heavy confirmation slice:

- `:,0.0:1.0,-30.0:30.0,-30.0:30.0`

## Phase Plan

### Phase 0 — Baseline Artifacts + Parity Harness

Deliverables:

- Scriptable benchmark command for agentic workflow
- Scriptable output parity comparison command
- Baseline artifacts saved to `/tmp` for later comparisons

Keep criteria:

- Baseline run + comparator both complete.

Commit:

- `phase0: add baseline/parity workflow scripts`

### Phase 1 — Reuse Loaded NXdata (Only)

Change scope:

- In-process workflow only.
- Load sliced NXdata once and reuse for all agentic sub-runs.
- Do **not** reuse thresholding or peak-averaging yet.

Keep criteria:

- Recommendation unchanged.
- BIC arrays unchanged.
- `results.h5` parity passes.
- Wall time improves vs baseline.

Commit:

- `phase1: reuse loaded data across in-process agentic runs`

### Phase 2 — Reuse Mask + Threshold for XTEC-d/BIC-d

Change scope:

- Cache `Mask_Zeros` and `Threshold_Background` for common config.
- Reuse in `bic-d` and `xtec-d` runs only.
- Do **not** change XTEC-s path yet.

Keep criteria:

- Same as Phase 1.
- Additional preprocessing stage-time reduction in benchmark.

Commit:

- `phase2: reuse threshold preprocessing for d-mode runs`

### Phase 3 — Reuse Peak-Averaging for XTEC-s/BIC-s

Change scope:

- Cache `Peak_averaging` output and reuse in `bic-s` + `xtec-s`.
- Keep math path unchanged.

Keep criteria:

- Same as previous.
- Additional stage-time reduction on s-mode path.

Commit:

- `phase3: reuse peak-averaging preprocessing for s-mode runs`

### Phase 4 — Streaming Full-Data Preprocessing (Future/Optional)

Change scope:

- Add opt-in streamed preprocessing mode for full data.
- Slab reads + approximate IQR (t-digest style) + FD bins + global threshold.

Status:

- Planned; not required for reuse-cache phases.

Commit:

- `phase4: add streamed preprocessing mode for full-data workflows`

## Progress Ledger

- [x] Plan file created.
- [x] Phase 0 complete.
- [x] Phase 1 complete.
- [ ] Phase 2 complete.
- [ ] Phase 3 complete.
- [ ] Phase 4 complete (optional).

Latest baseline artifacts:

- Baseline run root: `/tmp/agentic_phase0_baseline_cuda_medium`
- Timing JSON: `/tmp/agentic_phase0_baseline_cuda_medium/timing_breakdown.json`
- Self-check parity JSON: `/tmp/agentic_phase0_baseline_cuda_medium/parity_selfcheck.json`

Phase 1 artifacts:

- Run root: `/tmp/agentic_phase1_cuda_medium`
- Timing JSON: `/tmp/agentic_phase1_cuda_medium/timing_breakdown.json`
- Parity JSON vs baseline: `/tmp/agentic_phase1_cuda_medium/parity_vs_phase0.json`
- Result summary:
  - Recommendation parity: pass
  - BIC parity: pass
  - Final `results.h5` parity: pass
  - Wall time: `125.14s -> 31.16s` (`4.02x` speedup on this benchmark)

## Resume Instructions

When resuming:

1. Confirm branch and cleanliness:
   - `git status --short --branch`
2. Read this file and continue from first unchecked phase.
3. Re-run baseline/perf/parity scripts for the current phase.
4. Commit phase if keep criteria pass; otherwise revert and note decision in this file.
