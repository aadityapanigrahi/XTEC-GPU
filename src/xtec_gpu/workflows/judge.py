"""Offline scorer + recommender for full-sweep artifacts.

Reads every ``runs/<id>/results.h5`` under a sweep directory, computes a
per-candidate metric bundle (BIC, cluster degeneracy, trajectory separation,
trajectory smoothness, spatial coherence), and writes:

- ``runs/<id>/metrics.json``  — per-run metric bundle
- ``recommendation.json``     — winner + ranked top-N + reasoning trace

This module is intentionally **offline**: it never invokes GMM or
preprocessing. That makes it cheap to re-run with different metric weights
via the ``full-sweep-judge`` CLI subcommand.

Public entry points
-------------------
- ``score_and_recommend(output_root, weights=None, top_n=5)``
- ``inspect(output_root, top=5)``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from xtec_gpu.workflows.logging_utils import get_logger
from xtec_gpu.workflows.sweep_types import (
    DEFAULT_JUDGE_WEIGHTS,
    DEGENERACY_FRAC_THRESHOLD,
    JUMP_PENALTY_START,
    PER_RUN_METRICS_KEYS,
    RECOMMENDATION_REQUIRED_KEYS,
    SIZE_PENALTY_FRAC_THRESHOLD,
    SPATIAL_COHERENCE_KNN,
    SPATIAL_COHERENCE_MAX_SAMPLES,
)


logger = get_logger(__name__)


def _judge_device() -> torch.device:
    """Pick a torch device for judge metrics.

    Honors ``XTEC_JUDGE_DEVICE`` env var if set (e.g. ``cuda:1``); otherwise
    picks CUDA if available, then MPS, then CPU. The judge is metric-heavy
    and benefits from GPU even for small N because of the chunked pairwise
    distance.
    """
    forced = os.environ.get("XTEC_JUDGE_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def _cluster_sizes_t(cluster_assignments: torch.Tensor, k: int) -> torch.Tensor:
    """Counts per cluster, on the same device as the input."""
    return torch.bincount(cluster_assignments, minlength=k)[:k]


def _mean_pairwise_sep_t(
    means: torch.Tensor,
    covs: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Average pairwise normalized L2 distance between cluster means.

    Restricted to clusters flagged by ``valid_mask`` (default: all). Done on
    device, single vectorized cdist + diag-mask. Restricting to clusters that
    pass a size threshold prevents tiny outlier clusters from inflating the
    score with a huge mean displacement.
    """
    k = int(means.shape[0])
    if k < 2:
        return 0.0
    if valid_mask is None:
        sel = torch.arange(k, device=means.device)
    else:
        sel = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if sel.numel() < 2:
        return 0.0
    M = means[sel]                       # (K', T)
    C = covs[sel].mean(dim=1)            # (K',) mean of diagonal cov per cluster
    # Pairwise L2 distance between cluster means.
    dist = torch.cdist(M, M)             # (K', K')
    # Pooled stds for every pair: sqrt(0.5*(c_i + c_j))
    pooled = 0.5 * (C[:, None] + C[None, :])
    denom = torch.where(pooled > 0, torch.sqrt(pooled), torch.ones_like(pooled))
    # Upper triangle only (exclude diagonal).
    iu = torch.triu_indices(M.shape[0], M.shape[0], offset=1, device=M.device)
    seps = dist[iu[0], iu[1]] / denom[iu[0], iu[1]]
    return float(seps.mean().item())


def _max_rel_jump_t(
    means: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Worst per-cluster single-step |Δ|/range. Vectorized on device.

    Tiny clusters often have a single spike that makes their entire range
    coincide with one Δ. ``valid_mask`` lets the caller restrict to
    non-degenerate clusters if desired, but by default we look at *all*
    clusters since the metric exists precisely to surface degenerate ones.
    """
    k = int(means.shape[0])
    if k == 0 or means.shape[1] < 2:
        return 0.0
    if valid_mask is not None:
        sel = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if sel.numel() == 0:
            return 0.0
        means = means[sel]
    diffs = means[:, 1:] - means[:, :-1]      # (K, T-1)
    max_abs = diffs.abs().amax(dim=1)         # (K,)
    rng = means.amax(dim=1) - means.amin(dim=1)
    safe_rng = torch.where(rng > 0, rng, torch.ones_like(rng))
    rel = max_abs / safe_rng
    rel = torch.where(rng > 0, rel, torch.zeros_like(rel))
    return float(rel.amax().item())


def _mean_trajectory_smoothness_t(
    means: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean relative second-difference norm across clusters. On device.

    Restricted to clusters flagged by ``valid_mask``. Tiny spike clusters
    have huge second-differences; including them in this average makes the
    "real" clusters look noisier than they actually are. The separate
    ``jump_penalty`` term still surfaces the spike directly.
    """
    if means.shape[1] < 3:
        return 0.0
    if valid_mask is not None:
        sel = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if sel.numel() == 0:
            return 0.0
        means = means[sel]
    second_diff = means[:, 2:] - 2 * means[:, 1:-1] + means[:, :-2]   # (K, T-2)
    norms = torch.linalg.norm(second_diff, dim=1)
    base = torch.linalg.norm(means, dim=1)
    base = torch.where(base > 0, base, torch.ones_like(base))
    return float((norms / base).mean().item())


def _spatial_coherence_t(
    data_indices: torch.Tensor,
    cluster_assignments: torch.Tensor,
    k_neighbors: int = SPATIAL_COHERENCE_KNN,
    max_samples: int = SPATIAL_COHERENCE_MAX_SAMPLES,
    seed: int = 0,
) -> float:
    """Fraction of k-nearest-neighbors sharing the same cluster label.

    Fully on-device: subsample, chunked ``torch.cdist`` + ``topk``. Avoids the
    O(N²) full distance matrix by streaming queries against the full subsample.
    """
    n = int(data_indices.shape[0])
    if n <= 1:
        return 0.0
    device = data_indices.device

    if n > max_samples:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        sel = torch.randperm(n, generator=g)[:max_samples]
        # randperm doesn't support all devices; do it on CPU then move.
        sel = sel.to(device)
        pts = data_indices[sel].to(torch.float32)
        labels = cluster_assignments[sel]
    else:
        pts = data_indices.to(torch.float32)
        labels = cluster_assignments

    m = pts.shape[0]
    kk = min(k_neighbors + 1, m)  # +1 because nearest is self
    chunk = 2048
    coherences = torch.empty(m, device=device, dtype=torch.float32)

    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        # cdist returns (chunk, m). topk with largest=False gives smallest distances.
        d = torch.cdist(pts[start:stop], pts)            # (chunk, m)
        nn_idx = torch.topk(d, kk, dim=1, largest=False).indices   # (chunk, kk)
        nn_labels = labels[nn_idx]                       # (chunk, kk)
        self_labels = labels[start:stop, None]
        same = (nn_labels == self_labels).to(torch.float32)
        # The closest is always the point itself (distance 0); remove its
        # contribution (always a 1) before averaging over the remaining kk-1.
        n_same = same.sum(dim=1) - 1.0
        coherences[start:stop] = torch.clamp(
            n_same / max(1, kk - 1), 0.0, 1.0,
        )

    return float(coherences.mean().item())


def compute_run_metrics(
    results_h5_path: Path,
    bic: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Compute the per-run metric bundle from a results.h5 file.

    All heavy compute (pairwise distances for spatial coherence, pairwise
    separations, second differences) runs on ``device`` — defaults to CUDA
    when available. Override globally with ``XTEC_JUDGE_DEVICE=cuda:1``.
    """
    if device is None:
        device = _judge_device()

    with h5py.File(results_h5_path, "r") as f:
        cluster_assignments_np = f["cluster_assignments"][...].astype(np.int64)
        pixel_assignments_np = f["pixel_assignments"][...].astype(np.int64)
        cluster_means_np = f["cluster_means"][...].astype(np.float32)
        cluster_covs_np = f["cluster_covariances"][...].astype(np.float32)
        data_indices_np = f["data_indices"][...].astype(np.int64)

    # (K, T) means/covs. covs may come as (K, T) or (K, T, T) — coerce to (K, T).
    if cluster_covs_np.ndim == 3:
        cluster_covs_np = np.array(
            [np.diag(c) for c in cluster_covs_np], dtype=np.float32,
        )

    cluster_assignments = torch.from_numpy(cluster_assignments_np).to(device)
    pixel_assignments = torch.from_numpy(pixel_assignments_np).to(device)
    cluster_means = torch.from_numpy(cluster_means_np).to(device)
    cluster_covs = torch.from_numpy(cluster_covs_np).to(device)
    data_indices = torch.from_numpy(data_indices_np).to(device)

    k = int(cluster_means.shape[0])
    n = int(cluster_assignments.shape[0])

    sizes_t = _cluster_sizes_t(cluster_assignments, k)
    sizes = [int(x) for x in sizes_t.tolist()]
    min_frac = float(min(sizes)) / float(n) if n > 0 else 0.0

    # Continuous size penalty: ramps from 0 → 1 as min_frac falls from the
    # threshold to 0. Tiny clusters get maximum penalty.
    size_penalty = max(
        0.0, (SIZE_PENALTY_FRAC_THRESHOLD - min_frac) / SIZE_PENALTY_FRAC_THRESHOLD,
    )
    size_penalty = min(1.0, size_penalty)

    # Separation is computed only over clusters that pass the size threshold.
    # Otherwise an outlier 23-pixel cluster with a huge mean inflates pairwise
    # distances and crowds out the real signal.
    size_frac_t = sizes_t.to(torch.float32) / max(1, n)
    valid_mask = size_frac_t >= SIZE_PENALTY_FRAC_THRESHOLD
    mean_pairwise_sep = _mean_pairwise_sep_t(cluster_means, cluster_covs, valid_mask)

    max_rel_jump = _max_rel_jump_t(cluster_means)
    jump_penalty = max(
        0.0, (max_rel_jump - JUMP_PENALTY_START) / (1.0 - JUMP_PENALTY_START),
    )
    jump_penalty = min(1.0, jump_penalty)

    mean_trajectory_smoothness = _mean_trajectory_smoothness_t(
        cluster_means, valid_mask,
    )

    # Spatial coherence requires per-point labels aligned with data_indices.
    # xtec-d/label-smooth: cluster_assignments == pixel_assignments, both
    # aligned with data_indices. xtec-s: cluster_assignments is per-peak;
    # pixel_assignments is per-pixel and aligned with data_indices.
    if pixel_assignments.shape[0] == data_indices.shape[0]:
        spatial_labels = pixel_assignments
    elif cluster_assignments.shape[0] == data_indices.shape[0]:
        spatial_labels = cluster_assignments
    else:
        spatial_labels = None

    spatial_coherence = (
        _spatial_coherence_t(data_indices, spatial_labels)
        if spatial_labels is not None
        else 0.0
    )

    metrics = {
        "bic": float(bic) if bic is not None else None,
        "n_points": int(n),
        "cluster_sizes": sizes,
        "min_cluster_frac": float(min_frac),
        "degeneracy_flag": bool(min_frac < DEGENERACY_FRAC_THRESHOLD),
        "size_penalty": float(size_penalty),
        "max_rel_jump": float(max_rel_jump),
        "jump_penalty": float(jump_penalty),
        "mean_pairwise_sep": float(mean_pairwise_sep),
        "mean_trajectory_smoothness": float(mean_trajectory_smoothness),
        "spatial_coherence": float(spatial_coherence),
    }
    return metrics


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _bic_norm_per_mode(candidates: List[Dict[str, Any]]) -> None:
    """Add bic_norm in [0, 1] to each candidate, scoped per mode.

    bic_norm = (bic - min_bic_in_mode) / (max_bic_in_mode - min_bic_in_mode)
    Lower bic => lower bic_norm => higher (1 - bic_norm) contribution.
    """
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_mode.setdefault(c["mode"], []).append(c)
    for mode_cands in by_mode.values():
        bics = [c["bic"] for c in mode_cands if c.get("bic") is not None]
        if not bics:
            for c in mode_cands:
                c.setdefault("metrics", {})["bic_norm"] = 0.0
            continue
        bmin, bmax = float(min(bics)), float(max(bics))
        span = bmax - bmin
        for c in mode_cands:
            metrics = c.setdefault("metrics", {})
            if c.get("bic") is None or span <= 0:
                metrics["bic_norm"] = 0.0
            else:
                metrics["bic_norm"] = float((c["bic"] - bmin) / span)


def _score_candidate(metrics: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Compute the multi-metric score for one candidate.

    Replaces the old boolean ``degeneracy_flag`` term with two continuous
    penalties:

    - ``size_penalty``: ramps from 0 to 1 as ``min_cluster_frac`` falls from
      ``SIZE_PENALTY_FRAC_THRESHOLD`` (1%) to 0. Catches small clusters that
      slip under absolute-threshold cliffs.
    - ``jump_penalty``: ramps from 0 to 1 as the worst cluster's single-step
      ``|Δ|/range`` rises from 0.5 to 1.0. Catches the "ordered trajectory
      shouldn't split with a jump" failure mode where a few outlier voxels
      form a spurious cluster with a spike-shaped mean.

    Both penalties accept weights so users can re-tune via the
    ``full-sweep-judge --judge-weights`` path without re-clustering.
    """
    score = (
        + weights["w_sep"] * metrics["mean_pairwise_sep"]
        + weights["w_coh"] * metrics["spatial_coherence"]
        + weights["w_bic"] * (1.0 - metrics.get("bic_norm", 0.0))
        - weights["w_smooth"] * metrics["mean_trajectory_smoothness"]
        - weights["w_size"] * metrics.get("size_penalty", 0.0)
        - weights["w_jump"] * metrics.get("jump_penalty", 0.0)
    )
    return float(score)


# ---------------------------------------------------------------------------
# Reasoning trace
# ---------------------------------------------------------------------------

def _build_reasoning(
    winner: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    bic_argmin: Optional[Dict[str, Any]],
) -> List[str]:
    """Generate 2-4 deterministic human-readable sentences about the choice."""
    out: List[str] = []
    if bic_argmin and bic_argmin["id"] == winner["id"]:
        # Explicit agreement: removes ambiguity about why the recommendation
        # exists when BIC and the multi-metric scorer happen to converge.
        out.append(
            f"BIC argmin and multi-metric score agree on {winner['id']} "
            f"(BIC={winner.get('bic'):.2f})."
        )
    elif bic_argmin:
        bm = bic_argmin
        bm_metrics = bm.get("metrics", {})
        reasons = []
        if bm_metrics.get("size_penalty", 0.0) > 0.5:
            reasons.append(
                f"smallest cluster is {bm_metrics.get('min_cluster_frac', 0.0) * 100:.3f}% "
                f"of points (size_penalty={bm_metrics['size_penalty']:.2f})"
            )
        if bm_metrics.get("jump_penalty", 0.0) > 0.3:
            reasons.append(
                f"worst cluster has a single-step jump of "
                f"{bm_metrics.get('max_rel_jump', 0.0) * 100:.1f}% of its range "
                f"(jump_penalty={bm_metrics['jump_penalty']:.2f})"
            )
        if reasons:
            out.append(
                f"BIC argmin was {bm['id']} (BIC={bm.get('bic'):.2f}) but "
                + " and ".join(reasons) + "; overridden."
            )
        else:
            out.append(
                f"BIC argmin was {bm['id']} (BIC={bm.get('bic'):.2f}) but score "
                f"favored {winner['id']} on separation/coherence."
            )

    w_metrics = winner.get("metrics", {})
    out.append(
        f"{winner['id']} chosen with score={winner['score']:.3f}: "
        f"separation={w_metrics.get('mean_pairwise_sep', 0.0):.2f}, "
        f"coherence={w_metrics.get('spatial_coherence', 0.0):.2f}, "
        f"smoothness={w_metrics.get('mean_trajectory_smoothness', 0.0):.3f}."
    )

    if len(ranked) >= 2:
        runner_up = ranked[1]
        margin = winner["score"] - runner_up["score"]
        out.append(
            f"Runner-up {runner_up['id']} (score={runner_up['score']:.3f}); "
            f"margin {margin:+.3f}."
        )

    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def score_and_recommend(
    output_root: Path,
    weights: Optional[Dict[str, float]] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Score every run under ``output_root/runs/`` and write recommendation.json.

    Parameters
    ----------
    output_root : Path
        Sweep directory containing ``runs/<id>/results.h5`` files and a
        ``sweep_summary.json`` written by the orchestrator.
    weights : dict, optional
        Override for the scoring weights. Defaults to
        ``DEFAULT_JUDGE_WEIGHTS``.
    top_n : int
        Number of ranked candidates to include in ``recommendation.json``.

    Returns
    -------
    dict
        The recommendation payload (also written to disk).
    """
    output_root = Path(output_root)
    summary_path = output_root / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"sweep_summary.json not found at {summary_path}. "
            "Run xtec-gpu full-sweep first."
        )
    summary = json.loads(summary_path.read_text())
    weights = {**DEFAULT_JUDGE_WEIGHTS, **(weights or {})}

    device = _judge_device()
    logger.info("Judge running on %s", device)

    # 1. Per-candidate metrics
    candidates = summary["candidates"]
    for cand in candidates:
        run_dir = output_root / cand["output_dir"]
        results_path = run_dir / "results.h5"
        if not results_path.exists():
            logger.warning("Missing results.h5 for %s, skipping", cand["id"])
            cand["metrics"] = None
            continue
        metrics = compute_run_metrics(
            results_path, bic=cand.get("bic"), device=device,
        )
        for key in PER_RUN_METRICS_KEYS:
            if key not in metrics:
                raise RuntimeError(
                    f"metrics for {cand['id']} missing required key: {key!r}"
                )
        cand["metrics"] = metrics
        # Persist per-run metrics.json
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 2. Per-mode BIC normalization
    valid = [c for c in candidates if c.get("metrics") is not None]
    _bic_norm_per_mode(valid)

    # 3. Score
    for cand in valid:
        cand["score"] = _score_candidate(cand["metrics"], weights)

    # 4. Rank
    ranked = sorted(valid, key=lambda c: c["score"], reverse=True)
    if not ranked:
        raise RuntimeError("No valid candidates to score.")
    winner = ranked[0]

    # 5. BIC argmin (across all modes)
    with_bic = [c for c in valid if c.get("bic") is not None]
    bic_argmin = min(with_bic, key=lambda c: c["bic"]) if with_bic else None

    # 6. Reasoning
    reasoning = _build_reasoning(winner, ranked, bic_argmin)

    # 7. Assemble payload
    payload: Dict[str, Any] = {
        "winner": {
            "id": winner["id"],
            "mode": winner["mode"],
            "k": winner["k"],
            "init": winner["init"],
            "output_dir": winner["output_dir"],
            "score": winner["score"],
            "bic": winner.get("bic"),
            "metrics": winner["metrics"],
        },
        "top_n": [
            {
                "id": c["id"],
                "mode": c["mode"],
                "k": c["k"],
                "init": c["init"],
                "output_dir": c["output_dir"],
                "score": c["score"],
                "bic": c.get("bic"),
                "metrics": c["metrics"],
            }
            for c in ranked[: max(1, int(top_n))]
        ],
        "reasoning": reasoning,
        "weights": weights,
        "bic_argmin": (
            {
                "id": bic_argmin["id"],
                "mode": bic_argmin["mode"],
                "k": bic_argmin["k"],
                "bic": bic_argmin["bic"],
            }
            if bic_argmin
            else None
        ),
        "bic_argmin_overridden": (
            bool(bic_argmin and bic_argmin["id"] != winner["id"])
            if bic_argmin
            else False
        ),
    }

    for key in RECOMMENDATION_REQUIRED_KEYS:
        if key not in payload:
            raise RuntimeError(f"Recommendation missing required key: {key}")

    rec_path = output_root / "recommendation.json"
    rec_path.write_text(json.dumps(payload, indent=2))
    logger.info("Recommendation written to %s", rec_path)

    # 8. Write back the updated sweep_summary.json (with metrics + score merged in)
    summary["candidates"] = candidates
    summary_path.write_text(json.dumps(summary, indent=2))

    # 9. Refresh the final_run symlink to point at the winner.
    # Living here (not in sweep.py) means re-judging with new weights also
    # updates the symlink, and a standalone full-sweep-judge invocation
    # leaves the directory in a consistent state even if the original sweep
    # died before its own symlink step.
    _update_final_run_symlink(output_root, winner["id"])

    return payload


def _update_final_run_symlink(output_root: Path, winner_id: str) -> None:
    """Point ``output_root/final_run`` at ``runs/<winner_id>`` (relative link)."""
    link = output_root / "final_run"
    target = Path("runs") / winner_id
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(target, link)
    except OSError as exc:
        logger.warning("Could not create final_run symlink: %s", exc)


def inspect(output_root: Path, top: int = 5) -> None:
    """Print the top-N candidates from an existing recommendation.json."""
    output_root = Path(output_root)
    rec_path = output_root / "recommendation.json"
    if not rec_path.exists():
        raise FileNotFoundError(
            f"recommendation.json not found at {rec_path}. "
            "Run xtec-gpu full-sweep-judge first."
        )
    payload = json.loads(rec_path.read_text())
    winner = payload["winner"]
    print(f"WINNER: {winner['id']}  score={winner['score']:.4f}  bic={winner.get('bic')}")
    print(f"  dir: {output_root / winner['output_dir']}")
    for line in payload.get("reasoning", []):
        print(f"  - {line}")
    print()
    print(f"TOP {min(top, len(payload['top_n']))}:")
    for rank, c in enumerate(payload["top_n"][:top], start=1):
        m = c.get("metrics") or {}
        print(
            f"  {rank}. {c['id']:<28} score={c['score']:+.4f}  "
            f"bic={c.get('bic')}  min_frac={m.get('min_cluster_frac', 0.0):.4f}  "
            f"sep={m.get('mean_pairwise_sep', 0.0):.2f}  "
            f"coh={m.get('spatial_coherence', 0.0):.2f}"
        )
        print(f"     dir: {output_root / c['output_dir']}")


__all__ = [
    "compute_run_metrics",
    "score_and_recommend",
    "inspect",
]
