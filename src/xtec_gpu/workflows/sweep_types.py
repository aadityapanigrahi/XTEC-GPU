"""Shared types, constants, and helpers for the full-sweep workflow.

Used by:
- ``xtec_gpu.workflows.sweep``  — orchestrator
- ``xtec_gpu.workflows.judge``  — offline scorer
- ``xtec_gpu.xtec_cli``         — inline-artifact path inside BIC sweeps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Init-strategy classification
# ---------------------------------------------------------------------------

GPU_INITS: Tuple[str, ...] = ("kmeans++",)
CPU_INITS: Tuple[str, ...] = ("sklearn-kmeans", "xtec")
ALL_INITS: Tuple[str, ...] = GPU_INITS + CPU_INITS + ("cuml-kmeans",)

INIT_SLUGS: Dict[str, str] = {
    "kmeans++": "kmeanspp",
    "sklearn-kmeans": "sklearnkmeans",
    "cuml-kmeans": "cumlkmeans",
    "xtec": "xtec",
}


def init_slug(init: str) -> str:
    """Return filename-safe slug for an init strategy string."""
    try:
        return INIT_SLUGS[init]
    except KeyError as exc:
        raise ValueError(
            f"Unknown init strategy {init!r}. "
            f"Expected one of {sorted(INIT_SLUGS)}."
        ) from exc


def run_id(mode: str, k: int, init: str) -> str:
    """Canonical run identifier used for directory names and summary keys."""
    return f"{mode}_k{int(k):02d}_{init_slug(init)}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FullSweepConfig:
    """Configuration for one full-sweep invocation."""

    input_path: str
    output_root: Path

    # Preprocessing
    entry: str = "entry/data"
    slices: Optional[str] = None
    threshold: bool = True
    rescale: str = "mean"

    # Compute
    device: str = "auto"

    # Sweep
    min_nc: int = 2
    max_nc: int = 8  # exclusive upper bound, matches np.arange semantics
    modes: List[str] = field(default_factory=lambda: ["d", "s"])
    inits: List[str] = field(default_factory=lambda: list(GPU_INITS))
    include_cpu_inits: bool = False

    # Output
    plots_level: str = "all"  # "all" | "primary" | "none"
    random_state: int = 0
    reorder_clusters: bool = True

    # Streaming
    stream_mode: str = "auto"  # "auto" | "on" | "off"
    stream_threshold_gb: float = 2.0
    streamed_chunk_voxels: int = 0
    streamed_reservoir_size: int = 500000
    streamed_max_bins: int = 4096
    streamed_exact_log_limit: int = 50000000
    streamed_seed: int = 0

    # Orchestration
    resume: bool = False
    dry_run: bool = False

    # Judge
    judge_weights: Optional[Dict[str, float]] = None
    judge_top_n: int = 5


# ---------------------------------------------------------------------------
# Default judge weights
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_WEIGHTS: Dict[str, float] = {
    "w_sep": 1.0,
    "w_coh": 0.5,
    "w_bic": 0.2,
    "w_smooth": 0.3,
    "w_size": 3.0,
    "w_jump": 2.0,
}

# Continuous size-penalty kicks in below this fraction; cluster fraction at
# or below 0 yields penalty 1.0, at or above the threshold yields 0.
SIZE_PENALTY_FRAC_THRESHOLD: float = 0.01

# Jump penalty: a single-step trajectory delta is judged relative to the
# cluster's own trajectory range. Penalty starts at this fraction and
# saturates at 1.0 when the single-step delta IS the whole range.
JUMP_PENALTY_START: float = 0.5

# Legacy boolean degeneracy flag still reported in metrics.json so users have
# a quick "is anything below 0.5%" view, but it no longer enters the score.
DEGENERACY_FRAC_THRESHOLD: float = 0.005

# Subsample cap for the spatial-coherence k-NN computation. Coherence only needs
# a representative estimate; computing pairwise distances on the full data is
# wasteful when N is large.
SPATIAL_COHERENCE_MAX_SAMPLES: int = 50_000
SPATIAL_COHERENCE_KNN: int = 8


# ---------------------------------------------------------------------------
# Required keys for downstream consumers / contract tests
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_KEYS: Tuple[str, ...] = (
    "input",
    "output_root",
    "config",
    "device",
    "torch_version",
    "git_sha",
    "host",
    "plan",
    "started_at",
    "finished_at",
)

SWEEP_SUMMARY_REQUIRED_KEYS: Tuple[str, ...] = (
    "candidates",
    "manifest_ref",
)

RECOMMENDATION_REQUIRED_KEYS: Tuple[str, ...] = (
    "winner",
    "top_n",
    "reasoning",
    "weights",
    "bic_argmin",
    "bic_argmin_overridden",
)

PER_RUN_METRICS_KEYS: Tuple[str, ...] = (
    "bic",
    "n_points",
    "cluster_sizes",
    "min_cluster_frac",
    "degeneracy_flag",
    "size_penalty",
    "max_rel_jump",
    "jump_penalty",
    "mean_pairwise_sep",
    "mean_trajectory_smoothness",
    "spatial_coherence",
)


__all__ = [
    "GPU_INITS",
    "CPU_INITS",
    "ALL_INITS",
    "INIT_SLUGS",
    "init_slug",
    "run_id",
    "FullSweepConfig",
    "DEFAULT_JUDGE_WEIGHTS",
    "DEGENERACY_FRAC_THRESHOLD",
    "SPATIAL_COHERENCE_MAX_SAMPLES",
    "SPATIAL_COHERENCE_KNN",
    "MANIFEST_REQUIRED_KEYS",
    "SWEEP_SUMMARY_REQUIRED_KEYS",
    "RECOMMENDATION_REQUIRED_KEYS",
    "PER_RUN_METRICS_KEYS",
]
