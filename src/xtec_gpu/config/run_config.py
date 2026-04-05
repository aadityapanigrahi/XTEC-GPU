"""Shared run configuration models used across CLI and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class CommonRunConfig:
    """Common runtime options used across XTEC commands."""

    entry: str = "entry/data"
    slices: Optional[str] = None
    threshold: bool = True
    rescale: str = "mean"
    device: str = "auto"
    init_strategy_mode: str = "kmeans++"
    streamed_preprocess: bool = False
    streamed_chunk_voxels: int = 0
    streamed_reservoir_size: int = 500000
    streamed_max_bins: int = 4096
    streamed_exact_log_limit: int = 20000000
    streamed_seed: int = 0


@dataclass(frozen=True)
class SweepConfig:
    """Cluster sweep configuration."""

    min_nc: int = 2
    max_nc: int = 14


@dataclass(frozen=True)
class AgenticWorkflowConfig(CommonRunConfig, SweepConfig):
    """Configuration for agentic recommendation workflow runs."""

    input_path: str = ""
    output_root: Path = Path(".")
    candidate_modes: Sequence[str] = ("d", "s")
    random_state: int = 0
    run_final: bool = True
    save_sweep_artifacts: bool = True
    execution_backend: str = "inprocess"
