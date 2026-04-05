"""Typed data models for workflow orchestration outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class WorkflowRecommendation(TypedDict):
    mode: str
    n_clusters: int
    init_strategy_mode: Optional[str]


class WorkflowReport(TypedDict):
    input: str
    output_root: str
    settings: Dict[str, Any]
    bic_results: Dict[str, Dict[str, Any]]
    recommendation: WorkflowRecommendation
    final_command: Optional[List[str]]
    sweep_artifacts: Optional[Dict[str, List[Dict[str, Any]]]]
