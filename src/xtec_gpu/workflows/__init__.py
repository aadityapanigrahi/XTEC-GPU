"""Workflow orchestration package.

Phase 1 scaffolding: existing scripts remain the active entry points.
"""

from .agentic import config_from_args, recommend_workflow
from .shared import WORKFLOW_REPORT_REQUIRED_KEYS
from .types import WorkflowReport, WorkflowRecommendation

__all__ = [
    "config_from_args",
    "recommend_workflow",
    "WORKFLOW_REPORT_REQUIRED_KEYS",
    "WorkflowReport",
    "WorkflowRecommendation",
]
