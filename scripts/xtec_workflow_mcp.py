#!/usr/bin/env python3
"""MCP server entry point for the XTEC agentic workflow."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xtec_gpu.config import AgenticWorkflowConfig
from xtec_gpu.workflows.agentic import recommend_workflow

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "MCP SDK is required. Install with: pip install mcp"
    ) from exc


mcp = FastMCP("xtec-workflow")


@mcp.tool()
def recommend_xtec_workflow(
    input_path: str,
    output_root: str,
    entry: str = "entry/data",
    slices: Optional[str] = None,
    threshold: bool = True,
    rescale: str = "mean",
    device: str = "auto",
    min_nc: int = 2,
    max_nc: int = 14,
    candidate_modes: str = "d,s",
    init_strategy_mode: str = "kmeans++",
    random_state: int = 0,
    run_final: bool = True,
    save_sweep_artifacts: bool = True,
) -> dict:
    cfg = AgenticWorkflowConfig(
        input_path=input_path,
        output_root=Path(output_root),
        entry=entry,
        slices=slices,
        threshold=threshold,
        rescale=rescale,
        device=device,
        min_nc=min_nc,
        max_nc=max_nc,
        candidate_modes=[x.strip() for x in candidate_modes.split(",") if x.strip()],
        init_strategy_mode=init_strategy_mode,
        random_state=random_state,
        run_final=run_final,
        save_sweep_artifacts=save_sweep_artifacts,
    )
    return recommend_workflow(cfg)


if __name__ == "__main__":
    mcp.run()
