#!/usr/bin/env python3
"""
MCP server for the XTEC agentic workflow.

Exposes:
- recommend_workflow: run BIC sweeps and recommend mode/k.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from xtec_agentic_workflow import RunConfig, recommend_workflow

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
    random_state: int = 0,
    run_final: bool = True,
    save_sweep_artifacts: bool = True,
) -> dict:
    cfg = RunConfig(
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
        random_state=random_state,
        run_final=run_final,
        save_sweep_artifacts=save_sweep_artifacts,
    )
    return recommend_workflow(cfg)


if __name__ == "__main__":
    mcp.run()
