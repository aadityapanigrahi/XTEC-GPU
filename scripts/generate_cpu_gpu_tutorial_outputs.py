#!/usr/bin/env python3
"""Entry point for CPU/GPU tutorial comparison workflow.

Runtime behavior is implemented in `xtec_gpu.workflows.comparison`.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xtec_gpu.workflows.comparison import main


if __name__ == "__main__":
    main()
