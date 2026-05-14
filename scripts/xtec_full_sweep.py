"""Convenience wrapper around xtec_gpu.workflows.sweep.main.

Mirrors the existing scripts/xtec_agentic_workflow.py convention so users have
a familiar entry point. The same command is also available as the
``xtec-gpu full-sweep`` subcommand of the main CLI.
"""

from xtec_gpu.workflows.sweep import main


if __name__ == "__main__":
    main()
