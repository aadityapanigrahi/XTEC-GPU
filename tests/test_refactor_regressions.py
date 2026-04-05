from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xtec_gpu.config import AgenticWorkflowConfig
from xtec_gpu.workflows import agentic
from xtec_gpu.workflows import WORKFLOW_REPORT_REQUIRED_KEYS
from xtec_gpu.xtec_cli import build_parser


class RefactorRegressionTests(unittest.TestCase):
    def test_workflow_required_keys_constant(self) -> None:
        # Guards report schema stability for downstream tooling.
        expected = {
            "input",
            "output_root",
            "settings",
            "bic_results",
            "recommendation",
            "final_command",
        }
        self.assertEqual(set(WORKFLOW_REPORT_REQUIRED_KEYS), expected)

    def test_cli_default_init_strategy_unchanged(self) -> None:
        # Guards user-facing CLI defaults.
        parser = build_parser()
        args_d = parser.parse_args(["xtec-d", "in.nxs", "-o", "out"])
        args_s = parser.parse_args(["xtec-s", "in.nxs", "-o", "out"])
        self.assertEqual(args_d.init_strategy_mode, "kmeans++")
        self.assertEqual(args_s.init_strategy_mode, "kmeans++")

    def test_agentic_report_shape_with_mocked_execution(self) -> None:
        # Guards workflow recommendation/report assembly without heavy compute.
        with tempfile.TemporaryDirectory() as td:
            cfg = AgenticWorkflowConfig(
                input_path="input.nxs",
                output_root=Path(td),
                candidate_modes=("d", "s"),
                run_final=False,
                save_sweep_artifacts=False,
            )

            fake_bic = {
                "d": {
                    "mode": "d",
                    "command": ["cmd"],
                    "n_clusters": [2, 3],
                    "bic_scores": [10.0, 9.0],
                    "best_k": 3,
                    "best_bic": 9.0,
                },
                "s": {
                    "mode": "s",
                    "command": ["cmd"],
                    "n_clusters": [2, 3],
                    "bic_scores": [11.0, 10.0],
                    "best_k": 3,
                    "best_bic": 10.0,
                },
            }

            def fake_run_bic(mode, *_args, **_kwargs):
                return fake_bic[mode]

            with patch.object(agentic, "_run_bic", side_effect=fake_run_bic):
                report = agentic.recommend_workflow(cfg)

            for key in WORKFLOW_REPORT_REQUIRED_KEYS:
                self.assertIn(key, report)
            self.assertEqual(report["recommendation"]["mode"], "d")
            self.assertEqual(report["recommendation"]["n_clusters"], 3)
            self.assertIsNone(report["final_command"])


if __name__ == "__main__":
    unittest.main()
