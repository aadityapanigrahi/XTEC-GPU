from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from xtec_gpu.config import AgenticWorkflowConfig
from xtec_gpu.workflows import agentic
from xtec_gpu.workflows import WORKFLOW_REPORT_REQUIRED_KEYS
from xtec_gpu.workflows.shared import build_bic_command
from xtec_gpu import xtec_cli
from xtec_gpu.config.run_config import CommonRunConfig
from xtec_gpu.streamed_preprocessing import _estimate_cutoff, _read_block_full_temperature
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

    def test_runtime_cache_reuses_loaded_nxdata(self) -> None:
        # Guards Phase 1 optimization: repeated in-process calls should reuse NXdata.
        args = SimpleNamespace(input="input.nxs", runtime_cache={})
        with patch.object(xtec_cli, "_load_data", return_value=object()) as load_mock:
            a = xtec_cli._get_or_load_data(args, "entry/data", ":,0:1,-1:1,-1:1")
            b = xtec_cli._get_or_load_data(args, "entry/data", ":,0:1,-1:1,-1:1")
            self.assertIs(a, b)
            self.assertEqual(load_mock.call_count, 1)

    def test_runtime_cache_reuses_threshold_for_d_mode(self) -> None:
        # Guards Phase 2 optimization: d-mode thresholding is computed once per shared key.
        args = SimpleNamespace(input="input.nxs", runtime_cache={})
        cfg = CommonRunConfig(entry="entry/data", slices=":,0:1,-1:1,-1:1", threshold=True, device="cuda:1")
        fake_data = object()
        with patch.object(xtec_cli, "_build_threshold", return_value=object()) as build_mock:
            a = xtec_cli._get_or_build_threshold_d(args, fake_data, cfg, "cuda:1")
            b = xtec_cli._get_or_build_threshold_d(args, fake_data, cfg, "cuda:1")
            self.assertIs(a, b)
            self.assertEqual(build_mock.call_count, 1)

    def test_runtime_cache_reuses_s_mode_preprocessing(self) -> None:
        # Guards Phase 3 optimization: s-mode preprocessing bundle is computed once.
        args = SimpleNamespace(input="input.nxs", runtime_cache={})
        cfg = CommonRunConfig(entry="entry/data", slices=":,0:1,-1:1,-1:1", threshold=True, device="cuda:1")
        fake_data = object()
        fake_bundle = (object(), object(), object())
        with patch.object(xtec_cli, "_build_s_preprocessed", return_value=fake_bundle) as build_mock:
            a = xtec_cli._get_or_build_s_preprocessed(args, fake_data, cfg, "cuda:1")
            b = xtec_cli._get_or_build_s_preprocessed(args, fake_data, cfg, "cuda:1")
            self.assertIs(a, b)
            self.assertEqual(build_mock.call_count, 1)

    def test_cli_streamed_preprocess_defaults(self) -> None:
        # Guards Phase 4 CLI defaults: opt-in only.
        parser = build_parser()
        args_d = parser.parse_args(["xtec-d", "in.nxs", "-o", "out"])
        self.assertFalse(args_d.streamed_preprocess)
        self.assertEqual(args_d.streamed_chunk_voxels, 0)
        self.assertEqual(args_d.streamed_reservoir_size, 500000)

    def test_workflow_command_includes_streamed_flags_when_enabled(self) -> None:
        # Guards Phase 4 subprocess parity: shared command builders carry streamed options.
        cfg = AgenticWorkflowConfig(
            input_path="input.nxs",
            output_root=Path("."),
            streamed_preprocess=True,
            streamed_chunk_voxels=123,
            streamed_reservoir_size=456,
            streamed_max_bins=789,
            streamed_exact_log_limit=111,
            streamed_seed=7,
        )
        cmd = build_bic_command("d", "input.nxs", Path("out"), cfg)
        self.assertIn("--streamed-preprocess", cmd)
        self.assertIn("--streamed-chunk-voxels", cmd)
        self.assertIn("123", cmd)
        self.assertIn("--streamed-seed", cmd)
        self.assertIn("7", cmd)

    def test_runtime_cache_reuses_streamed_threshold_for_d_mode(self) -> None:
        # Guards Phase 4 in-process caching for streamed d-mode preprocessing.
        args = SimpleNamespace(input="input.nxs", runtime_cache={})
        cfg = CommonRunConfig(
            entry="entry/data",
            slices=None,
            threshold=True,
            device="cpu",
            streamed_preprocess=True,
            streamed_chunk_voxels=32,
            streamed_reservoir_size=64,
            streamed_max_bins=128,
            streamed_exact_log_limit=256,
            streamed_seed=3,
        )
        fake_data = object()
        fake_threshold = object()
        with patch.object(xtec_cli, "_build_threshold_d", return_value=fake_threshold) as build_mock:
            a = xtec_cli._get_or_build_threshold_d(args, fake_data, cfg, "cpu")
            b = xtec_cli._get_or_build_threshold_d(args, fake_data, cfg, "cpu")
            self.assertIs(a, b)
            self.assertEqual(build_mock.call_count, 1)

    def test_streamed_preprocess_rejects_runtime_slices(self) -> None:
        # Guards fail-fast behavior: streamed preprocessing must not silently fallback.
        cfg = CommonRunConfig(
            entry="entry/data",
            slices=":,0:1,-1:1,-1:1",
            threshold=True,
            device="cpu",
            streamed_preprocess=True,
        )
        with self.assertRaises(ValueError):
            xtec_cli._build_threshold_d(
                SimpleNamespace(input="input.nxs", runtime_cache={}),
                object(),
                cfg,
                "cpu",
            )

    def test_streamed_cutoff_uses_exact_kl_path(self) -> None:
        # Guards exact streamed cutoff path and metadata shape.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.nxs"
            data = np.full((4, 3, 3), 10.0, dtype=np.float64)
            with h5py.File(path, "w") as h5:
                grp = h5.create_group("entry")
                grp.attrs["signal"] = "data"
                grp.create_dataset("data", data=data)
            with h5py.File(path, "r") as h5:
                ds = h5["entry/data"]
                with patch(
                    "xtec_gpu.streamed_preprocessing._exact_kl_cutoff_from_logs",
                    return_value=(1.23, "exact-kl", True, 0.5, 8),
                ) as exact_mock:
                    stats = _estimate_cutoff(
                        ds,
                        threshold_enabled=True,
                        chunk_voxels=4,
                        reservoir_size=2,
                        max_bins=2,
                        exact_log_limit=10_000,
                        seed=0,
                        compute_device=xtec_cli._get_device("cpu"),
                    )
        self.assertTrue(exact_mock.called)
        self.assertEqual(stats.mode, "exact-kl")
        self.assertTrue(stats.success)
        self.assertAlmostEqual(stats.cutoff, 1.23, places=12)
        self.assertIsNotNone(stats.exact_iqr)
        self.assertIsNotNone(stats.approx_iqr)
        self.assertAlmostEqual(float(stats.exact_iqr), float(stats.approx_iqr), places=12)

    def test_streamed_cutoff_raises_when_exact_limit_exceeded(self) -> None:
        # Guards fail-fast behavior: no silent fallback when exact budget is exceeded.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny_skip.nxs"
            data = np.full((3, 4, 4), 5.0, dtype=np.float64)
            with h5py.File(path, "w") as h5:
                grp = h5.create_group("entry")
                grp.attrs["signal"] = "data"
                grp.create_dataset("data", data=data)
            with h5py.File(path, "r") as h5:
                ds = h5["entry/data"]
                with self.assertRaises(RuntimeError):
                    _estimate_cutoff(
                        ds,
                        threshold_enabled=True,
                        chunk_voxels=4,
                        reservoir_size=2,
                        max_bins=2,
                        exact_log_limit=1,
                        seed=0,
                        compute_device=xtec_cli._get_device("cpu"),
                    )

    def test_streamed_block_keeps_full_temperature_axis(self) -> None:
        # Guards chunking contract: chunk spatial/momentum axes only, never temperature.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny_temp_axis.nxs"
            data = np.arange(5 * 4 * 6 * 7, dtype=np.float64).reshape(5, 4, 6, 7)
            with h5py.File(path, "w") as h5:
                grp = h5.create_group("entry")
                grp.attrs["signal"] = "data"
                grp.create_dataset("data", data=data)
            with h5py.File(path, "r") as h5:
                ds = h5["entry/data"]
                block = _read_block_full_temperature(
                    ds,
                    (slice(1, 3), slice(2, 5), slice(0, 4)),
                )
        self.assertEqual(block.shape[0], data.shape[0])
        self.assertEqual(block.shape[1:], (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
