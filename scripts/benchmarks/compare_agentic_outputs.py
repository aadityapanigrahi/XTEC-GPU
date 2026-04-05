#!/usr/bin/env python3
"""Compare two agentic workflow output directories for parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _load_bic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        ks = np.asarray(f["n_clusters"][...], dtype=int)
        bics = np.asarray(f["bic_scores"][...], dtype=float)
    return ks, bics


def _load_results(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {
            "cluster_assignments": np.asarray(f["cluster_assignments"][...]),
            "pixel_assignments": np.asarray(f["pixel_assignments"][...]),
            "data_indices": np.asarray(f["data_indices"][...]),
            "data_thresholded": np.asarray(f["data_thresholded"][...]),
            "cluster_means": np.asarray(f["cluster_means"][...]),
            "cluster_covariances": np.asarray(f["cluster_covariances"][...]),
        }


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old-root", required=True, help="Reference workflow output root")
    p.add_argument("--new-root", required=True, help="Candidate workflow output root")
    p.add_argument("--out-json", default=None, help="Optional output json path")
    args = p.parse_args()

    old_root = Path(args.old_root)
    new_root = Path(args.new_root)

    old_report = _load_json(old_root / "workflow_report.json")
    new_report = _load_json(new_root / "workflow_report.json")

    out: Dict[str, object] = {
        "old_root": str(old_root),
        "new_root": str(new_root),
        "recommendation_equal": old_report.get("recommendation") == new_report.get("recommendation"),
        "recommendation_old": old_report.get("recommendation"),
        "recommendation_new": new_report.get("recommendation"),
        "bic": {},
    }

    for mode in ("d", "s"):
        old_bic_h5 = old_root / "bic_sweeps" / f"bic_{mode}" / f"bic_xtec_{mode}.h5"
        new_bic_h5 = new_root / "bic_sweeps" / f"bic_{mode}" / f"bic_xtec_{mode}.h5"
        if old_bic_h5.exists() and new_bic_h5.exists():
            old_ks, old_bics = _load_bic(old_bic_h5)
            new_ks, new_bics = _load_bic(new_bic_h5)
            out["bic"][mode] = {
                "n_clusters_equal": bool(np.array_equal(old_ks, new_ks)),
                "scores_max_abs_diff": _safe_float(np.max(np.abs(old_bics - new_bics))) if old_bics.shape == new_bics.shape else None,
                "scores_mean_abs_diff": _safe_float(np.mean(np.abs(old_bics - new_bics))) if old_bics.shape == new_bics.shape else None,
            }

    old_mode = old_report["recommendation"]["mode"]
    new_mode = new_report["recommendation"]["mode"]
    old_res = old_root / "final_run" / f"xtec_{old_mode}" / "results.h5"
    new_res = new_root / "final_run" / f"xtec_{new_mode}" / "results.h5"
    if old_res.exists() and new_res.exists():
        old_h5 = _load_results(old_res)
        new_h5 = _load_results(new_res)
        same_assign_shape = old_h5["cluster_assignments"].shape == new_h5["cluster_assignments"].shape
        assign_equal = bool(np.array_equal(old_h5["cluster_assignments"], new_h5["cluster_assignments"])) if same_assign_shape else False
        assign_match_ratio = (
            float(np.mean(old_h5["cluster_assignments"] == new_h5["cluster_assignments"]))
            if same_assign_shape
            else None
        )
        out["final_results"] = {
            "mode_old": old_mode,
            "mode_new": new_mode,
            "cluster_assignments_equal": assign_equal,
            "cluster_assignments_match_ratio": assign_match_ratio,
            "cluster_means_mae": _mae(old_h5["cluster_means"], new_h5["cluster_means"]) if old_h5["cluster_means"].shape == new_h5["cluster_means"].shape else None,
            "cluster_means_linf": _linf(old_h5["cluster_means"], new_h5["cluster_means"]) if old_h5["cluster_means"].shape == new_h5["cluster_means"].shape else None,
            "cluster_covariances_mae": _mae(old_h5["cluster_covariances"], new_h5["cluster_covariances"]) if old_h5["cluster_covariances"].shape == new_h5["cluster_covariances"].shape else None,
            "cluster_covariances_linf": _linf(old_h5["cluster_covariances"], new_h5["cluster_covariances"]) if old_h5["cluster_covariances"].shape == new_h5["cluster_covariances"].shape else None,
        }

    print(json.dumps(out, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

