"""Streamed preprocessing helpers for large NX datasets.

This module provides an opt-in thresholding path for XTEC-d style workflows
that need to process large datasets without materializing the full input
volume in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterator, List, Sequence, Tuple

import h5py
import numpy as np
import torch

from .Preprocessing import Threshold_Background

_DEFAULT_STREAM_CHUNK_BYTES = 1 << 30  # 1 GiB


def _decode_attr_value(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar = value.item()
        if isinstance(scalar, bytes):
            return scalar.decode()
        return scalar
    return value


def _resolve_signal_dataset(h5: h5py.File, entry_path: str) -> h5py.Dataset:
    group = h5[entry_path]
    signal_name = _decode_attr_value(group.attrs.get("signal", "data"))
    if isinstance(signal_name, np.ndarray):
        if signal_name.size == 0:
            signal_name = "data"
        else:
            signal_name = _decode_attr_value(signal_name.flat[0])
    if not isinstance(signal_name, str):
        signal_name = str(signal_name)
    return group[signal_name]


def _chunk_shape(spatial_shape: Sequence[int], target_voxels: int) -> Tuple[int, ...]:
    ndim = len(spatial_shape)
    if ndim == 0:
        return tuple()
    # Fill chunk sizes progressively across axes so the product approaches the
    # target even for anisotropic shapes (e.g., small first momentum axis).
    remaining = max(1.0, float(target_voxels))
    out: List[int] = []
    for axis, size in enumerate(spatial_shape):
        remaining_axes = max(1, ndim - axis)
        ideal = int(round(remaining ** (1.0 / float(remaining_axes))))
        c = max(1, min(int(size), ideal))
        out.append(c)
        remaining = max(1.0, remaining / float(c))
    return tuple(out)


def _resolve_chunk_voxels(data_shape: Sequence[int], requested_chunk_voxels: int) -> Tuple[int, bool]:
    """Resolve spatial chunk voxels from request; <=0 means auto (~1 GiB target)."""
    req = int(requested_chunk_voxels)
    if req > 0:
        return req, False
    t = max(1, int(data_shape[0]))
    bytes_per_voxel = t * np.dtype(np.float64).itemsize
    auto_voxels = max(1, int(_DEFAULT_STREAM_CHUNK_BYTES // max(1, bytes_per_voxel)))
    return auto_voxels, True


def _iter_spatial_chunks(
    spatial_shape: Sequence[int], target_voxels: int
) -> Iterator[Tuple[Tuple[int, ...], Tuple[slice, ...]]]:
    cshape = _chunk_shape(spatial_shape, target_voxels)
    starts_per_axis = [range(0, int(size), int(csize)) for size, csize in zip(spatial_shape, cshape)]
    for starts in product(*starts_per_axis):
        slices = tuple(
            slice(int(s), min(int(s) + int(csize), int(size)))
            for s, csize, size in zip(starts, cshape, spatial_shape)
        )
        yield tuple(int(x) for x in starts), slices


def _read_block_full_temperature(
    dataset: h5py.Dataset,
    spatial_slices: Sequence[slice],
) -> np.ndarray:
    """Read a streamed block while preserving the full temperature axis.

    Chunking is only permitted across spatial/momentum axes. The leading
    temperature axis is always read in full.
    """
    block = np.asarray(dataset[(slice(None),) + tuple(spatial_slices)], dtype=np.float64)
    if block.shape[0] != int(dataset.shape[0]):
        raise RuntimeError(
            "Streamed block must preserve the full temperature axis; "
            f"expected {int(dataset.shape[0])}, got {int(block.shape[0])}."
        )
    return block


@dataclass
class _CutoffStats:
    cutoff: float
    mode: str
    success: bool
    n_valid: int
    min_log: float
    max_log: float
    bin_width: float
    n_bins: int
    approx_iqr: float | None = None
    exact_iqr: float | None = None


def _exact_kl_cutoff_from_logs(
    log_sample: np.ndarray,
    *,
    mean_global: float,
    std_global: float,
    compute_device: torch.device,
) -> Tuple[float, str, bool, float, int]:
    helper = Threshold_Background.__new__(Threshold_Background)
    log_t = torch.as_tensor(log_sample, dtype=torch.float64, device=compute_device)
    log_sorted_t = torch.sort(log_t).values
    bin_size_t = helper.Freedman_Diaconis_for_bin_width(log_sorted_t, device=compute_device)
    y_bins_t, x_bins_t = helper.hist(log_t, bin_size_t, device=compute_device)
    helper.bin_size = torch.as_tensor(float(bin_size_t.item()), dtype=torch.float64, device=compute_device)
    try:
        opt_idx = helper.Truncate(
            x_bins_t,
            y_bins_t,
            log_sorted_t,
            max_iter=100,
            device=compute_device,
            dtype=torch.float64,
        )
        cutoff = float(x_bins_t[int(opt_idx)].item())
    except Exception as exc:
        raise RuntimeError(
            "Exact streamed KL cutoff failed during Truncate(). "
            "No fallback path is enabled."
        ) from exc

    sanity_ok = not (
        (cutoff > mean_global + 2.0 * std_global) or (cutoff < mean_global - std_global)
    )
    if not sanity_ok:
        raise RuntimeError(
            "Exact streamed KL cutoff failed sanity bounds. "
            "No fallback path is enabled."
        )
    mode = "exact-kl"
    success = True
    return cutoff, mode, success, float(bin_size_t.item()), int(x_bins_t.numel())


class StreamedThresholdResult:
    """Threshold result object compatible with xtec_cli d-mode pathways."""

    Rescale_mean = Threshold_Background.Rescale_mean
    Rescale_zscore = Threshold_Background.Rescale_zscore

    def __init__(
        self,
        *,
        device: torch.device,
        data_shape_orig: Tuple[int, ...],
        threshold_type: str,
        logi_cutoff: float,
        data_thresholded: torch.Tensor,
        ind_thresholded: torch.Tensor,
        diagnostics: Dict[str, object],
    ) -> None:
        self.device = torch.device(device)
        self.data_shape_orig = tuple(int(x) for x in data_shape_orig)
        self.threshold_type = str(threshold_type)
        self.success = bool(diagnostics.get("success", True))
        self.LogI_cutoff = torch.as_tensor(float(logi_cutoff), device=self.device, dtype=torch.float64)
        self.data_thresholded = data_thresholded
        self.ind_thresholded = ind_thresholded
        # Full dense mask is intentionally omitted to avoid O(spatial_volume) memory.
        self.thresholded = None
        self.streaming = dict(diagnostics)


def _estimate_cutoff(
    dataset: h5py.Dataset,
    *,
    threshold_enabled: bool,
    chunk_voxels: int,
    reservoir_size: int,
    max_bins: int,
    exact_log_limit: int,
    seed: int,
    compute_device: torch.device,
) -> _CutoffStats:
    # Keep older args for CLI/workflow compatibility; exact streamed KL is now
    # the default robust path.
    _ = (reservoir_size, max_bins, seed)

    if not threshold_enabled:
        return _CutoffStats(
            cutoff=-1e6,
            mode="no-threshold",
            success=True,
            n_valid=0,
            min_log=float("nan"),
            max_log=float("nan"),
            bin_width=float("nan"),
            n_bins=0,
            approx_iqr=None,
            exact_iqr=None,
        )

    spatial_shape = tuple(int(x) for x in dataset.shape[1:])
    n_valid = 0
    sum_log = 0.0
    sumsq_log = 0.0
    min_log = float("inf")
    max_log = float("-inf")
    exact_logs: List[np.ndarray] = []

    for _starts, spatial_slices in _iter_spatial_chunks(spatial_shape, chunk_voxels):
        block = _read_block_full_temperature(dataset, spatial_slices)
        mean_t = block.mean(axis=0)
        valid = np.isfinite(mean_t) & (mean_t > 0.0)
        if not np.any(valid):
            continue
        log_vals = np.log(mean_t[valid]).astype(np.float64, copy=False)
        n = int(log_vals.size)
        n_valid += n
        sum_log += float(log_vals.sum())
        sumsq_log += float(np.square(log_vals).sum())
        min_log = min(min_log, float(log_vals.min()))
        max_log = max(max_log, float(log_vals.max()))
        exact_logs.append(log_vals.copy())
        if int(exact_log_limit) > 0 and n_valid > int(exact_log_limit):
            raise RuntimeError(
                "Streamed exact cutoff exceeded --streamed-exact-log-limit. "
                f"n_valid={n_valid}, limit={int(exact_log_limit)}. "
                "Increase --streamed-exact-log-limit or reduce input volume."
            )

    if n_valid == 0:
        return _CutoffStats(
            cutoff=-1e6,
            mode="empty-valid-set",
            success=True,
            n_valid=0,
            min_log=float("nan"),
            max_log=float("nan"),
            bin_width=float("nan"),
            n_bins=0,
            approx_iqr=None,
            exact_iqr=None,
        )

    log_sample = np.concatenate(exact_logs, axis=0) if exact_logs else np.empty(0, dtype=np.float64)
    if log_sample.size == 0:
        return _CutoffStats(
            cutoff=-1e6,
            mode="empty-log-sample",
            success=True,
            n_valid=n_valid,
            min_log=min_log,
            max_log=max_log,
            bin_width=float("nan"),
            n_bins=0,
            approx_iqr=None,
            exact_iqr=None,
        )

    n_float = float(max(1, n_valid))
    mean_global = sum_log / n_float
    var_global = max(sumsq_log / n_float - mean_global * mean_global, 0.0)
    std_global = float(np.sqrt(var_global))

    q25_exact, q75_exact = np.quantile(log_sample, [0.25, 0.75], method="linear")
    iqr_exact = float(q75_exact - q25_exact)
    cutoff, mode, success, bin_width, n_bins = _exact_kl_cutoff_from_logs(
        log_sample,
        mean_global=mean_global,
        std_global=std_global,
        compute_device=compute_device,
    )

    return _CutoffStats(
        cutoff=cutoff,
        mode=mode,
        success=success,
        n_valid=n_valid,
        min_log=min_log,
        max_log=max_log,
        bin_width=bin_width,
        n_bins=n_bins,
        approx_iqr=iqr_exact,
        exact_iqr=iqr_exact,
    )


def build_streamed_threshold_result(
    *,
    input_path: str,
    entry_path: str,
    threshold_enabled: bool,
    device: torch.device | str,
    chunk_voxels: int = 0,
    reservoir_size: int = 500_000,
    max_bins: int = 4_096,
    exact_log_limit: int = 2_000_000,
    seed: int = 0,
) -> StreamedThresholdResult:
    """Build threshold outputs for d-mode using streamed slab reads.

    This path is intended for large/full datasets and currently supports
    unsliced NXdata reads only.
    """

    target_device = torch.device(device)
    with h5py.File(input_path, "r") as h5:
        signal_ds = _resolve_signal_dataset(h5, entry_path)
        shape = tuple(int(x) for x in signal_ds.shape)
        if len(shape) < 3:
            raise ValueError(
                "Streamed preprocessing expects data with shape (T, *spatial) "
                f"with at least 2 spatial dimensions. Got: {shape}"
            )
        resolved_chunk_voxels, chunk_auto = _resolve_chunk_voxels(shape, int(chunk_voxels))

        cutoff_stats = _estimate_cutoff(
            signal_ds,
            threshold_enabled=bool(threshold_enabled),
            chunk_voxels=int(resolved_chunk_voxels),
            reservoir_size=int(reservoir_size),
            max_bins=int(max_bins),
            exact_log_limit=int(exact_log_limit),
            seed=int(seed),
            compute_device=target_device,
        )

        data_chunks: List[np.ndarray] = []
        ind_chunks: List[np.ndarray] = []
        spatial_shape = shape[1:]
        for starts, spatial_slices in _iter_spatial_chunks(spatial_shape, int(resolved_chunk_voxels)):
            block = _read_block_full_temperature(signal_ds, spatial_slices)
            mean_t = block.mean(axis=0)
            valid = np.isfinite(mean_t) & (mean_t > 0.0)
            if bool(threshold_enabled):
                pass_mask = np.zeros_like(valid, dtype=bool)
                if np.any(valid):
                    pass_mask[valid] = np.log(mean_t[valid]).astype(np.float64, copy=False) > float(cutoff_stats.cutoff)
            else:
                pass_mask = valid
            if not np.any(pass_mask):
                continue
            flat_mask = pass_mask.reshape(-1)
            selected = block.reshape(shape[0], -1)[:, flat_mask]
            data_chunks.append(selected.astype(np.float64, copy=False))

            local_idx = np.argwhere(pass_mask)
            if local_idx.size > 0:
                starts_arr = np.asarray(starts, dtype=np.int64).reshape(1, -1)
                ind_chunks.append((local_idx.astype(np.int64, copy=False) + starts_arr))

        if data_chunks:
            data_thresholded_np = np.concatenate(data_chunks, axis=1)
            ind_thresholded_np = np.concatenate(ind_chunks, axis=0) if ind_chunks else np.empty((0, len(shape) - 1))
        else:
            data_thresholded_np = np.empty((shape[0], 0), dtype=np.float64)
            ind_thresholded_np = np.empty((0, len(shape) - 1), dtype=np.int64)

    data_thresholded = torch.as_tensor(data_thresholded_np, device=target_device)
    ind_thresholded = torch.as_tensor(ind_thresholded_np, device=target_device, dtype=torch.long)
    resolved_chunk_shape = _chunk_shape(shape[1:], int(resolved_chunk_voxels))
    est_chunk_voxels = int(np.prod(np.asarray(resolved_chunk_shape, dtype=np.int64)))
    est_chunk_bytes = int(shape[0]) * est_chunk_voxels * np.dtype(np.float64).itemsize
    diagnostics: Dict[str, object] = {
        "mode": cutoff_stats.mode,
        "success": bool(cutoff_stats.success),
        "n_valid_log_means": int(cutoff_stats.n_valid),
        "log_min": float(cutoff_stats.min_log) if np.isfinite(cutoff_stats.min_log) else None,
        "log_max": float(cutoff_stats.max_log) if np.isfinite(cutoff_stats.max_log) else None,
        "bin_width": float(cutoff_stats.bin_width) if np.isfinite(cutoff_stats.bin_width) else None,
        "n_bins": int(cutoff_stats.n_bins),
        "chunk_voxels_requested": int(chunk_voxels),
        "chunk_voxels": int(resolved_chunk_voxels),
        "chunk_auto_1gib": bool(chunk_auto),
        "chunk_shape": [int(x) for x in resolved_chunk_shape],
        "estimated_chunk_bytes": int(est_chunk_bytes),
        "reservoir_size": int(reservoir_size),
        "exact_log_limit": int(exact_log_limit),
        "cutoff_compute_device": str(target_device),
        "approx_iqr": (
            float(cutoff_stats.approx_iqr)
            if cutoff_stats.approx_iqr is not None and np.isfinite(cutoff_stats.approx_iqr)
            else None
        ),
        "exact_iqr": (
            float(cutoff_stats.exact_iqr)
            if cutoff_stats.exact_iqr is not None and np.isfinite(cutoff_stats.exact_iqr)
            else None
        ),
    }
    if diagnostics["approx_iqr"] is not None and diagnostics["exact_iqr"] is not None:
        exact_iqr = float(diagnostics["exact_iqr"])
        if abs(exact_iqr) > np.finfo(np.float64).eps:
            diagnostics["iqr_rel_err"] = float(
                abs(float(diagnostics["approx_iqr"]) - exact_iqr) / abs(exact_iqr)
            )
        else:
            diagnostics["iqr_rel_err"] = None

    return StreamedThresholdResult(
        device=target_device,
        data_shape_orig=tuple(int(x) for x in shape),
        threshold_type="KL-streamed" if threshold_enabled else "none-streamed",
        logi_cutoff=float(cutoff_stats.cutoff),
        data_thresholded=data_thresholded,
        ind_thresholded=ind_thresholded,
        diagnostics=diagnostics,
    )
