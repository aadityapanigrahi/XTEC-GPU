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
    base = max(1, int(round(float(target_voxels) ** (1.0 / float(ndim)))))
    return tuple(max(1, min(int(size), base)) for size in spatial_shape)


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


class _ReservoirSketch:
    """Uniform reservoir sketch for approximate quantiles."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = max(1, int(capacity))
        self._rng = np.random.default_rng(int(seed))
        self._values = np.empty(0, dtype=np.float64)
        self._priorities = np.empty(0, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if vals.size == 0:
            return
        pr = self._rng.random(vals.size)
        if self._values.size == 0:
            merged_vals = vals
            merged_pr = pr
        else:
            merged_vals = np.concatenate([self._values, vals], axis=0)
            merged_pr = np.concatenate([self._priorities, pr], axis=0)
        if merged_vals.size > self.capacity:
            keep = np.argpartition(merged_pr, self.capacity - 1)[: self.capacity]
            self._values = merged_vals[keep]
            self._priorities = merged_pr[keep]
        else:
            self._values = merged_vals
            self._priorities = merged_pr

    def values(self) -> np.ndarray:
        return np.asarray(self._values, dtype=np.float64)


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
) -> _CutoffStats:
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
        )

    spatial_shape = tuple(int(x) for x in dataset.shape[1:])
    n_valid = 0
    sum_log = 0.0
    sumsq_log = 0.0
    min_log = float("inf")
    max_log = float("-inf")
    exact_logs: List[np.ndarray] = []
    exact_count = 0
    exact_mode = True
    sketch = _ReservoirSketch(reservoir_size, seed=seed)

    for _starts, spatial_slices in _iter_spatial_chunks(spatial_shape, chunk_voxels):
        block = np.asarray(dataset[(slice(None),) + spatial_slices], dtype=np.float64)
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

        if exact_mode and (exact_count + n) <= int(exact_log_limit):
            exact_logs.append(log_vals.copy())
            exact_count += n
        else:
            if exact_mode and exact_logs:
                sketch.update(np.concatenate(exact_logs, axis=0))
                exact_logs.clear()
            exact_mode = False
            sketch.update(log_vals)

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
        )

    if exact_mode:
        log_sample = np.concatenate(exact_logs, axis=0) if exact_logs else np.empty(0, dtype=np.float64)
        mode = "exact"
    else:
        log_sample = sketch.values()
        mode = "reservoir"

    if log_sample.size == 0:
        # Defensive fallback
        log_sample = np.array([sum_log / float(n_valid)], dtype=np.float64)

    n_float = float(max(1, n_valid))
    mean_global = sum_log / n_float
    var_global = max(sumsq_log / n_float - mean_global * mean_global, 0.0)
    std_global = float(np.sqrt(var_global))

    if exact_mode:
        helper = Threshold_Background.__new__(Threshold_Background)
        log_t = torch.as_tensor(log_sample, dtype=torch.float64)
        log_sorted_t = torch.sort(log_t).values
        bin_size_t = helper.Freedman_Diaconis_for_bin_width(log_sorted_t, device=torch.device("cpu"))
        y_bins_t, x_bins_t = helper.hist(log_t, bin_size_t, device=torch.device("cpu"))
        helper.bin_size = torch.as_tensor(float(bin_size_t.item()), dtype=torch.float64)
        try:
            opt_idx = helper.Truncate(
                x_bins_t,
                y_bins_t,
                log_sorted_t,
                max_iter=100,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            cutoff = float(x_bins_t[int(opt_idx)].item())
            naive_mean = float(torch.mean(log_sorted_t).item())
            naive_std = float(torch.std(log_sorted_t, unbiased=False).item())
            sanity_ok = not (
                (cutoff > naive_mean + 2.0 * naive_std) or (cutoff < naive_mean - naive_std)
            )
            mode = "exact-kl" if sanity_ok else "exact-kl-sanity-fail"
            success = bool(sanity_ok)
        except Exception:
            cutoff = float(mean_global + 2.0 * std_global)
            mode = "exact+simple-fallback"
            success = False
        return _CutoffStats(
            cutoff=cutoff,
            mode=mode,
            success=success,
            n_valid=n_valid,
            min_log=min_log,
            max_log=max_log,
            bin_width=float(bin_size_t.item()),
            n_bins=int(x_bins_t.numel()),
        )

    q25, q75 = np.quantile(log_sample, [0.25, 0.75], method="linear")
    iqr = max(float(q75 - q25), np.finfo(np.float64).eps)
    bin_width = max(iqr / (n_float ** (1.0 / 3.0)), np.finfo(np.float64).eps)

    hist_span = max(max_log - min_log, np.finfo(np.float64).eps)
    n_bins = int(np.ceil(hist_span / bin_width))
    n_bins = max(2, min(int(max_bins), n_bins))
    counts = np.zeros(n_bins, dtype=np.float64)
    for _starts, spatial_slices in _iter_spatial_chunks(spatial_shape, chunk_voxels):
        block = np.asarray(dataset[(slice(None),) + spatial_slices], dtype=np.float64)
        mean_t = block.mean(axis=0)
        valid = np.isfinite(mean_t) & (mean_t > 0.0)
        if not np.any(valid):
            continue
        log_vals = np.log(mean_t[valid]).astype(np.float64, copy=False)
        h, _ = np.histogram(log_vals, bins=n_bins, range=(min_log, max_log))
        counts += h.astype(np.float64)

    x_bins = min_log + np.arange(n_bins, dtype=np.float64) * bin_width
    y_bins = counts / (n_float * bin_width)
    helper = Threshold_Background.__new__(Threshold_Background)
    helper.bin_size = torch.as_tensor(bin_width, dtype=torch.float64)
    x_bins_t = torch.as_tensor(x_bins, dtype=torch.float64)
    y_bins_t = torch.as_tensor(y_bins, dtype=torch.float64)
    log_sorted_t = torch.as_tensor(np.sort(log_sample), dtype=torch.float64)
    try:
        opt_idx = helper.Truncate(
            x_bins_t,
            y_bins_t,
            log_sorted_t,
            max_iter=100,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        cutoff = float(x_bins_t[int(opt_idx)].item())
        upper = mean_global + 2.0 * std_global
        lower = mean_global - 1.0 * std_global
        sanity_ok = not (cutoff > upper or cutoff < lower)
        mode = "reservoir-kl" if sanity_ok else "reservoir-kl-sanity-fail"
        success = bool(sanity_ok)
    except Exception:
        cutoff = float(mean_global + 2.0 * std_global)
        mode = "reservoir+simple-fallback"
        success = False

    return _CutoffStats(
        cutoff=cutoff,
        mode=mode,
        success=success,
        n_valid=n_valid,
        min_log=min_log,
        max_log=max_log,
        bin_width=bin_width,
        n_bins=n_bins,
    )


def build_streamed_threshold_result(
    *,
    input_path: str,
    entry_path: str,
    threshold_enabled: bool,
    device: torch.device | str,
    chunk_voxels: int = 200_000,
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

        cutoff_stats = _estimate_cutoff(
            signal_ds,
            threshold_enabled=bool(threshold_enabled),
            chunk_voxels=int(chunk_voxels),
            reservoir_size=int(reservoir_size),
            max_bins=int(max_bins),
            exact_log_limit=int(exact_log_limit),
            seed=int(seed),
        )

        data_chunks: List[np.ndarray] = []
        ind_chunks: List[np.ndarray] = []
        spatial_shape = shape[1:]
        for starts, spatial_slices in _iter_spatial_chunks(spatial_shape, int(chunk_voxels)):
            block = np.asarray(signal_ds[(slice(None),) + spatial_slices], dtype=np.float64)
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
    diagnostics: Dict[str, object] = {
        "mode": cutoff_stats.mode,
        "success": bool(cutoff_stats.success),
        "n_valid_log_means": int(cutoff_stats.n_valid),
        "log_min": float(cutoff_stats.min_log) if np.isfinite(cutoff_stats.min_log) else None,
        "log_max": float(cutoff_stats.max_log) if np.isfinite(cutoff_stats.max_log) else None,
        "bin_width": float(cutoff_stats.bin_width) if np.isfinite(cutoff_stats.bin_width) else None,
        "n_bins": int(cutoff_stats.n_bins),
        "chunk_voxels": int(chunk_voxels),
        "reservoir_size": int(reservoir_size),
        "exact_log_limit": int(exact_log_limit),
    }

    return StreamedThresholdResult(
        device=target_device,
        data_shape_orig=tuple(int(x) for x in shape),
        threshold_type="KL-streamed" if threshold_enabled else "none-streamed",
        logi_cutoff=float(cutoff_stats.cutoff),
        data_thresholded=data_thresholded,
        ind_thresholded=ind_thresholded,
        diagnostics=diagnostics,
    )
