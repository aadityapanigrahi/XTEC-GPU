"""Streamed preprocessing helpers for large NX datasets.

This module provides opt-in streamed preprocessing paths for XTEC workflows
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
from scipy import ndimage

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


class StreamedPeakAveraging:
    """Peak-averaging container compatible with GMM.Get_pixel_labels."""

    def __init__(
        self,
        *,
        peak_avg_data: torch.Tensor,
        peak_avg_ind_list: List[torch.Tensor],
        peak_max_data: torch.Tensor,
        peak_max_ind_list: List[torch.Tensor],
        diagnostics: Dict[str, object],
    ) -> None:
        self.peak_avg_data = peak_avg_data
        self.peak_avg_ind_list = peak_avg_ind_list
        self.peak_max_data = peak_max_data
        self.peak_max_ind_list = peak_max_ind_list
        self.streaming = dict(diagnostics)


def _structure_element_for_spatial_dims(ndim: int) -> np.ndarray:
    if ndim == 2:
        return np.ones((3, 3), dtype=np.uint8)
    if ndim == 3:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError(
        "Streamed peak averaging supports only 2D/3D spatial data. "
        f"Got ndim={ndim}."
    )


def _component_labels_from_sparse_indices(
    ind_thresholded_np: np.ndarray,
    spatial_shape: Sequence[int],
) -> Tuple[np.ndarray, int]:
    if ind_thresholded_np.size == 0:
        return np.empty((0,), dtype=np.int64), 0
    if ind_thresholded_np.ndim != 2:
        raise ValueError(
            "Expected thresholded indices with shape (N, D). "
            f"Got shape={ind_thresholded_np.shape}."
        )
    if ind_thresholded_np.shape[1] != len(spatial_shape):
        raise ValueError(
            "Threshold index dimensionality does not match spatial shape. "
            f"indices D={ind_thresholded_np.shape[1]}, spatial D={len(spatial_shape)}."
        )

    structure = _structure_element_for_spatial_dims(len(spatial_shape))
    thresholded_mask = np.zeros(tuple(int(x) for x in spatial_shape), dtype=bool)
    thresholded_mask[tuple(ind_thresholded_np.T)] = True
    labeled_array, num_features = ndimage.label(thresholded_mask, structure=structure)

    labels_1_based = labeled_array[tuple(ind_thresholded_np.T)]
    if np.any(labels_1_based <= 0):
        raise RuntimeError(
            "Connected-component labeling produced unlabeled thresholded points."
        )
    labels_0_based = labels_1_based.astype(np.int64, copy=False) - 1
    return labels_0_based, int(num_features)


def build_streamed_peak_averaging(
    threshold_result: StreamedThresholdResult,
) -> StreamedPeakAveraging:
    """Compute peak-averaged trajectories from streamed threshold outputs.

    This reproduces the connected-component semantics of Peak_averaging while
    avoiding full intensity-volume materialization.
    """

    data_thresholded = threshold_result.data_thresholded
    ind_thresholded = threshold_result.ind_thresholded
    device = data_thresholded.device
    dtype = data_thresholded.dtype
    n_temp = int(data_thresholded.shape[0])
    spatial_shape = tuple(int(x) for x in threshold_result.data_shape_orig[1:])

    ind_thresholded_np = (
        ind_thresholded.detach().cpu().numpy().astype(np.int64, copy=False)
        if torch.is_tensor(ind_thresholded)
        else np.asarray(ind_thresholded, dtype=np.int64)
    )

    labels_0_based, num_features = _component_labels_from_sparse_indices(
        ind_thresholded_np,
        spatial_shape,
    )

    if num_features == 0:
        empty = torch.empty((n_temp, 0), device=device, dtype=dtype)
        diagnostics = {
            "num_peaks": 0,
            "spatial_dims": len(spatial_shape),
        }
        return StreamedPeakAveraging(
            peak_avg_data=empty,
            peak_avg_ind_list=[],
            peak_max_data=empty,
            peak_max_ind_list=[],
            diagnostics=diagnostics,
        )

    labels_t = torch.as_tensor(labels_0_based, device=device, dtype=torch.long)
    data_nt = data_thresholded.transpose(0, 1).contiguous()  # (N, T)

    peak_sum_nt = torch.zeros((num_features, n_temp), device=device, dtype=dtype)
    peak_sum_nt.index_add_(0, labels_t, data_nt)
    peak_counts = torch.bincount(labels_t, minlength=num_features).to(dtype).clamp_min(1)
    peak_avg_data = (peak_sum_nt / peak_counts.unsqueeze(1)).transpose(0, 1).contiguous()

    # Preserve peak_max_data field parity with Peak_averaging.
    neg_inf = (
        torch.finfo(dtype).min
        if torch.is_floating_point(data_nt)
        else float(torch.iinfo(dtype).min)
    )
    peak_max_nt = torch.full((num_features, n_temp), neg_inf, device=device, dtype=dtype)
    peak_label_expand = labels_t.unsqueeze(1).expand(-1, n_temp)
    peak_max_nt.scatter_reduce_(
        0,
        peak_label_expand,
        data_nt,
        reduce="amax",
        include_self=True,
    )
    peak_max_data = peak_max_nt.transpose(0, 1).contiguous()

    flat_inds = np.ravel_multi_index(ind_thresholded_np.T, dims=spatial_shape)
    sort_order = np.lexsort((flat_inds, labels_0_based))
    sorted_inds = ind_thresholded_np[sort_order]
    sorted_labels = labels_0_based[sort_order]
    counts = np.bincount(sorted_labels, minlength=num_features)
    split_points = np.cumsum(counts[:-1], dtype=np.int64)
    split_inds = (
        np.split(sorted_inds, split_points.tolist())
        if num_features > 1
        else [sorted_inds]
    )
    peak_avg_ind_list = [
        torch.as_tensor(arr, device=device, dtype=torch.long) for arr in split_inds
    ]
    peak_max_ind_list = list(peak_avg_ind_list)

    diagnostics = {
        "num_peaks": int(num_features),
        "spatial_dims": len(spatial_shape),
    }
    return StreamedPeakAveraging(
        peak_avg_data=peak_avg_data,
        peak_avg_ind_list=peak_avg_ind_list,
        peak_max_data=peak_max_data,
        peak_max_ind_list=peak_max_ind_list,
        diagnostics=diagnostics,
    )


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
        exact_logs.append(log_vals)
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
    collect_indices: bool = True,
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
        flat_ind_chunks: List[np.ndarray] = []
        ind_chunks: List[np.ndarray] | None = [] if bool(collect_indices) else None
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
                global_idx = local_idx.astype(np.int64, copy=False) + starts_arr
                flat_ind_chunks.append(
                    np.ravel_multi_index(global_idx.T, dims=spatial_shape).astype(np.int64, copy=False)
                )
                if ind_chunks is not None:
                    ind_chunks.append(global_idx)

        if data_chunks:
            data_thresholded_np = np.concatenate(data_chunks, axis=1)
            flat_inds_np = (
                np.concatenate(flat_ind_chunks, axis=0)
                if flat_ind_chunks
                else np.empty((0,), dtype=np.int64)
            )
            if ind_chunks is not None:
                ind_thresholded_np = (
                    np.concatenate(ind_chunks, axis=0)
                    if ind_chunks
                    else np.empty((0, len(shape) - 1), dtype=np.int64)
                )
            else:
                ind_thresholded_np = np.empty((0, len(shape) - 1), dtype=np.int64)

            # Match CPU preprocessing ordering exactly: flatten spatial indices in
            # C-order and keep thresholded trajectories in that global order.
            # Without this, chunk traversal can permute points and introduce
            # small downstream drift in BIC/EM despite identical threshold sets.
            if bool(threshold_enabled) and flat_inds_np.size > 1:
                order = np.argsort(flat_inds_np, kind="stable")
                data_thresholded_np = data_thresholded_np[:, order]
                if ind_thresholded_np.size > 0:
                    ind_thresholded_np = ind_thresholded_np[order]
        else:
            data_thresholded_np = np.empty((shape[0], 0), dtype=np.float64)
            ind_thresholded_np = np.empty((0, len(shape) - 1), dtype=np.int64)

    data_thresholded = torch.as_tensor(data_thresholded_np, device=target_device)
    ind_thresholded = torch.as_tensor(ind_thresholded_np, device=target_device, dtype=torch.long)
    resolved_chunk_shape = _chunk_shape(shape[1:], int(resolved_chunk_voxels))
    chunks_per_axis = [
        int(np.ceil(float(dim) / float(max(1, cdim))))
        for dim, cdim in zip(shape[1:], resolved_chunk_shape)
    ]
    n_streaming_chunks = int(np.prod(np.asarray(chunks_per_axis, dtype=np.int64)))
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
        "chunks_per_axis": chunks_per_axis,
        "n_streaming_chunks": int(n_streaming_chunks),
        "estimated_chunk_bytes": int(est_chunk_bytes),
        "reservoir_size": int(reservoir_size),
        "exact_log_limit": int(exact_log_limit),
        "collect_indices": bool(collect_indices),
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


def build_streamed_s_preprocessed(
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
) -> Tuple[StreamedThresholdResult, StreamedPeakAveraging, torch.Tensor]:
    """Build streamed preprocessing bundle for XTEC-s workflows."""

    threshold = build_streamed_threshold_result(
        input_path=input_path,
        entry_path=entry_path,
        threshold_enabled=threshold_enabled,
        device=device,
        chunk_voxels=chunk_voxels,
        reservoir_size=reservoir_size,
        max_bins=max_bins,
        exact_log_limit=exact_log_limit,
        seed=seed,
    )
    peak_avg = build_streamed_peak_averaging(threshold)
    return threshold, peak_avg, peak_avg.peak_avg_data
