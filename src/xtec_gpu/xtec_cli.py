"""
XTEC-GPU Command-Line Interface
================================

Provides the same clustering functionality as the NeXpy plugin
(``cluster_data.py``) but driven entirely from the terminal.

Usage examples
--------------
::

    # XTEC-d (direct GMM via torchgmm, GPU preprocessing)
    xtec-gpu xtec-d data.nxs -o results/ -n 4 --rescale mean

    # Tutorial-faithful XTEC-d workflow (threshold plots + second pass)
    xtec-gpu tutorial-d data.nxs -o tutorial_results/ --device cuda:1

    # XTEC-s (peak-averaged GMM via torchgmm on GPU)
    xtec-gpu xtec-s data.nxs -o results/ -n 4

    # XTEC label smooth (GPU GMM + Markov label smoothing)
    xtec-gpu label-smooth data.nxs -o results/ -n 4 --L-scale 0.05

    # BIC sweep for XTEC-d
    xtec-gpu bic-d data.nxs -o results/ --min-nc 2 --max-nc 14

    # BIC sweep for XTEC-s
    xtec-gpu bic-s data.nxs -o results/ --min-nc 2 --max-nc 14

All methods save results (cluster assignments, means, covariances,
plots) into the specified output directory.
"""

import argparse
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CLI
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from nexusformat.nexus import nxload, NXdata, NXfield

from .Preprocessing import (
    Mask_Zeros,
    Peak_averaging,
    Threshold_Background,
)
from .GMM import GMM, GMM_kernels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a torch.Tensor to NumPy; pass through if already ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_device(device_arg="auto"):
    """Return the torch device based on user preference or automatic fallback."""
    if device_arg and device_arg.lower() != "auto":
        return torch.device(device_arg)
    
    # Auto-detection sequence (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        # Find the CUDA device with the most free memory
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            best_gpu = 0
            max_free = -1
            for i in range(num_gpus):
                # mem_get_info returns (free, total)
                free, total = torch.cuda.mem_get_info(i)
                if free > max_free:
                    max_free = free
                    best_gpu = i
            return torch.device(f"cuda:{best_gpu}")
        else:
            return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _load_data(filepath, entry_path, slices):
    """Load a NeXus file and return the NXdata object.

    Parameters
    ----------
    filepath : str
        Path to the ``.nxs`` file.
    entry_path : str
        HDF5 path inside the file to the NXdata group
        (e.g. ``'entry/data'``).
    slices : str or None
        Optional Python-style slice string applied after loading,
        e.g. ``':,0.0:1.0,-10:10,-15:15'``.

    Returns
    -------
    data : NXdata
    """
    f = nxload(filepath, "r")
    data = f[entry_path]

    if slices:
        # Parse the user-provided slice string, e.g. ":,0.0:1.0,-10:10"
        slice_objs = []
        for s in slices.split(","):
            s = s.strip()
            if s == ":":
                slice_objs.append(slice(None))
            elif ":" in s:
                parts = s.split(":")
                start = float(parts[0]) if parts[0] else None
                stop = float(parts[1]) if parts[1] else None
                slice_objs.append(slice(start, stop))
            else:
                slice_objs.append(float(s))
        data = data[tuple(slice_objs)]

    return data


def _rescale(threshold, Data_thresh, rescale_text, device):
    """Apply rescaling; returns a torch tensor."""
    if rescale_text == "mean":
        return threshold.Rescale_mean(Data_thresh)
    elif rescale_text == "z-score":
        return threshold.Rescale_zscore(Data_thresh)
    elif rescale_text == "log-mean":
        dt = Data_thresh.float() if torch.is_tensor(Data_thresh) \
            else torch.as_tensor(Data_thresh, dtype=torch.float32, device=device)
        Rescaled = torch.log(1 + dt)
        return Rescaled - torch.mean(Rescaled, dim=0)
    else:  # "None"
        return Data_thresh


def _axis_display_name(nxaxis):
    """Return a compact display name for an NX axis."""
    name = getattr(nxaxis, "nxname", str(nxaxis))
    if name.lower().startswith("q") and len(name) == 2:
        return name[1:].upper()
    return name


def _parse_int_list(text):
    """Parse a comma-separated integer list."""
    if text is None:
        return None
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _parse_zoom_window(text):
    """Parse tutorial zoom text of the form '200:300,0:100'."""
    if text is None:
        return None
    try:
        row_text, col_text = [piece.strip() for piece in text.split(",", 1)]
        row_start, row_stop = [int(v) for v in row_text.split(":", 1)]
        col_start, col_stop = [int(v) for v in col_text.split(":", 1)]
    except Exception as exc:
        raise ValueError(
            f"Invalid zoom window '{text}'. Expected format 'row0:row1,col0:col1'."
        ) from exc

    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError(
            f"Invalid zoom window '{text}'. Start indices must be < stop indices."
        )
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _resolve_slice_index(data, axis_ind, slice_value):
    """Resolve the requested slice value to an index, using nearest if needed."""
    axis = data.nxaxes[axis_ind + 1]
    axis_values = np.asarray(axis.nxvalue)
    try:
        slice_ind = int(axis.index(slice_value))
        actual_value = float(axis_values[slice_ind])
    except Exception:
        slice_ind = int(np.argmin(np.abs(axis_values - slice_value)))
        actual_value = float(axis_values[slice_ind])
        print(f"  Requested slice value {slice_value} not found exactly; "
              f"using nearest value {actual_value} at index {slice_ind}")
    return slice_ind, actual_value


def _slice_plot_axes(data, axis_ind):
    """Return x/y axes metadata for a 2D plot extracted from a 3D volume."""
    if axis_ind == 0:
        x_axis = data.nxaxes[3]
        y_axis = data.nxaxes[2]
    elif axis_ind == 1:
        x_axis = data.nxaxes[3]
        y_axis = data.nxaxes[1]
    elif axis_ind == 2:
        x_axis = data.nxaxes[2]
        y_axis = data.nxaxes[1]
    else:
        raise ValueError(f"axis_ind must be 0, 1, or 2. Received: {axis_ind}")
    return x_axis, y_axis


def _apply_slice_axes_metadata(ax, data, axis_ind, slice_value, label_size=None):
    """Apply extent, labels, and tutorial-style title to an image axis."""
    x_axis, y_axis = _slice_plot_axes(data, axis_ind)
    ax.get_images()[0].set_extent(
        (x_axis.nxvalue[0], x_axis.nxvalue[-1], y_axis.nxvalue[0], y_axis.nxvalue[-1])
    )
    axis_label = _axis_display_name(data.nxaxes[axis_ind + 1])
    x_label = _axis_display_name(x_axis)
    y_label = _axis_display_name(y_axis)
    label_kwargs = {}
    if label_size is not None:
        label_kwargs["size"] = label_size
    ax.set_xlabel(x_label, **label_kwargs)
    ax.set_ylabel(y_label, **label_kwargs)
    ax.set_title(f"{axis_label}={slice_value}", **label_kwargs)


def _sync_cluster_model(clusterGMM, cluster_assigns, cluster_means, cluster_covs):
    """Update a fitted GMM wrapper after externally reordering cluster labels."""
    clusterGMM.cluster_assignments = np.asarray(cluster_assigns)
    clusterGMM.means = np.asarray(cluster_means)
    clusterGMM.covs = np.asarray(cluster_covs)
    clusterGMM.num_per_cluster = [
        int(np.sum(clusterGMM.cluster_assignments == k))
        for k in range(clusterGMM.cluster_num)
    ]
    for k in range(clusterGMM.cluster_num):
        clusterGMM.cluster[k].mean = np.asarray(cluster_means[k])
        clusterGMM.cluster[k].cov = np.asarray(cluster_covs[k])


def _run_direct_gmm(data, threshold, Data_thresh, Data_ind, nc, rescale_text,
                    device, random_state=None, reorder=False, rescaled_data=None,
                    init_strategy_mode="kmeans++", post_stepwise_epochs=0,
                    post_stepwise_tol=None, solver_mode="torchgmm",
                    batch_num=1, max_batch_epoch=50, max_full_epoch=500):
    """Run XTEC-d clustering and optionally reorder cluster labels."""
    t0 = time.time()

    if rescaled_data is None:
        Rescaled_data = _rescale(threshold, Data_thresh, rescale_text, device)
    else:
        Rescaled_data = rescaled_data

    if torch.is_tensor(Rescaled_data):
        Data_for_GMM = Rescaled_data.T
    else:
        Data_for_GMM = torch.as_tensor(
            np.asarray(Rescaled_data).T, dtype=torch.float32, device=device
        )

    gmm_kwargs = {
        "cov_type": "diag",
        "solver_mode": solver_mode,
        "init_strategy_mode": init_strategy_mode,
        "post_stepwise_epochs": int(post_stepwise_epochs),
        "post_stepwise_tol": post_stepwise_tol,
        "batch_num": int(batch_num),
        "max_batch_epoch": int(max_batch_epoch),
        "max_full_epoch": int(max_full_epoch),
    }
    if random_state is not None:
        gmm_kwargs["random_state"] = int(random_state)
    clusterGMM = GMM(Data_for_GMM, nc, **gmm_kwargs)
    clusterGMM.RunEM()

    cluster_assigns = _to_numpy(clusterGMM.cluster_assignments)
    cluster_means = _to_numpy(clusterGMM.means)
    cluster_covs = [_to_numpy(clusterGMM.cluster[i].cov) for i in range(nc)]
    Data_thresh_np = _to_numpy(Data_thresh)
    Data_ind_np = _to_numpy(Data_ind)
    Rescaled_data_np = _to_numpy(Rescaled_data)

    if reorder:
        temp_values = data.nxaxes[0].nxvalue
        cluster_assigns, cluster_assigns, cluster_means, cluster_covs = \
            _reorder_clusters(cluster_assigns, cluster_assigns,
                              cluster_means, cluster_covs,
                              Data_thresh_np, nc, temp_values)
        _sync_cluster_model(clusterGMM, cluster_assigns, cluster_means, cluster_covs)

    elapsed = time.time() - t0
    return {
        "clusterGMM": clusterGMM,
        "elapsed": elapsed,
        "cluster_assigns": cluster_assigns,
        "cluster_means": cluster_means,
        "cluster_covs": cluster_covs,
        "Data_thresh_np": Data_thresh_np,
        "Data_ind_np": Data_ind_np,
        "Rescaled_data_np": Rescaled_data_np,
    }


def _gmm_parameter_count(n_components, n_features, cov_type="diag"):
    """Return free-parameter count for a Gaussian mixture model."""
    if cov_type == "diag":
        cov_params = n_components * n_features
    elif cov_type == "full":
        cov_params = n_components * (n_features * (n_features + 1) // 2)
    else:
        raise ValueError(f"Unsupported covariance type: {cov_type}")

    mean_params = n_components * n_features
    weight_params = n_components - 1
    return int(cov_params + mean_params + weight_params)


def _bic_from_loglikelihood(log_likelihood, n_components, n_features, n_samples,
                            cov_type="diag"):
    """Compute BIC from log-likelihood and model dimensionality."""
    p = _gmm_parameter_count(n_components, n_features, cov_type=cov_type)
    return -2.0 * float(log_likelihood) + p * np.log(float(n_samples))


def _reorder_clusters(cluster_assigns, pixel_assigns, cluster_means,
                      cluster_covs, Data_thresh_np, nc, temp_values):
    """Relabel clusters in descending order of low-T raw intensity.

    Sorting criterion: for each cluster *k*, compute the average raw
    intensity over the lowest 10%% of temperature data-points (by actual
    temperature value, minimum 1).  Clusters are then numbered 0, 1, …
    in *descending* order of that average, ensuring a deterministic
    labelling across runs.

    Parameters
    ----------
    cluster_assigns : ndarray (N,)
    pixel_assigns   : ndarray (N,) or (N_pix,)
    cluster_means   : ndarray (K, T)
    cluster_covs    : list/ndarray, each element shape (T,) or (T, T)
    Data_thresh_np  : ndarray (T, N)   — raw thresholded intensities
    nc              : int
    temp_values     : ndarray (T,) — actual temperature values

    Returns
    -------
    (cluster_assigns, pixel_assigns, cluster_means, cluster_covs)
        All relabelled / reordered.
    """
    T = len(temp_values)
    n_low = max(1, int(T * 0.1))  # lowest 10%, at least 1

    # Find the indices of the n_low lowest *temperature values*
    low_T_indices = np.argsort(temp_values)[:n_low]

    # Average raw intensity at those temperatures for each cluster
    low_T_avg = np.zeros(nc)
    for k in range(nc):
        mask_k = cluster_assigns == k
        if mask_k.any():
            low_T_avg[k] = Data_thresh_np[low_T_indices][:, mask_k].mean()
        else:
            low_T_avg[k] = -np.inf  # empty clusters go last

    # Descending order
    new_order = np.argsort(-low_T_avg)  # old_label → position
    old_to_new = np.empty(nc, dtype=int)
    for new_label, old_label in enumerate(new_order):
        old_to_new[old_label] = new_label

    cluster_assigns = old_to_new[cluster_assigns]
    pixel_assigns = old_to_new[pixel_assigns]
    cluster_means = cluster_means[new_order]
    cluster_covs = [cluster_covs[i] for i in new_order]

    return cluster_assigns, pixel_assigns, cluster_means, cluster_covs


def _save_results(outdir, cluster_assigns, pixel_assigns, Data_ind,
                  Data_thresh, cluster_means, cluster_covs, prefix=""):
    """Save clustering results as an HDF5 file."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{prefix}results.h5")

    with h5py.File(path, "w") as f:
        f.create_dataset("cluster_assignments", data=cluster_assigns)
        f.create_dataset("pixel_assignments", data=pixel_assigns)
        f.create_dataset("data_indices", data=Data_ind)
        # Persist thresholded input used for clustering to support manual oversight.
        f.create_dataset("data_thresholded", data=Data_thresh)
        f.create_dataset("cluster_means", data=cluster_means)
        f.create_dataset("cluster_covariances", data=np.array(cluster_covs))

    print(f"  Results saved to {path}")


def _save_tutorial_pickle(path, temp_values, data_values, data_ind,
                          cluster_assigns, cluster_means, cluster_covs):
    """Save the final tutorial-style pickle bundle."""
    obj = {
        "Temp": np.asarray(temp_values),
        "Data": np.asarray(data_values).transpose(),
        "Data_ind": np.asarray(data_ind),
        "cluster_assignments": np.asarray(cluster_assigns),
        "Num_clusters": int(len(cluster_means)),
        "cluster_mean": np.asarray(cluster_means),
        "cluster_var": np.asarray(cluster_covs),
    }
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    print(f"  Tutorial pickle saved to {path}")


def _save_figure(path):
    """Save and close the current Matplotlib figure."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Plot saved to {path}")


def _save_threshold_plots(data, threshold, outdir, axis_ind, slice_value,
                          prefix=""):
    """Save tutorial-style threshold diagnostic plots."""
    slice_ind, actual_slice_value = _resolve_slice_index(data, axis_ind, slice_value)

    threshold.plot_cutoff((10, 5))
    _save_figure(os.path.join(outdir, f"{prefix}cutoff.png"))

    threshold.plot_thresholding_2D_slice((8, 8), slice_ind, axis_ind)
    ax = plt.gca()
    _apply_slice_axes_metadata(ax, data, axis_ind, actual_slice_value)
    _save_figure(os.path.join(outdir, f"{prefix}threshold_slice.png"))

    return slice_ind, actual_slice_value


def _save_traj_plot(data, clusterGMM, outdir, filename, title_text):
    """Save a tutorial-style cluster trajectory plot."""
    temp_values = data.nxaxes[0].nxvalue
    clusterGMM.Plot_Cluster_Results_traj(temp_values)
    plt.xlabel("T(K)", size=18)
    plt.ylabel(r"$\widetilde{I}_q(T)$", size=18)
    plt.title(title_text)
    _save_figure(os.path.join(outdir, filename))


def _save_qmap_plot(data, threshold, clusterGMM, data_ind, outdir, filename,
                    axis_ind, slice_ind, slice_value, figsize=(8, 8)):
    """Save a tutorial-style reciprocal-space cluster map."""
    clusterGMM.Plot_Cluster_kspace_2D_slice(
        threshold, figsize, data_ind, slice_ind, axis_ind
    )
    ax = plt.gca()
    _apply_slice_axes_metadata(ax, data, axis_ind, slice_value, label_size=22)
    _save_figure(os.path.join(outdir, filename))


def _save_zoom_plot(data, clusterGMM, outdir, filename, axis_ind, slice_value,
                    zoom_window):
    """Save the zoomed-in cluster map used in the tutorial notebooks."""
    if zoom_window is None:
        return

    row_slice, col_slice = zoom_window
    x_axis, y_axis = _slice_plot_axes(data, axis_ind)
    x_stop = min(col_slice.stop, len(x_axis.nxvalue) - 1)
    y_stop = min(row_slice.stop, len(y_axis.nxvalue) - 1)
    x_extent = [x_axis.nxvalue[col_slice.start], x_axis.nxvalue[x_stop]]
    y_extent = [y_axis.nxvalue[row_slice.start], y_axis.nxvalue[y_stop]]

    plt.figure(figsize=(10, 10))
    plt.imshow(
        clusterGMM.plot_image[row_slice, col_slice],
        origin="lower",
        cmap=clusterGMM.plot_cmap,
        norm=clusterGMM.plot_norm,
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
    )
    plt.xlabel(_axis_display_name(x_axis), size=22)
    plt.ylabel(_axis_display_name(y_axis), size=22)
    axis_label = _axis_display_name(data.nxaxes[axis_ind + 1])
    plt.title(f"{axis_label}={slice_value}", size=22)
    _save_figure(os.path.join(outdir, filename))


def _save_avg_intensity_plot(temp_values, data_values, cluster_assigns, outdir,
                             filename):
    """Save the tutorial-style cluster-averaged raw intensity plot."""
    color_list = [
        "red", "blue", "green", "purple", "yellow",
        "orange", "pink", "black", "grey", "cyan",
    ]

    plt.figure()
    nc = int(np.max(cluster_assigns)) + 1 if len(cluster_assigns) else 0
    for i in range(nc):
        cluster_mask = cluster_assigns == i
        yc = np.mean(data_values[:, cluster_mask], axis=1).flatten()
        plt.plot(temp_values, yc, color=color_list[i], lw=2)

    plt.yscale("linear")
    plt.xlabel("T")
    plt.ylabel("I (cluster avged)")
    _save_figure(os.path.join(outdir, filename))


def _plot_qmap(data, Data_ind, pixel_assigns, nc, outdir, prefix=""):
    """Plot and save the cluster Q-map."""
    cluster_image = np.zeros(data.nxsignal.nxvalue[0].shape)

    for i in range(nc):
        cluster_mask = pixel_assigns == i
        c_ind = Data_ind[cluster_mask]
        c_ind = tuple(np.array(c_ind).T)
        cluster_image[c_ind] = i + 1

    fig, ax = plt.subplots(figsize=(8, 8))
    if cluster_image.ndim == 3:
        mid = cluster_image.shape[0] // 2
        im = ax.imshow(cluster_image[mid], origin="lower", cmap="viridis")
        ax.set_title(f"Cluster Q Map (Slice {mid})")
    elif cluster_image.ndim == 2:
        im = ax.imshow(cluster_image, origin="lower", cmap="viridis")
        ax.set_title("Cluster Q Map")
    else:
        print("  Q-map has unsupported shape, skipping plot.")
        plt.close(fig)
        return
        
    plt.colorbar(im, ax=ax, label="Cluster ID")
    path = os.path.join(outdir, f"{prefix}qmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Q-map saved to {path}")


def _plot_trajectories(data, cluster_means, cluster_covs, nc, rescale,
                       outdir, prefix=""):
    """Plot and save cluster mean trajectories with ±1 std envelopes."""
    xc = data.nxaxes[0].nxvalue

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(nc):
        yc = cluster_means[i]
        yc_std = np.sqrt(np.abs(cluster_covs[i]))
        ax.plot(xc, yc, color=f"C{i}", lw=2, label=f"Cluster {i+1}")
        ax.fill_between(xc, yc - yc_std, yc + yc_std,
                        color=f"C{i}", alpha=0.3)
    ax.set_xlabel("Temperature / independent variable")
    ax.set_ylabel(f"Intensity (rescaled = {rescale})")
    ax.set_title("Cluster Means and Variances")
    ax.legend()
    path = os.path.join(outdir, f"{prefix}trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectories saved to {path}")


def _plot_avg_intensities(data, Data_thresh, cluster_assigns, nc, outdir,
                          prefix=""):
    """Plot and save average raw intensities per cluster."""
    xc = data.nxaxes[0].nxvalue

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(nc):
        cluster_mask = cluster_assigns == i
        data_c = Data_thresh[:, cluster_mask]
        yc = np.mean(data_c, axis=1)
        ax.plot(xc, yc, color=f"C{i}", lw=1, marker="o", markersize=2,
                label=f"Cluster {i+1}")
    ax.set_xlabel("Temperature / independent variable")
    ax.set_ylabel("Cluster Intensity")
    ax.set_title("Average Intensity in Cluster")
    ax.legend()
    path = os.path.join(outdir, f"{prefix}avg_intensities.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Average intensities saved to {path}")


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def run_xtec_d(args):
    """XTEC-d: direct GMM clustering (torchgmm, GPU preprocessing)."""
    data = _load_data(args.input, args.entry, args.slices)
    device = _get_device(args.device)
    nc = args.n_clusters

    print(f"[XTEC-d] {nc} clusters | threshold={args.threshold} | "
          f"rescale={args.rescale} | device={device}")

    t0 = time.time()
    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    threshold = Threshold_Background(masked, threshold_type=thresh_type,
                                     device=device)
    results = _run_direct_gmm(
        data,
        threshold,
        threshold.data_thresholded,
        threshold.ind_thresholded,
        nc,
        args.rescale,
        device,
        random_state=0 if args.random_state is None else args.random_state,
        reorder=args.reorder_clusters,
        init_strategy_mode=args.init_strategy_mode,
        post_stepwise_epochs=args.post_stepwise_epochs,
        post_stepwise_tol=args.post_stepwise_tol,
        solver_mode=args.solver_mode,
        batch_num=args.batch_num,
        max_batch_epoch=args.max_batch_epoch,
        max_full_epoch=args.max_full_epoch,
    )
    cluster_assigns = results["cluster_assigns"]
    cluster_means = results["cluster_means"]
    cluster_covs = results["cluster_covs"]
    Data_thresh_np = results["Data_thresh_np"]
    Data_ind_np = results["Data_ind_np"]

    elapsed = time.time() - t0
    print(f"  Clustering completed in {elapsed:.2f} s")
    print(f"  Cluster sizes: {[int(np.sum(cluster_assigns == k)) for k in range(nc)]}")
    if args.reorder_clusters:
        print("  Cluster ordering: deterministic low-temperature intensity ordering")

    _save_results(args.output, cluster_assigns, cluster_assigns, Data_ind_np,
                  Data_thresh_np, cluster_means, cluster_covs)
    _plot_qmap(data, Data_ind_np, cluster_assigns, nc, args.output)
    _plot_trajectories(data, cluster_means, cluster_covs, nc, args.rescale,
                       args.output)
    _plot_avg_intensities(data, Data_thresh_np, cluster_assigns, nc,
                          args.output)


def run_tutorial_d(args):
    """Tutorial-faithful XTEC-d workflow mirroring the reference notebooks."""
    data = _load_data(args.input, args.entry, args.slices)
    device = _get_device(args.device)
    first_nc = args.first_pass_clusters

    print(f"[Tutorial XTEC-d] first-pass clusters={first_nc} | "
          f"second-pass clusters={args.second_pass_clusters} | "
          f"threshold={args.threshold} | rescale={args.rescale} | device={device}")

    os.makedirs(args.output, exist_ok=True)
    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    threshold = Threshold_Background(masked, threshold_type=thresh_type,
                                     device=device)

    slice_ind, actual_slice_value = _save_threshold_plots(
        data, threshold, args.output, args.slice_axis, args.slice_value
    )
    zoom_window = _parse_zoom_window(args.zoom_window)

    first_pass = _run_direct_gmm(
        data,
        threshold,
        threshold.data_thresholded,
        threshold.ind_thresholded,
        first_nc,
        args.rescale,
        device,
        random_state=args.random_state,
        reorder=args.reorder_clusters,
        init_strategy_mode=args.init_strategy_mode,
        post_stepwise_epochs=args.post_stepwise_epochs,
        post_stepwise_tol=args.post_stepwise_tol,
        solver_mode=args.solver_mode,
        batch_num=args.batch_num,
        max_batch_epoch=args.max_batch_epoch,
        max_full_epoch=args.max_full_epoch,
    )

    clusterGMM_1 = first_pass["clusterGMM"]
    cluster_assigns_1 = first_pass["cluster_assigns"]
    cluster_means_1 = first_pass["cluster_means"]
    cluster_covs_1 = first_pass["cluster_covs"]
    Data_thresh_np = first_pass["Data_thresh_np"]
    Data_ind_np = first_pass["Data_ind_np"]
    Rescaled_data_np = first_pass["Rescaled_data_np"]

    print(f"  First-pass cluster sizes: "
          f"{[int(np.sum(cluster_assigns_1 == k)) for k in range(first_nc)]}")

    _save_results(
        args.output,
        cluster_assigns_1,
        cluster_assigns_1,
        Data_ind_np,
        Data_thresh_np,
        cluster_means_1,
        cluster_covs_1,
        prefix="pass1_",
    )
    _save_traj_plot(
        data,
        clusterGMM_1,
        args.output,
        "trajectories_pass1.png",
        "Cluster mean and variance \n (rescaled) intensity trajectory ",
    )
    _save_qmap_plot(
        data,
        threshold,
        clusterGMM_1,
        Data_ind_np,
        args.output,
        "qmap_pass1.png",
        args.slice_axis,
        slice_ind,
        actual_slice_value,
    )
    _save_zoom_plot(
        data,
        clusterGMM_1,
        args.output,
        "qmap_zoom_pass1.png",
        args.slice_axis,
        actual_slice_value,
        zoom_window,
    )

    if args.good_cluster is not None:
        good_mask = cluster_assigns_1 == args.good_cluster
        print(f"  Retaining user-selected first-pass cluster {args.good_cluster} "
              f"for the second pass")
    elif args.bad_clusters:
        bad_clusters = set(_parse_int_list(args.bad_clusters))
        good_mask = ~np.isin(cluster_assigns_1, sorted(bad_clusters))
        print(f"  Removing first-pass clusters {sorted(bad_clusters)} "
              f"for the second pass")
    else:
        std_scores = np.array([
            0.0 if np.sum(cluster_assigns_1 == k) < 2 else np.std(cluster_means_1[k])
            for k in range(first_nc)
        ])
        discovery_cluster = int(np.argmax(std_scores))
        good_mask = cluster_assigns_1 == discovery_cluster
        print(f"  Auto-selected discovery cluster {discovery_cluster} "
              f"for the second pass")

    if not np.any(good_mask):
        raise RuntimeError("Second-pass selection is empty. Adjust --good-cluster "
                           "or --bad-clusters.")

    Good_data = Data_thresh_np[:, good_mask]
    Good_rescaled_data = Rescaled_data_np[:, good_mask]
    Good_ind = Data_ind_np[good_mask]

    second_pass = _run_direct_gmm(
        data,
        threshold,
        Good_data,
        Good_ind,
        args.second_pass_clusters,
        args.rescale,
        device,
        random_state=args.random_state,
        reorder=args.reorder_clusters,
        rescaled_data=Good_rescaled_data,
        init_strategy_mode=args.init_strategy_mode,
        post_stepwise_epochs=args.post_stepwise_epochs,
        post_stepwise_tol=args.post_stepwise_tol,
        solver_mode=args.solver_mode,
        batch_num=args.batch_num,
        max_batch_epoch=args.max_batch_epoch,
        max_full_epoch=args.max_full_epoch,
    )

    clusterGMM_2 = second_pass["clusterGMM"]
    cluster_assigns_2 = second_pass["cluster_assigns"]
    cluster_means_2 = second_pass["cluster_means"]
    cluster_covs_2 = second_pass["cluster_covs"]

    print(f"  Second-pass cluster sizes: "
          f"{[int(np.sum(cluster_assigns_2 == k)) for k in range(args.second_pass_clusters)]}")

    _save_results(
        args.output,
        cluster_assigns_2,
        cluster_assigns_2,
        Good_ind,
        Good_data,
        cluster_means_2,
        cluster_covs_2,
        prefix="pass2_",
    )
    _save_traj_plot(
        data,
        clusterGMM_2,
        args.output,
        "trajectories_pass2.png",
        " ",
    )
    _save_qmap_plot(
        data,
        threshold,
        clusterGMM_2,
        Good_ind,
        args.output,
        "qmap_pass2.png",
        args.slice_axis,
        slice_ind,
        actual_slice_value,
    )
    _save_zoom_plot(
        data,
        clusterGMM_2,
        args.output,
        "qmap_zoom_pass2.png",
        args.slice_axis,
        actual_slice_value,
        zoom_window,
    )
    _save_avg_intensity_plot(
        data.nxaxes[0].nxvalue,
        Good_data,
        cluster_assigns_2,
        args.output,
        "avg_intensities_pass2.png",
    )

    pickle_name = args.pickle_name
    if pickle_name is None:
        axis_label = _axis_display_name(data.nxaxes[args.slice_axis + 1]).lower()
        if axis_label == "l" and abs(actual_slice_value) < 1e-12:
            pickle_name = "csrs_0x0_clustering.p"
        else:
            pickle_name = "tutorial_d_final_clustering.p"

    _save_tutorial_pickle(
        os.path.join(args.output, pickle_name),
        data.nxaxes[0].nxvalue,
        Good_data,
        Good_ind,
        cluster_assigns_2,
        cluster_means_2,
        cluster_covs_2,
    )


def run_xtec_s(args):
    """XTEC-s: peak-averaged GMM clustering (GPU via torchgmm)."""
    data = _load_data(args.input, args.entry, args.slices)
    device = _get_device(args.device)
    nc = args.n_clusters

    print(f"[XTEC-s] {nc} clusters | threshold={args.threshold} | "
          f"rescale={args.rescale} | device={device}")

    t0 = time.time()
    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    threshold = Threshold_Background(masked, threshold_type=thresh_type,
                                     device=device)

    Peak_avg = Peak_averaging(data.nxsignal.nxvalue, threshold, device=device)
    Data_thresh = Peak_avg.peak_avg_data

    Rescaled_data = _rescale(threshold, Data_thresh, args.rescale, device)

    Data_for_GMM = Rescaled_data.T
    gmm_kwargs = {
        "cov_type": "diag",
        "solver_mode": args.solver_mode,
        "init_strategy_mode": args.init_strategy_mode,
        "post_stepwise_epochs": int(args.post_stepwise_epochs),
        "post_stepwise_tol": args.post_stepwise_tol,
        "batch_num": int(args.batch_num),
        "max_batch_epoch": int(args.max_batch_epoch),
        "max_full_epoch": int(args.max_full_epoch),
    }
    if args.random_state is not None:
        gmm_kwargs["random_state"] = int(args.random_state)
    clusterGMM = GMM(Data_for_GMM, nc, **gmm_kwargs)
    clusterGMM.RunEM()
    clusterGMM.Get_pixel_labels(Peak_avg)

    Data_thresh_np = _to_numpy(Data_thresh)
    Data_ind_np = _to_numpy(clusterGMM.Data_ind)
    cluster_assigns = _to_numpy(clusterGMM.cluster_assignments)
    pixel_assigns = _to_numpy(clusterGMM.Pixel_assignments)
    cluster_means = _to_numpy(clusterGMM.means)
    cluster_covs = [_to_numpy(clusterGMM.cluster[i].cov) for i in range(nc)]

    elapsed = time.time() - t0
    print(f"\n  Clustering completed in {elapsed:.2f} s")
    print(f"  Cluster sizes: {clusterGMM.num_per_cluster}")

    if args.reorder_clusters:
        # Deterministic cluster ordering (descending low-T intensity)
        temp_values = data.nxaxes[0].nxvalue
        cluster_assigns, pixel_assigns, cluster_means, cluster_covs = \
            _reorder_clusters(cluster_assigns, pixel_assigns,
                              cluster_means, cluster_covs,
                              Data_thresh_np, nc, temp_values)
        print(f"  Reordered cluster sizes: "
              f"{[int(np.sum(cluster_assigns == k)) for k in range(nc)]}")

    _save_results(args.output, cluster_assigns, pixel_assigns, Data_ind_np,
                  Data_thresh_np, cluster_means, cluster_covs)
    _plot_qmap(data, Data_ind_np, pixel_assigns, nc, args.output)
    _plot_trajectories(data, cluster_means, cluster_covs, nc, args.rescale,
                       args.output)
    _plot_avg_intensities(data, Data_thresh_np, cluster_assigns, nc,
                          args.output)


def run_label_smooth(args):
    """XTEC label smooth: GMM with Markov label smoothing (GPU)."""
    print("\n[Under Construction] Label smoothening is currently disabled due to "
          "out-of-memory issues on extremely large datasets. It will be patched soon.")
    
    # PATCH INSTRUCTIONS FOR FUTURE REFERNCE:
    # Use a 3-layer strategy.
    # 
    # Immediate (no algorithm change)
    # Lower chunk_size in Build_Markov_Matrix (4096 -> 512 or 256).
    # Increase zero_cutoff (1e-2 -> 5e-2) if acceptable.
    # Reduce L_scale slightly.
    # Force cov_type="diag" for GMM (already default).
    # 
    # Keep exact functionality, fix memory properly
    # In GMM.py, replace (row_chunk x N) build with 2D tiling (row_chunk x col_chunk).
    # Accumulate row/col/val on CPU torch tensors (.cpu()), not GPU lists.
    # Keep kernel/cutoff/row-normalization unchanged, so behavior matches old code.
    # 
    # Add CLI args in xtec_cli.py:
    # --markov-row-chunk
    # --markov-col-chunk
    # --zero-cutoff
    # 
    # Robust scaling
    # Auto-select chunk sizes from free VRAM (torch.cuda.mem_get_info) with a safety margin (e.g., 60%).
    # Optional fallback: if CUDA OOM during Markov build, retry with half chunk sizes automatically.
    
    # --- CURRENT CODE (COMMENTED OUT FOR FUTURE USE) ---
    # data = _load_data(args.input, args.entry, args.slices)
    # device = _get_device(args.device)
    # nc = args.n_clusters
    #
    # print(f"[Label Smooth] {nc} clusters | threshold={args.threshold} | "
    #       f"rescale={args.rescale} | L_scale={args.L_scale} | "
    #       f"smooth_type={args.smooth_type} | device={device}")
    #
    # t0 = time.time()
    # thresh_type = "KL" if args.threshold else "No threshold"
    # masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    # threshold = Threshold_Background(masked, threshold_type=thresh_type,
    #                                  device=device)
    # Data_thresh = threshold.data_thresholded
    # Data_ind = threshold.ind_thresholded
    #
    # Rescaled_data = _rescale(threshold, Data_thresh, args.rescale, device)
    #
    # # Build unit cell shape for periodic kernel
    # kernel_type = args.smooth_type
    # unit_cell_shape = None
    # if kernel_type == "periodic":
    #     unit_cell_shape = []
    #     for nxq in data.nxaxes[1:]:
    #         q = nxq.nxvalue
    #         x = np.min(q[q % 1 == 0])
    #         l = len(nxq[x : x + 1]) - 1
    #         unit_cell_shape.append(l)
    #     unit_cell_shape = np.array(unit_cell_shape)
    #
    # # Build Markov matrix on GPU
    # Markov_matrix = GMM_kernels.Build_Markov_Matrix(
    #     Data_ind, args.L_scale, kernel_type, unit_cell_shape, device=device
    # )
    #
    # Data_for_GMM = Rescaled_data.T
    # clusterGMM = GMM(Data_for_GMM, nc)
    # clusterGMM.RunEM(
    #     label_smoothing_flag=True, Markov_matrix=Markov_matrix
    # )
    #
    # Data_thresh_np = _to_numpy(Data_thresh)
    # Data_ind_np = _to_numpy(Data_ind)
    # cluster_assigns = _to_numpy(clusterGMM.cluster_assignments)
    # cluster_means = _to_numpy(clusterGMM.means)
    # cluster_covs = [_to_numpy(clusterGMM.cluster[i].cov) for i in range(nc)]
    #
    # elapsed = time.time() - t0
    # print(f"\n  Clustering completed in {elapsed:.2f} s")
    # print(f"  Cluster sizes: {clusterGMM.num_per_cluster}")
    #
    # # Deterministic cluster ordering (descending low-T intensity)
    # temp_values = data.nxaxes[0].nxvalue
    # cluster_assigns, cluster_assigns, cluster_means, cluster_covs = \
    #     _reorder_clusters(cluster_assigns, cluster_assigns,
    #                       cluster_means, cluster_covs,
    #                       Data_thresh_np, nc, temp_values)
    # print(f"  Reordered cluster sizes: "
    #       f"{[int(np.sum(cluster_assigns == k)) for k in range(nc)]}")
    #
    # _save_results(args.output, cluster_assigns, cluster_assigns, Data_ind_np,
    #               Data_thresh_np, cluster_means, cluster_covs)
    # _plot_qmap(data, Data_ind_np, cluster_assigns, nc, args.output)
    # _plot_trajectories(data, cluster_means, cluster_covs, nc, args.rescale,
    #                    args.output)
    # _plot_avg_intensities(data, Data_thresh_np, cluster_assigns, nc,
    #                       args.output)
    
    return


def run_bic_d(args):
    """BIC score sweep for XTEC-d (torchgmm)."""
    data = _load_data(args.input, args.entry, args.slices)
    device = _get_device(args.device)

    print(f"[BIC XTEC-d] nc={args.min_nc}..{args.max_nc} | "
          f"threshold={args.threshold} | rescale={args.rescale}")

    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    threshold = Threshold_Background(masked, threshold_type=thresh_type,
                                     device=device)
    Data_thresh = threshold.data_thresholded

    Rescaled_data = _rescale(threshold, Data_thresh, args.rescale, device)

    Data_for_GMM = Rescaled_data.T
    n_samples, n_features = map(int, Data_for_GMM.shape)

    ks = np.arange(args.min_nc, args.max_nc)
    bics = []
    for k in ks:
        clusterGMM = GMM(Data_for_GMM, int(k), cov_type="diag", random_state=0)
        clusterGMM.RunEM()
        bics.append(
            _bic_from_loglikelihood(
                clusterGMM.log_likelihood,
                n_components=int(k),
                n_features=n_features,
                n_samples=n_samples,
                cov_type="diag",
            )
        )
        print(f"  k={k}: BIC={bics[-1]:.2f}")

    os.makedirs(args.output, exist_ok=True)
    with h5py.File(os.path.join(args.output, "bic_xtec_d.h5"), "w") as f:
        f.create_dataset("n_clusters", data=ks)
        f.create_dataset("bic_scores", data=bics)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, "-o", lw=2, markersize=6)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("BIC Score")
    ax.set_title("XTEC-D BIC Score")
    path = os.path.join(args.output, "bic_xtec_d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  BIC plot saved to {path}")


def run_bic_s(args):
    """BIC score sweep for XTEC-s (peak averaging, torchgmm)."""
    data = _load_data(args.input, args.entry, args.slices)
    device = _get_device(args.device)

    print(f"[BIC XTEC-s] nc={args.min_nc}..{args.max_nc} | "
          f"threshold={args.threshold} | rescale={args.rescale}")

    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    threshold = Threshold_Background(masked, threshold_type=thresh_type,
                                     device=device)
    Peak_avg = Peak_averaging(data.nxsignal.nxvalue, threshold, device=device)
    Data_thresh = Peak_avg.peak_avg_data

    Rescaled_data = _rescale(threshold, Data_thresh, args.rescale, device)

    Data_for_GMM = Rescaled_data.T
    n_samples, n_features = map(int, Data_for_GMM.shape)

    ks = np.arange(args.min_nc, args.max_nc)
    bics = []
    for k in ks:
        clusterGMM = GMM(Data_for_GMM, int(k), cov_type="diag", random_state=0)
        clusterGMM.RunEM()
        bics.append(
            _bic_from_loglikelihood(
                clusterGMM.log_likelihood,
                n_components=int(k),
                n_features=n_features,
                n_samples=n_samples,
                cov_type="diag",
            )
        )
        print(f"  k={k}: BIC={bics[-1]:.2f}")

    os.makedirs(args.output, exist_ok=True)
    with h5py.File(os.path.join(args.output, "bic_xtec_s.h5"), "w") as f:
        f.create_dataset("n_clusters", data=ks)
        f.create_dataset("bic_scores", data=bics)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, "-o", lw=2, markersize=6)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("BIC Score")
    ax.set_title("XTEC-S (Peak Avg) BIC Score")
    path = os.path.join(args.output, "bic_xtec_s.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  BIC plot saved to {path}")


def run_test(args):
    """Run a quick sanity check of XTEC-GPU hardware and GMM code."""
    device = _get_device(args.device)
    print("=========================================")
    print("XTEC-GPU Hardware & Environment Check")
    print("=========================================")
    try:
        from importlib.metadata import version
        pkg_version = version("XTEC-GPU")
    except Exception:
        pkg_version = "unknown"
    
    print(f"XTEC-GPU version:{pkg_version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available:   {has_mps}")
    print(f"Selected device: {device}")

    print("\n[Running synthetic Preprocessing test]")
    prep_data = np.random.rand(50, 30, 30).astype(np.float32)
    # Inject zeros to test Mask_Zeros
    prep_data[:, 0:10, :] = 0.0
    
    t0_prep = time.time()
    try:
        prep_torch = torch.as_tensor(prep_data).to(device)
        print("Testing Mask_Zeros... ", end="", flush=True)
        mask = Mask_Zeros(prep_torch, mask_type="zero_mean")
        print(f"Retained {mask.data_nonzero.shape[1]} / {mask.data_shape[1]*mask.data_shape[2]} pixels.")
        
        print("Testing Threshold_Background... ", end="", flush=True)
        thresh = Threshold_Background(mask, threshold_type="simple")
        print(f"Cutoff: {thresh.LogI_cutoff:.2f}. Passed: {thresh.data_thresholded.shape[1]}")
        
        print("Testing Peak_averaging... ", end="", flush=True)
        peaks = Peak_averaging(prep_torch, thresh, device=device)
        print(f"Found {peaks.peak_avg_data.shape[1]} peaks.")

        print("Testing Markov Matrix builder... ", end="", flush=True)
        M = GMM_kernels.Build_Markov_Matrix(thresh.ind_thresholded, L_scale=1.0, device=device)
        print(f"Created {M.shape} sparse matrix.")

        print(f"Preprocessing test completed in {time.time() - t0_prep:.2f} s")
    except Exception as e:
        print(f"\n❌ Preprocessing Error:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[Running synthetic GMM test]")
    # Create simple synthetic data
    N_data = 2000
    N_temp = 50
    N_clusters = 3
    print(f"Generating synthetic data: {N_data} points, {N_temp} temperatures")
    data_np = np.random.rand(N_data, N_temp).astype(np.float32)

    t0 = time.time()
    try:
        data_torch = torch.as_tensor(data_np).to(device)
        print("Data transferred to device successfully.")

        print(f"Instantiating torchgmm ({N_clusters} clusters)... ", end="", flush=True)
        clusterGMM = GMM(data_torch.T, N_clusters)
        print("Done.")

        print("Running EM... ")
        clusterGMM.RunEM()
        print("\nDone.")
        print(f"Found cluster sizes: {clusterGMM.num_per_cluster}")
        print(f"Synthetic test completed in {time.time() - t0:.2f} s")
        print("\nAll systems nominal. ✅")
    except Exception as e:
        print(f"\n❌ Error during execution:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="xtec-gpu",
        description="XTEC-GPU: GPU-accelerated clustering for "
                    "temperature-dependent scattering data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Common arguments added to each subparser --------------------------
    def _add_common(sp):
        sp.add_argument("input", help="Path to the input .nxs (NeXus) file")
        sp.add_argument("-o", "--output", required=True,
                        help="Output directory for results")
        sp.add_argument("--entry", default="entry/data",
                        help="HDF5 path to the NXdata group "
                             "(default: 'entry/data')")
        sp.add_argument("--slices", default=None,
                        help="Comma-separated slice string, e.g. "
                             "':,0.0:1.0,-10:10,-15:15'")
        sp.add_argument("--threshold", action="store_true", default=True,
                        help="Apply KL background thresholding (default: on)")
        sp.add_argument("--no-threshold", dest="threshold",
                        action="store_false",
                        help="Disable background thresholding")
        sp.add_argument("--rescale",
                        choices=["mean", "z-score", "log-mean", "None"],
                        default="mean",
                        help="Rescaling method (default: mean)")
        sp.add_argument("--device", default="auto", type=str,
                        help="Compute device: 'auto', 'cpu', 'cuda', 'cuda:1', 'mps' (default: 'auto')")

    def _add_gmm_parity_args(sp, random_state_default=None,
                             reorder_default=False,
                             random_state_help_suffix="",
                             solver_mode_default="torchgmm",
                             init_strategy_default="sklearn-kmeans",
                             post_stepwise_epochs_default=0):
        sp.add_argument("--random-state", type=int, default=random_state_default,
                        help="Random seed for GMM initialization "
                             f"(default: {random_state_help_suffix})")
        sp.add_argument("--solver-mode",
                        choices=["torchgmm", "legacy-stepwise"],
                        default=solver_mode_default,
                        help="GMM solver backend "
                             f"(default: {solver_mode_default})")
        sp.add_argument("--init-strategy-mode",
                        choices=["kmeans++", "xtec", "sklearn-kmeans", "cuml-kmeans"],
                        default=init_strategy_default,
                        help="Cluster-mean initialization strategy "
                             f"(default: {init_strategy_default})")
        sp.add_argument("--post-stepwise-epochs", type=int, default=post_stepwise_epochs_default,
                        help="GPU-native stepwise EM refinement iterations "
                             "after the initial torchgmm fit "
                             f"(default: {post_stepwise_epochs_default})")
        sp.add_argument("--post-stepwise-tol", type=float, default=None,
                        help="Tolerance for post-fit GPU stepwise refinement "
                             "(default: use the GMM tolerance)")
        sp.add_argument("--batch-num", type=int, default=1,
                        help="Number of mini-batches for legacy-stepwise mode "
                             "(default: 1)")
        sp.add_argument("--max-batch-epoch", type=int, default=50,
                        help="Maximum batch-phase epochs for legacy-stepwise mode "
                             "(default: 50)")
        sp.add_argument("--max-full-epoch", type=int, default=500,
                        help="Maximum full-data epochs for legacy-stepwise mode "
                             "(default: 500)")
        sp.add_argument("--reorder-clusters", dest="reorder_clusters",
                        action="store_true", default=reorder_default,
                        help="Reorder clusters by descending low-temperature "
                             "raw intensity")
        sp.add_argument("--no-reorder-clusters", dest="reorder_clusters",
                        action="store_false",
                        help="Keep the native GMM cluster ordering")

    # -- xtec-d -----------------------------------------------------------
    sp_d = subparsers.add_parser(
        "xtec-d", help="XTEC-d: direct GMM (torchgmm) with GPU preprocessing")
    _add_common(sp_d)
    sp_d.add_argument("-n", "--n-clusters", type=int, default=4,
                      help="Number of clusters (default: 4)")
    _add_gmm_parity_args(sp_d, random_state_default=None,
                         reorder_default=True,
                         random_state_help_suffix="0 for xtec-d",
                         solver_mode_default="torchgmm",
                         init_strategy_default="sklearn-kmeans",
                         post_stepwise_epochs_default=0)
    sp_d.set_defaults(func=run_xtec_d)

    # -- tutorial-d -------------------------------------------------------
    sp_td = subparsers.add_parser(
        "tutorial-d",
        help="Tutorial-faithful XTEC-d workflow with threshold plots and a second clustering pass",
    )
    _add_common(sp_td)
    sp_td.add_argument("--first-pass-clusters", type=int, default=4,
                       help="Number of clusters for the first pass (default: 4)")
    sp_td.add_argument("--second-pass-clusters", type=int, default=3,
                       help="Number of clusters for the second pass (default: 3)")
    sp_td.add_argument("--good-cluster", type=int, default=None,
                       help="Retain only this first-pass cluster for the second pass")
    sp_td.add_argument("--bad-clusters", default=None,
                       help="Comma-separated first-pass clusters to drop before "
                            "the second pass, e.g. '0,1,3'")
    sp_td.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=0,
                       help="Orthogonal axis for the tutorial slice plot: "
                            "0=L, 1=K, 2=H (default: 0)")
    sp_td.add_argument("--slice-value", type=float, default=0.0,
                       help="Axis value for the tutorial slice plot "
                            "(default: 0.0)")
    sp_td.add_argument("--zoom-window", default="200:300,0:100",
                       help="Zoom window for q-map images in the form "
                            "'row0:row1,col0:col1' (default: 200:300,0:100)")
    _add_gmm_parity_args(sp_td, random_state_default=None,
                         reorder_default=False,
                         random_state_help_suffix="None, matching the tutorial notebooks",
                         solver_mode_default="torchgmm",
                         init_strategy_default="sklearn-kmeans",
                         post_stepwise_epochs_default=0)
    sp_td.add_argument("--pickle-name", default=None,
                       help="Optional filename for the final tutorial pickle")
    sp_td.set_defaults(func=run_tutorial_d)

    # -- xtec-s -----------------------------------------------------------
    sp_s = subparsers.add_parser(
        "xtec-s", help="XTEC-s: peak-averaged GMM (torchgmm on GPU)")
    _add_common(sp_s)
    sp_s.add_argument("-n", "--n-clusters", type=int, default=4,
                      help="Number of clusters (default: 4)")
    _add_gmm_parity_args(sp_s, random_state_default=None,
                         reorder_default=True,
                         random_state_help_suffix="None for xtec-s",
                         solver_mode_default="torchgmm",
                         init_strategy_default="sklearn-kmeans",
                         post_stepwise_epochs_default=0)
    sp_s.set_defaults(func=run_xtec_s)

    # -- label-smooth -----------------------------------------------------
    sp_ls = subparsers.add_parser(
        "label-smooth",
        help="XTEC label smooth: GMM with Markov label smoothing (GPU)")
    _add_common(sp_ls)
    sp_ls.add_argument("-n", "--n-clusters", type=int, default=4,
                       help="Number of clusters (default: 4)")
    sp_ls.add_argument("--L-scale", type=float, default=0.05,
                       help="Smoothing length scale in pixel units "
                            "(default: 0.05)")
    sp_ls.add_argument("--smooth-type", choices=["local", "periodic"],
                       default="local",
                       help="Smoothing kernel type (default: local)")
    sp_ls.set_defaults(func=run_label_smooth)

    # -- bic-d ------------------------------------------------------------
    sp_bd = subparsers.add_parser(
        "bic-d", help="BIC score sweep for XTEC-d")
    _add_common(sp_bd)
    sp_bd.add_argument("--min-nc", type=int, default=2,
                       help="Minimum number of clusters (default: 2)")
    sp_bd.add_argument("--max-nc", type=int, default=14,
                       help="Maximum number of clusters (default: 14)")
    sp_bd.set_defaults(func=run_bic_d)

    # -- bic-s ------------------------------------------------------------
    sp_bs = subparsers.add_parser(
        "bic-s", help="BIC score sweep for XTEC-s (peak averaging)")
    _add_common(sp_bs)
    sp_bs.add_argument("--min-nc", type=int, default=2,
                       help="Minimum number of clusters (default: 2)")
    sp_bs.add_argument("--max-nc", type=int, default=14,
                       help="Maximum number of clusters (default: 14)")
    sp_bs.set_defaults(func=run_bic_s)

    # -- test -------------------------------------------------------------
    sp_t = subparsers.add_parser(
        "test", help="Run a quick sanity check of hardware and GMM code")
    sp_t.add_argument("--device", default="auto", type=str,
                      help="Compute device: 'auto', 'cpu', 'cuda', 'cuda:1', 'mps' (default: 'auto')")
    sp_t.set_defaults(func=run_test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
