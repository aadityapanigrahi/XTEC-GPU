"""
XTEC-GPU Command-Line Interface
================================

Provides the same clustering functionality as the NeXpy plugin
(``cluster_data.py``) but driven entirely from the terminal.

Usage examples
--------------
::

    # XTEC-d (direct GMM via sklearn, GPU preprocessing)
    xtec-gpu xtec-d data.nxs -o results/ -n 4 --rescale mean

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
        f.create_dataset("cluster_means", data=cluster_means)
        f.create_dataset("cluster_covariances", data=np.array(cluster_covs))

    print(f"  Results saved to {path}")


def _plot_qmap(data, Data_ind, pixel_assigns, nc, outdir, prefix=""):
    """Plot and save the cluster Q-map."""
    cluster_image = np.zeros(data.nxsignal.nxvalue[0].shape)

    for i in range(nc):
        cluster_mask = pixel_assigns == i
        c_ind = Data_ind[cluster_mask]
        c_ind = tuple(np.array(c_ind).T)
        cluster_image[c_ind] = i + 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cluster_image, origin="lower", cmap="viridis")
    ax.set_title("Cluster Q Map")
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
    """XTEC-d: direct GMM clustering via sklearn (GPU preprocessing)."""
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
    Data_thresh = threshold.data_thresholded
    Data_ind = threshold.ind_thresholded

    Rescaled_data = _rescale(threshold, Data_thresh, args.rescale, device)

    from sklearn.mixture import GaussianMixture
    Data_for_GMM = _to_numpy(Rescaled_data).transpose()
    gm = GaussianMixture(
        n_components=nc, covariance_type="diag", random_state=0
    ).fit(Data_for_GMM)
    cluster_assigns = gm.predict(Data_for_GMM)

    cluster_means = gm.means_
    cluster_covs = gm.covariances_

    Data_thresh_np = _to_numpy(Data_thresh)
    Data_ind_np = _to_numpy(Data_ind)

    elapsed = time.time() - t0
    print(f"  Clustering completed in {elapsed:.2f} s")
    print(f"  Cluster sizes: {[int(np.sum(cluster_assigns == k)) for k in range(nc)]}")

    # Deterministic cluster ordering (descending low-T intensity)
    temp_values = data.nxaxes[0].nxvalue
    cluster_assigns, cluster_assigns, cluster_means, cluster_covs = \
        _reorder_clusters(cluster_assigns, cluster_assigns,
                          cluster_means, cluster_covs,
                          Data_thresh_np, nc, temp_values)
    print(f"  Reordered cluster sizes: "
          f"{[int(np.sum(cluster_assigns == k)) for k in range(nc)]}")

    _save_results(args.output, cluster_assigns, cluster_assigns, Data_ind_np,
                  Data_thresh_np, cluster_means, cluster_covs)
    _plot_qmap(data, Data_ind_np, cluster_assigns, nc, args.output)
    _plot_trajectories(data, cluster_means, cluster_covs, nc, args.rescale,
                       args.output)
    _plot_avg_intensities(data, Data_thresh_np, cluster_assigns, nc,
                          args.output)


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
    clusterGMM = GMM(Data_for_GMM, nc)
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
    data = _load_data(args.input, args.entry, args.slices)
    device = _get_device(args.device)
    nc = args.n_clusters

    print(f"[Label Smooth] {nc} clusters | threshold={args.threshold} | "
          f"rescale={args.rescale} | L_scale={args.L_scale} | "
          f"smooth_type={args.smooth_type} | device={device}")

    t0 = time.time()
    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(data.nxsignal.nxvalue, device=device)
    threshold = Threshold_Background(masked, threshold_type=thresh_type,
                                     device=device)
    Data_thresh = threshold.data_thresholded
    Data_ind = threshold.ind_thresholded

    Rescaled_data = _rescale(threshold, Data_thresh, args.rescale, device)

    # Build unit cell shape for periodic kernel
    kernel_type = args.smooth_type
    unit_cell_shape = None
    if kernel_type == "periodic":
        unit_cell_shape = []
        for nxq in data.nxaxes[1:]:
            q = nxq.nxvalue
            x = np.min(q[q % 1 == 0])
            l = len(nxq[x : x + 1]) - 1
            unit_cell_shape.append(l)
        unit_cell_shape = np.array(unit_cell_shape)

    # Build Markov matrix on GPU
    Markov_matrix = GMM_kernels.Build_Markov_Matrix(
        Data_ind, args.L_scale, kernel_type, unit_cell_shape, device=device
    )

    Data_for_GMM = Rescaled_data.T
    clusterGMM = GMM(Data_for_GMM, nc)
    clusterGMM.RunEM(
        label_smoothing_flag=True, Markov_matrix=Markov_matrix
    )

    Data_thresh_np = _to_numpy(Data_thresh)
    Data_ind_np = _to_numpy(Data_ind)
    cluster_assigns = _to_numpy(clusterGMM.cluster_assignments)
    cluster_means = _to_numpy(clusterGMM.means)
    cluster_covs = [_to_numpy(clusterGMM.cluster[i].cov) for i in range(nc)]

    elapsed = time.time() - t0
    print(f"\n  Clustering completed in {elapsed:.2f} s")
    print(f"  Cluster sizes: {clusterGMM.num_per_cluster}")

    # Deterministic cluster ordering (descending low-T intensity)
    temp_values = data.nxaxes[0].nxvalue
    cluster_assigns, cluster_assigns, cluster_means, cluster_covs = \
        _reorder_clusters(cluster_assigns, cluster_assigns,
                          cluster_means, cluster_covs,
                          Data_thresh_np, nc, temp_values)
    print(f"  Reordered cluster sizes: "
          f"{[int(np.sum(cluster_assigns == k)) for k in range(nc)]}")

    _save_results(args.output, cluster_assigns, cluster_assigns, Data_ind_np,
                  Data_thresh_np, cluster_means, cluster_covs)
    _plot_qmap(data, Data_ind_np, cluster_assigns, nc, args.output)
    _plot_trajectories(data, cluster_means, cluster_covs, nc, args.rescale,
                       args.output)
    _plot_avg_intensities(data, Data_thresh_np, cluster_assigns, nc,
                          args.output)


def run_bic_d(args):
    """BIC score sweep for XTEC-d."""
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

    from sklearn.mixture import GaussianMixture
    Data_for_GMM = _to_numpy(Rescaled_data).transpose()

    ks = np.arange(args.min_nc, args.max_nc)
    bics = []
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="diag")
        gm.fit(Data_for_GMM)
        bics.append(gm.bic(Data_for_GMM))
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
    """BIC score sweep for XTEC-s (peak averaging)."""
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

    from sklearn.mixture import GaussianMixture
    Data_for_GMM = _to_numpy(Rescaled_data).transpose()

    ks = np.arange(args.min_nc, args.max_nc)
    bics = []
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="diag")
        gm.fit(Data_for_GMM)
        bics.append(gm.bic(Data_for_GMM))
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

    # -- xtec-d -----------------------------------------------------------
    sp_d = subparsers.add_parser(
        "xtec-d", help="XTEC-d: direct GMM (sklearn) with GPU preprocessing")
    _add_common(sp_d)
    sp_d.add_argument("-n", "--n-clusters", type=int, default=4,
                      help="Number of clusters (default: 4)")
    sp_d.set_defaults(func=run_xtec_d)

    # -- xtec-s -----------------------------------------------------------
    sp_s = subparsers.add_parser(
        "xtec-s", help="XTEC-s: peak-averaged GMM (torchgmm on GPU)")
    _add_common(sp_s)
    sp_s.add_argument("-n", "--n-clusters", type=int, default=4,
                      help="Number of clusters (default: 4)")
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
