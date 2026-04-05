#!/usr/bin/env python3
"""Generate side-by-side CPU and GPU tutorial outputs in labeled folders.

Owns:
- workflow orchestration for CPU/GPU comparison runs
- comparison metadata and artifact layout

Does not own:
- core clustering implementations for CPU or GPU backends
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nexusformat.nexus import nxload


CPU_REPO_URL = "https://github.com/KimGroup/XTEC.git"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate labeled CPU and GPU tutorial outputs side by side.",
    )
    parser.add_argument("input", help="Path to the NeXus .nxs file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output root containing cpu_reference/ and gpu_cli/")
    parser.add_argument("--entry", default="entry/data",
                        help="HDF5 path to the NXdata group (default: entry/data)")
    parser.add_argument("--slices", default=":,0.0:1.0,-10.0:10.0,-15.0:15.0",
                        help="Comma-separated slice string matching the tutorial")
    parser.add_argument("--gpu-device", default="cuda:1",
                        help="GPU device passed to the XTEC-GPU CLI "
                             "(default: cuda:1)")
    parser.add_argument("--rescale", choices=["mean", "z-score", "log-mean", "None"],
                        default="mean",
                        help="Rescaling mode for both CPU and GPU runs")
    parser.add_argument("--threshold", dest="threshold", action="store_true",
                        default=True,
                        help="Enable KL thresholding (default: on)")
    parser.add_argument("--no-threshold", dest="threshold", action="store_false",
                        help="Disable thresholding")
    parser.add_argument("--first-pass-clusters", type=int, default=4,
                        help="Number of clusters in the first pass (default: 4)")
    parser.add_argument("--second-pass-clusters", type=int, default=3,
                        help="Number of clusters in the second pass (default: 3)")
    parser.add_argument("--good-cluster", type=int, default=None,
                        help="Keep only this first-pass cluster in the second pass")
    parser.add_argument("--bad-clusters", default=None,
                        help="Comma-separated first-pass clusters to remove "
                             "before the second pass")
    parser.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=0,
                        help="Orthogonal axis for 2D slice plots: 0=L, 1=K, 2=H")
    parser.add_argument("--slice-value", type=float, default=0.0,
                        help="Slice value for 2D plots (default: 0.0)")
    parser.add_argument("--zoom-window", default="200:300,0:100",
                        help="Zoom window in the form row0:row1,col0:col1")
    parser.add_argument("--cpu-repo",
                        default=str(Path(__file__).resolve().parents[1] / "external" / "KimGroup-XTEC"),
                        help="Path to the original KimGroup/XTEC repository")
    parser.add_argument("--no-clone-cpu-repo", dest="clone_cpu_repo",
                        action="store_false", default=True,
                        help="Fail instead of cloning the CPU repo when it is missing")
    parser.add_argument("--cpu-random-state", type=int, default=None,
                        help="Optional NumPy random seed for the CPU GMM")
    parser.add_argument("--gpu-random-state", type=int, default=None,
                        help="Optional random seed passed to the GPU tutorial CLI")
    parser.add_argument("--gpu-solver-mode",
                        choices=["torchgmm", "legacy-stepwise"],
                        default="torchgmm",
                        help="GPU tutorial solver backend (default: torchgmm)")
    parser.add_argument("--gpu-init-strategy-mode",
                        choices=["kmeans++", "xtec", "sklearn-kmeans", "cuml-kmeans"],
                        default="sklearn-kmeans",
                        help="GPU tutorial initialization strategy "
                             "(default: sklearn-kmeans)")
    parser.add_argument("--gpu-post-stepwise-epochs", type=int, default=0,
                        help="GPU tutorial post-fit stepwise refinement epochs "
                             "(default: 0)")
    parser.add_argument("--gpu-post-stepwise-tol", type=float, default=None,
                        help="GPU tutorial post-fit refinement tolerance "
                             "(default: None)")
    parser.add_argument("--gpu-batch-num", type=int, default=1,
                        help="GPU legacy-stepwise batch count (default: 1)")
    parser.add_argument("--gpu-max-batch-epoch", type=int, default=50,
                        help="GPU legacy-stepwise max batch epochs "
                             "(default: 50)")
    parser.add_argument("--gpu-max-full-epoch", type=int, default=500,
                        help="GPU legacy-stepwise max full-data epochs "
                             "(default: 500)")
    parser.add_argument("--cpu-max-batch-epoch", type=int, default=50,
                        help="CPU reference GMM max_batch_epoch "
                             "(default: 50, matching the reference repo)")
    parser.add_argument("--cpu-max-full-epoch", type=int, default=500,
                        help="CPU reference GMM max_full_epoch "
                             "(default: 500, matching the reference repo)")
    return parser.parse_args()


def parse_slice_string(slices_text):
    parts = []
    for s in slices_text.split(","):
        s = s.strip()
        if s == ":":
            parts.append(slice(None))
        elif ":" in s:
            start_text, stop_text = s.split(":", 1)
            start = float(start_text) if start_text else None
            stop = float(stop_text) if stop_text else None
            parts.append(slice(start, stop))
        else:
            parts.append(float(s))
    return tuple(parts)


def load_data(filepath, entry_path, slices_text):
    root = nxload(filepath, "r")
    data = root[entry_path]
    if slices_text:
        data = data[parse_slice_string(slices_text)]
    return data


def parse_int_list(text):
    if text is None:
        return None
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    return values


def parse_zoom_window(text):
    if text is None:
        return None
    row_text, col_text = [piece.strip() for piece in text.split(",", 1)]
    row_start, row_stop = [int(v) for v in row_text.split(":", 1)]
    col_start, col_stop = [int(v) for v in col_text.split(":", 1)]
    return slice(row_start, row_stop), slice(col_start, col_stop)


def axis_display_name(nxaxis):
    name = getattr(nxaxis, "nxname", str(nxaxis))
    if name.lower().startswith("q") and len(name) == 2:
        return name[1:].upper()
    return name


def resolve_slice_index(data, axis_ind, slice_value):
    axis = data.nxaxes[axis_ind + 1]
    values = np.asarray(axis.nxvalue)
    try:
        slice_ind = int(axis.index(slice_value))
        actual_value = float(values[slice_ind])
    except Exception:
        slice_ind = int(np.argmin(np.abs(values - slice_value)))
        actual_value = float(values[slice_ind])
    return slice_ind, actual_value


def slice_plot_axes(data, axis_ind):
    if axis_ind == 0:
        return data.nxaxes[3], data.nxaxes[2]
    if axis_ind == 1:
        return data.nxaxes[3], data.nxaxes[1]
    if axis_ind == 2:
        return data.nxaxes[2], data.nxaxes[1]
    raise ValueError(f"Unsupported axis index: {axis_ind}")


def apply_slice_axes_metadata(ax, data, axis_ind, slice_value, label_size=None):
    x_axis, y_axis = slice_plot_axes(data, axis_ind)
    ax.get_images()[0].set_extent(
        (x_axis.nxvalue[0], x_axis.nxvalue[-1], y_axis.nxvalue[0], y_axis.nxvalue[-1])
    )
    kwargs = {}
    if label_size is not None:
        kwargs["size"] = label_size
    ax.set_xlabel(axis_display_name(x_axis), **kwargs)
    ax.set_ylabel(axis_display_name(y_axis), **kwargs)
    ax.set_title(f"{axis_display_name(data.nxaxes[axis_ind + 1])}={slice_value}", **kwargs)


def save_current_figure(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def save_results_h5(path, cluster_assigns, data_ind, cluster_means, cluster_covs,
                    pixel_assignments=None):
    if pixel_assignments is None:
        pixel_assignments = cluster_assigns
    with h5py.File(path, "w") as handle:
        handle["cluster_assignments"] = np.asarray(cluster_assigns)
        handle["pixel_assignments"] = np.asarray(pixel_assignments)
        handle["data_indices"] = np.asarray(data_ind)
        handle["cluster_means"] = np.asarray(cluster_means)
        handle["cluster_covariances"] = np.asarray(cluster_covs)


def save_tutorial_pickle(path, temp_values, data_values, data_ind,
                         cluster_assigns, cluster_means, cluster_covs):
    payload = {
        "Temp": np.asarray(temp_values),
        "Data": np.asarray(data_values).transpose(),
        "Data_ind": np.asarray(data_ind),
        "cluster_assignments": np.asarray(cluster_assigns),
        "Num_clusters": int(len(cluster_means)),
        "cluster_mean": np.asarray(cluster_means),
        "cluster_var": np.asarray(cluster_covs),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def save_avg_intensity_plot(path, temp_values, data_values, cluster_assigns):
    colors = [
        "red", "blue", "green", "purple", "yellow",
        "orange", "pink", "black", "grey", "cyan",
    ]
    plt.figure()
    nc = int(np.max(cluster_assigns)) + 1 if len(cluster_assigns) else 0
    for i in range(nc):
        mask = cluster_assigns == i
        yc = np.mean(data_values[:, mask], axis=1).flatten()
        plt.plot(temp_values, yc, color=colors[i], lw=2)
    plt.yscale("linear")
    plt.xlabel("T")
    plt.ylabel("I (cluster avged)")
    save_current_figure(path)


def apply_rescale(threshold, data_values, rescale_text):
    if rescale_text == "mean":
        return threshold.Rescale_mean(data_values)
    if rescale_text == "z-score":
        return threshold.Rescale_zscore(data_values)
    if rescale_text == "log-mean":
        rescaled = np.log1p(np.asarray(data_values))
        return rescaled - np.mean(rescaled, axis=0)
    return np.asarray(data_values)


def ensure_cpu_repo(cpu_repo, clone_if_missing):
    cpu_repo = Path(cpu_repo)
    if cpu_repo.exists():
        return cpu_repo
    if not clone_if_missing:
        raise FileNotFoundError(
            f"CPU repo not found at {cpu_repo}. "
            f"Clone {CPU_REPO_URL} there or omit --no-clone-cpu-repo."
        )
    cpu_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", CPU_REPO_URL, str(cpu_repo)], check=True)
    return cpu_repo


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text(path, text):
    path.write_text(text)


def run_cpu_reference(args, repo_root, output_dir):
    cpu_repo = ensure_cpu_repo(args.cpu_repo, args.clone_cpu_repo)
    prep_module = load_module_from_path(
        "kimgroup_xtec_preprocessing", cpu_repo / "src" / "xtec" / "Preprocessing.py"
    )
    gmm_module = load_module_from_path(
        "kimgroup_xtec_gmm", cpu_repo / "src" / "xtec" / "GMM.py"
    )
    Mask_Zeros = prep_module.Mask_Zeros
    Threshold_Background = prep_module.Threshold_Background
    GMM = gmm_module.GMM

    if args.cpu_random_state is not None:
        np.random.seed(args.cpu_random_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.input, args.entry, args.slices)
    I = data.nxsignal.nxvalue
    temp_values = np.asarray(data.nxaxes[0].nxvalue)

    thresh_type = "KL" if args.threshold else "No threshold"
    masked = Mask_Zeros(I)
    threshold = Threshold_Background(masked, None, thresh_type)
    Data_thresh = np.asarray(threshold.data_thresholded)
    Data_ind = np.asarray(threshold.ind_thresholded)
    Rescaled_data = np.asarray(apply_rescale(threshold, Data_thresh, args.rescale))

    slice_ind, actual_slice_value = resolve_slice_index(
        data, args.slice_axis, args.slice_value
    )

    threshold.plot_cutoff((10, 5))
    save_current_figure(output_dir / "cutoff.png")

    threshold.plot_thresholding_2D_slice((8, 8), slice_ind, args.slice_axis)
    ax = plt.gca()
    apply_slice_axes_metadata(ax, data, args.slice_axis, actual_slice_value)
    save_current_figure(output_dir / "threshold_slice.png")

    clusterGMM_1 = GMM(
        Rescaled_data.transpose(),
        args.first_pass_clusters,
        max_batch_epoch=args.cpu_max_batch_epoch,
        max_full_epoch=args.cpu_max_full_epoch,
    )
    clusterGMM_1.RunEM()
    cluster_assigns_1 = np.asarray(clusterGMM_1.cluster_assignments)
    cluster_means_1 = np.asarray(clusterGMM_1.means)
    cluster_covs_1 = np.asarray(clusterGMM_1.covs)

    save_results_h5(
        output_dir / "pass1_results.h5",
        cluster_assigns_1,
        Data_ind,
        cluster_means_1,
        cluster_covs_1,
    )

    clusterGMM_1.Plot_Cluster_Results_traj(temp_values)
    plt.xlabel("T(K)", size=18)
    plt.ylabel(r"$\widetilde{I}_q(T)$", size=18)
    plt.title("Cluster mean and variance \n (rescaled) intensity trajectory ")
    save_current_figure(output_dir / "trajectories_pass1.png")

    clusterGMM_1.Plot_Cluster_kspace_2D_slice(
        threshold, (8, 8), Data_ind, slice_ind, args.slice_axis
    )
    ax = plt.gca()
    apply_slice_axes_metadata(ax, data, args.slice_axis, actual_slice_value, label_size=22)
    save_current_figure(output_dir / "qmap_pass1.png")

    zoom_window = parse_zoom_window(args.zoom_window)
    if zoom_window is not None:
        row_slice, col_slice = zoom_window
        x_axis, y_axis = slice_plot_axes(data, args.slice_axis)
        x_stop = min(col_slice.stop, len(x_axis.nxvalue) - 1)
        y_stop = min(row_slice.stop, len(y_axis.nxvalue) - 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(
            clusterGMM_1.plot_image[row_slice, col_slice],
            origin="lower",
            cmap=clusterGMM_1.plot_cmap,
            norm=clusterGMM_1.plot_norm,
            extent=[
                x_axis.nxvalue[col_slice.start],
                x_axis.nxvalue[x_stop],
                y_axis.nxvalue[row_slice.start],
                y_axis.nxvalue[y_stop],
            ],
        )
        plt.xlabel(axis_display_name(x_axis), size=22)
        plt.ylabel(axis_display_name(y_axis), size=22)
        plt.title(f"{axis_display_name(data.nxaxes[args.slice_axis + 1])}={actual_slice_value}", size=22)
        save_current_figure(output_dir / "qmap_zoom_pass1.png")

    bad_clusters = parse_int_list(args.bad_clusters)
    if args.good_cluster is not None:
        selected_mode = "good_cluster"
        selected_value = args.good_cluster
        good_mask = cluster_assigns_1 == args.good_cluster
    elif bad_clusters:
        selected_mode = "bad_clusters"
        selected_value = bad_clusters
        good_mask = ~np.isin(cluster_assigns_1, bad_clusters)
    else:
        selected_mode = "auto_discovery_cluster"
        selected_value = int(clusterGMM_1.discovery_cluster_ind)
        good_mask = cluster_assigns_1 == selected_value

    Good_data = Data_thresh[:, good_mask]
    Good_rescaled_data = Rescaled_data[:, good_mask]
    Good_ind = Data_ind[good_mask]

    clusterGMM_2 = GMM(
        Good_rescaled_data.transpose(),
        args.second_pass_clusters,
        max_batch_epoch=args.cpu_max_batch_epoch,
        max_full_epoch=args.cpu_max_full_epoch,
    )
    clusterGMM_2.RunEM()
    cluster_assigns_2 = np.asarray(clusterGMM_2.cluster_assignments)
    cluster_means_2 = np.asarray(clusterGMM_2.means)
    cluster_covs_2 = np.asarray(clusterGMM_2.covs)

    save_results_h5(
        output_dir / "pass2_results.h5",
        cluster_assigns_2,
        Good_ind,
        cluster_means_2,
        cluster_covs_2,
    )

    clusterGMM_2.Plot_Cluster_Results_traj(temp_values)
    plt.xlabel("T(K)", size=18)
    plt.ylabel(r"$\widetilde{I}_q(T)$", size=18)
    plt.title(" ")
    save_current_figure(output_dir / "trajectories_pass2.png")

    clusterGMM_2.Plot_Cluster_kspace_2D_slice(
        threshold, (8, 8), Good_ind, slice_ind, args.slice_axis
    )
    ax = plt.gca()
    apply_slice_axes_metadata(ax, data, args.slice_axis, actual_slice_value, label_size=22)
    save_current_figure(output_dir / "qmap_pass2.png")

    if zoom_window is not None:
        row_slice, col_slice = zoom_window
        x_axis, y_axis = slice_plot_axes(data, args.slice_axis)
        x_stop = min(col_slice.stop, len(x_axis.nxvalue) - 1)
        y_stop = min(row_slice.stop, len(y_axis.nxvalue) - 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(
            clusterGMM_2.plot_image[row_slice, col_slice],
            origin="lower",
            cmap=clusterGMM_2.plot_cmap,
            norm=clusterGMM_2.plot_norm,
            extent=[
                x_axis.nxvalue[col_slice.start],
                x_axis.nxvalue[x_stop],
                y_axis.nxvalue[row_slice.start],
                y_axis.nxvalue[y_stop],
            ],
        )
        plt.xlabel(axis_display_name(x_axis), size=22)
        plt.ylabel(axis_display_name(y_axis), size=22)
        plt.title(f"{axis_display_name(data.nxaxes[args.slice_axis + 1])}={actual_slice_value}", size=22)
        save_current_figure(output_dir / "qmap_zoom_pass2.png")

    save_avg_intensity_plot(
        output_dir / "avg_intensities_pass2.png",
        temp_values,
        Good_data,
        cluster_assigns_2,
    )

    save_tutorial_pickle(
        output_dir / "csrs_0x0_clustering.p",
        temp_values,
        Good_data,
        Good_ind,
        cluster_assigns_2,
        cluster_means_2,
        cluster_covs_2,
    )

    metadata = {
        "implementation": "cpu_reference",
        "repo_path": str(cpu_repo),
        "input": str(args.input),
        "entry": args.entry,
        "slices": args.slices,
        "threshold": args.threshold,
        "rescale": args.rescale,
        "first_pass_clusters": args.first_pass_clusters,
        "second_pass_clusters": args.second_pass_clusters,
        "selection_mode": selected_mode,
        "selection_value": selected_value,
        "slice_axis": args.slice_axis,
        "slice_value_requested": args.slice_value,
        "slice_value_used": actual_slice_value,
        "zoom_window": args.zoom_window,
        "cpu_random_state": args.cpu_random_state,
        "cpu_max_batch_epoch": args.cpu_max_batch_epoch,
        "cpu_max_full_epoch": args.cpu_max_full_epoch,
        "pass1_cluster_sizes": np.bincount(cluster_assigns_1).tolist(),
        "pass2_cluster_sizes": np.bincount(cluster_assigns_2).tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def run_gpu_cli(args, repo_root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = repo_root / "src"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mpl")
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", "/tmp/xdg-cache")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_dir) if not existing_pythonpath else (
        f"{src_dir}{os.pathsep}{existing_pythonpath}"
    )

    cmd = [
        sys.executable,
        "-m",
        "xtec_gpu.xtec_cli",
        "tutorial-d",
        str(args.input),
        "-o",
        str(output_dir),
        "--device",
        args.gpu_device,
        "--rescale",
        args.rescale,
        "--entry",
        args.entry,
        "--first-pass-clusters",
        str(args.first_pass_clusters),
        "--second-pass-clusters",
        str(args.second_pass_clusters),
        "--slice-axis",
        str(args.slice_axis),
        "--slice-value",
        str(args.slice_value),
        "--zoom-window",
        args.zoom_window,
        "--solver-mode",
        args.gpu_solver_mode,
        "--init-strategy-mode",
        args.gpu_init_strategy_mode,
        "--post-stepwise-epochs",
        str(args.gpu_post_stepwise_epochs),
        "--batch-num",
        str(args.gpu_batch_num),
        "--max-batch-epoch",
        str(args.gpu_max_batch_epoch),
        "--max-full-epoch",
        str(args.gpu_max_full_epoch),
    ]

    if args.slices:
        cmd.extend(["--slices", args.slices])
    if not args.threshold:
        cmd.append("--no-threshold")
    if args.good_cluster is not None:
        cmd.extend(["--good-cluster", str(args.good_cluster)])
    elif args.bad_clusters:
        cmd.extend(["--bad-clusters", args.bad_clusters])
    if args.gpu_random_state is not None:
        cmd.extend(["--random-state", str(args.gpu_random_state)])
    if args.gpu_post_stepwise_tol is not None:
        cmd.extend(["--post-stepwise-tol", str(args.gpu_post_stepwise_tol)])

    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, env=env)

    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}
    metadata.update({
        "implementation": "gpu_cli",
        "gpu_device": args.gpu_device,
        "command": cmd,
        "gpu_random_state": args.gpu_random_state,
        "gpu_solver_mode": args.gpu_solver_mode,
        "gpu_init_strategy_mode": args.gpu_init_strategy_mode,
        "gpu_post_stepwise_epochs": args.gpu_post_stepwise_epochs,
        "gpu_post_stepwise_tol": args.gpu_post_stepwise_tol,
        "gpu_batch_num": args.gpu_batch_num,
        "gpu_max_batch_epoch": args.gpu_max_batch_epoch,
        "gpu_max_full_epoch": args.gpu_max_full_epoch,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def write_top_level_readme(path, args, cpu_dir, gpu_dir):
    text = (
        "CPU vs GPU tutorial outputs\n"
        "===========================\n\n"
        f"Input file: {args.input}\n"
        f"Entry: {args.entry}\n"
        f"Slices: {args.slices}\n"
        f"GPU device: {args.gpu_device}\n\n"
        f"cpu_reference/\n"
        f"  Outputs generated with the original KimGroup/XTEC NumPy reference code.\n"
        f"  See {cpu_dir / 'metadata.json'} for configuration details.\n\n"
        f"gpu_cli/\n"
        f"  Outputs generated with the XTEC-GPU CLI tutorial workflow.\n"
        f"  See {gpu_dir / 'metadata.json'} and {gpu_dir / 'command.txt'}.\n"
    )
    path.write_text(text)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output).resolve()
    cpu_dir = output_root / "cpu_reference"
    gpu_dir = output_root / "gpu_cli"

    output_root.mkdir(parents=True, exist_ok=True)
    cpu_metadata = run_cpu_reference(args, repo_root, cpu_dir)
    gpu_metadata = run_gpu_cli(args, repo_root, gpu_dir)

    comparison_metadata = {
        "input": str(args.input),
        "entry": args.entry,
        "slices": args.slices,
        "threshold": args.threshold,
        "rescale": args.rescale,
        "first_pass_clusters": args.first_pass_clusters,
        "second_pass_clusters": args.second_pass_clusters,
        "slice_axis": args.slice_axis,
        "slice_value": args.slice_value,
        "zoom_window": args.zoom_window,
        "gpu_device": args.gpu_device,
        "cpu_reference": cpu_metadata,
        "gpu_cli": gpu_metadata,
    }
    (output_root / "comparison_metadata.json").write_text(
        json.dumps(comparison_metadata, indent=2)
    )
    write_top_level_readme(output_root / "README.txt", args, cpu_dir, gpu_dir)

    print(f"Comparison outputs written to {output_root}")
    print(f"  CPU reference: {cpu_dir}")
    print(f"  GPU CLI:       {gpu_dir}")


if __name__ == "__main__":
    main()
