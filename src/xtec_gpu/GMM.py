"""
Gaussian Mixture Model (GMM) clustering (torchgmm + KMeans seeding)
===================================================================

This module provides a PyTorch-based Gaussian Mixture Model workflow for
clustering temperature trajectories shaped ``(num_data, num_T)``. It uses
**torchgmm** for EM optimization and initializes component means with
scikit-learn **KMeans** for stable starts. After fitting, it exposes
responsibilities, hard labels, component means/covariances, and convenient
plotting utilities.

Contents
--------
- :class:`GMM` — main wrapper around ``torchgmm.bayes.GaussianMixture`` with
  KMeans seeding and NumPy-friendly outputs.
- :class:`Cluster_Gaussian` — lightweight container exposing per-cluster
  mean and (diagonal or full) covariance for plotting/inspection.

Quick start
-----------
>>> import torch
>>> from GMM import GMM
>>>
>>> # data: (num_data, num_T) — one trajectory per row
>>> data = torch.randn(500, 64)
>>> gmm = GMM(data, cluster_num=3, cov_type="diag", random_state=0)
>>> gmm.RunEM()
>>> labels = gmm.cluster_assignments           # (num_data,)
>>> probs  = gmm.cluster_probs                 # (K, num_data)
>>> means  = gmm.means                         # (K, num_T)
>>> covs   = gmm.covs                          # (K, num_T) or (K, num_T, num_T)

Notes
-----
- **Shapes:** Input ``data`` is ``(num_data, num_T)``. Responsibilities are
  returned as ``(K, num_data)`` and hard labels as ``(num_data,)``.
- **Covariance types:** ``cov_type='diag'`` yields per-component diagonal
  variances of shape ``(K, num_T)``; ``'full'`` yields full matrices
  ``(K, num_T, num_T)``.
- **Initialization:** Means are seeded via scikit-learn KMeans and passed to
  ``torchgmm``; this is typically more stable than the built-in init for
  trajectory data.
- **Devices:** If ``data`` is a CUDA tensor, fitting runs on the same device.
  Public arrays are converted to NumPy on CPU for downstream interoperability.
- **Mixing weights:** The mixture weights :math:`\pi_k` are computed as the
  average responsibility per component and stored in ``mixing_weights``; these
  support plotting helpers that optionally re-shift means.

Dependencies
------------
- PyTorch
- NumPy
- scikit-learn (for KMeans seeding)
- torchgmm
- Matplotlib (optional; for plotting helpers)

Attribution
-----------
Written by **Yanjun Liu** and **Aaditya Panigrahi**. 

API overview
------------
- :class:`GMM(data, cluster_num, cov_type={'diag','full'}, tol=1e-4, reg_covar=1e-6, ...)`
  - :meth:`RunEM(label_smoothing_flag=False, Markov_matrix=None, smoothing_iterations=1)`
  - :meth:`Smooth_Labels(it_num=1)`
  - :meth:`Plot_Cluster_Results_traj(x_train, traj_flag=False, data_means=None)`
  - :meth:`Plot_Cluster_kspace_2D_slice(threshold, ...)`
  - :meth:`Plot_Cluster_Results_kspace_3D(threshold)`
  - :meth:`Get_pixel_labels(Peak_avg)`
- :class:`Cluster_Gaussian(mean, cov)`
- :class:`GMM_kernels`
  - :staticmethod:`Build_Markov_Matrix(data_inds, L_scale=1, kernel_type='local', ...)`
"""

from sklearn.cluster import KMeans
import numpy as np
from torchgmm.bayes import GaussianMixture
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


class GMM(object):
    """
    Independent PyTorch implementation using **torchgmm** for EM and
    scikit-learn **KMeans** for mean initialization.

    The model clusters samples shaped as `(num_data, num_T)` into
    `cluster_num` Gaussian components. Means are seeded by running KMeans
    on the input and passed to `torchgmm.bayes.GaussianMixture`.

    Attributes
    ----------
    cluster : list[Cluster_Gaussian]
        Convenience wrappers exposing per-cluster mean and covariance.
    cluster_assignments : ndarray, shape (num_data,)
        Hard assignments (argmax of responsibilities) for each sample.
    cluster_probs : ndarray, shape (num_clusters, num_data)
        Responsibilities (posterior probabilities) for each sample.
    num_per_cluster : list[int], length = num_clusters
        Number of samples assigned to each cluster (via hard labels).
    means : ndarray, shape (num_clusters, num_T)
        Component means from the fitted mixture.
    covs : ndarray
        Component covariances from the fitted mixture. With `'diag'`, this
        will be `(num_clusters, num_T)` (diagonal entries). With `'full'`,
        this will be `(num_clusters, num_T, num_T)`.
    mixing_weights : ndarray, shape (num_clusters,)
        Mixture weights π_k estimated as the mean responsibility across samples.

    Examples
    --------
    >>> num_clusters = 2
    >>> gmm = GMM(data, num_clusters)
    >>> gmm.RunEM()
    >>> print(gmm.num_per_cluster)
    >>> labels = gmm.cluster_assignments
    >>> gmm.Plot_Cluster_Results_traj(Temp, traj_flag=False)
    """

    def __init__(
        self,
        data,
        cluster_num,
        cov_type="diag",
        n_init=1,
        tol=1e-4,
        reg_covar=1e-6,
        max_iter=100,
        init_params="kmeans",
        device=None,
        alpha=0.7,
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        color_list=None,
        trainer_params=None,
    ):
        """
        Initialize GMM parameters and underlying torchgmm model.

        Parameters
        ----------
        data : array-like or torch.Tensor, shape (num_data, num_T)
            Input samples (rows = samples, columns = temperature/time index).
        cluster_num : int
            Number of mixture components.
        cov_type : {'diag', 'full'}, default='diag'
            Covariance structure for the mixture components.
        n_init : int, default=1
            (Kept for interface compatibility; KMeans seeding inside uses its own `n_init`.)
        tol : float, default=1e-4
            Convergence tolerance for EM.
        reg_covar : float, default=1e-6
            Non-negative regularization added to covariance to keep it PSD.
        max_iter : int, default=100
            Maximum EM iterations (passed via `trainer_params` if supported).
        init_params : str, default='kmeans'
            Init strategy name saved in the model. Actual means are set from
            scikit-learn KMeans for stability.
        device : str or torch.device, optional
            Target device for the torch tensors.
        random_state : int or None, optional
            Random seed used for KMeans seeding (and possibly torchgmm).
        warm_start, verbose, verbose_interval, weights_init, means_init, precisions_init :
            Kept for API compatibility; not used directly in this wrapper.
        color_list : list[str], optional
            Colors used in plotting routines.
        trainer_params : dict, optional
            Extra parameters forwarded to torchgmm's trainer (if applicable).
        """
        self.color_list = (
            color_list
            if color_list
            else ["red", "blue", "green", "purple", "yellow", "orange", "pink"]
        )

        if device is None:
            if torch.is_tensor(data):
                self.device = data.device
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.data = data
        self.cluster_num = cluster_num
        self.cov_type = cov_type
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.alpha = alpha
        self.epoch = 0

        if torch.is_tensor(self.data):
            self.data = self.data.to(self.device)
        else:
            self.data = torch.as_tensor(self.data, dtype=torch.float32, device=self.device)

        # Use scikit-learn KMeans to obtain robust initial means.
        # Tip: use an integer (e.g., 10) for broader sklearn version compatibility.
        km = KMeans(n_clusters=cluster_num, n_init='auto', random_state=random_state)
        km.fit(self.data.detach().cpu().numpy())
        init_means = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)

        # Build torchgmm model
        self.GaussianMixture = GaussianMixture(
            num_components=cluster_num,
            covariance_type=cov_type,
            convergence_tolerance=tol,
            covariance_regularization=reg_covar,
            init_strategy=init_params,
            init_means=init_means,
            trainer_params={"max_epochs": max_iter, **(trainer_params or {})}
        )

    def RunEM(self, label_smoothing_flag=False, Markov_matrix=None,
              smoothing_iterations=1, max_smooth_epoch=500, tol=None):
        """
        Fit the torchgmm GaussianMixture to the data, optionally with
        label smoothing.

        When ``label_smoothing_flag`` is True, the standard torchgmm EM fit is
        used as a warm start, then additional **E → Smooth → M** iterations are
        run on the GPU until convergence.  This matches the behaviour of the
        original CPU implementation where label smoothing is applied **inside**
        the EM loop — the smoothed responsibilities feed back into the M-step
        to update means and covariances each iteration.

        Parameters
        ----------
        label_smoothing_flag : bool, default=False
            If True, run smoothed EM iterations after the initial fit.
        Markov_matrix : torch sparse tensor, optional
            Row-normalised adjacency matrix (N, N) for label smoothing.
            Required when ``label_smoothing_flag`` is True.  Build it with
            :meth:`GMM_kernels.Build_Markov_Matrix`.
        smoothing_iterations : int, default=1
            Number of times the Markov matrix is applied to
            ``cluster_probs`` between E and M steps.
        max_smooth_epoch : int, default=500
            Maximum number of smoothed EM iterations.
        tol : float or None
            Convergence tolerance for the smoothed EM loop.
            Defaults to ``self.tol``.
        """
        if tol is None:
            tol = self.tol

        # Ensure tensor on the right device
        X = self.data
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        # ---- Standard EM via torchgmm (warm start) --------------------------
        self.GaussianMixture.fit(X)

        # Extract initial parameters as torch tensors on device
        self.means = self.GaussianMixture.model_.means.detach().clone()  # (K, T)
        self.covs = self.GaussianMixture.model_.covariances.detach().clone()  # (K, T) or (K,T,T)

        # Compute initial responsibilities on device
        probs = self.GaussianMixture.predict_proba(X).T  # (K, N) torch on device
        self.cluster_probs = probs.detach().clone()

        # Mixing weights from responsibilities
        self.mixing_weights = self.cluster_probs.mean(dim=1)  # (K,)

        # ---- Optional: smoothed E→Smooth→M loop on GPU ----------------------
        if label_smoothing_flag:
            if Markov_matrix is None:
                raise ValueError(
                    "Markov_matrix is required when label_smoothing_flag=True. "
                    "Use GMM_kernels.Build_Markov_Matrix() to construct one."
                )
            self.Markov_matrix = Markov_matrix

            loglikelihood_diff = 1e6
            loglikelihood_new = torch.tensor(-1e6, device=self.device)

            for epoch in range(max_smooth_epoch):
                loglikelihood_old = loglikelihood_new

                # E-step
                self._E_Step_torch(X)

                # Smooth labels: diffuse cluster_probs through Markov matrix
                self._Smooth_Labels_torch(smoothing_iterations)

                # M-step with stepwise blending (matches old code)
                self._M_Step_torch(X)
                self.epoch += 1

                # Log-likelihood for convergence check
                loglikelihood_new = self._LogLikelihood_torch(X)
                loglikelihood_diff = torch.abs(
                    loglikelihood_new - loglikelihood_old
                ).item()

                if loglikelihood_diff <= tol:
                    break

        # ---- Finalise: move everything to numpy for downstream use -----------
        self.cluster_assignments = (
            torch.argmax(self.cluster_probs, dim=0).detach().cpu().numpy()
        )
        self.cluster_probs = self.cluster_probs.detach().cpu().numpy()
        self.means = self.means.detach().cpu().numpy()
        self.covs = self.covs.detach().cpu().numpy()
        self.mixing_weights = self.mixing_weights.detach().cpu().numpy()

        # Number of samples per hard cluster
        self.num_per_cluster = [
            int(np.sum(self.cluster_assignments == k))
            for k in range(self.cluster_num)
        ]

        # Convenience wrappers
        self.cluster = []
        for k in range(self.cluster_num):
            self.cluster.append(
                Cluster_Gaussian(self.means[k], self.covs[k])
            )

    # ------------------------------------------------------------------
    # Torch-native E / M / LogLikelihood for the smoothed EM loop
    # ------------------------------------------------------------------

    def _E_Step_torch(self, X):
        """E-step on GPU: compute responsibilities from current means/covs.

        Updates ``self.cluster_probs`` in-place (K, N) on device.
        """
        log_gauss = self._LogGaussian_torch(X)  # (K, N)
        # Convert log-space to probabilities (numerically stable)
        log_gauss_max = log_gauss.max(dim=0, keepdim=True).values
        p = torch.exp(log_gauss - log_gauss_max)  # (K, N)
        weighted = self.mixing_weights.unsqueeze(1) * p  # (K, N)
        denom = weighted.sum(dim=0, keepdim=True).clamp(min=1e-300)
        self.cluster_probs = weighted / denom

    def _M_Step_torch(self, X):
        """M-step on GPU: update means, covs, mixing_weights from probs.

        Uses **stepwise EM** blending (cf. Liang & Klein 2009) to match
        the original CPU implementation::

            eta_k = (epoch + 2) ^ (-alpha)
            means = (1 - eta_k) * means_old + eta_k * means_new
            covs  = (1 - eta_k) * covs_old  + eta_k * covs_new

        ``X`` has shape ``(N, T)``; ``self.cluster_probs`` has shape ``(K, N)``.
        """
        K = self.cluster_num
        N, T = X.shape

        # Mixing weights: <C_k> averaged over data
        self.mixing_weights = self.cluster_probs.mean(dim=1)  # (K,)
        mw = self.mixing_weights.clamp(min=1e-10)

        # New means: (K, T)
        means_new = (
            (self.cluster_probs @ X) / (N * mw.unsqueeze(1))
        )

        # New covariances
        if self.cov_type == "diag":
            # cov_k = sum_i C_ki * (x_i - mu_k)^2 / (N * w_k)
            covs_new = torch.zeros(K, T, device=X.device, dtype=X.dtype)
            for k in range(K):
                diff = X - means_new[k].unsqueeze(0)  # (N, T)
                covs_new[k] = (
                    (self.cluster_probs[k].unsqueeze(1) * diff * diff).sum(dim=0)
                    / (N * mw[k])
                )
        else:
            # Full covariance: (K, T, T)
            covs_new = torch.zeros(K, T, T, device=X.device, dtype=X.dtype)
            for k in range(K):
                diff = X - means_new[k].unsqueeze(0)  # (N, T)
                weighted_diff = self.cluster_probs[k].unsqueeze(1) * diff
                covs_new[k] = (weighted_diff.T @ diff) / (N * mw[k])

        # Stepwise EM blending (matches old code)
        eta_k = (self.epoch + 2) ** (-self.alpha)
        self.means = (1 - eta_k) * self.means + eta_k * means_new
        self.covs = (1 - eta_k) * self.covs + eta_k * covs_new

    def _LogGaussian_torch(self, X):
        """Compute log N(x | mu_k, cov_k) for each cluster k.

        Returns shape ``(K, N)`` on device.
        """
        K = self.cluster_num
        N, T = X.shape

        if self.cov_type == "diag":
            # covs: (K, T)  — diagonal variances
            covs_safe = self.covs.clamp(min=1e-12)
            log_det = covs_safe.log().sum(dim=1)  # (K,)
            results = []
            for k in range(K):
                diff = X - self.means[k].unsqueeze(0)  # (N, T)
                mahal = (diff * diff / covs_safe[k].unsqueeze(0)).sum(dim=1)  # (N,)
                log_p = -0.5 * (T * np.log(2 * np.pi) + log_det[k] + mahal)
                results.append(log_p)
            return torch.stack(results, dim=0)  # (K, N)
        else:
            # covs: (K, T, T) — full covariance
            results = []
            for k in range(K):
                cov_k = self.covs[k]
                # Add regularisation for stability
                cov_k = cov_k + 1e-6 * torch.eye(T, device=X.device)
                L = torch.linalg.cholesky(cov_k)
                log_det = 2.0 * L.diagonal().log().sum()
                diff = X - self.means[k].unsqueeze(0)  # (N, T)
                solved = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (T, N)
                mahal = (solved * solved).sum(dim=0)  # (N,)
                log_p = -0.5 * (T * np.log(2 * np.pi) + log_det + mahal)
                results.append(log_p)
            return torch.stack(results, dim=0)  # (K, N)

    def _LogLikelihood_torch(self, X):
        """Compute total log-likelihood: sum_n log sum_k w_k N(x_n|mu_k,cov_k).

        Returns a scalar torch tensor on device.
        """
        log_gauss = self._LogGaussian_torch(X)  # (K, N)
        log_w = torch.log(self.mixing_weights.clamp(min=1e-300))  # (K,)
        # log(w_k) + log N(x|mu_k, cov_k) → log(w_k * N(...))
        log_weighted = log_w.unsqueeze(1) + log_gauss  # (K, N)
        # log-sum-exp over K for each data point, then sum over data
        return torch.logsumexp(log_weighted, dim=0).sum()

    def _Smooth_Labels_torch(self, it_num=1):
        """Diffuse cluster probabilities through the Markov matrix on GPU.

        Modifies ``self.cluster_probs`` (K, N) in-place.
        """
        M = self.Markov_matrix
        probs = self.cluster_probs  # (K, N), already on device
        for _ in range(it_num):
            # M @ probs^T → (N, K), then transpose → (K, N)
            probs = torch.sparse.mm(M, probs.T).T
        self.cluster_probs = probs

    def Plot_Cluster_Results_traj(self, x_train, traj_flag: bool = False, data_means=None):
        """
        Plot trajectories and cluster means ± 1*std.

        Parameters
        ----------
        x_train : array-like, shape (num_T,)
            X-axis values (e.g., temperature).
        traj_flag : bool, default=False
            If True, plot all trajectories color coded by cluster labels.
            If False, plot only cluster mean ± std envelopes.
        data_means : array-like or None, optional
            Optional per-sample mean offsets to re-center the cluster means.
            If provided, the displayed mean for cluster k is:
                mean_k + mean_shift_k,  where
                mean_shift = (cluster_probs @ data_means) / (N * mixing_weights[k])
        """
        if self.cluster_num > len(self.color_list):
            print("Error: cluster num larger than color list")
            return

        color_list = self.color_list

        if traj_flag is True:
            plt.figure()
            # plot each trajectory colored by its assigned label
            data_TN = self.data.transpose(0, 1) if self.data.ndim == 2 else self.data.T
            for i in range(data_TN.shape[1]):
                plt.plot(
                    x_train,
                    data_TN[:, i],
                    color=color_list[self.cluster_assignments[i]],
                    alpha=0.7,
                )
            # overlay cluster means
            for i in range(self.cluster_num):
                plt.plot(x_train, self.cluster[i].mean, "k--", lw=2)
            return plt

        # Means-only view with ±1 std envelope
        plt.figure()
        std_dev_num = 1

        if data_means is None:
            traj_means = [self.cluster[k].mean for k in range(self.cluster_num)]
        else:
            data_means = np.asarray(data_means).reshape(-1)
            # mean_shift_k = sum_i r_ki * data_means_i / (N * pi_k)
            N = self.cluster_probs.shape[1]
            denom = (N * np.maximum(self.mixing_weights, 1e-12))
            mean_shift = (self.cluster_probs @ data_means) / denom
            traj_means = [self.cluster[k].mean + mean_shift[k] for k in range(self.cluster_num)]

        for i in range(self.cluster_num):
            mu_i = traj_means[i]
            # cov may be diag (num_T,) or full (num_T, num_T)
            if self.covs.ndim == 2:  # diag case stacked as (K, T)
                std_i = np.sqrt(self.covs[i])
            else:
                std_i = np.sqrt(np.clip(np.diag(self.covs[i]), 0, None))
            plt.plot(x_train, mu_i, color=color_list[i], lw=2)
            plt.gca().fill_between(
                x_train,
                mu_i - std_dev_num * std_i,
                mu_i + std_dev_num * std_i,
                color=color_list[i],
                alpha=0.35,
            )
        return

    def Plot_Cluster_kspace_2D_slice(
        self,
        threshold,
        figsize_=None,
        data_ind=None,
        slice_ind=None,
        axis_=None,
        cluster_assignments=None,
        cluster_list=None,
    ):
        """
        Plot a 2D image slice color-coded by cluster label over the threshold mask.

        Parameters
        ----------
        threshold : Threshold_Background
            Thresholding result providing `.thresholded` and `.data_shape_orig`.
        figsize_ : tuple or None
            Matplotlib figure size.
        data_ind : ndarray or tensor, optional
            Indices of clustered data points; defaults to `threshold.ind_thresholded`.
        slice_ind : int, optional
            For 3D data, the slice index along `axis_`.
        axis_ : {0, 1, 2}, optional
            Axis along which to slice for 3D data.
        cluster_assignments : ndarray, optional
            Labels for the thresholded points; defaults to `self.cluster_assignments`.
        cluster_list : iterable, optional
            Subset of clusters to display (others remain grey). Defaults to all.
        """
        def _to_numpy(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            return x

        if cluster_list is None:
            cluster_list = range(self.cluster_num)

        if data_ind is None:
            data_ind = threshold.ind_thresholded
        data_ind = _to_numpy(data_ind)

        plotting_matrix = _to_numpy(threshold.thresholded).copy()
        data_shape = threshold.data_shape_orig[1:]
        if torch.is_tensor(data_shape):
            data_shape = tuple(_to_numpy(data_shape))

        if cluster_assignments is None:
            cluster_assignments = self.cluster_assignments
        cluster_assignments = _to_numpy(cluster_assignments)

        for k in cluster_list:
            cluster_mask = cluster_assignments == k
            cluster_ind = data_ind[cluster_mask]
            if len(data_shape) == 2:
                plotting_matrix[cluster_ind[:, 0], cluster_ind[:, 1]] = k + 2
            elif len(data_shape) == 3:
                plotting_matrix[
                    cluster_ind[:, 0], cluster_ind[:, 1], cluster_ind[:, 2]
                ] = (k + 2)

        color_list = ["white", "lightgrey"] + self.color_list
        cluster_cmap = colors.ListedColormap(color_list)

        bounds = [i - 0.5 for i in range(len(color_list) + 1)]
        norm = colors.BoundaryNorm(bounds, cluster_cmap.N)

        if len(data_shape) == 2:
            self.plot_image = plotting_matrix
        elif len(data_shape) == 3:
            if slice_ind is None or axis_ is None:
                raise ValueError("Provide slice_ind and axis_ for 3D data.")
            self.plot_image = plotting_matrix.take(slice_ind, axis=axis_)

        self.plot_norm = norm
        self.plot_cmap = cluster_cmap

        if figsize_ is not None:
            plt.figure(figsize=figsize_)
            plt.imshow(self.plot_image, origin="lower", cmap=cluster_cmap, norm=norm)

    def Plot_Cluster_Results_kspace_3D(self, threshold):
        """
        Plot a 3D scatter of thresholded data color-coded by cluster assignments.

        Parameters
        ----------
        threshold : Threshold_Background
            Thresholding result with fields `ind_low_std_dev`, `ind_high_std_dev`,
            and `data_shape_orig`.
        """
        def _to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return x

        # Spatial shape
        data_shape = threshold.data_shape_orig[1:]
        if torch.is_tensor(data_shape):
            data_shape = tuple(_to_numpy(data_shape).astype(int).tolist())
        elif isinstance(data_shape, (list, np.ndarray)):
            data_shape = tuple(np.array(data_shape).astype(int).tolist())

        Ql_cell = np.arange(data_shape[0]) / (data_shape[0] - 1)
        Qk_cell = np.arange(data_shape[1]) / (data_shape[1] - 1)
        Qh_cell = np.arange(data_shape[2]) / (data_shape[2] - 1)

        X, Y, Z = np.meshgrid(Ql_cell, Qk_cell, Qh_cell, indexing="ij")
        X = np.reshape(X, np.prod(data_shape))
        Y = np.reshape(Y, np.prod(data_shape))
        Z = np.reshape(Z, np.prod(data_shape))

        ind_low_std_dev = _to_numpy(threshold.ind_low_std_dev).astype(int, copy=False)
        ind_high_std_dev = _to_numpy(threshold.ind_high_std_dev).astype(int, copy=False)

        low_std_dev_cluster = np.zeros(data_shape).astype(bool)
        if len(data_shape) == 2:
            low_std_dev_cluster[
                ind_low_std_dev[:, 0], ind_low_std_dev[:, 1]
            ] = True
        elif len(data_shape) == 3:
            low_std_dev_cluster[
                ind_low_std_dev[:, 0], ind_low_std_dev[:, 1], ind_low_std_dev[:, 2]
            ] = True

        masks = [low_std_dev_cluster.reshape(np.prod(data_shape))]

        for k in range(self.cluster_num):
            cluster_mask = self.cluster_assignments == k
            cluster_ind = ind_high_std_dev[cluster_mask]
            cluster_kspace_mask = np.zeros(data_shape).astype(bool)
            cluster_kspace_mask[
                cluster_ind[:, 0], cluster_ind[:, 1], cluster_ind[:, 2]
            ] = True
            masks.append(cluster_kspace_mask.reshape(np.prod(data_shape)))

        color_list = ["white", "lightgrey"] + self.color_list
        RGBs = [colors.to_rgb(color) for color in color_list]

        fig = plt.figure(dpi=600)
        ax = fig.add_subplot(111, projection="3d")

        for mask_num, mask in enumerate(masks):
            if mask_num > 0:
                alpha = 0.3
                sval = 20
                rgba_mask = np.append(RGBs[mask_num + 1], alpha)[np.newaxis, :]
                ax.scatter(
                    (5 * X[mask] - 5),
                    (5 * Y[mask] - 5),
                    (5 * Z[mask] - 5),
                    c=rgba_mask,
                    marker=".",
                    s=sval,
                    edgecolors="none",
                )

        ax.view_init(azim=50)
        ax.set_xlabel("$Q_l$", fontsize=15)
        ax.set_ylabel("$Q_k$", fontsize=15)
        ax.set_zlabel("$Q_h$", fontsize=15)
        plt.tight_layout()

    def Get_pixel_labels(self, Peak_avg):
        """
        Propagate each peak-average cluster label to all pixels in that peak.

        Parameters
        ----------
        Peak_avg : Peak_averaging
            Contains `peak_avg_data` (T × num_peaks) and `peak_avg_ind_list`
            (list of (n_i, D) index arrays per peak).

        Sets
        ----
        Data_ind : ndarray, shape (num_data, D)
            Stacked spatial indices for all pixels belonging to any peak.
        Pixel_assignments : ndarray, shape (num_data,)
            Cluster label for each pixel (propagated from its parent peak).
        """
        Peak_avg_data = Peak_avg.peak_avg_data
        Peak_avg_ind_list = Peak_avg.peak_avg_ind_list

        Data_ind = []
        Pixel_assignments = []

        # For each peak, assign its label to all of its member pixels.
        # The label for a peak is taken from the hard assignment of the
        # corresponding peak-average trajectory (column index).
        num_peaks = Peak_avg_data.shape[1]
        for i in range(num_peaks):
            inds_i = Peak_avg_ind_list[i]  # (n_i, D) tensor/ndarray
            Data_ind.append(np.asarray(inds_i))
            # cluster_assignments is over samples (rows in data); here we map
            # the i-th peak-average trajectory's label to all its pixels.
            Pixel_assignments.append(
                np.full((np.asarray(inds_i).shape[0],), self.cluster_assignments[i], dtype=int)
            )

        self.Data_ind = np.vstack(Data_ind) if Data_ind else np.empty((0, 0), dtype=int)
        self.Pixel_assignments = np.concatenate(Pixel_assignments, axis=0) if Pixel_assignments else np.empty((0,),
                                                                                                              dtype=int)


class Cluster_Gaussian(object):
    """
    Lightweight container for a single Gaussian component.

    Attributes
    ----------
    mean : array-like, shape (num_T,)
        Component mean trajectory.
    cov : array-like
        Diagonal variances (num_T,) if original covariance is diagonal,
        otherwise the full covariance matrix (num_T, num_T).
    """
    def __init__(self, mean, cov):
        self.mean = mean
        # If cov came as a full matrix, keep it; if diag vector, keep as vector
        # (plotting code handles both).
        self.cov = cov


class GMM_kernels(object):
    """Builds the Adjacency (Markov) matrix for label smoothing (GPU-accelerated).

    Unlike the original CPU implementation which uses a Python for-loop over
    every data point, this version vectorizes the pairwise kernel computation
    on the GPU using batched tensor operations.  For datasets that exceed GPU
    memory a configurable ``chunk_size`` is used to process rows in batches.

    The returned matrix is a **torch sparse CSR tensor** on the same device
    as the input, and can be used directly with ``torch.sparse.mm`` for
    label smoothing.
    """

    @staticmethod
    def Build_Markov_Matrix(
        data_inds,
        L_scale=1,
        kernel_type="local",
        unit_cell_shape=None,
        uniform_similarity=True,
        zero_cutoff=1e-2,
        device=None,
        chunk_size=4096,
    ):
        """Build a row-normalised adjacency (Markov) matrix on GPU.

        Parameters
        ----------
        data_inds : array-like or torch.Tensor, shape (N, D)
            HKL indices of the preprocessed data (D = 2 or 3).
        L_scale : float, optional
            Length scale (pixel units) for local correlations, by default 1.
        kernel_type : {'local', 'periodic'}, optional
            Kernel for pairwise similarity, by default ``'local'``.
        unit_cell_shape : array-like or torch.Tensor, optional
            Number of pixels defining the unit cell (required when
            ``kernel_type='periodic'``).
        uniform_similarity : bool, optional
            If True, all nonzero entries (above ``zero_cutoff``) are set to 1
            before row-normalisation, by default True.
        zero_cutoff : float, optional
            Entries below this value are treated as zero, by default 1e-2.
        device : torch.device or str, optional
            Target device; inferred from ``data_inds`` if it is a tensor,
            else defaults to CUDA if available.
        chunk_size : int, optional
            Number of rows processed per batch to limit peak GPU memory,
            by default 4096.

        Returns
        -------
        Markov_matrix : torch.Tensor (sparse_csr, float32)
            Row-normalised (L1) adjacency matrix of shape ``(N, N)``
            on ``device``.
        """
        import time

        print("\n\tBuilding Adjacency Matrix (GPU) ...")
        start_time = time.time()

        # ---- Device handling ---------------------------------------------------
        if device is None:
            if torch.is_tensor(data_inds):
                device = data_inds.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # ---- Convert inputs to tensors on device ------------------------------
        if not torch.is_tensor(data_inds):
            data_inds = torch.as_tensor(data_inds, dtype=torch.float32, device=device)
        else:
            data_inds = data_inds.to(device=device, dtype=torch.float32)

        if unit_cell_shape is not None:
            if not torch.is_tensor(unit_cell_shape):
                unit_cell_shape = torch.as_tensor(
                    unit_cell_shape, dtype=torch.float32, device=device
                )
            else:
                unit_cell_shape = unit_cell_shape.to(device=device, dtype=torch.float32)

        N = data_inds.shape[0]
        inv_L2 = 1.0 / (L_scale ** 2)

        # ---- Build COO indices in chunks --------------------------------------
        # Processing in chunks of rows avoids allocating a full (N, N) dense
        # matrix which would be prohibitive for large datasets.
        row_list = []
        col_list = []
        val_list = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # chunk_inds: (chunk, D),  data_inds: (N, D)
            chunk_inds = data_inds[start:end]  # (C, D)

            # diff: (C, N, D)
            diff = chunk_inds.unsqueeze(1) - data_inds.unsqueeze(0)

            if kernel_type == "local":
                # exp[-sum_d (q_i - q_j)^2 / L^2]
                kernel_vals = torch.exp(
                    -torch.sum(diff ** 2, dim=2) * inv_L2
                )  # (C, N)
            elif kernel_type == "periodic":
                # exp[-sum_d sin(pi/L_cell * (q_i - q_j))^2 / L^2]
                kernel_vals = torch.exp(
                    -torch.sum(
                        torch.sin(torch.pi / unit_cell_shape * diff) ** 2,
                        dim=2,
                    ) * inv_L2
                )  # (C, N)
            else:
                raise ValueError(f"Invalid kernel_type: {kernel_type!r}")

            # Sparsify: keep entries above cutoff
            mask = kernel_vals > zero_cutoff  # (C, N)
            rows_chunk, cols_chunk = torch.where(mask)  # local-row, global-col
            rows_chunk = rows_chunk + start  # shift to global row index

            if uniform_similarity:
                vals_chunk = torch.ones(rows_chunk.shape[0], device=device)
            else:
                vals_chunk = kernel_vals[mask]

            row_list.append(rows_chunk)
            col_list.append(cols_chunk)
            val_list.append(vals_chunk)

            # Free intermediate memory
            del diff, kernel_vals, mask

        # ---- Assemble sparse CSR tensor ---------------------------------------
        rows = torch.cat(row_list)
        cols = torch.cat(col_list)
        vals = torch.cat(val_list)

        # COO → CSR via torch.sparse_coo_tensor → to_sparse_csr
        coo = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=(N, N), device=device
        ).coalesce()

        Markov_matrix = coo.to_sparse_csr().to(torch.float32)

        # ---- L1 row-normalise -------------------------------------------------
        # row_sum via sparse mm with ones vector
        ones = torch.ones(N, 1, device=device)
        row_sums = torch.sparse.mm(Markov_matrix, ones).squeeze(1)  # (N,)
        row_sums = row_sums.clamp(min=1e-12)

        # Scale values: divide each nonzero by its row sum
        crow = Markov_matrix.crow_indices()
        vals_csr = Markov_matrix.values().clone()
        for i in range(N):
            vals_csr[crow[i]:crow[i + 1]] /= row_sums[i]

        Markov_matrix = torch.sparse_csr_tensor(
            crow, Markov_matrix.col_indices(), vals_csr, size=(N, N), device=device
        )

        elapsed = time.time() - start_time
        print(f"\tFinished Building Adjacency Matrix in {elapsed:.2f} s\n")

        return Markov_matrix