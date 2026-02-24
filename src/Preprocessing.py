import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import colors


"""
Preprocessing utilities for scattering/intensity data (PyTorch adaptation)
=========================================================================

This module provides preprocessing steps commonly applied before clustering
and peak analysis on temperature-stacked scattering or intensity datasets.
Given data shaped ``(num_T, *spatial)``, it supports the following:

- Masking sites with nonpositive or invalid values across temperature
  (:class:`Mask_Zeros`)
- Estimating and applying a background-intensity cutoff in log space via a
  KL-guided or simple heuristic (:class:`Threshold_Background`)
- Aggregating intensity trajectories per connected spatial peak
  (:class:`Peak_averaging`)

The current implementation is a **PyTorch adaptation** of an earlier NumPy
version. Masking, histogramming, and per-peak reductions are implemented in
Torch and run on the chosen device (CPU or GPU). Connected-component labeling
uses :mod:`scipy.ndimage` and therefore runs on CPU; tensors are moved only
when necessary to minimize memory transfers.

Contents
--------
- :class:`Mask_Zeros` — remove sites that are nonpositive (``<=0``) for any T
  or whose mean over T is ``<=0``; optionally remove NaNs.
- :class:`Threshold_Background` — compute ``log(mean_T(I))`` per site, select
  a cutoff either by minimizing a KL objective over a left-truncated
  histogram (``threshold_type='KL'``) or by a simple
  ``mean + 2*std`` heuristic (``'simple'``), and apply it.
- :class:`Peak_averaging` — find connected spatial peaks from the binary mask
  and compute per-peak mean and maximum intensity trajectories.

Quick start
-----------
>>> import torch
>>> from Preprocessing import Mask_Zeros, Threshold_Background, Peak_averaging
>>>
>>> # Example: 2D scattering data (temperature × spatial)
>>> I = torch.rand(20, 64, 64)
>>>
>>> # 1) Remove sites that are zero/nonpositive/NaN
>>> mask = Mask_Zeros(I, mask_type="zero_mean", check4NaN=True)
>>>
>>> # 2) Threshold low background (KL or simple)
>>> tb = Threshold_Background(mask, threshold_type="KL", max_iter=100)
>>> data_pass = tb.data_thresholded  # shape (num_T, N_pass)
>>>
>>> # Optional: rescale trajectories (choose one)
>>> data_rescaled = tb.Rescale_mean(data_pass)      # x / mean(x_T) - 1
>>> # or:
>>> # data_rescaled = tb.Rescale_zscore(data_pass)  # (x - mean) / std
>>>
>>> # 3) Aggregate intensity per connected peak
>>> peaks = Peak_averaging(I, tb)
>>> peaks.peak_avg_data.shape, peaks.peak_max_data.shape
(torch.Size([20, P]), torch.Size([20, P]))

Notes
-----
- **Devices:** If inputs are Torch tensors on CUDA, computations inside the
  classes run on the same device when possible. Connected-component labeling
  (``scipy.ndimage.label``) executes on CPU; only the binary mask is moved for
  labeling. Per-peak reductions are vectorized Torch operations.
- **Histogram units:** Thresholding operates on ``log(mean_T(I)))`` (natural
  log). Histogram outputs are **densities** (counts divided by ``N * bin_width``).
- **Sanity check (KL):** The selected cutoff must lie within
  ``[mean - 1*std, mean + 2*std]`` of the global ``log(mean_T(I))``; otherwise
  ``success`` is set to ``False``.

Dependencies
------------
- PyTorch
- NumPy
- SciPy (for connected-component labeling)
- Matplotlib (for optional diagnostic plots)

Attribution
-----------
Original NumPy code by **Jordan Venderley**, revised by **Krishnanand Mallayya**.
PyTorch adaptation by **Yanjun Liu** and **Aaditya Panigrahi**, with additional
refinements provided here.

API overview
------------
- :class:`Mask_Zeros(data, mask_type={'zero_mean','any_zeros'}, check4NaN=False, device=None)`
- :class:`Threshold_Background(mask, bin_size=None, threshold_type={'KL','simple','none'}, max_iter=100, set_cutoff=-1e6, device=None)`
- :class:`Peak_averaging(intensity, threshold, device=None)`
"""


class Mask_Zeros(object):
    """
    Mask out nonpositive (<= 0) or invalid (NaN) data across the temperature
    dimension and return flattened subsets for further processing.

    This class removes or flags data positions that are nonpositive or NaN in
    any temperature slice, providing both the retained and masked subsets as
    tensors. It supports both NumPy arrays and PyTorch tensors and maintains
    device placement for GPU acceleration.

    The masking operates in one of two modes:
    - ``'zero_mean'``: positions whose **mean over temperature** ≤ 0.
    - ``'any_zeros'``: positions that are ≤ 0 in **any** temperature slice.

    Parameters
    ----------
    data : array_like or torch.Tensor, shape (num_T, n1, n2, ..., nk)
        Intensity array as a function of temperature. The leading axis indexes
        temperature, and the remaining axes correspond to spatial or
        reciprocal-space coordinates.
    mask_type : {'zero_mean', 'any_zeros'}, default='zero_mean'
        Type of masking criterion. See Notes for details.
    check4NaN : bool, default=False
        If True, marks positions that contain NaN values at any temperature.
    device : torch.device or str, optional
        Device on which to place the tensors. If None, uses the device of
        ``data`` (if tensor), otherwise CPU.

    Attributes
    ----------
    data_shape : tuple of int
        Original shape of ``data``.
    device : torch.device
        Device on which internal tensors are stored.
    zero_mask : torch.BoolTensor, shape (n1, n2, ..., nk)
        Boolean mask of positions identified for removal.
    ind_cols_with_zeros : torch.LongTensor, shape (n_zero, k)
        Indices of masked (nonpositive or invalid) positions.
    ind_cols_nonzeros : torch.LongTensor, shape (n_nonzero, k)
        Indices of retained positions.
    data_with_zeros : torch.Tensor, shape (num_T, n_zero)
        Flattened data for masked positions at each temperature.
    data_nonzero : torch.Tensor, shape (num_T, n_nonzero)
        Flattened data for retained positions at each temperature.

    Notes
    -----
    Negative values are treated as zero for the purpose of masking since the
    condition uses ``<= 0``. All non-temperature axes are flattened when
    forming ``data_with_zeros`` and ``data_nonzero`` to mimic NumPy-style
    slicing: ``data[:, mask]``.

    Examples
    --------
    >>> import numpy as np
    >>> from Preprocessing import Mask_Zeros
    >>>
    >>> I = np.random.rand(5, 4, 4, 4)
    >>> I[:, 1, 2, 3] = 0.0
    >>> mask = Mask_Zeros(I, mask_type='zero_mean')
    >>> mask.data_nonzero.shape
    torch.Size([5, 63])
    """

    def __init__(self, data, mask_type="zero_mean", check4NaN=False, device=None):
        """
        Initialize the masking operation.

        Parameters
        ----------
        data : array_like or torch.Tensor, shape (num_T, n1, n2, ..., nk)
            Intensity array as a function of temperature.
        mask_type : {'zero_mean', 'any_zeros'}, default='zero_mean'
            Masking rule. ``'zero_mean'`` removes positions whose mean ≤ 0,
            while ``'any_zeros'`` removes those that are ≤ 0 at any temperature.
        check4NaN : bool, default=False
            If True, positions containing NaN are also masked.
        device : torch.device or str, optional
            Device on which to store tensors. Defaults to device of ``data`` if
            it's a tensor, otherwise CPU.

        Notes
        -----
        The constructor automatically performs the masking operation and
        populates attributes such as ``data_nonzero`` and ``zero_mask``.
        """
        # Resolve and set device
        if device is None:
            self.device = data.device if torch.is_tensor(data) else torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Ensure `data` is a tensor on the target device
        data = torch.as_tensor(data).to(self.device)

        self.data_shape = tuple(data.shape)

        # --- Build zero mask over the temperature-averaged (or any-zero) signal ---
        if mask_type == "zero_mean":
            zero_mask = (data.float().mean(dim=0) <= 0)
        elif mask_type == "any_zeros":
            zero_mask = (data <= 0).any(dim=0)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}. Expected 'zero_mean' or 'any_zeros'.")

        if check4NaN:
            NaN_mask = torch.isnan(data.float()).any(dim=0)
            zero_mask = torch.logical_or(zero_mask, NaN_mask)

        self.zero_mask = zero_mask

        # Indices of zero and nonzero positions
        self.ind_cols_with_zeros = torch.nonzero(zero_mask, as_tuple=False)
        self.ind_cols_nonzeros = torch.nonzero(~zero_mask, as_tuple=False)

        # Flatten spatial axes and apply mask
        data_flat = data.reshape(data.shape[0], -1)
        zm_flat = zero_mask.reshape(-1)

        self.data_with_zeros = data_flat[:, zm_flat]
        self.data_nonzero = data_flat[:, ~zm_flat]


class Threshold_Background(object):
    """
    Estimate and apply an intensity cutoff to remove low-background signal
    from scattering data using either a Kullback–Leibler (KL)-guided
    truncation or a simple mean+2*std heuristic.

    The algorithm operates on the **nonzero** subset produced by a previous
    masking step (e.g., :class:`Mask_Zeros`). For each spatial position
    (flattened over non-temperature axes), it computes the log of the mean
    intensity across temperature, builds a histogram, and selects a cutoff
    in log-intensity space. Positions with ``log(mean_I)`` above the cutoff
    are retained.

    When ``threshold_type == "KL"``, the cutoff is determined by minimizing
    a directional derivative of the KL divergence between the **left-truncated**
    histogram and a Gaussian background model, iteratively updating an index
    in the histogram until convergence or a maximum number of iterations is
    reached. A sanity check ensures the cutoff lies in a plausible region
    relative to the global mean and standard deviation.

    Parameters
    ----------
    mask : object
        A mask object (typically from :class:`Mask_Zeros`) that provides:
        - ``data_nonzero`` : torch.Tensor of shape (num_T, num_data)
          Flattened nonzero data over spatial axes.
        - ``ind_cols_nonzeros`` : torch.LongTensor of shape (num_data, k)
          Indices of retained (nonzero) spatial positions.
        - ``data_shape`` : tuple
          The original multi-dimensional data shape (including temperature).
    bin_size : float, optional
        Histogram bin size (in log-intensity units). If None, computed using
        the Freedman–Diaconis rule on ``log(mean_I)`` (sorted).
    threshold_type : {'KL', 'simple', 'none'}, default='KL'
        - ``'KL'`` : Use KL-guided truncation based on a Gaussian background fit.
        - ``'simple'`` : Use ``naive_mean + 2*naive_std`` over ``log(mean_I)``.
        - Any other value is treated as "no threshold", and the cutoff is set
          to ``set_cutoff`` directly.
    max_iter : int, default=100
        Maximum iterations for the KL-guided truncation search.
    set_cutoff : float, default=-1e6
        Fallback cutoff value (log-intensity) used when ``threshold_type`` is
        not recognized (treated as "no threshold").
    device : torch.device or str, optional
        Target device for computations. If None, inferred from ``mask`` or
        falls back to CPU.

    Attributes
    ----------
    device : torch.device
        Device on which internal tensors are stored.
    bin_size : float or torch.Tensor
        Histogram bin width in log-intensity. Computed if not given.
    data_shape : tuple
        Shape of the flattened input data, ``(num_T, num_data)``.
    data_shape_orig : tuple
        Original shape **before** flattening, taken from ``mask.data_shape``.
    threshold_type : str
        Chosen thresholding method.
    success : bool
        Whether KL-based thresholding passed the sanity check and completed.
    y_bins : torch.Tensor, shape (n_bins,), optional
        Histogram probability densities for ``log(mean_I)`` (left edges).
        Populated when ``threshold_type == "KL"``.
    x_bins : torch.Tensor, shape (n_bins,), optional
        Histogram bin **left edges** for ``log(mean_I)``. Populated for KL mode.
    optimal_x_ind_cut : int, optional
        Optimal histogram index for truncation (KL mode).
    LogI_cutoff : torch.Scalar
        Selected cutoff in log-intensity units.
    LTS_ind_cut : int, optional
        Index in sorted ``log(mean_I)`` that corresponds to ``LogI_cutoff`` (KL).
    mean_opt : torch.Scalar, optional
        Gaussian mean fitted to the left-truncated region (KL).
    std_dev_opt : torch.Scalar, optional
        Gaussian std fitted to the left-truncated region (KL).
    naive_mean : torch.Scalar, optional
        Mean of ``log(mean_I)`` (simple mode).
    naive_std : torch.Scalar, optional
        Std of ``log(mean_I)`` with ``unbiased=False`` (simple mode).
    mask_threshold : torch.BoolTensor, shape (num_data,)
        Boolean mask over flattened spatial positions indicating
        ``log(mean_I) > LogI_cutoff``.
    data_thresholded : torch.Tensor, shape (num_T, n_pass)
        Subset of ``mask.data_nonzero`` that passed the cutoff.
    ind_thresholded : torch.LongTensor, shape (n_pass, k)
        Indices (in original spatial axes) of positions that passed.
    thresholded : torch.LongTensor, shape data_shape_orig[1:]
        Binary volume/image (0/1) in the original spatial shape indicating
        positions that passed the cutoff.

    Notes
    -----
    - The log intensity is computed as ``log(mean_T(I)))`` with natural log.
    - Histogram densities are returned as probability **densities**
      (frequency divided by ``N * dx``).
    - In KL mode, a sanity check requires the cutoff to be within
      ``[mean - 1*std, mean + 2*std]`` of the global ``log(mean_I)``.
    - All non-temperature axes are flattened for histogramming and masking.

    Examples
    --------
    >>> # `mask` is an instance of Mask_Zeros computed earlier
    >>> tb = Threshold_Background(mask, threshold_type="KL", max_iter=50)
    >>> tb.LogI_cutoff
    tensor(...)
    >>> tb.data_thresholded.shape  # subset that passed the cutoff
    torch.Size([num_T, n_pass])

    """

    def __init__(self, mask, bin_size=None, threshold_type="KL", max_iter=100, set_cutoff=-1e6, device=None):
        """
        Initialize thresholding state, compute a log-intensity histogram,
        and select a cutoff according to the chosen strategy.

        Parameters
        ----------
        mask : object
            An object providing at least ``data_nonzero``, ``ind_cols_nonzeros``,
            and ``data_shape`` as described in the class docstring.
        bin_size : float, optional
            Histogram bin width (in log-intensity). If None, computed using the
            Freedman–Diaconis rule.
        threshold_type : {'KL', 'simple', 'none'}, default='KL'
            Thresholding strategy. See class docstring.
        max_iter : int, default=100
            Maximum iterations in the KL-guided truncation loop.
        set_cutoff : float, default=-1e6
            Direct cutoff value used when ``threshold_type`` is not recognized.
        device : torch.device or str, optional
            Target device; if None, inferred from ``mask`` or defaults to CPU.

        Raises
        ------
        RuntimeError
            If KL mode is requested but internal sanity checks fail during
            plotting (only when plotting is attempted).
        """
        #  shape=(num_T, num_data), the input data to be thresholded.

        if device is None:
            if hasattr(mask, "device"):
                self.device = mask.device
            elif hasattr(mask, "data_nonzero") and torch.is_tensor(mask.data_nonzero):
                self.device = mask.data_nonzero.device
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        data = mask.data_nonzero
        if torch.is_tensor(data):
            data = data.to(self.device)
        else:
            data = torch.as_tensor(data, device=self.device)
        #  shape=(num_T, num_data), the input data to be thresholded.

        self.bin_size = bin_size  # bin size for histogram
        self.data_shape = tuple(data.shape)  # data_shape = (num_T, num_data)
        self.threshold_type = threshold_type  #
        self.data_shape_orig = tuple(mask.data_shape)  # original data shape from the mask

        # data.shape= (num_T, num_data)
        data = data.reshape(data.shape[0], -1)

        # = log[avg_T[I(q)]]
        logData_mean = torch.log(torch.mean(data, dim=0))
        logData_mean_sorted, _ = torch.sort(logData_mean)

        # Calculate optimal bin_size from Freedman-Diaconis rule
        if self.bin_size is None:
            self.bin_size = self.Freedman_Diaconis_for_bin_width(logData_mean_sorted, device=data.device)

        # becomes true only when thresholding is successful and complete
        self.success = False
        if self.threshold_type == "KL":
            # returns prob density y= freq/[(total #x)dx] , and x-bins (left edges)
            self.y_bins, self.x_bins = self.hist(logData_mean, self.bin_size, device=data.device)

            self.bin_size = torch.as_tensor(float(self.bin_size), device=data.device, dtype=logData_mean.dtype)

            try:
                self.optimal_x_ind_cut = self.Truncate(
                    self.x_bins, self.y_bins, logData_mean_sorted, max_iter,
                    device=data.device, dtype=logData_mean.dtype
                )
                # optimal intensity cutoff
                self.LogI_cutoff = self.x_bins[self.optimal_x_ind_cut]

                # Sanity Check:
                naive_mean = torch.mean(logData_mean_sorted)
                naive_std = torch.std(logData_mean_sorted, unbiased=False)  # match numpy ddof=0
                if (self.LogI_cutoff > naive_mean + 2 * naive_std) or (self.LogI_cutoff < naive_mean - naive_std):
                    self.success = False  # over/under threshold
                else:
                    # cut off lies in (mean-sigma,mean+2*sigma) of the full data log[avg_T[I(q)]]
                    self.success = True

            except Exception:
                self.success = False

            self.LogI_cutoff = self.x_bins[self.optimal_x_ind_cut]
            self.LTS_ind_cut = int(torch.searchsorted(logData_mean_sorted, self.LogI_cutoff).item())
            # avg_q of {log[avg_T[I(q)]]} of cutoff data
            self.mean_opt = torch.mean(logData_mean_sorted[: self.LTS_ind_cut])
            # std_q of {log[avg_T[I(q)]]} of cutoff data
            self.std_dev_opt = torch.var(logData_mean_sorted[: self.LTS_ind_cut], unbiased=False) ** 0.5

        elif self.threshold_type == "simple":
            self.naive_mean = torch.mean(logData_mean_sorted)
            self.naive_std = torch.std(logData_mean_sorted, unbiased=False)
            self.LogI_cutoff = self.naive_mean + 2 * self.naive_std
        else:
            # 'no thresh' sets the cutoff as given
            self.LogI_cutoff = torch.as_tensor(set_cutoff, device=data.device, dtype=logData_mean.dtype)

        # True/False whether data above I cutoff
        self.mask_threshold = (logData_mean > self.LogI_cutoff)

        # True/False whether non zero masked data above I cutoff
        self.data_thresholded = mask.data_nonzero[:, self.mask_threshold]
        self.ind_thresholded = mask.ind_cols_nonzeros[self.mask_threshold]

        # build thresholded volume/image (0/1 int), same shape as original spatial dims
        self.thresholded = torch.zeros(self.data_shape_orig[1:], device=data.device, dtype=torch.int64)
        if len(self.data_shape_orig[1:]) == 2:
            if self.ind_thresholded.numel() > 0:
                ii = self.ind_thresholded[:, 0].long()
                jj = self.ind_thresholded[:, 1].long()
                self.thresholded[ii, jj] = 1
        elif len(self.data_shape_orig[1:]) == 3:
            if self.ind_thresholded.numel() > 0:
                ii = self.ind_thresholded[:, 0].long()
                jj = self.ind_thresholded[:, 1].long()
                kk = self.ind_thresholded[:, 2].long()
                self.thresholded[ii, jj, kk] = 1
        self.thresholded = self.thresholded.to(torch.int64)


    def Rescale_mean(self, data, device=None, dtype=torch.float32):
        """
        Mean-center each spatial trajectory across temperature: ``x / mean(x_T) - 1``.

        Parameters
        ----------
        data : array_like or torch.Tensor, shape (num_T, num_data)
            Input trajectories with temperature along axis 0.
        device : torch.device or str, optional
            Target device; if None, inferred from ``self.device`` or ``data``.
        dtype : torch.dtype, default=torch.float32
            Output dtype and dtype used to store trajectory means.

        Returns
        -------
        rescale_data : torch.Tensor, shape (num_T, num_data)
            Mean-centered data with the specified dtype on the chosen device.

        Notes
        -----
        - If ``data`` is not floating, it is cast to ``float32`` prior to
          computation to avoid integer division.
        - The per-position mean is stored in ``self.traj_means``.
        """
        # Parse/set the device (prefer the function argument; then self.device; then data.device; finally CPU)

        if device is not None:
            dev = torch.device(device)
        elif hasattr(self, "device"):
            dev = self.device
        elif torch.is_tensor(data):
            dev = data.device
        else:
            dev = torch.device("cpu")

        if torch.is_tensor(data):
            x = data.to(dev)
            if not torch.is_floating_point(x):
                x = x.to(torch.float32)
        else:
            x = torch.as_tensor(data, dtype=torch.float32, device=dev)

        self.traj_means = torch.mean(x, dim=0).to(dtype)

        rescale_data = x / self.traj_means - 1.0
        rescale_data = rescale_data.to(dtype)
        return rescale_data

    def Rescale_zscore(self, data, device=None, dtype=torch.float32):
        """
        Z-score each spatial trajectory across temperature: ``(x - mean) / std``.

        Parameters
        ----------
        data : array_like or torch.Tensor, shape (num_T, num_data)
            Input trajectories with temperature along axis 0.
        device : torch.device or str, optional
            Target device; if None, inferred from ``self.device`` or ``data``.
        dtype : torch.dtype, default=torch.float32
            Output dtype and dtype used to store trajectory statistics.

        Returns
        -------
        rescale_data : torch.Tensor, shape (num_T, num_data)
            Z-scored data with the specified dtype on the chosen device.

        Notes
        -----
        - Uses population standard deviation (``unbiased=False``) to match the
          NumPy default (``ddof=0``).
        - The per-position mean/std are stored in ``self.traj_means`` and
          ``self.traj_std``.
        """
        # Parse/set the device (prefer the function argument; then self.device; then data.device; finally CPU)
        if device is not None:
            dev = torch.device(device)
        elif hasattr(self, "device"):
            dev = self.device
        elif torch.is_tensor(data):
            dev = data.device
        else:
            dev = torch.device("cpu")

        if torch.is_tensor(data):
            x = data.to(dev)
            if not torch.is_floating_point(x):
                x = x.to(torch.float32)
        else:
            x = torch.as_tensor(data, dtype=torch.float32, device=dev)

        self.traj_means = torch.mean(x, dim=0).to(dtype)
        self.traj_std = torch.std(x, dim=0, unbiased=False).to(dtype)


        rescale_data = (x - self.traj_means) / self.traj_std
        rescale_data = rescale_data.to(dtype)
        return rescale_data

    def Get_High_Variance(self, data, std_dev_cutoff=0.5):
        """Gets data whose std dev in temperature > std_dev_cutoff.

        Parameters
        ----------
        data : array-like or torch.Tensor, shape=(num_T, num_data)
            Thresholded and rescaled data.
        std_dev_cutoff : float, optional
            Standard deviation cut-off, by default 0.5.
        """
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, device=self.device)
        std_dev = torch.std(data.float(), dim=0, unbiased=False)
        mask_std_dev = std_dev >= std_dev_cutoff
        self.data_high_std_dev = data[:, mask_std_dev]
        self.ind_high_std_dev = self.ind_thresholded[mask_std_dev]
        self.data_low_std_dev = data[:, ~mask_std_dev]
        self.ind_low_std_dev = self.ind_thresholded[~mask_std_dev]
        self.mask_std_dev = mask_std_dev

    def Freedman_Diaconis_for_bin_width(self, sorted_data_: torch.Tensor, device=None) -> torch.Tensor:
        """
        Compute the Freedman–Diaconis bin width (Torch).

        Parameters
        ----------
        sorted_data_ : torch.Tensor, shape (N,)
            **Ascending-sorted** 1-D tensor of samples.
        device : torch.device or str, optional
            Device for the computation and returned value. If None, uses
            ``sorted_data_.device``.

        Returns
        -------
        bin_size : torch.Tensor, shape ()
            Scalar bin width as a tensor on ``device`` with dtype of
            ``sorted_data_``.

        Raises
        ------
        AssertionError
            If ``sorted_data_`` is not 1-D.

        Notes
        -----
        Uses ``IQR / n^(1/3)`` (half of ``2*IQR / n^(1/3)``) consistent with the
        original implementation in this project.
        """
        assert sorted_data_.dim() == 1, "sorted_data_ must be a 1-D sorted tensor."

        # ---- Normalize device and dtype ----
        target_device = torch.device(device) if device is not None else sorted_data_.device
        x = sorted_data_.to(target_device)
        dtype = x.dtype
        n = x.shape[0]

        # Quartile indices (nearest rounding)
        upper_ind = int(round(n * 0.75))
        lower_ind = int(round(n * 0.25))

        # IQR
        IQR = x[upper_ind] - x[lower_ind]

        # Half of the FD width: 0.5 * (2 * IQR / n^(1/3)) == IQR / n^(1/3)
        n_float = torch.tensor(float(n), device=target_device, dtype=dtype)
        bin_size = 0.5 * (2.0 * IQR / torch.pow(n_float, 1.0 / 3.0))

        return bin_size

    def hist(self, x: torch.Tensor, bin_size, device=None):
        """
        Histogram as a probability density over left-edge bins (Torch).

        Parameters
        ----------
        x : torch.Tensor, shape (N,)
            One-dimensional data (e.g., ``logData_mean``).
        bin_size : float or torch.Tensor
            Bin width. If a tensor, only its scalar value is used.
        device : torch.device or str, optional
            Target device; if None, use ``x.device``.

        Returns
        -------
        y_density : torch.Tensor, shape (n_bins,)
            Probability **density** per bin (counts divided by ``N * bin_size``).
        x_left_edges : torch.Tensor, shape (n_bins,)
            Left edges of the histogram bins.

        Notes
        -----
        - The number of bins is ``ceil((max - min) / bin_size)``.
        - ``torch.histc`` requires explicit ``min`` and ``max``.
        """
        target_device = torch.device(device) if device is not None else x.device
        x = torch.as_tensor(x, device=target_device)

        max_val = torch.max(x)
        min_val = torch.min(x)

        self.bin_size = float(bin_size.item() if torch.is_tensor(bin_size) else bin_size)

        bin_num = int(torch.ceil((max_val - min_val) / self.bin_size).item())

        counts = torch.histc(x, bins=bin_num,
                             min=float(min_val.item()),
                             max=float(max_val.item()))

        edges = torch.linspace(min_val, max_val, steps=bin_num + 1, device=target_device, dtype=x.dtype)
        x_hist = (counts, edges)

        return (
            x_hist[0] / (x.shape[0] * self.bin_size),
            x_hist[1][:-1],
        )

    def Gaussian(
            self,
            x,
            mean,
            std_dev,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        """
        Evaluate the univariate Gaussian probability density function.

        Parameters
        ----------
        x : array_like or torch.Tensor
            Support points at which to evaluate the PDF.
        mean : float or torch.Scalar
            Gaussian mean.
        std_dev : float or torch.Scalar
            Gaussian standard deviation. Values are clamped to
            ``torch.finfo(dtype).eps`` for stability.
        device : torch.device or str, optional
            Target device; defaults to that of ``x`` (or CPU if ``x`` is not a tensor).
        dtype : torch.dtype, optional
            Target dtype; defaults to that of ``x`` (or default float dtype).

        Returns
        -------
        pdf : torch.Tensor
            Gaussian PDF evaluated at ``x`` on the chosen device/dtype.
        """
        # Resolve target device/dtype
        if torch.is_tensor(x):
            x_device = x.device
            x_dtype = x.dtype
        else:
            x_device = torch.device("cpu")
            x_dtype = torch.get_default_dtype()

        device = torch.device(device) if device is not None else x_device
        dtype = dtype if dtype is not None else x_dtype

        # Convert inputs to tensors on the target device/dtype
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=device, dtype=dtype)
        else:
            if x.device != device or x.dtype != dtype:
                x = x.to(device=device, dtype=dtype)

        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std_dev = torch.as_tensor(std_dev, device=device, dtype=dtype)

        # Numerical stability: avoid zero/near-zero std
        eps = torch.finfo(dtype).eps
        std_dev = torch.clamp(std_dev, min=eps)

        # Compute PDF: exp(-0.5 * z^2) / (std * sqrt(2*pi))
        z = (x - mean) / std_dev
        return torch.exp(-0.5 * z * z) / (std_dev * (2 * torch.pi) ** 0.5)

    def KL(
            self,
            x_bins: torch.Tensor,
            y_bins: torch.Tensor,
            ind: int,
            mean,
            std_dev,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a **left-truncated** histogram and a
        Gaussian background model.

        The histogram ``(x_bins, y_bins)`` is interpreted as a left-edge
        discretization with uniform spacing. The left region up to ``ind`` is
        normalized to integrate to one, and the KL divergence
        ``KL(p || q) = sum p * log(p/q) * dx`` is computed in discretized form.

        Parameters
        ----------
        x_bins : torch.Tensor, shape (n_bins,)
            Left edges of the histogram bins (uniform spacing assumed).
        y_bins : torch.Tensor, shape (n_bins,)
            Histogram **densities** per bin.
        ind : int
            Rightmost bin index (exclusive boundary) for the left region.
        mean : float or torch.Scalar
            Mean of the Gaussian background model.
        std_dev : float or torch.Scalar
            Standard deviation of the Gaussian model.
        device : torch.device or str, optional
            Target device; defaults to that of ``x_bins``.
        dtype : torch.dtype, optional
            Target dtype; defaults to that of ``x_bins``.

        Returns
        -------
        kl : torch.Tensor, shape ()
            Scalar KL divergence as a tensor.

        Notes
        -----
        - If ``ind <= 0``, returns ``inf``.
        - Uses ``torch.xlogy`` when available for numerical stability.
        """
        # Resolve device/dtype (default to x_bins' if not specified)
        if torch.is_tensor(x_bins):
            default_device, default_dtype = x_bins.device, x_bins.dtype
        else:
            default_device, default_dtype = torch.device("cpu"), torch.get_default_dtype()

        device = torch.device(device) if device is not None else default_device
        dtype = dtype if dtype is not None else default_dtype

        # Ensure tensors on target device/dtype
        if not torch.is_tensor(x_bins):
            x_bins = torch.as_tensor(x_bins, device=device, dtype=dtype)
        else:
            x_bins = x_bins.to(device=device, dtype=dtype)

        if not torch.is_tensor(y_bins):
            y_bins = torch.as_tensor(y_bins, device=device, dtype=dtype)
        else:
            y_bins = y_bins.to(device=device, dtype=dtype)

        # --- Guard ind and basic sizes ---
        n = x_bins.numel()
        ind = int(ind)
        if ind <= 0:
            return torch.tensor(float("inf"), device=device, dtype=dtype)
        if ind > n:
            ind = n

        # Bin width (assumes uniform bins)
        dx = x_bins[1] - x_bins[0]

        # Model: Gaussian evaluated on the left part + small epsilon
        x_left = x_bins[:ind]
        Model = self.Gaussian(x_left, mean, std_dev, device=device, dtype=dtype) + 1e-12

        # Normalize truncated histogram to integrate to 1 over the left region
        y_left = y_bins[:ind]
        norm_factor = y_left.sum() * dx
        eps = torch.finfo(dtype).eps
        norm_factor = torch.clamp(norm_factor, min=eps)
        y_bins_normed = y_left / norm_factor

        # KL vector: p*log(p/q)
        if hasattr(torch, "xlogy"):
            KL_vec = -torch.xlogy(y_bins_normed, Model) + torch.xlogy(y_bins_normed, y_bins_normed)
        else:
            KL_vec = y_bins_normed * (torch.log(torch.clamp(y_bins_normed, min=eps)) -
                                      torch.log(torch.clamp(Model, min=eps)))

        KL = KL_vec.sum()
        return KL

    def Calc_KL(
            self,
            x_index: int,
            x_bins,
            y_bins,
            logData_mean_sorted,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Fit a Gaussian background on the **left-truncated** samples and compute
        the KL divergence to the truncated histogram at a given bin index.

        Parameters
        ----------
        x_index : int
            Histogram index defining the right boundary of the left region.
        x_bins : array_like or torch.Tensor, shape (n_bins,)
            Left edges of the histogram bins.
        y_bins : array_like or torch.Tensor, shape (n_bins,)
            Histogram densities per bin.
        logData_mean_sorted : array_like or torch.Tensor, shape (N,)
            Sorted samples of ``log(mean_I)``.
        device : torch.device or str, optional
            Target device; defaults to that of ``x_bins``.
        dtype : torch.dtype, optional
            Target dtype; defaults to that of ``x_bins``.

        Returns
        -------
        kl : torch.Tensor, shape ()
            Scalar KL divergence for the proposed left region.

        Notes
        -----
        - The mean and std are computed on the truncated sample using population
          variance (``ddof=0``) to mirror NumPy defaults.
        """
        # ---- resolve device/dtype ----
        if torch.is_tensor(x_bins):
            default_device, default_dtype = x_bins.device, x_bins.dtype
        else:
            default_device, default_dtype = torch.device("cpu"), torch.get_default_dtype()

        device = torch.device(device) if device is not None else default_device
        dtype = dtype if dtype is not None else default_dtype

        # ---- convert inputs to tensors ----
        if not torch.is_tensor(x_bins):
            x_bins = torch.as_tensor(x_bins, device=device, dtype=dtype)
        else:
            x_bins = x_bins.to(device=device, dtype=dtype)

        if not torch.is_tensor(y_bins):
            y_bins = torch.as_tensor(y_bins, device=device, dtype=dtype)
        else:
            y_bins = y_bins.to(device=device, dtype=dtype)

        if not torch.is_tensor(logData_mean_sorted):
            logData_mean_sorted = torch.as_tensor(logData_mean_sorted, device=device, dtype=dtype)
        else:
            logData_mean_sorted = logData_mean_sorted.to(device=device, dtype=dtype)

        # ---- index & boundaries ----
        x_index = int(x_index)
        x_index = max(0, min(x_index, x_bins.numel() - 1))
        x_val = x_bins[x_index]

        # left-truncated slice boundary via searchsorted
        LTS_ind = int(torch.searchsorted(logData_mean_sorted, x_val).item())

        # protect against empty slice
        region = logData_mean_sorted[: max(LTS_ind, 1)]

        # population mean/std (ddof=0)
        mean_guess = region.mean()
        std_dev_guess = torch.sqrt(torch.mean((region - mean_guess) ** 2))

        # KL on left region
        KL_div = self.KL(
            x_bins=x_bins,
            y_bins=y_bins,
            ind=x_index,
            mean=mean_guess,
            std_dev=std_dev_guess,
            device=device,
            dtype=dtype,
        )
        return KL_div

    @torch.no_grad()
    def Truncate(self, x_bins, y_bins, logData_mean_sorted, max_iter, device=None, dtype=None) -> int:
        """
        Iteratively update the left-region boundary (histogram index) by
        descending the directional derivative of log KL, until convergence.

        Parameters
        ----------
        x_bins : array_like or torch.Tensor, shape (n_bins,)
            Left edges of the histogram bins.
        y_bins : array_like or torch.Tensor, shape (n_bins,)
            Histogram densities per bin.
        logData_mean_sorted : array_like or torch.Tensor, shape (N,)
            Sorted samples of ``log(mean_I)``.
        max_iter : int
            Maximum number of iterations for the index update.
        device : torch.device or str, optional
            Target device; defaults to that of ``x_bins``.
        dtype : torch.dtype, optional
            Target dtype; defaults to that of ``x_bins``.

        Returns
        -------
        index_opt : int
            Optimal histogram index (left-region boundary).

        Notes
        -----
        - Uses a finite-difference step in index-space and updates via
          a Newton-like rule on ``log(KL)``; stops when the step magnitude
          is less than one bin width or when ``max_iter`` is reached.
        """
        # normalize inputs
        if torch.is_tensor(x_bins):
            default_device, default_dtype = x_bins.device, x_bins.dtype
        else:
            default_device, default_dtype = torch.device("cpu"), torch.get_default_dtype()
        device = torch.device(device) if device is not None else default_device
        dtype = dtype if dtype is not None else default_dtype

        if not torch.is_tensor(x_bins):
            x_bins = torch.as_tensor(x_bins, device=device, dtype=dtype)
        else:
            x_bins = x_bins.to(device=device, dtype=dtype)
        if not torch.is_tensor(y_bins):
            y_bins = torch.as_tensor(y_bins, device=device, dtype=dtype)
        else:
            y_bins = y_bins.to(device=device, dtype=dtype)
        if not torch.is_tensor(logData_mean_sorted):
            logData_mean_sorted = torch.as_tensor(logData_mean_sorted, device=device, dtype=dtype)
        else:
            logData_mean_sorted = logData_mean_sorted.to(device=device, dtype=dtype)

        init_mean = logData_mean_sorted.mean()
        init_std = torch.sqrt(torch.mean((logData_mean_sorted - init_mean) ** 2))
        init_ind = int(torch.searchsorted(x_bins, init_mean + init_std).item())

        peak_ind = int(torch.argmax(y_bins).item())
        x_ind_guess = int(0.5 * peak_ind + 0.5 * init_ind)
        if x_ind_guess >= y_bins.shape[0]:
            x_ind_guess = peak_ind - 2
        if x_ind_guess < 0:
            x_ind_guess = 0
        x_ind_guess_plus_dx = x_ind_guess + 1

        # use tensor bin_size stored in __init__
        bin_size_t = self.bin_size.to(device=device, dtype=dtype)

        delta_x_guess = torch.tensor(1e6, device=device, dtype=dtype)
        counter = 0
        eps = torch.finfo(dtype).eps

        while delta_x_guess.item() > bin_size_t.item() and counter < max_iter:
            KL_div = self.Calc_KL(x_ind_guess, x_bins, y_bins, logData_mean_sorted, device=device, dtype=dtype)
            KL_div_plus_dx = self.Calc_KL(x_ind_guess_plus_dx, x_bins, y_bins, logData_mean_sorted, device=device,
                                                dtype=dtype)

            DLogKL = (torch.log(torch.clamp(KL_div_plus_dx, min=eps)) -
                      torch.log(torch.clamp(KL_div, min=eps))) / bin_size_t

            delta_x_guess = (bin_size_t * DLogKL).abs()
            x_guess = x_bins[x_ind_guess] - bin_size_t * DLogKL

            new_index = int(torch.searchsorted(x_bins, x_guess.unsqueeze(0)).item())
            if new_index >= x_bins.shape[0]:
                new_index = x_bins.shape[0] - 2
            if new_index < 0:
                new_index = 0

            x_ind_guess = new_index
            x_ind_guess_plus_dx = x_ind_guess + 1
            counter += 1

        return int(x_ind_guess)

    def plot_cutoff(self, figsize_=(15, 15)):
        """
        Plot the histogram of log-mean intensities and the chosen background fit
        (Gaussian) with the cutoff marker, mirroring the original NumPy version.

        Parameters
        ----------
        figsize_ : tuple, default=(15, 15)
            Figure size passed to Matplotlib.

        Returns
        -------
        None
            Displays a Matplotlib figure.

        Raises
        ------
        RuntimeError
            If called before the corresponding attributes have been computed
            (e.g., missing ``x_bins``/``y_bins`` or, in KL mode, missing
            ``mean_opt``/``std_dev_opt``/``optimal_x_ind_cut``).
        """

        # Utility: convert torch.Tensor to numpy (without or with minimal copy)
        def _np(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        # Safety check for required attributes
        if not hasattr(self, "x_bins") or not hasattr(self, "y_bins"):
            raise RuntimeError("plot_cutoff: x_bins / y_bins are not computed; cannot plot.")

        x_bins_np = _np(self.x_bins)
        y_bins_np = _np(self.y_bins)

        # Create figure and plot Raw Data (same as original logic)
        plt.figure(figsize=figsize_)
        plt.plot(x_bins_np, y_bins_np, label="Raw Data")
        plt.ylabel(r"Density at $\log(\overline{I_q(T)})$")
        plt.xlabel(r"$\log(\overline{I_q(T)})$")

        # Case 1: KL
        if getattr(self, "threshold_type", "KL") == "KL":
            # Need mean_opt / std_dev_opt; compute with Torch Gaussian and convert to numpy
            if not hasattr(self, "mean_opt") or not hasattr(self, "std_dev_opt"):
                raise RuntimeError("plot_cutoff(KL): missing mean_opt / std_dev_opt.")

            gauss_t = self.Gaussian(
                self.x_bins, self.mean_opt, self.std_dev_opt,
                device=self.x_bins.device if torch.is_tensor(self.x_bins) else None,
                dtype=self.x_bins.dtype if torch.is_tensor(self.x_bins) else None
            )
            plt.plot(x_bins_np, _np(gauss_t), label="Background Fit")

            if not hasattr(self, "optimal_x_ind_cut"):
                raise RuntimeError("plot_cutoff(KL): missing optimal_x_ind_cut.")
            x_cut = float(_np(self.x_bins[int(self.optimal_x_ind_cut)]))
            plt.scatter(x_cut, 0, color="red", zorder=10, label="Background Truncation")

        # Case 2: simple
        elif getattr(self, "threshold_type") == "simple":
            if not hasattr(self, "naive_mean") or not hasattr(self, "naive_std"):
                raise RuntimeError("plot_cutoff(simple): missing naive_mean / naive_std.")

            gauss_t = self.Gaussian(
                self.x_bins, self.naive_mean, self.naive_std,
                device=self.x_bins.device if torch.is_tensor(self.x_bins) else None,
                dtype=self.x_bins.dtype if torch.is_tensor(self.x_bins) else None
            )
            plt.plot(
                x_bins_np,
                _np(gauss_t),
                label="Naive Background Fit",
            )
            x_trunc = float(_np(self.naive_mean)) + float(_np(self.naive_std)) * 2.0
            plt.scatter(
                x_trunc,
                0,
                color="green",
                zorder=10,
                label="Background Truncation",
            )

        # Case 3: others
        else:
            print("No thresholding performed")

        # Match original: update font size and add legend
        plt.rcParams.update({"font.size": 14})
        plt.legend()

    def plot_thresholding_2D_slice(
            self, figsize_=(10, 10), slice_ind=None, axis_=None
    ):
        """
        Plot a 2D map of threshold pass/fail (0/1) for 2D data or a 2D slice of 3D data.

        Parameters
        ----------
        figsize_ : tuple, default=(10, 10)
            Figure size for Matplotlib.
        slice_ind : int, optional
            Slice index when the original spatial data are 3D.
        axis_ : {0, 1, 2}, optional
            Axis along which to slice in the 3D case (0: L, 1: K, 2: H).

        Returns
        -------
        plt : matplotlib.pyplot
            The pyplot module with the figure rendered.

        Raises
        ------
        ValueError
            If ``slice_ind`` or ``axis_`` is missing for 3D data.
        RuntimeError
            If the spatial dimensionality is not 2 or 3.
        """

        # Torch -> NumPy helper
        def _np(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        # fixed color map (0 -> white, 1 -> lightgrey)
        color_list = ["white", "lightgrey"]
        cluster_cmap = colors.ListedColormap(color_list)
        bounds = np.arange(-0.5, len(color_list) + 0.5, 1.0)
        norm = colors.BoundaryNorm(bounds, cluster_cmap.N)

        # plot 2D thresholded data, or a 2D slice of 3D thresholded data
        data_shape = self.data_shape_orig[1:]  # (num_l, num_k, num_h) or (num_k, num_h)
        thr_np = _np(self.thresholded).astype(int, copy=False)

        if len(data_shape) == 2:
            plotting_matrix = thr_np
        elif len(data_shape) == 3:
            if slice_ind is None or axis_ is None:
                raise ValueError("For 3D data, please provide both slice_ind and axis_.")
            plotting_matrix = np.take(thr_np, indices=int(slice_ind), axis=int(axis_))
        else:
            raise RuntimeError(f"Unsupported data dimensionality: {len(data_shape)}")

        plt.figure(figsize=figsize_)
        plt.imshow(plotting_matrix, origin="lower", cmap=cluster_cmap, norm=norm)

        return plt

    def plot_thresholding_3D(self):
        """
        Scatter a 3D point cloud of positions that passed thresholding.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axes with the scatter.

        Notes
        -----
        - Coordinates are normalized to the unit cube per axis using the
          original spatial extents.
        - Alpha channel reflects pass/fail (1 for pass, 0 for fail).
        """


        # Torch -> NumPy helper
        def _np(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        # Cell coordinates in [0,1] along each axis (use original spatial shape)
        L, K, H = self.data_shape_orig[1], self.data_shape_orig[2], self.data_shape_orig[3]
        Ql_cell = np.arange(L) / (L - 1)
        Qk_cell = np.arange(K) / (K - 1)
        Qh_cell = np.arange(H) / (H - 1)

        X, Y, Z = np.meshgrid(Ql_cell, Qk_cell, Qh_cell, indexing="ij")

        # Base RGB color (blue) and alpha from thresholded mask
        blue = np.array([66, 134, 244], dtype=float) / 255.0
        blue = blue[np.newaxis, np.newaxis, np.newaxis, :] + np.zeros(self.data_shape_orig[1:] + (3,))

        thr_np = _np(self.thresholded).astype(float, copy=False)  # 0/1 for alpha channel
        rgba_mat = np.concatenate([blue, thr_np[:, :, :, np.newaxis]], axis=3)  # shape (L,K,H,4)

        # Flatten everything
        N = int(np.prod(self.data_shape_orig[1:]))
        X = X.reshape(N)
        Y = Y.reshape(N)
        Z = Z.reshape(N)
        rgba_mat = rgba_mat.reshape(N, 4)

        mask = thr_np.astype(bool).reshape(N)
        rgba_mat_subset = rgba_mat[mask, :]
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        # Plot
        fig = plt.figure(dpi=70)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X, Y, Z, c=rgba_mat_subset, s=30, edgecolors="none")
        ax.set_xlabel("$Q_l$", fontsize=15)
        ax.set_ylabel("$Q_k$", fontsize=15)
        ax.set_zlabel("$Q_h$", fontsize=15)
        ax.view_init(azim=20)
        plt.tight_layout()

        return fig, ax

class Peak_averaging_old(object):
    """
    Average (and max) the intensity over each connected spatial peak, per temperature.

    Given a temperature-stacked intensity tensor and a binary threshold mask
    (from :class:`Threshold_Background`), this class finds **connected components**
    (peaks) in the spatial dimensions using ``scipy.ndimage.label`` and computes,
    for each detected peak, the **mean** and **maximum** intensity across all
    voxels/pixels in that peak at every temperature. Results are returned as
    time series with shape ``(num_T, num_peaks)``. Component membership indices
    (in the original spatial coordinates) are also stored.

    Connectivity is defined by a full-ones structuring element of size ``3x3``
    for 2D data and ``3x3x3`` for 3D data, matching the original logic.

    Parameters
    ----------
    intensity : torch.Tensor, shape (num_T, *spatial)
        Temperature-stacked intensity data. The leading axis corresponds to
        temperature; the remaining one or more axes are spatial coordinates
        (must match the spatial shape in ``threshold.data_shape_orig[1:]``).
    threshold : Threshold_Background
        Result object that provides the binary mask ``threshold.thresholded``
        (same spatial shape as ``*spatial``) and ``threshold.data_shape_orig``.
        Peaks are defined as connected components in ``threshold.thresholded``.
    device : torch.device or str, optional
        Device on which to place tensors derived from ``intensity`` (and
        where peak-wise averages/maxima are computed). If None, uses
        ``intensity.device`` if ``intensity`` is a tensor; otherwise CPU.

    Attributes
    ----------
    peak_avg_data : torch.Tensor, shape (num_T, num_peaks)
        Per-peak **mean** intensity time series.
    peak_avg_ind_list : list[torch.Tensor]
        List of length ``num_peaks``; each element is an integer tensor of
        shape ``(n_i, D)`` containing spatial indices (per spatial axis) that
        belong to the i-th peak. Here ``D`` is the number of spatial dims.
    peak_max_data : torch.Tensor, shape (num_T, num_peaks)
        Per-peak **maximum** intensity time series.
    peak_max_ind_list : list[torch.Tensor]
        Same structure as ``peak_avg_ind_list``; indices correspond to the
        voxels/pixels used when computing ``peak_max_data``.
    
    Notes
    -----
    - Connected-component labeling is run on **CPU** using
      ``scipy.ndimage.label``; the threshold volume is converted to NumPy.
      Subsequent tensor computations (averages/maxima) occur on ``device``.
    - The number of detected peaks is ``peak_avg_data.shape[1]`` and equals
      ``len(peak_avg_ind_list) == len(peak_max_ind_list)``.
    - Spatial connectivity uses a ones-structuring element:
      ``np.ones((3, 3))`` for 2D, and ``np.ones((3, 3, 3))`` for 3D.

    Examples
    --------
    >>> # intensity: (T, L, K, H) or (T, K, H) torch.Tensor
    >>> # threshold: Threshold_Background instance with .thresholded and .data_shape_orig
    >>> peaks = Peak_averaging_old(intensity, threshold)
    >>> peaks.peak_avg_data.shape  # (num_T, num_peaks)
    torch.Size([T, P])
    >>> len(peaks.peak_avg_ind_list) == peaks.peak_avg_data.shape[1]
    True
    """

    def __init__(self, intensity, threshold, device=None):
        """
        Identify connected components in the threshold mask and compute
        per-peak mean and max intensity trajectories.

        Parameters
        ----------
        intensity : torch.Tensor, shape (num_T, *spatial)
            Intensity values with temperature along axis 0.
        threshold : Threshold_Background
            Object providing ``threshold.thresholded`` (binary mask in spatial
            shape) and ``threshold.data_shape_orig`` (including temperature).
        device : torch.device or str, optional
            Device for tensor computations; defaults to ``intensity.device`` if
            available, otherwise CPU.

        Notes
        -----
        - Uses ``scipy.ndimage.label`` on CPU with a 3x3 (2D) or 3x3x3 (3D)
          structuring element to define connectivity.
        - The binary mask is taken from ``threshold.thresholded``; only those
          spatial sites contribute to each peak.
        """
        # Device handling (do not change original logic; only decide tensor placement)
        if device is None:
            dev = intensity.device if torch.is_tensor(intensity) else torch.device("cpu")
        else:
            dev = torch.device(device)

        # Keep values/shape; move to the chosen device
        x = torch.as_tensor(intensity, device=dev)

        # --- Same as original: choose structuring element and run connected-component labeling (on CPU) ---
        data_shape = threshold.data_shape_orig[1:]  # (num_l, num_k, [num_h])
        if len(data_shape) == 2:
            structure_element = np.ones((3, 3))
        elif len(data_shape) == 3:
            structure_element = np.ones((3, 3, 3))

        thr_cpu_np = torch.as_tensor(threshold.thresholded).detach().to("cpu").numpy()
        labeled_array, num_features = ndimage.label(thr_cpu_np, structure=structure_element)
        # labeled_array has features labeled with integers 1..num_features

        P_avg_data = []
        P_avg_ind_list = []

        P_max_data = []
        P_max_ind_list = []

        # --- Same as original: process each connected component ---
        for i in range(1, num_features + 1):
            label_i = np.isin(labeled_array, i)                             # numpy bool, shape=*spatial
            mask_t = torch.as_tensor(label_i, dtype=torch.bool, device=dev)  # torch.bool mask on device

            data_i = x[:, mask_t]                                           # (T, n_i) — mirrors intensity[:, label_i]
            ind_i = np.array(np.where(label_i)).transpose()                 # (n_i, D)
            ind_i_t = torch.as_tensor(ind_i, dtype=torch.long, device=dev)

            peak_avg_i = torch.mean(data_i, dim=1)                          # (T,)
            P_avg_data.append(peak_avg_i)
            P_avg_ind_list.append(ind_i_t)

            peak_max_i = torch.amax(data_i, dim=1)                          # (T,)
            P_max_data.append(peak_max_i)
            P_max_ind_list.append(ind_i_t)

        # --- Same as original: vstack + transpose -> (num_T, num_peaks) ---
        self.peak_avg_data = torch.vstack(P_avg_data).transpose(0, 1)
        self.peak_avg_ind_list = P_avg_ind_list

        self.peak_max_data = torch.vstack(P_max_data).transpose(0, 1)
        self.peak_max_ind_list = P_max_ind_list


class Peak_averaging(object):
    """
    Aggregate intensity per connected spatial peak across temperature.

    Given a temperature-stacked intensity tensor and a binary threshold mask
    produced by :class:`Threshold_Background`, this class finds connected
    components (peaks) in the spatial dimensions using
    :func:`scipy.ndimage.label` and computes, for each peak, the **mean** and
    **maximum** intensity across all voxels/pixels at every temperature.
    Results are returned as time series with shape ``(num_T, num_peaks)``.
    Component membership indices (in the original spatial coordinates) are
    also stored.

    Connectivity is defined by a ones-valued structuring element of size
    ``3×3`` for 2D data and ``3×3×3`` for 3D data, matching the original logic.

    Parameters
    ----------
    intensity : torch.Tensor, shape (num_T, *spatial)
        Temperature-stacked intensity data. The leading axis corresponds to
        temperature; remaining axes are spatial coordinates and must match
        ``threshold.data_shape_orig[1:]``.
    threshold : Threshold_Background
        Object providing the binary mask ``threshold.thresholded`` (same
        spatial shape as ``*spatial``) and ``threshold.data_shape_orig``. Peaks
        are the connected components of ``threshold.thresholded``.
    device : torch.device or str, optional
        Device on which to place tensors derived from ``intensity`` and to
        perform the per-peak reductions. If ``None``, uses
        ``intensity.device`` when ``intensity`` is a tensor; otherwise CPU.

    Attributes
    ----------
    peak_avg_data : torch.Tensor, shape (num_T, num_peaks)
        Per-peak **mean** intensity time series.
    peak_avg_ind_list : list of torch.Tensor
        List of length ``num_peaks``; each element is an integer tensor of
        shape ``(n_i, D)`` containing spatial indices (per spatial axis) that
        belong to the i-th peak. Here ``D`` is the number of spatial dims.
    peak_max_data : torch.Tensor, shape (num_T, num_peaks)
        Per-peak **maximum** intensity time series.
    peak_max_ind_list : list of torch.Tensor
        Same structure as ``peak_avg_ind_list``; indices correspond to the
        voxels/pixels used when computing ``peak_max_data``.

    Notes
    -----
    - Connected-component labeling is run on **CPU** with
      :func:`scipy.ndimage.label`; the threshold volume is converted to NumPy.
      Subsequent tensor computations (sums/maxima) are performed on ``device``.
    - If no connected components are found (i.e., ``num_peaks == 0``),
      ``peak_avg_data`` and ``peak_max_data`` are empty tensors of shape
      ``(num_T, 0)``, and the index lists are empty.
    - Per-peak means are computed as sums divided by counts. For integer
      ``intensity`` inputs, the mean is returned as floating point to mirror
      ``torch.mean`` semantics.

    Examples
    --------
    >>> # intensity: (T, L, K, H) or (T, K, H) torch.Tensor
    >>> # threshold: Threshold_Background instance with .thresholded and .data_shape_orig
    >>> peaks = Peak_averaging(intensity, threshold)
    >>> peaks.peak_avg_data.shape  # (num_T, num_peaks)
    torch.Size([T, P])
    >>> len(peaks.peak_avg_ind_list) == peaks.peak_avg_data.shape[1]
    True
    """

    def __init__(self, intensity, threshold, device=None):
        """
        Identify connected components in the spatial mask and compute per-peak
        mean and max intensity trajectories.

        Parameters
        ----------
        intensity : torch.Tensor, shape (num_T, *spatial)
            Intensity values with temperature along axis 0 and spatial axes
            matching ``threshold.data_shape_orig[1:]``.
        threshold : Threshold_Background
            Object providing:
              - ``threshold.thresholded`` : binary mask (NumPy-compatible) in
                spatial shape indicating sites retained by thresholding.
              - ``threshold.data_shape_orig`` : original data shape including
                the temperature axis.
        device : torch.device or str, optional
            Target device for tensor reductions; if ``None``, inferred from
            ``intensity`` (if tensor), else set to CPU.

        Raises
        ------
        ValueError
            If the number of spatial dimensions is not 2 or 3.

        Notes
        -----
        - Connectivity uses a ones-structuring element: ``np.ones((3, 3))`` for
          2D and ``np.ones((3, 3, 3))`` for 3D data.
        - Per-peak aggregation is vectorized via indexed reductions:
          ``index_add_`` for sums (means) and ``scatter_reduce_(..., 'amax')``
          for maxima, preserving the original outputs and shapes.
        """
        # Device handling (do not change original logic; only decide tensor placement)
        if device is None:
            dev = intensity.device if torch.is_tensor(intensity) else torch.device("cpu")
        else:
            dev = torch.device(device)

        # Keep values/shape; move to the chosen device
        x = torch.as_tensor(intensity, device=dev)  # (T, *spatial)

        # --- Same as original: choose structuring element and run connected-component labeling (on CPU) ---
        data_shape = threshold.data_shape_orig[1:]  # (num_l, num_k, [num_h])
        if len(data_shape) == 2:
            structure_element = np.ones((3, 3))
        elif len(data_shape) == 3:
            structure_element = np.ones((3, 3, 3))
        else:
            raise ValueError(f"Only 2D/3D spatial dims supported, got len(data_shape)={len(data_shape)}")

        thr_cpu_np = torch.as_tensor(threshold.thresholded).detach().to("cpu").numpy()
        labeled_array, num_features = ndimage.label(thr_cpu_np, structure=structure_element)
        # labeled_array has features labeled with integers 1..num_features

        # ===== Faster replacement for the original per-label Python loop =====
        # Vectorized per-label aggregation on GPU/CPU using segment reductions.
        # This preserves output shapes & semantics:
        #   - self.peak_avg_data: (T, num_peaks)
        #   - self.peak_max_data: (T, num_peaks)
        #   - self.peak_*_ind_list: list of length num_peaks with (n_i, D) long indices

        T = x.shape[0]
        spatial = labeled_array.shape
        N = int(np.prod(spatial))
        L = int(num_features)

        if L == 0:
            # No connected components -> return empty tensors/lists (keeps original behavior)
            self.peak_avg_data = torch.empty((T, 0), device=dev)
            self.peak_avg_ind_list = []
            self.peak_max_data = torch.empty((T, 0), device=dev)
            self.peak_max_ind_list = []
            return

        # 1) Flatten labels once and select only labeled (label>0) pixels
        labels_np = labeled_array.reshape(-1)       # (N,)
        valid_np = labels_np > 0                    # (N,)
        labels1_np = labels_np[valid_np] - 1        # to 0..L-1

        # 2) Prepare tensors on device; gather valid pixels once
        x_flat = x.view(T, N)  # (T, N)
        idx_all = torch.from_numpy(np.nonzero(valid_np)[0]).to(dev)                # (M,)
        labels_all = torch.from_numpy(labels1_np.astype(np.int64)).to(dev)         # (M,)
        x_valid = x_flat[:, idx_all]                                               # (T, M)

        # 3) Per-label SUM (for mean)
        sum_per_label = torch.zeros((T, L), device=dev, dtype=x.dtype)
        sum_per_label.index_add_(1, labels_all, x_valid)  # reduce-sum along label dim

        # 4) Per-label COUNT
        counts = torch.bincount(labels_all, minlength=L)  # (L,) on device

        # 5) Per-label MAX (vectorized); initialize with dtype-appropriate minimum
        if x.dtype.is_floating_point:
            neg_inf = torch.finfo(x.dtype).min
            max_dtype = x.dtype
        else:
            # integer inputs: use dtype min for seeding; max stays integer like original torch.amax
            neg_inf = torch.iinfo(x.dtype).min
            max_dtype = x.dtype

        label_expand = labels_all.unsqueeze(0).expand(T, -1)  # (T, M)
        max_per_label = torch.full((T, L), neg_inf, device=dev, dtype=max_dtype)
        # PyTorch 2.0+: scatter_reduce_ with 'amax'
        max_per_label.scatter_reduce_(1, label_expand, x_valid.to(max_dtype), reduce='amax', include_self=True)

        # 6) Means (match torch.mean semantics: floating output)
        if x.dtype.is_floating_point:
            peak_avg = sum_per_label / counts.clamp_min(1).to(sum_per_label.dtype).unsqueeze(0)  # (T, L)
        else:
            # If input were integer, original torch.mean would require float; emit float32
            peak_avg = (sum_per_label.to(torch.float32) /
                        counts.clamp_min(1).to(torch.float32).unsqueeze(0))  # (T, L)

        peak_max = max_per_label  # (T, L)

        # 7) Build index lists per label (same grouping as original, label order 1..L)
        #    Do this in one pass by sorting by label, then splitting at boundaries.
        coords = np.column_stack(np.unravel_index(np.nonzero(valid_np)[0], spatial))  # (M, D)
        order = np.argsort(labels1_np, kind='mergesort')   # stable; groups by label 0..L-1
        labels_sorted = labels1_np[order]
        coords_sorted = coords[order]
        boundaries = np.flatnonzero(np.diff(labels_sorted)) + 1
        splits = np.split(coords_sorted, boundaries)       # list length = L, order label 0..L-1

        peak_avg_ind_list = [torch.as_tensor(s, dtype=torch.long, device=dev) for s in splits]
        peak_max_ind_list = [torch.as_tensor(s, dtype=torch.long, device=dev) for s in splits]

        # 8) Assign results (columns correspond to label ids 1..L like the original loop)
        self.peak_avg_data = peak_avg
        self.peak_avg_ind_list = peak_avg_ind_list
        self.peak_max_data = peak_max
        self.peak_max_ind_list = peak_max_ind_list


