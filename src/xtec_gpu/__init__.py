"""
XTEC-GPU — GPU-accelerated X-ray Temperature Clustering.
"""
import os

# Prevent OpenBLAS/OMP from spawning too many threads on restricted HPC login nodes.
# This must happen before any numeric libraries (numpy, scipy, sklearn) are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "1.0.0"

from .GMM import GMM, GMM_kernels, Cluster_Gaussian
from .Preprocessing import Mask_Zeros, Threshold_Background, Peak_averaging
