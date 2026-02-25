"""
XTEC-GPU — GPU-accelerated X-ray Temperature Clustering.
"""
import os

# Prevent OpenBLAS/OMP from spawning too many threads on restricted HPC login nodes.
# This must happen before any numeric libraries (numpy, scipy, sklearn) are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Suppress annoying PyTorch Lightning outputs (like the LitLogger tip)
import logging
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*GPU available but not used.*")

__version__ = "1.0.0"

from .GMM import GMM, GMM_kernels, Cluster_Gaussian
from .Preprocessing import Mask_Zeros, Threshold_Background, Peak_averaging
