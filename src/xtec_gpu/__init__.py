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

try:
    import pytorch_lightning
except ImportError:
    pass

class _LightningQuietFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Suppress everything except actual errors/warnings and progress bars (which bypass logging)
        if "LitLogger" in msg: return False
        if "GPU available" in msg: return False
        if "TPU available" in msg: return False
        if "LOCAL_RANK" in msg: return False
        if "stopped:" in msg: return False
        return True

for logger_name in ["pytorch_lightning", "lightning.pytorch", "lightning.fabric", "lightning_utilities.core.rank_zero", "lightning_utilities"]:
    logger = logging.getLogger(logger_name)
    logger.addFilter(_LightningQuietFilter())
    for handler in logger.handlers:
        handler.addFilter(_LightningQuietFilter())

warnings.filterwarnings("ignore", ".*GPU available but not used.*")
warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

__version__ = "1.0.0"

from .GMM import GMM, GMM_kernels, Cluster_Gaussian
from .Preprocessing import Mask_Zeros, Threshold_Background, Peak_averaging
