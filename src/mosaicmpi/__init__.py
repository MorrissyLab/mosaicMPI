
# detect version from package metadata
import importlib.metadata
__version__ = importlib.metadata.version('mosaicmpi')

import anndata as ad

# get CPU affinity for MP-enabled tasks
import os
if hasattr(os, "sched_getaffinity"):
    cpus_available = len(os.sched_getaffinity(0))
else:
    cpus_available = os.cpu_count()

logging_started = False

if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True

from . import factorization
from .factorization import register_factorizer
from .dataset import Dataset
from .config import Config
from .integration import Integration
from .network import Network
from .colors import Colors
from .plots import *
from .utils import start_logging
