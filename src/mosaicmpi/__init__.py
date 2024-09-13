
# detect version from package metadata
import importlib.metadata
__version__ = importlib.metadata.version('mosaicmpi')

# get CPU affinity for MP-enabled tasks
import os
if hasattr(os, "sched_getaffinity"):
    cpus_available = len(os.sched_getaffinity(0))
else:
    cpus_available = os.cpu_count()

logging_started = False

from .dataset import Dataset
from .config import Config
from .integration import Integration
from .network import Network
from .colors import Colors
from .plots import *
from .utils import start_logging