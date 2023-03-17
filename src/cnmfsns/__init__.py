# detect version from package metadata
import importlib.metadata
__version__ = importlib.metadata.version('cnmfsns')

# get CPU affinity for MP-enabled tasks
import os
if hasattr(os, "sched_getaffinity"):
    cpus_available = len(os.sched_getaffinity(0))
else:
    cpus_available = os.cpu_count()

from .dataset import Dataset
from .config import Config
from .integration import Integration
from .sns import SNS
from .colors import Colors
from .plots import *
from .utils import start_logging


# module level doc-string
__doc__ = """
cNMF-SNS - consensus Non-negative Matrix Factorization Solution Network Space
=====================================================================

**cNMF-SNS** is a Python package enabling integration of bulk, single-cell, and
spatial expression data between and within datasets. cNMF provides a **robust, 
unsupervised** deconvolution of each dataset into gene expression programs (GEPs).
**Network-based integration** of GEPs enables flexible integration of many datasets
across assays (eg. Protein, RNA-Seq) and patient cohorts.

Communities with GEPs from multiple datasets can be annotated with dataset-specific
annotations to facilitate interpretation.

Main Features
-------------
Here are just a few of the things that cNMF-SNS does well:

  - Integration of expression data does not require subsetting features/genes to
    a shared subset
  - Ideal for incremental integration (adding datasets one at a time) since
    deconvolution is performed independently on each dataset
"""
