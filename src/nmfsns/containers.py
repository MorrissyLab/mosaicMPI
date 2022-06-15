from anndata import AnnData
import pandas as pd
import numpy as np
import os
from glob import glob

class CnmfResult(AnnData):
    def __init__(self, cnmf_result_dir, result_types = ["gene_spectra_score", "gene_spectra_tpm", "spectra"], local_density_threshold=None):
        
        self.run_name = os.path.basename(os.path.normpath(cnmf_result_dir))

        # Import GEP matrices
        geps = {}
        for result_type in result_types:
            meta_w = []
            for fn in glob(os.path.join(cnmf_result_dir, f"*.{result_type}.k_*")):
                splitf = os.path.basename(fn).split(".")
                k, dt = splitf[2:4]
                k = int(k.replace("k_", ""))
                dt = float(dt.replace("dt_", "").replace("_", "."))
                if local_density_threshold is not None and local_density_threshold != dt:
                    continue
                w = pd.read_table(fn, index_col=0)
                w.index = pd.MultiIndex.from_arrays(([dt] * w.shape[0], [k] * w.shape[0], w.index))
                meta_w.append(w)
            meta_w = pd.concat(meta_w, axis=0).sort_index(axis=0).rename_axis(["local_density_threshold", "k", "gep"], axis=0)
            geps[result_type] = meta_w

        # Import Usages matrix
        meta_h = []
        for fn in glob(os.path.join(cnmf_result_dir, f"*.usages.k_*")):
            splitf = os.path.basename(fn).split(".")
            k, dt = splitf[2:4]
            k = int(k.replace("k_", ""))
            dt = float(dt.replace("dt_", "").replace("_", "."))
            if local_density_threshold is not None and local_density_threshold != dt:
                continue
            h = pd.read_table(fn, index_col=0)
            h.columns = pd.MultiIndex.from_arrays(([dt] * h.shape[1], [k] * h.shape[1], h.columns.astype(int)))
            meta_h.append(h)
        meta_h = pd.concat(meta_h, axis=1).sort_index(axis=1).rename_axis(["local_density_threshold", "k", "gep"], axis=1)
        
        # Unfortunately, usage matrices must be stored in `.obsm`, which is not ideal, since these samples will also themselves have annotations.
        # This is a limitation of AnnData.
        
        AnnData.__init__(self, X=geps["gene_spectra_score"], dtype=np.float32, layers=geps, obsm=meta_h.T)

