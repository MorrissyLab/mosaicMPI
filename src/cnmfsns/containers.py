from mudata import MuData
from anndata import AnnData
import pandas as pd
import numpy as np
import os
from glob import glob

class CnmfResult(object):
    
    def __init__(self, run_name, ldt, geps, usage):
        self.run_name = run_name
        self.ldt = ldt
        # # conform GEP matrices to identical axes
        # gene_union = geps["gene_spectra_tpm"].columns.union(geps["gene_spectra_score"].columns).union(geps["spectra"].columns) # get union of genes in all matrices
        # gep_union = geps["gene_spectra_tpm"].index.union(geps["gene_spectra_score"].index).union(geps["spectra"].index)
        # {k: v.reindex(gep_union).reindex(gene_union, axis=1) for k, v in geps.items()}
        self.geps = geps
        self.usage = usage # .reindex(gep_union, axis=1)
    
    def __repr__(self):
        repstr = [
            "CnmfResult",
            f"Run Name: {self.run_name}",
            f"Local Density Threshold: {self.ldt}",
            f"Usage: {self.usage.shape}",
            f"Gene Expression Programs (GEPs):"
        ]
        for result_type, df in self.geps.items():
            shape_no_nan = df.dropna(how="all").dropna(how="all", axis=1).shape
            repstr.append(f"  - {result_type}: {shape_no_nan}")
        return "\n".join(repstr)
    @property
    def gep_tpm(self):
        return self.geps["gene_spectra_tpm"]

    @property
    def gep_score(self):
        return self.geps["gene_spectra_score"]
    
    @property
    def gep_raw(self):
        return self.geps["spectra"]

    @classmethod
    def from_h5ad(cls, h5ad_file):
        pass

    @classmethod
    def from_dir(cls, cnmf_result_dir, local_density_threshold: float = None):
        """
        if local_density_threshold is None, then infer from filenames (assuming only 1 threshold was used)
        """
        run_name = os.path.basename(os.path.normpath(cnmf_result_dir))
        
        # infer from filenames which local density threshold was used
        ldt_options = set()
        for fn in glob(os.path.join(cnmf_result_dir, f"*.*spectra*.k_*")):
            ldt_str = os.path.basename(fn).split(".")[3]
            ldt = float(ldt_str.replace("dt_", "").replace("_", "."))
            ldt_options.add((ldt_str, ldt))
        if local_density_threshold is None and len(ldt_options) == 1:
            ldt_str, ldt = ldt_options.pop()
        elif local_density_threshold in (ldt[1] for ldt in ldt_options):
            ldt_str, ldt = [(ldt_str, ldt) for ldt_str, ldt in ldt_options if ldt == local_density_threshold].pop()
        else:
            raise RuntimeError(f"local_density_threshold of {local_density_threshold} "
                               f"does not match what is in the cNMF result directory: {ldt_options}")
        
        # Import GEP files
        result_types = ["gene_spectra_score", "gene_spectra_tpm", "spectra"]
        geps = {}
        for result_type in result_types:
            meta_w = []
            for fn in glob(os.path.join(cnmf_result_dir, f"*.{result_type}.k_*.{ldt_str}.*txt")):
                k = int(os.path.basename(fn).split(".")[2].replace("k_", ""))
                w = pd.read_table(fn, index_col=0)
                w.index = pd.MultiIndex.from_arrays(([k] * w.shape[0], w.index))
                meta_w.append(w)
            meta_w = pd.concat(meta_w, axis=0).sort_index(axis=0).rename_axis(["k", "gep"], axis=0)
            geps[result_type] = meta_w

        # Import Usages matrix
        usage = []
        for fn in glob(os.path.join(cnmf_result_dir, f"*.usages.k_*.{ldt_str}.*txt")):
            k = int(os.path.basename(fn).split(".")[2].replace("k_", ""))
            h = pd.read_table(fn, index_col=0)
            h.columns = pd.MultiIndex.from_arrays(([k] * h.shape[1], h.columns.astype(int)))
            usage.append(h)
        usage = pd.concat(usage, axis=1).sort_index(axis=1).rename_axis(["k", "gep"], axis=1)
        
        return cls(run_name, ldt, geps, usage)

    def to_anndata(self, gep_type="gene_spectra_score"):
        df = self.geps[gep_type]
        varm = {}
        for k in df.index.get_level_values(0).unique():
            subdf = df.loc[k].T.copy()
            subdf.columns = subdf.columns.astype("str")
        varm[str(k)] = subdf
        obsm = {}
        for k in self.usage.columns.get_level_values(0).unique():
            subdf = self.usage.loc(axis=1)[k].copy()
            subdf.columns = str(k) + "." + subdf.columns.astype("str")
        obsm[str(k)] = subdf
        uns = {"run_name": self.run_name, "ldt": self.ldt, "gep_type": gep_type}
        return AnnData(X=pd.DataFrame(np.NaN, index=self.usage.index, columns=df.columns), varm=varm, obsm=obsm, uns=uns)

    def to_mudata(self):
        mu_dict = {gep_type: self.to_anndata(gep_type) for gep_type in self.geps.keys()}
        uns = {"run_name": self.run_name, "ldt": self.ldt}
        return MuData(mu_dict, uns=uns)