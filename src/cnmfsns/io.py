import numpy as np
import pandas as pd
import scipy as sp
import logging
from datetime import datetime
import sys
import typing
import semantic_version
import anndata as ad
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import cnmfsns as cn

# lower-level functions

def save_df_to_text(obj, filename):
    obj.to_csv(filename, sep='\t')

def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def load_df_from_npz(filename, multiindex=False):
    with np.load(filename, allow_pickle=True) as f:
        if any([isinstance(c, tuple) for c in (f["index"])]):
            index = pd.MultiIndex.from_tuples(f["index"])
        else:
            index = f["index"]
        if any([isinstance(c, tuple) for c in (f["columns"])]):
            columns = pd.MultiIndex.from_tuples(f["columns"])
        else:
            columns = f["columns"]
        obj = pd.DataFrame(f["data"], index=index, columns=columns)
    return obj

def migrate_anndata(adata:ad.AnnData, force: bool = False):
    # migrates pre-1.0.0 anndata objects to newer format.
    X = adata.to_df()
    raw = pd.DataFrame(adata.raw.X, index=X.index, columns=X.columns)
    corrdist = X.corrwith(raw, axis=1)
    # checks that all samples are perfectly correlated between counts and normalized data
    if ((corrdist - 1).abs() > 1e-6).any():
        logging.warning("Counts and normalized expression matrices are not perfectly correlated. Counts data will be retained in migrated object.")
        if not force:
            errormsg = "Could not migrate AnnData object."
            logging.error(errormsg)
            raise ValueError(errormsg)
    # check for whether user originally input normalized or counts data
    is_normalized = ((X - raw).abs() < 1e-6).all().all()
    X_is_tpm = ((X.sum(axis=1) - 1e6).abs() > 1e2).any()
    if is_normalized and not X_is_tpm:
        logging.warning("AnnData object contains non-TPM normalized data. New AnnData object will retain the count (unnormalized) data only.")
        
    # create new AnnData object
    new_adata = ad.AnnData(X=raw, obs=adata.obs, var=adata.var, uns=adata.uns)
    if "history" not in new_adata.uns:
        new_adata.uns["history"] = {}
    return new_adata, is_normalized


class Dataset():
    
    def __init__(self,
                 adata: ad.AnnData,
                 name: typing.Optional[str] = None,
                 color: typing.Optional[str] = None
                 ):
        
        
        self.name = name
        self.color = color
        self.adata = adata
    
    @classmethod
    def from_df(cls,
                  data: pd.DataFrame,
                  is_normalized: bool,
                  sparsify: bool = False,
                  obs: typing.Optional[pd.DataFrame] = None,
                  var: typing.Optional[pd.DataFrame] = None,
                  name: typing.Optional[str] = None,
                  color: typing.Optional[str] = None,
                  ):
        if sparsify:
            data = sp.csr_matrix(data.values)
        uns = {"history": {}}
        adata = ad.AnnData(X=data, obs=obs, var=var, uns=uns)  
        dataset = cls(adata=adata, name=name, color=color)
        dataset.is_normalized = is_normalized
        dataset.cnmfsns_version = cn.__version__
        dataset.append_to_history("Initialized new AnnData object")  
        return dataset
    
    @classmethod
    def from_h5ad(cls,
                  h5ad_file: str,
                  name: typing.Optional[str] = None,
                  color: typing.Optional[str] = None, force_migrate=False
                  ):
        adata = ad.read_h5ad(h5ad_file)
        dataset = cls(adata=adata, name=name, color=color)
        version = semantic_version.Version(dataset.cnmfsns_version)
        if version.major == 0:
            # importing old versions requires updating
            dataset.adata, dataset.is_normalized = migrate_anndata(adata, force=force_migrate)
            dataset.cnmfsns_version = cn.__version__
            dataset.append_to_history("Migrated pre-1.0.0 AnnData object")  
        if dataset.adata.X is None:
            logging.error(f".h5ad file contains no expression data (adata.X)")
            raise ValueError()
        return dataset
        
    
    def append_to_history(self, text):
        self.adata.uns["history"][datetime.utcnow().isoformat()] = text
        
    def get_history(self):
        return self.adata.uns["history"]
    
    @property
    def is_normalized(self):
        return self.adata.uns["is_normalized"]
    
    @is_normalized.setter
    def is_normalized(self, value: bool):
        self.adata.uns["is_normalized"] = value
        
    @property
    def cnmfsns_version(self):
        if "cnmfsns_version" in self.adata.uns:
            version = self.adata.uns["cnmfsns_version"]
        else:
            version = "0.0.0"
        return version
    
    @cnmfsns_version.setter
    def cnmfsns_version(self, value: bool):
        self.adata.uns["cnmfsns_version"] = value
        
    def update_metadata(self, obs_df):
        # convert 'object' dtype to categorical, converting bool values to strings as these are not supported by AnnData on-disk format
        for col in obs_df.select_dtypes(include="object").columns:
            obs_df[col] = obs_df[col].replace({True: "True", False: "False"}).astype("category")
        missing_samples_in_X = obs_df.index.difference(self.adata.obs.index).astype(str).to_list()
        if missing_samples_in_X:
            logging.warning("The following samples in the metadata were not present in the data (`adata.X`):\n  - " + "\n  - ".join(missing_samples_in_X))
        missing_samples_in_md = self.adata.obs.index.difference(obs_df.index).astype(str).to_list()
        if missing_samples_in_md:
            logging.warning("The following samples in the data (`adata.X`) were absent in the metadata:\n  - " + "\n  - ".join(missing_samples_in_md))
        self.adata.obs = obs_df.reindex(self.adata.obs.index)
    
    def get_metadata_type_summary(self):
        msg = ""
        for col in self.adata.obs.columns:
            msg += "    Column: " + col + "\n"
            for value_type, count in self.adata.obs[col].dropna().map(type).value_counts().items():
                msg += f"        {value_type}: {count}\n"
        return msg
    
    def write_h5ad(self, filename):
        filename = os.path.abspath(filename)
        logging.info(f"Writing to {filename}")
        self.adata.write_h5ad(filename)
        logging.info(f"Done")
    
    def to_df(self, normalized=False):
        df = self.adata.to_df()
        if normalized and not self.is_normalized:
            df = df.div(df.sum(axis=1), axis=0) * 1e6  # TPM normalization
        return df
        
    def remove_unfactorizable_genes(self):
        df = self.to_df(normalized=False)
        
        # Check for variables with missing values
        genes_with_missingvalues = df.isnull().any()
        
        if genes_with_missingvalues.any():
            n_missing = genes_with_missingvalues.sum()
            logging.warning(f"{n_missing} of {dataset.adata.n_vars} variables are missing values (`adata.X`).")
            logging.warning(f"Subsetting variables to those with no missing values.")
                
        # Check for genes with zero variance
        zerovargenes = (df.var() == 0).sum()
        if zerovargenes:
            logging.warning(f"{zerovargenes} of {dataset.adata.n_vars} variables have a variance of zero in counts data (`adata.raw.X`).")
            logging.warning(f"Subsetting variables to those with nonzero variance.")
        
        genes_to_keep = ~genes_with_missingvalues & ~zerovargenes
        
        self.adata = self.adata[:,genes_to_keep]

        
def add_cnmf_results_to_h5ad(cnmf_output_dir, cnmf_name, h5ad_path, local_density_threshold: float = None, local_neighborhood_size: float = None, force=False):
    adata = read_h5ad(h5ad_path)
    adata.uns["cnmf_name"] = cnmf_name
    cnmf_data_loaded =  "cnmf_usage" in adata.obsm or\
                        "cnmf_gep_score" in adata.varm or\
                        "cnmf_gep_tpm" in adata.varm or\
                        "cnmf_gep_raw" in adata.varm
    if cnmf_data_loaded and not force:
        logging.error(f"Error: {h5ad_path} already contains cNMF results. Use --force_h5ad_update to overwrite.")
        sys.exit(1)

    # infer from filenames which local density threshold was used
    sensed_ldts = set()
    for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.*spectra*.k_*")):
        ldt_str = os.path.basename(fn).split(".")[-3]
        try:
            ldt = float(ldt_str.replace("dt_", "").replace("_", "."))
        except ValueError:
            pass
        else:
            sensed_ldts.add((ldt_str, ldt))
    if local_density_threshold is None and len(sensed_ldts) == 1:
        ldt_str, ldt = sensed_ldts.pop()
    elif local_density_threshold in (ldt[1] for ldt in sensed_ldts):
        ldt_str, ldt = [(ldt_str, ldt) for ldt_str, ldt in sensed_ldts if ldt == local_density_threshold].pop()
    else:
        logging.error(f"local_density_threshold of {local_density_threshold} does not match what is in the cNMF result directory: {sensed_ldts}")
        sys.exit(1)
    adata.uns["ldt"] = ldt
    adata.uns["lns"] = local_neighborhood_size
        
    # Import GEPs
    result_types = {
        "gene_spectra_score": "cnmf_gep_score",
        "gene_spectra_tpm": "cnmf_gep_tpm",
        "spectra": "cnmf_gep_raw"
        }
    for matchstr, result_type in result_types.items():
        logging.info(f"Importing GEPs: {matchstr}")  
        meta_w = []
        for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.{matchstr}.k_*.{ldt_str}.*txt")):
            k = int(os.path.basename(fn).removeprefix(f"{cnmf_name}.{matchstr}.").split(".")[0].replace("k_", ""))
            w = pd.read_table(fn, index_col=0)
            w.index = str(k) + "." + w.index.astype(str)
            meta_w.append(w)
        meta_w = pd.concat(meta_w, axis=0).T.reindex(adata.var.index).rename_axis(["k.gep"], axis=1)
        adata.varm[result_type] = meta_w

    # Import Usages
    logging.info(f"Importing Usages")  
    usage = []
    for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.usages.k_*.{ldt_str}.*txt")):
        k = int(os.path.basename(fn).removeprefix(f"{cnmf_name}.usages.").split(".")[0].replace("k_", ""))
        h = pd.read_table(fn, index_col=0)
        h.columns = str(k) + "." + h.columns.astype(str)
        usage.append(h)
    adata.obsm["cnmf_usage"] = pd.concat(usage, axis=1).sort_index(axis=1).rename_axis(["k.gep"], axis=1)
    
    # Import gene list used for factorization
    with open(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}.overdispersed_genes.txt")) as f:
        adata.uns["gene_list"] = [line.strip() for line in f.readlines()]

    # Import K-selection stats
    kvals = pd.DataFrame(**np.load(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}.k_selection_stats.df.npz"), allow_pickle=True)).set_index("k")[["stability", "prediction_error"]]
    kvals.index = kvals.index.astype(int)
    adata.uns["kvals"] = kvals
    logging.info(f"Writing h5ad file")  

    adata.write_h5ad(h5ad_path)

