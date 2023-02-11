import numpy as np
import pandas as pd
import logging
import sys
from anndata import AnnData, read_h5ad
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from cnmfsns.config import Config


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
