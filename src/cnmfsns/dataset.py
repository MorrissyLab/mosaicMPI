
from . import utils, cnmf, __version__

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import sys
from typing import Union, Optional
from collections.abc import Iterable, Collection
import os
from glob import glob

import scipy as sp
import anndata as ad
import pandas as pd
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines

def migrate_anndata(adata:ad.AnnData, force: bool = False):
    # migrates external and pre-1.0.0 anndata objects to newer format.
    X = adata.to_df()
    
    if adata.raw is None:  # eg., PBMC dataset from scanpy
        is_normalized=False
        raw = adata.to_df()
    else:
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
        
    if "odg" in adata.uns and "gene_stats" in adata.uns["odg"]:
        gene_stat_columns = adata.uns["odg"]["gene_stats"].columns
        if adata.var.columns.isin(gene_stat_columns).any():
            overlapping_cols = adata.var.columns.isin(gene_stat_columns)
            overlapping_colstr = ", ".join(overlapping_cols[overlapping_cols].index.to_list())
            logging.warning(f"AnnData object contains cNMF gene stats which will override columns in adata.var: {overlapping_colstr}")
        adata.var = pd.merge(left=adata.var, right=adata.uns["odg"]["gene_stats"], how="left", left_index=True, right_index=True)
        del adata.uns["odg"]["gene_stats"]
    
    # create new AnnData object
    new_adata = ad.AnnData(X=raw, obs=adata.obs, var=adata.var, varm=adata.varm, obsm=adata.obsm, uns=adata.uns)
    if "history" not in new_adata.uns:
        new_adata.uns["history"] = {}
    return new_adata, is_normalized

class Dataset():
    
    def __init__(self,
                 adata: ad.AnnData,
                 name: Optional[str] = None,
                 color: Optional[str] = None,
                 patient_id_col: Optional[str] = None
                 ):
        
        
        self.name = name
        self.color = color
        self.adata = adata
        self.patient_id_col = patient_id_col
    
    @classmethod
    def from_df(cls,
                  data: pd.DataFrame,
                  is_normalized: bool,
                  sparsify: bool = False,
                  obs: Optional[pd.DataFrame] = None,
                  var: Optional[pd.DataFrame] = None,
                  name: Optional[str] = None,
                  color: Optional[str] = None,
                  patient_id_col: Optional[str] = None
                  ):
        if var is not None:
            var = var.reindex(data.columns)
        if sparsify:
            data = sp.csr_matrix(data.values)
        data = data.astype("float32")
        uns = {"history": {}, "odg":{}}
        adata = ad.AnnData(X=data, var=var, uns=uns)
        dataset = cls(adata=adata, name=name, color=color, patient_id_col=patient_id_col)
        if obs is not None:
            dataset.update_obs(obs_df=obs)
        dataset.is_normalized = is_normalized
        dataset.cnmfsns_version = __version__
        dataset.append_to_history("Initialized new AnnData object")  
        return dataset
    
    @classmethod
    def from_anndata(cls,
                     adata: ad.AnnData,
                     name: Optional[str] = None,
                     color: Optional[str] = None,
                     patient_id_col: Optional[str] = None,
                     force_migrate: bool = False
                     ):
        dataset = cls(adata=adata, name=name, color=color, patient_id_col=patient_id_col)
        if dataset.cnmfsns_version is None:
            # importing old versions requires updating
            dataset.adata, dataset.is_normalized = migrate_anndata(adata, force=force_migrate)
            dataset.cnmfsns_version = __version__
            dataset.append_to_history("Migrated pre-1.0.0 or external AnnData object")
        if dataset.adata.X is None:
            logging.error(f".h5ad file contains no expression data (adata.X)")
            raise ValueError()
        return dataset
    
    @classmethod
    def from_h5ad(cls,
                  h5ad_file: str,
                  name: Optional[str] = None,
                  color: Optional[str] = None,
                  patient_id_col: Optional[str] = None,
                  force_migrate=False, backed=False
                  ):
        adata = ad.read_h5ad(h5ad_file, backed=backed)
        print(adata.to_df())
        dataset = Dataset.from_anndata(adata=adata, name=name, color=color,
                                       patient_id_col=patient_id_col, 
                                       force_migrate=force_migrate)
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
            version = None
        return version
    
    @cnmfsns_version.setter
    def cnmfsns_version(self, value: bool):
        self.adata.uns["cnmfsns_version"] = value
        
    @property
    def has_cnmf_results(self):
        matrix_checks = [
            "cnmf_usage" in self.adata.obsm,
            "cnmf_gep_score" in self.adata.varm,
            "cnmf_gep_tpm" in self.adata.varm,
            "cnmf_gep_raw" in self.adata.varm
        ]
        return all(matrix_checks)
        
    @property
    def overdispersed_genes(self):
        return self.adata.uns["gene_list"]
        
    def update_obs(self, obs_df):
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
            logging.warning(f"{n_missing} of {self.adata.n_vars} variables are missing values (`adata.X`).")
            logging.warning(f"Subsetting variables to those with no missing values.")
                
        # Check for genes with zero variance
        zerovargenes = (df.var() == 0).sum()
        if zerovargenes:
            logging.warning(f"{zerovargenes} of {self.adata.n_vars} variables have a variance of zero in counts data (`adata.raw.X`).")
            logging.warning(f"Subsetting variables to those with nonzero variance.")
        
        genes_to_keep = ~genes_with_missingvalues & ~zerovargenes
        
        self.adata = self.adata[:,genes_to_keep]

    def compute_gene_stats(self, odg_default_spline_degree: int = 3, odg_default_dof: int = 8):

        data_raw = self.to_df()
        data_normalized = self.to_df(normalized=True)
        
        # create dataframe of per-gene statistics
        self.adata.var["mean"] = data_normalized.mean()
        self.adata.var["rank_mean"] = self.adata.var["mean"].rank()
        self.adata.var["variance"] = data_normalized.var()
        self.adata.var["sd"] = data_normalized.std()
        self.adata.var["missingness"] = data_normalized.isnull().sum() / data_normalized.shape[0]
        self.adata.var[["log_mean", "log_variance"]] = np.log10(self.adata.var[["mean", "variance"]])
        self.adata.var["mean_counts"] = data_raw.mean()
        self.adata.var["odscore_excluded"] = ((self.adata.var["missingness"] > 0) |
                                              self.adata.var["log_mean"].isnull() |
                                              (self.adata.var["mean"] == 0) |
                                              self.adata.var["log_variance"].isnull())

        # model mean-variance relationship using generalized additive model with smooth components
        df_model = self.adata.var[~self.adata.var["odscore_excluded"]]
        bs = BSplines(df_model["mean"], df=odg_default_dof, degree=odg_default_spline_degree)
        gam = GLMGam.from_formula("log_variance ~ log_mean", data=df_model, smoother=bs).fit()
        self.adata.var["resid_log_variance"] = gam.resid_response
        self.adata.var["odscore"] = np.sqrt(10 ** self.adata.var["resid_log_variance"])
        self.adata.var["gam_fittedvalues"] = gam.fittedvalues


        # model mean-variance relationship using cNMF's method based on v-score and minimum expression threshold
        vscore_stats = pd.DataFrame(cnmf.get_highvar_genes(input_counts=data_normalized.values, minimal_mean=0)[0])
        vscore_stats.index = data_normalized.columns
        self.adata.var["vscore"] = vscore_stats["fano_ratio"]
        self.adata.uns["odg"]["odg_default_spline_degree"] = odg_default_spline_degree
        self.adata.uns["odg"]["odg_default_dof"] = odg_default_dof
        self.append_to_history("Gene-level statistics and overdispersion modelling completed.")
        
    def select_overdispersed_genes_from_genelist(self, genes: Collection, min_mean=0):
        self.adata.var["selected"] = self.adata.var.index.isin(genes) & self.adata.var["mean_counts"] >= min_mean
        self.adata.uns["odg"]["overdispersion_metric"] = ""
        self.adata.uns["odg"]["min_mean"] = min_mean
        self.adata.uns["odg"]["min_score"] = ""
        self.adata.uns["odg"]["top_n"] = ""
        self.adata.uns["odg"]["quantile"] = ""
        self.append_to_history("Overdispersed genes selected from custom gene list")
        
    def select_overdispersed_genes(self,
                                   overdispersion_metric: str = "odscore",
                                   min_mean: float = 0,
                                   min_score: float = 1.0,
                                   top_n: int = None,
                                   quantile: float = None):
        
        if overdispersion_metric not in self.adata.var.columns:
            if overdispersion_metric in ("odscore", "vscore"):
                raise ValueError(
                    f"{overdispersion_metric} has not been calculated for this dataset. "
                    "Ensure that you call the `Dataset.compute_gene_stats()` first."
                    )
            else:
                raise ValueError(
                    f"{overdispersion_metric} is not a valid overdispersion metric."
                    )
        
        # warn if multiple methods are selected
        selected_methods = []
        if min_score is not None:
            selected_methods.append("min_score")
        if top_n is not None:
            selected_methods.append("top_n")
        if quantile is not None:
            selected_methods.append("quantile")
        if len(selected_methods) > 1:
            methodwarnstr = ", ".join(selected_methods)
            logging.warning(f"Multiple conflicting overdispersed gene selection criteria have been selected: {methodwarnstr}. "
                            "Only the intersection of these methods will be selected.")
                
        # min_mean filter
        selected_genes = self.adata.var["mean_counts"] >= min_mean
        # min_score filter
        if min_score is not None:
            selected_genes = selected_genes & (self.adata.var[overdispersion_metric] >= min_score)
        # top_n filter
        if top_n is not None:
            genes = self.adata.var[overdispersion_metric].sort_values(ascending=False).head(int(top_n)).index
            selected_genes = selected_genes & self.adata.var.index.isin(genes)
        # quantile filter
        if quantile is not None:
            n_total_genes = self.adata.var[overdispersion_metric].notnull().sum()
            genes = self.adata.var[overdispersion_metric].sort_values(ascending=False).head(int(quantile * n_total_genes)).index
            selected_genes = selected_genes & self.adata.var.index.isin(genes)

        n_selected_genes = selected_genes.sum()
        logging.info(f"{n_selected_genes} genes selected for factorization")
        
        # make changes to Dataset object
        self.adata.var["selected"] = selected_genes
        self.adata.uns["odg"]["overdispersion_metric"] = overdispersion_metric
        self.adata.uns["odg"]["min_mean"] = min_mean
        self.adata.uns["odg"]["min_score"] = min_score if min_score is not None else ""
        self.adata.uns["odg"]["top_n"] = top_n if top_n is not None else ""
        self.adata.uns["odg"]["quantile"] = quantile if quantile is not None else ""
        self.append_to_history("Overdispersed genes selected")
    
    def initialize_cnmf(self, output_dir: str,
                        name: str,
                        kvals: Collection = range(2, 61),
                        n_iter: int = 200,
                        beta_loss: str = "kullback-leibler",
                        seed: Optional[int] = None):
        cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)
        
        # write TPM (normalized) data
        tpm = ad.AnnData(self.to_df(normalized=True))
        tpm.write_h5ad(cnmf_obj.paths["tpm"])

        gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
        gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)
        input_tpm_stats = pd.DataFrame([gene_tpm_mean, gene_tpm_stddev], index = ['__mean', '__std']).T
        utils.save_df_to_npz(input_tpm_stats, cnmf_obj.paths['tpm_stats'])
        overdispersed_genes = self.adata.var["selected"][self.adata.var["selected"]].index
        norm_counts = cnmf_obj.get_norm_counts(self.adata, tpm, high_variance_genes_filter=overdispersed_genes)
        if norm_counts.X.dtype != np.float64:
            norm_counts.X = norm_counts.X.astype(np.float64)
        cnmf_obj.save_norm_counts(norm_counts)

        # save parameters for factorization step
        cnmf_obj.save_nmf_iter_params(*cnmf_obj.get_nmf_iter_params(ks=kvals, n_iter=n_iter, random_state_seed=seed, beta_loss=beta_loss))

        # save parameters in AnnData object
        self.adata.uns["cnmf"] = cnmf_obj.get_nmf_iter_params(ks=kvals, n_iter=n_iter, random_state_seed=seed, beta_loss=beta_loss)[1]  # dict of cnmf parameters
        
        self.append_to_history("cNMF parameters added. cNMF inputs initialized in {output_dir}/{name}")
        return cnmf_obj
    
    def add_cnmf_results(self, cnmf_output_dir, cnmf_name, local_density_threshold: float = None, local_neighborhood_size: float = None):
        self.adata.uns["cnmf_name"] = cnmf_name

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
        self.adata.uns["ldt"] = ldt
        self.adata.uns["lns"] = local_neighborhood_size
            
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
            meta_w = pd.concat(meta_w, axis=0).T.reindex(self.adata.var.index).rename_axis(["k.gep"], axis=1)
            self.adata.varm[result_type] = meta_w

        # Import Usages
        logging.info(f"Importing Usages")  
        usage = []
        for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.usages.k_*.{ldt_str}.*txt")):
            k = int(os.path.basename(fn).removeprefix(f"{cnmf_name}.usages.").split(".")[0].replace("k_", ""))
            h = pd.read_table(fn, index_col=0)
            h.columns = str(k) + "." + h.columns.astype(str)
            usage.append(h)
        self.adata.obsm["cnmf_usage"] = pd.concat(usage, axis=1).sort_index(axis=1).rename_axis(["k.gep"], axis=1)
        
        # Import gene list used for factorization
        with open(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}.overdispersed_genes.txt")) as f:
            self.adata.uns["gene_list"] = [line.strip() for line in f.readlines()]

        # Import K-selection stats
        kvals = pd.DataFrame(**np.load(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}.k_selection_stats.df.npz"), allow_pickle=True)).set_index("k")[["stability", "prediction_error"]]
        kvals.index = kvals.index.astype(int)
        self.adata.uns["kvals"] = kvals
        self.append_to_history("cNMF results added from output directory {cnmf_output_dir}/{cnmf_name}")

    def get_usages(self,
                   k: int = None,
                   discretize: bool = False,
                   normalize: bool = False):
        df = self.adata.obsm["cnmf_usage"].copy()
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(".").to_list())
        df.columns = df.columns.set_levels([l.astype("int") for l in df.columns.levels])
        if normalize:
            normalized = []
            for _, subdf in df.groupby(axis=1, level=0):
                normalized.append(subdf.div(subdf.sum(axis=1), axis=0))
            df = pd.concat(normalized, axis=1)
        if discretize:
            discretized = []
            for _, subdf in df.groupby(axis=1, level=0):
                discretized.append(subdf.eq(subdf.max(axis=1), axis=0).astype(int))
            df = pd.concat(discretized, axis=1)        
        if k is not None:
            df = df.loc[:, k]
        df = df.sort_index(axis=0).sort_index(axis=1)   
        return df
    
    def get_geps(self, k: Union[int, Iterable] = None, type="cnmf_gep_score"):
        df = self.adata.varm[type].copy()
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(".").to_list())
        df.columns = df.columns.set_levels([l.astype("int") for l in df.columns.levels])
        if isinstance(k, (int, Iterable)):
            df = df.loc[:, k]
        df = df.sort_index(axis=1)
        return df
    
    def get_metadata_df(self,
                        include_categorical: bool = True,
                        include_numerical: bool = True
                        ) -> pd.DataFrame:
        dtypes = []
        if include_categorical:
            dtypes.append("category")
        if include_numerical:
            dtypes += ["float", "int"]
        unexplained_cols = self.adata.obs.select_dtypes(exclude=("category", "float", "int")).columns
        if len(unexplained_cols) > 0:
            unexplained_col_str = ", ".join(unexplained_cols)
            raise ValueError(f"{unexplained_col_str} metadata columns have unrecognized dtypes.")
        df = self.adata.obs.select_dtypes(include=dtypes)
        return df
    
    def get_category_overrepresentation(self,
                                        layer: str,
                                        truncate_negative: bool = True
                                        ) -> pd.DataFrame:
        usage = self.get_usages().copy()
        sample_to_class = self.get_metadata_df()[layer]
        usage.index = usage.index.map(sample_to_class)
        observed = usage.groupby(axis=0, level=0).sum()
        expected = []
        for k, obs_k in observed.groupby(axis=1, level=1):
            exp_k = pd.DataFrame(obs_k.sum(axis=1)) @ pd.DataFrame(obs_k.sum(axis=0)).T / obs_k.sum().sum()
            expected.append(exp_k)
        expected = pd.concat(expected, axis=1)
        chisq_resid = (observed - expected) / np.sqrt(expected)  # pearson residual of chi-squared test of contingency table
        if truncate_negative:
            chisq_resid = chisq_resid.clip(lower=0)
        return chisq_resid
    
        
    def get_metadata_correlation(self, 
                                 layer: str,
                                 method: str = "pearson"
                                 ) -> pd.Series:
        usage = self.get_usages().copy()
        metadata = self.get_metadata_df()[layer]
        md_corr = usage.corrwith(metadata, method=method)
        return md_corr