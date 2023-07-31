
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

class Dataset():
    
    def __init__(self,
                 adata: ad.AnnData,
                 force_migrate: bool = False
                 ):
        """Creates a :class:`~cnmfsns.dataset.Dataset` object from an `ad.AnnData` object.

        :param adata: AnnData object with data
        :type adata: ad.AnnData
        :param force_migrate: forces conversion of AnnData objects even when adata.X and adata.raw.X are not linearly scaled relative to each other, defaults to False
        :type force_migrate: bool, optional
        :raises RuntimeError: Backed-mode Anndata objects cannot be migrated
        :raises ValueError: Error is raised when force is False and adata is non-linearly scaled.
        :return: Object with expression and metadata
        :rtype: :class:`~cnmfsns.dataset.Dataset`
        """

        if adata.X is None:
            logging.error(f"adata contains no expression data (adata.X)")
            raise ValueError()
        
        
        if "cnmfsns_version" in adata.uns and adata.uns["cnmfsns_version"] is not None:
            self.adata = adata
        else:
            # check and update old or external h5ad files for cNMF-SNS compliance
    
            X = adata.to_df()
            
            if adata.isbacked:
                raise RuntimeError("adata is a backed AnnData object. AnnData objects opened in backed mode cannot be migrated.")

            if adata.raw is None:
                is_normalized=False
                raw = adata.to_df()
            else:  # reconciles two matrices: X and raw.X
                raw = pd.DataFrame(adata.raw.X, index=X.index, columns=X.columns)
                corrdist = X.corrwith(raw, axis=1)
                # checks that all samples are perfectly correlated between counts and normalized data
                if ((corrdist - 1).abs() > 1e-6).any():
                    logging.warning("Counts and normalized expression matrices are not perfectly correlated. Counts data will be retained in migrated object.")
                    if not force_migrate:
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

            # update dataset object
            self.adata = new_adata
            self.is_normalized = is_normalized
            self.cnmfsns_version = __version__
    
    @classmethod
    def from_df(cls,
                  data: pd.DataFrame,
                  is_normalized: bool,
                  sparsify: bool = False,
                  obs: Optional[pd.DataFrame] = None,
                  var: Optional[pd.DataFrame] = None,
                  patient_id_col: Optional[str] = None
                  ):
        """Creates a :class:`~cnmfsns.dataset.Dataset` object from a pandas DataFrame.

        :param data: An observations × variables data
        :type data: pd.DataFrame
        :param is_normalized: Specify if data is already normalized or whether not. Raw data will be TPM normalized prior to overdispersed gene selection, whereas already normalized data will not.
        :type is_normalized: bool
        :param sparsify: Store data as a sparse matrix. [Note that this feature is experimental], defaults to False
        :type sparsify: bool, optional
        :param obs: An observations × metadata matrix, defaults to None
        :type obs: `pd.DataFrame`, optional
        :param var: A variables × metadata matrix, defaults to None
        :type var: `pd.DataFrame`, optional
        :param patient_id_col: Name of metadata layer with patient ID information, defaults to None
        :type patient_id_col: str, optional
        :return: Object with expression and metadata
        :rtype: :class:`~cnmfsns.dataset.Dataset`
        """
        if var is not None:
            var = var.reindex(data.columns)
        if sparsify:
            data = sp.csr_matrix(data.values)
        data = data.astype("float32")
        uns = {"history": {}, "odg":{}}
        adata = ad.AnnData(X=data, var=var, uns=uns)
        dataset = cls(adata)
        if obs is not None:
            dataset.update_obs(obs=obs) 
        dataset.patient_id_col = patient_id_col  # must be done after updating the obs frame since it will check for the columns existence in obs
        dataset.is_normalized = is_normalized
        return dataset
    
    @classmethod
    def from_h5ad(cls,
                  h5ad_file: str,
                  force_migrate=False, backed=False
                  ):
        """Creates a :class:`~cnmfsns.dataset.Dataset` object from an AnnData-compatible .h5ad file.

        :param h5ad_file: Path to .h5ad file produced by scanpy, AnnData, or cNMF-SNS
        :type h5ad_file: str
        :param force_migrate: forces conversion of AnnData objects even when adata.X and adata.raw.X are not linearly scaled relative to each other, defaults to False
        :type force_migrate: bool, optional
        :param backed: Use backed mode to open h5ad file. This can save memory when the dataset is very large, but is not compatible with h5ad files produced outside of cNMF-SNS, defaults to False
        :type backed: bool, optional
        :return: Object with expression and metadata
        :rtype: :class:`~cnmfsns.dataset.Dataset`
        """
        adata = ad.read_h5ad(h5ad_file, backed=backed)
        dataset = cls(adata=adata, force_migrate=force_migrate)
        return dataset
    
    @property
    def is_normalized(self):
        """Outputs the normalization status of the dataset.

        :return: True if dataset contains normalized data, False if it is raw data.
        :rtype: bool
        """
        return self.adata.uns["is_normalized"]
    
    @is_normalized.setter
    def is_normalized(self, value: bool):
        self.adata.uns["is_normalized"] = value
    
    @property
    def patient_id_col(self):
        """Outputs the normalization status of the dataset.

        :return: True if dataset contains normalized data, False if it is raw data.
        :rtype: bool
        """
        return self.adata.uns["patient_id_col"]
    
    @patient_id_col.setter
    def patient_id_col(self, value: str):
        
        if value is not None and value not in self.adata.obs.columns:
            avail_columns = ", ".join(self.adata.obs.columns)
            raise ValueError(f"{value} is not a valid column in the metadata matrix. Available columns are: {avail_columns}")
        self.adata.uns["patient_id_col"] = value

    @property
    def cnmfsns_version(self):
        """cNMF-SNS version used to create the dataset

        :return: version
        :rtype: str
        """
        if "cnmfsns_version" in self.adata.uns:
            version = self.adata.uns["cnmfsns_version"]
        else:
            version = None
        return version
        
    @property
    def has_cnmf_results(self):
        """Test for wehther Dataset contains cNMF results for the dataset

        :return: Whether complete cNMF results are contained for at least 1 rank (k)
        :rtype: bool
        """
        matrix_checks = [
            "cnmf_usage" in self.adata.obsm,
            "cnmf_gep_score" in self.adata.varm,
            "cnmf_gep_tpm" in self.adata.varm,
            "cnmf_gep_raw" in self.adata.varm,
            "kvals" in self.adata.uns
        ]
        return all(matrix_checks)
        
    @property
    def overdispersed_genes(self):
        """Overdispersed gene list used for cNMF

        :return: gene list
        :rtype: list
        """
        return self.adata.uns["gene_list"]


    @cnmfsns_version.setter
    def cnmfsns_version(self, value: bool):
        self.adata.uns["cnmfsns_version"] = value

        
    def update_obs(self, obs):
        """Update the observation metadata with a new metadata matrix

        :param obs: An observations × metadata matrix, defaults to None
        :type obs: `pd.DataFrame`, optional
        """
        # convert 'object' dtype to categorical, converting bool values to strings as these are not supported by AnnData on-disk format
        for col in obs.select_dtypes(include=("bool", "object")).columns:
            obs[col] = obs[col].astype("str").astype("category")
        missing_samples_in_X = obs.index.difference(self.adata.obs.index).astype(str).to_list()
        if missing_samples_in_X:
            logging.warning("The following samples in the metadata were not present in the data (`adata.X`):\n  - " + "\n  - ".join(missing_samples_in_X))
        missing_samples_in_md = self.adata.obs.index.difference(obs.index).astype(str).to_list()
        if missing_samples_in_md:
            logging.warning("The following samples in the data (`adata.X`) were absent in the metadata:\n  - " + "\n  - ".join(missing_samples_in_md))
        self.adata.obs = obs.reindex(self.adata.obs.index)
    
    def get_metadata_type_summary(self):
        """Return a printable summary of metadata and value types for each metadata layer.

        :return: Summary of metadata
        :rtype: str
        """
        msg = ""
        for col in self.adata.obs.columns:
            msg += "    Column: " + col + "\n"
            for value_type, count in self.adata.obs[col].dropna().map(type).value_counts().items():
                msg += f"        {value_type}: {count}\n"
        return msg
    
    def write_h5ad(self,
                   filename: str):
        """Write dataset to .h5ad file.

        :param filename: filepath
        :type filename: str
        """
        filename = os.path.abspath(filename)
        logging.info(f"Writing to {filename}")
        self.adata.write_h5ad(filename)
        logging.info(f"Done")
    
    def to_df(self,
              normalized: bool = False):
        """Get data matrix as a `pd.DataFrame`

        :param normalized: Set true for TPM normalized output, defaults to False
        :type normalized: bool, optional
        :return: observations × variables data matrix
        :rtype: pd.DataFrame
        """
        df = self.adata.to_df()
        if normalized and not self.is_normalized:
            df = df.div(df.sum(axis=1), axis=0) * 1e6  # TPM normalization
        return df
        
    def remove_unfactorizable_genes(self):
        """Removes genes with missing values or zero variance from the data matrix.
        """
        df = self.to_df(normalized=False)
        # Check for variables with missing values
        genes_with_missingvalues = df.isnull().any()
        
        if genes_with_missingvalues.any():
            n_missing = genes_with_missingvalues.sum()
            logging.warning(f"{n_missing} of {self.adata.n_vars} variables are missing values (`adata.X`).")
            logging.warning(f"Subsetting variables to those with no missing values.")
                
        # Check for genes with zero variance
        zerovargenes = (df.var() == 0)
        if zerovargenes.any():
            n_zerovar = zerovargenes.sum()
            logging.warning(f"{n_zerovar} of {self.adata.n_vars} variables have a variance of zero in counts data (`adata.raw.X`).")
            logging.warning(f"Subsetting variables to those with nonzero variance.")
        
        genes_to_keep = (~genes_with_missingvalues) & (~zerovargenes)
        
        self.adata = self.adata[:,genes_to_keep]

    def compute_gene_stats(self, odg_default_spline_degree: int = 3, odg_default_dof: int = 8):
        """
        Computes gene statistics and fits two models of mean and variance of genes in the dataset. The first method is the
        generalized additive model with smooth components (B-splines) to model the relationship of mean and variance
        between genes in the dataset. It produces an odscore metric for overdispersion. The second is the count-statistics
        method found in the cNMF package, which produces a modified v-score metric. All gene statistics are stored within the
        dataset object and are accessible using `dataset.anndata.var`.

        :param odg_default_spline_degree: B-Spline degree for GLM-GAM modelling of mean-variance relationship, defaults to 3
        :type odg_default_spline_degree: int, optional
        :param odg_default_dof: Degrees of freedom for GLM-GAM modelling of mean-varance, defaults to 8
        :type odg_default_dof: int, optional
        """
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
        """Select overdispersed genes/features using a custom list. Genes/features not present in the dataset are automatically filtered out.

        :param genes: gene list
        :type genes: Collection
        :param min_mean: minimum gene expression for genes to be counted as overdispersed, defaults to 0
        :type min_mean: int, optional
        """
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
        """Select overdispersed genes/features using an overdispersion metric. Optionally set a minimum gene expression level.
        Set a threshold using the top N ('top_n'), minimum score ('min_score'), or proportion of features ('quantile') methods.
        Overdispersed gene list is saved in the Dataset object.

        :param overdispersion_metric: "odscore" or "vscore", defaults to "odscore"
        :type overdispersion_metric: str, optional
        :param min_mean: minimum gene expression for genes to be counted as overdispersed, defaults to 0
        :type min_mean: int, optional
        :param min_score: minimum score for overdispersion, defaults to 1.0
        :type min_score: float, optional
        :param top_n: Choose the top N most overdispersed genes, defaults to None
        :type top_n: int, optional
        :param quantile: Choose a quantile of overdispersion. For example, the top 10% of overdispersed genes would be 0.10. Defaults to None
        :type quantile: float, optional
        :raises ValueError: Error if invalid overdispersion metric is chosen.
        """
        
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
    
    def initialize_cnmf(self, cnmf_output_dir: str,
                        cnmf_name: str,
                        kvals: Collection = range(2, 61),
                        n_iter: int = 200,
                        beta_loss: str = "kullback-leibler",
                        seed: Optional[int] = None) -> cnmf.cNMF:
        """Initialize a cNMF run for subsequent factorization.

        :param cnmf_output_dir: Output directory for cNMF results
        :type cnmf_output_dir: str
        :param cnmf_name: Name of the cNMF results. Files will be output to [cnmf_output_dir]/[cnmf_name]/
        :type cnmf_name: str
        :param kvals: Ranks for cNMF factorization, defaults to range(2, 61)
        :type kvals: Collection, optional
        :param n_iter: Number of iterations from which to build a consensus solution, defaults to 200
        :type n_iter: int, optional
        :param beta_loss: beta-loss function, either "kullback-leibler" or "frobenius". Defaults to "kullback-leibler"
        :type beta_loss: str, optional
        :param seed: Random seed for reproducibility, defaults to None
        :type seed: Optional[int], optional
        :return: cNMF object
        :rtype: :class:`cnmfsns.cnmf.cNMF`
        """
        cnmf_obj = cnmf.cNMF(output_dir=cnmf_output_dir, name=cnmf_name)
        
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
        
        self.append_to_history(f"cNMF parameters added. cNMF inputs initialized in {cnmf_output_dir}/{cnmf_name}")
        return cnmf_obj
    
    def add_cnmf_results(self, cnmf_output_dir, cnmf_name, local_density_threshold: float = None, local_neighborhood_size: float = None):
        """
        After factorization, add completed cNMF results in [cnmf_output_dir]/[cnmf_name] to the dataset object.

        :param cnmf_output_dir: Output directory for cNMF results
        :type cnmf_output_dir: str
        :param cnmf_name: Name of the cNMF results. Files will be output to [cnmf_output_dir]/[cnmf_name]/
        :type cnmf_name: str
        :param local_density_threshold: Threshold for the local density filtering prior to GEP consensus. Acceptable thresholds are > 0 and <= 2 (2.0 is no filtering). Defaults to None.
        :type local_density_threshold: float, optional
        :param local_neighborhood_size: Fraction of the number of replicates to use as nearest neighbors for local density filtering. Defaults to None
        :type local_neighborhood_size: float, optional
        """
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

    def remove_cnmf_results(self):
        for result_type in ("cnmf_gep_score", "cnmf_gep_tpm", "cnmf_gep_raw"):
            self.adata.varm.pop(result_type)
        self.adata.obsm.pop("cnmf_usage")
        self.adata.uns.pop("gene_list")
        self.adata.uns.pop("kvals")
        self.append_to_history("cNMF results removed.")


    def get_usages(self,
                   k: Union[int, Iterable] = None,
                   discretize: bool = False,
                   normalize: bool = False
                   ) -> pd.DataFrame:
        """
        Calculate usage of each GEP in each sample/observation.

        :param k: If an integer or list of integers, returns usages only for specified ranks. Otherwise, returns usage of all GEPs across ranks. Defaults to None
        :type k: int, optional
        :param discretize: Discretizes the usage matrix such that for each value of k, each sample has usage of only 1 GEP (the one with the maximum usage). Defaults to False
        :type discretize: bool, optional
        :param normalize: Normalize the GEP usage matrix such that for each value of k, usage of all GEPs sums to 1. Defaults to False
        :type normalize: bool, optional
        :return: observation × GEP matrix
        :rtype: pd.DataFrame
        """
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
    
    def get_geps(self,
                 k: Union[int, Iterable] = None,
                 type="cnmf_gep_score"
                 ) -> pd.DataFrame:
        """
        Get GEPs.

        :param k: If an integer or list of integers, returns GEPs only for specified ranks. Otherwise, returns GEPs from all ranks. Defaults to None
        :type k: Union[int, Iterable], optional
        :param type: "cnmf_gep_score" or "cnmf_gep_tpm", defaults to "cnmf_gep_score"
        :type type: str, optional
        :return: features × GEP matrix
        :rtype: pd.DataFrame
        """
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
        """Get sample/observation metadata.

        :param include_categorical: Include categorical metadata layers, defaults to True
        :type include_categorical: bool, optional
        :param include_numerical: Include numerical metadata layers, defaults to True
        :type include_numerical: bool, optional
        :raises ValueError: Error if metadata types are not recognized
        :return: observations × metadata matrix
        :rtype: pd.DataFrame
        """
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
        df = df.dropna(axis=1, how="all")
        return df
    
    def get_category_overrepresentation(self,
                                        layer: str,
                                        truncate_negative: bool = True
                                        ) -> pd.DataFrame:
        """Calculate Pearson residual of chi-squared test, associating GEPs for each rank (k) to categories of samples/observations. By default, truncates negative values.

        :param layer: name of categorical data layer
        :type layer: str
        :param truncate_negative: Truncate negative residuals to 0, defaults to True
        :type truncate_negative: bool, optional
        :return: category × GEP matrix of overrepresentation values
        :rtype: pd.DataFrame
        """
        usage = self.get_usages(normalize=True).copy()
        sample_to_class = self.get_metadata_df()[layer]
        usage.index = usage.index.map(sample_to_class)
        observed = usage.groupby(axis=0, level=0).sum()

        n_categories = observed.shape[0]
        if n_categories < 2:
            logging.warning(f"Overrepresentation could not be calculated for layer '{layer}', as only {n_categories} categories were found in the data. "
                            f"Note that empty values in the metadata are not considered a category. "
                            f"Overrepresentation cannot be calculated with fewer than 2 categories for each layer. ")
            return pd.DataFrame(np.NaN, index = observed.index, columns=observed.columns)
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
        """Calculate Pearson correlation of GEP usage to numerical metadata across samples/observations.

        :param layer: name of numerical data layer
        :type layer: str
        :param method: Correlation method: "pearson", "spearman", or "kendall". Defaults to "pearson"
        :type method: str, optional
        :return: correlation of GEP to metadata
        :rtype: pd.Series
        """
        usage = self.get_usages().copy()
        metadata = self.get_metadata_df()[layer]
        md_corr = usage.corrwith(metadata, method=method)
        return md_corr
        
    def append_to_history(self, entry):
        """Add entry to Dataset history.

        :param entry: Description of event to record in the history.
        :type entry: str
        """
        self.adata.uns["history"][datetime.utcnow().isoformat()] = entry
        
    def get_history(self):
        """Returns timestamped history of Dataset object.

        :return: history
        :rtype: dict
        """
        return self.adata.uns["history"]