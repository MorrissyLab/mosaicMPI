
from . import utils, cnmf, biomart, __version__

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import sys
from typing import Union, Optional, Literal
from collections.abc import Iterable, Collection
import os
import shutil
from io import StringIO
from glob import glob

import scipy.sparse as sp
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from statsmodels.gam.api import GLMGam, BSplines
from tqdm import tqdm

class Dataset():
    
    def __init__(self,
                 adata: ad.AnnData,
                 force_migrate: bool = False
                 ):
        """Creates a :class:`~mosaicmpi.dataset.Dataset` object from an :class:`anndata.AnnData` object.

        :param adata: AnnData object with data
        :type adata: :class:`anndata.AnnData`
        :param force_migrate: forces conversion of AnnData objects even when adata.X and adata.raw.X are not linearly scaled relative to each other, defaults to False
        :type force_migrate: bool, optional
        :raises RuntimeError: Backed-mode Anndata objects cannot be migrated
        :raises ValueError: Error is raised when force is False and adata is non-linearly scaled.
        :return: Object with expression and metadata
        :rtype: :class:`~mosaicmpi.dataset.Dataset`
        """

        if adata.X is None:
            logging.error(f"adata contains no data")
        
        if "mosaicmpi_version" in adata.uns and adata.uns["mosaicmpi_version"] is not None:
            self.adata = adata
        else:
            # check and update old or external h5ad files for mosaicMPI compliance
    
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
            
            # create new AnnData object
            new_adata = ad.AnnData(X=raw, obs=adata.obs, var=adata.var, varm=adata.varm, obsm=adata.obsm, uns=adata.uns)
            
            # check/initialize unstructured metadata
            if "history" not in new_adata.uns:
                new_adata.uns["history"] = {}
            if "odg" not in new_adata.uns:
                new_adata.uns["odg"] = {}
            # add missing value calculations on input data
            if "missing_values" not in new_adata.var:
                new_adata.var["missing_values"] = raw.isnull().sum()
            if "missingness" not in new_adata.var:
                new_adata.var["missingness"] = new_adata.var["missing_values"] / raw.shape[0]

            # update dataset object
            self.adata = new_adata
            self.is_normalized = is_normalized
            self.mosaicmpi_version = __version__
    
        # fix limitations of AnnData on-disk format


    @classmethod
    def from_df(cls,
                  data: pd.DataFrame,
                  is_normalized: bool,
                  sparsify: bool = False,
                  obs: Optional[pd.DataFrame] = None,
                  var: Optional[pd.DataFrame] = None,
                  patient_id_col: Optional[str] = None
                  ):
        """Creates a :class:`~mosaicmpi.dataset.Dataset` object from a pandas DataFrame.

        :param data: An observations × features data
        :type data: pd.DataFrame
        :param is_normalized: Specify if data is already normalized or whether not. Raw data will be TPM normalized prior to overdispersed gene selection, whereas already normalized data will not.
        :type is_normalized: bool
        :param sparsify: Store data as a sparse matrix. [Note that this feature is experimental], defaults to False
        :type sparsify: bool, optional
        :param obs: An observations × metadata matrix, defaults to None
        :type obs: `pd.DataFrame`, optional
        :param var: A features × metadata matrix, defaults to None
        :type var: `pd.DataFrame`, optional
        :param patient_id_col: Name of metadata layer with patient ID information, defaults to None
        :type patient_id_col: str, optional
        :return: Object with expression and metadata
        :rtype: :class:`~mosaicmpi.dataset.Dataset`
        """
        if var is not None:
            var = var.reindex(data.columns)
        if sparsify:
            data = sp.csr_matrix(data.values)
        data = data.astype("float32")
        adata = ad.AnnData(X=data, var=var)
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
        """Creates a :class:`~mosaicmpi.dataset.Dataset` object from an AnnData-compatible .h5ad file.

        :param h5ad_file: Path to .h5ad file produced by scanpy, AnnData, or mosaicMPI
        :type h5ad_file: str
        :param force_migrate: forces conversion of AnnData objects even when adata.X and adata.raw.X are not linearly scaled relative to each other, defaults to False
        :type force_migrate: bool, optional
        :param backed: Use backed mode to open h5ad file. This can save memory when the dataset is very large, but is not compatible with h5ad files produced outside of mosaicMPI, defaults to False
        :type backed: bool, optional
        :return: Object with expression and metadata
        :rtype: :class:`~mosaicmpi.dataset.Dataset`
        """
        adata = ad.read_h5ad(h5ad_file, backed=backed)
        dataset = cls(adata=adata, force_migrate=force_migrate)
        return dataset
    
    @classmethod
    def from_anndata(cls,
                  adata: ad.AnnData,
                  force_migrate=False
                  ):
        """Creates a :class:`~mosaicmpi.dataset.Dataset` object from an :class:`anndata.AnnData` object.

        :param adata: AnnData object with data
        :type adata: :class:`anndata.AnnData`
        :param force_migrate: forces conversion of AnnData objects even when adata.X and adata.raw.X are not linearly scaled relative to each other, defaults to False
        :type force_migrate: bool, optional
        :return: Object with expression and metadata
        :rtype: :class:`~mosaicmpi.dataset.Dataset`
        """
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
    def is_imputed(self):
        """Outputs the imputation status of the dataset.

        :return: True if dataset contains normalized data, False if it is raw data.
        :rtype: bool
        """
        return "imputation" in self.adata.uns

    @property
    def patient_id_col(self):
        """Outputs the normalization status of the dataset.

        :return: True if dataset contains normalized data, False if it is raw data.
        :rtype: bool
        """
        if "patient_id_col" in self.adata.uns:
            return self.adata.uns["patient_id_col"]
        else:
            return None
    
    @patient_id_col.setter
    def patient_id_col(self, value: str):
        
        if value is not None and value not in self.get_metadata_df():
            avail_columns = ", ".join(self.get_metadata_df())
            raise ValueError(f"{value} is not a valid column in the metadata matrix. Available columns are: {avail_columns}")
        self.adata.uns["patient_id_col"] = value

    @property
    def mosaicmpi_version(self):
        """mosaicMPI version used to create the dataset

        :return: version
        :rtype: str
        """
        if "mosaicmpi_version" in self.adata.uns:
            version = self.adata.uns["mosaicmpi_version"]
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
        return self.adata.var[self.adata.var["selected"]].index


    @mosaicmpi_version.setter
    def mosaicmpi_version(self, value: bool):
        self.adata.uns["mosaicmpi_version"] = value

    def map_gene_ids(self,
                        source_species: Literal["hsapiens", "mmusculus", "rnorvegicus", "sscrofa", "dmelanogaster", "drerio", "celegans"],
                        dest_species: Literal["hsapiens", "mmusculus", "rnorvegicus", "sscrofa", "dmelanogaster", "drerio", "celegans"],
                        source_ids: Literal["ensembl_gene", "gene_name"],
                        dest_ids: Literal["ensembl_gene", "gene_name"],
                        one_to_one: bool = True,
                        one_to_many: Literal[False, "duplicate", "divide"] = False,
                        many_to_one: Literal[False, "mean", "sum"] = False,
                        many_to_many: Literal[False, "mean", "sum"] = False,
                        unmapped_prefix: str = "unmapped_",
                        case_sensitive: bool = False,
                        biomart_url: str = "http://www.ensembl.org:80/biomart/martservice"):
        """Map the feature IDs in place for a dataset. Mapping occurs from a source to the dest species, and can be gene names or ensembl gene IDs (eg., ENSG..., ENSMUSG...).

        :param source_species: Species name for IDs in the dataset.
        :type source_species: Literal["hsapiens", "mmusculus", "rnorvegicus", "sscrofa", "dmelanogaster", "drerio", "celegans"]
        :param dest_species: Species name for IDs after mapping
        :type dest_species: Literal["hsapiens", "mmusculus", "rnorvegicus", "sscrofa", "dmelanogaster", "drerio", "celegans"]
        :param source_ids: Whether the IDs are gene names (eg., EGFR), or Ensembl genes (eg., ENSG00000146648)
        :type source_ids: Literal["ensembl_gene", "gene_name"]
        :param dest_ids: Whether the IDs should be mapped to gene names (eg., EGFR), or Ensembl genes (eg., ENSG00000146648)
        :type dest_ids: Literal["ensembl_gene", "gene_name"]
        :param one_to_one: Whether to map genes that have a one-to-one mapping, defaults to True
        :type one_to_one: bool, optional
        :param one_to_many: Whether to map genes that have a one-to-many mapping, defaults to False
        :type one_to_many: Literal[False, "duplicate", "divide"], optional
        :param many_to_one: Whether and how to map genes that have a many-to-one mapping, defaults to False
        :type many_to_one: Literal[False, "mean", "sum"], optional
        :param many_to_many: Whether and how to map genes that have a many-to-many mapping, defaults to False
        :type many_to_many: Literal[False, "mean", "sum"], optional
        :param unmapped_prefix: For unmapped features, prepend this text to their ID, defaults to "unmapped\\_"
        :type unmapped_prefix: str, optional
        :param case_sensitive: Case-sensitive ID matching, defaults to False
        :type case_sensitive: bool, optional
        :param biomart_url: URL to connect to the Biomart web server, defaults to
                            "http://www.ensembl.org:80/biomart/martservice"
        :type biomart_url: str, optional
        :raises NotImplementedError: for features net yet implemented, including many-to-one and many-to-many gene mappings
        """
        logging.info("Downloading gene ID table from Ensembl Biomart")
        gene_dataset = biomart.BiomartServer(url=biomart_url).datasets[f"{source_species}_gene_ensembl"]
        if source_species == dest_species:
            result = gene_dataset.search(params={"attributes": ["external_gene_name",
                                                                "ensembl_gene_id"]})
            columns=['source_gene_name','source_ensembl_gene']
            result = pd.read_csv(StringIO(result.text), sep="\t", header=None, names=columns)
            result_dest = result.copy()
            result_dest.columns = ["dest_gene_name", "dest_ensembl_gene"]
            result = pd.concat([result, result_dest], axis=1)

        else:
            result = gene_dataset.search(params={"attributes": ["external_gene_name",
                                                            "ensembl_gene_id",
                                                            f"{dest_species}_homolog_associated_gene_name",
                                                            f"{dest_species}_homolog_ensembl_gene"]})
            columns=['source_gene_name','source_ensembl_gene', 'dest_gene_name', 'dest_ensembl_gene']
            result = pd.read_csv(StringIO(result.text), sep="\t", header=None, names=columns)

        logging.info("Mapping gene IDs")

        if not case_sensitive:
            result[f"source_{source_ids}"] = result[f"source_{source_ids}"].str.casefold()

        id_mapping = result.groupby([f"source_{source_ids}", f"dest_{dest_ids}"]).apply(lambda x : x.count()).iloc[:,0]
        id_mapping = pd.DataFrame(id_mapping.rename("path_counts"))
        assert id_mapping.index.is_unique
        source_id_counts = id_mapping.index.get_level_values(0).value_counts()
        dest_id_counts = id_mapping.index.get_level_values(1).value_counts()
        id_mapping["source_id_count"] = id_mapping.index.get_level_values(0).map(source_id_counts)
        id_mapping["dest_id_count"] = id_mapping.index.get_level_values(1).map(dest_id_counts)

        # create ordered lists of the IDs for indexing the anndata object
        source_id_list = []
        dest_id_list= []
        mapping_relationship = []

        # One-to-none: IDs that are not found in the ID mapping table
        if case_sensitive:
            id_idx = ~self.adata.var_names.isin(id_mapping.index.get_level_values(0))
        else:
            id_idx = ~self.adata.var_names.str.casefold().isin(id_mapping.index.get_level_values(0))
        src_ids = self.adata.var_names[id_idx]
        dst_ids = unmapped_prefix + src_ids
        source_id_list.extend(src_ids)
        dest_id_list.extend(unmapped_prefix + src_ids)
        mapping_relationship.extend(len(src_ids) * ["one-to-none"])

        # One-to-one
        o2o = id_mapping[(id_mapping["source_id_count"] == 1) & (id_mapping["dest_id_count"] == 1)]
        if case_sensitive:
            id_idx = self.adata.var_names.isin(o2o.index.get_level_values(0))
            ids_to_map = self.adata.var_names[id_idx]
        else:
            id_idx = self.adata.var_names.str.casefold().isin(o2o.index.get_level_values(0).str.casefold())
            ids_to_map = self.adata.var_names[id_idx].str.casefold()
        src_ids = self.adata.var_names[id_idx]
        if one_to_one:
            dst_ids = o2o.index.to_frame().reset_index(level=1, drop=True)[f"dest_{dest_ids}"].loc[ids_to_map]
        else:
            dst_ids = unmapped_prefix + src_ids
        source_id_list.extend(src_ids)
        dest_id_list.extend(dst_ids)
        mapping_relationship.extend(len(src_ids) * ["one-to-one"])

        # One-to-many
        o2m = id_mapping[(id_mapping["source_id_count"] == 1) & (id_mapping["dest_id_count"] > 1)]
        if case_sensitive:
            id_idx = self.adata.var_names.isin(o2m.index.get_level_values(0))
            ids_to_map = self.adata.var_names[id_idx]
        else:
            id_idx = self.adata.var_names.str.casefold().isin(o2m.index.get_level_values(0).str.casefold())
            ids_to_map = self.adata.var_names[id_idx].str.casefold()
        src_ids = self.adata.var_names[id_idx]
        if one_to_many == "duplicate":
            # duplicates the values of the existing ID for the new multiple IDs
            for src_id, id_to_map in zip(src_ids, ids_to_map):
                dest = o2m.loc[id_to_map].index
                source_id_list.extend(len(dest) * [src_id])
                dest_id_list.extend(dest)
                mapping_relationship.extend(len(dest) * ["one-to-many"])
        elif one_to_many is False:
            dst_ids = unmapped_prefix + src_ids
            source_id_list.extend(src_ids)
            dest_id_list.extend(dst_ids)
            mapping_relationship.extend(len(src_ids) * ["one-to-many"])
        else:
            raise NotImplementedError(f"`{one_to_many}` is not currently implemented for one-to-many gene mappings.")

        # Many-to-one only
        m2o = id_mapping[(id_mapping["source_id_count"] > 1) & (id_mapping["dest_id_count"] == 1)]
        m2m = id_mapping[(id_mapping["source_id_count"] > 1) & (id_mapping["dest_id_count"] > 1)]

        if case_sensitive:
            id_idx = self.adata.var_names.isin(m2o.index.get_level_values(0)) & ~self.adata.var_names.isin(m2m.index.get_level_values(0))
            ids_to_map = self.adata.var_names[id_idx]
        else:
            id_idx = self.adata.var_names.str.casefold().isin(m2o.index.get_level_values(0).str.casefold()) & ~self.adata.var_names.str.casefold().isin(m2m.index.get_level_values(0).str.casefold())
            ids_to_map = self.adata.var_names[id_idx].str.casefold()
        src_ids = self.adata.var_names[id_idx]
        if many_to_one is False:
            dst_ids = unmapped_prefix + src_ids
            source_id_list.extend(src_ids)
            dest_id_list.extend(dst_ids)
            mapping_relationship.extend(len(src_ids) * ["many-to-one"])
        else:
            raise NotImplementedError(f"`{many_to_one}` is not currently implemented for many-to-one gene mappings.")

        # Many-to-many only
        if case_sensitive:
            id_idx = self.adata.var_names.isin(m2m.index.get_level_values(0)) & ~self.adata.var_names.isin(m2o.index.get_level_values(0))
            ids_to_map = self.adata.var_names[id_idx]
        else:
            id_idx = self.adata.var_names.str.casefold().isin(m2m.index.get_level_values(0).str.casefold()) & ~self.adata.var_names.str.casefold().isin(m2o.index.get_level_values(0).str.casefold())
            ids_to_map = self.adata.var_names[id_idx].str.casefold()
        src_ids = self.adata.var_names[id_idx]
        if many_to_many is False:
            dst_ids = unmapped_prefix + src_ids
            source_id_list.extend(src_ids)
            dest_id_list.extend(dst_ids)
            mapping_relationship.extend(len(src_ids) * ["many-to-many"])
        else:
            raise NotImplementedError(f"`{many_to_many}` is not currently implemented for many-to-many gene mappings.")

        # IDs with both many-to-many and many-to-one relationship
        if case_sensitive:
            id_idx = self.adata.var_names.isin(m2m.index.get_level_values(0)) & self.adata.var_names.isin(m2o.index.get_level_values(0))
            ids_to_map = self.adata.var_names[id_idx]
        else:
            id_idx = self.adata.var_names.str.casefold().isin(m2m.index.get_level_values(0).str.casefold()) & self.adata.var_names.str.casefold().isin(m2o.index.get_level_values(0).str.casefold())
            ids_to_map = self.adata.var_names[id_idx].str.casefold()
        src_ids = self.adata.var_names[id_idx]
        dst_ids = unmapped_prefix + src_ids
        source_id_list.extend(src_ids)
        dest_id_list.extend(dst_ids)
        mapping_relationship.extend(len(src_ids) * ["many-to-many; many-to-one"])

        # create new anndata object
        new_adata = self.adata[:, source_id_list]
        new_adata.var_names = dest_id_list
        new_adata.var["source_id"] = source_id_list
        new_adata.var["mapping_relationship"] = mapping_relationship
        new_adata.var["mapping_relationship"] = new_adata.var["mapping_relationship"].astype("category")
        self.adata = new_adata
        self.append_to_history(f"Mapped feature IDs. source species={source_species}, source_ids={source_ids}, dest_species={dest_species}, dest_ids={dest_ids}, "
                               f"one_to_one={one_to_one}, one_to_many={one_to_many}, many_to_one={many_to_one}, many_to_many={many_to_many}, unmapped_prefix={unmapped_prefix}")

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
        self.append_to_history(f"metadata (adata.obs) updated.")
    
    def get_printable_metadata_type_summary(self):
        """Return a printable summary of metadata and the types.

        :return: Summary of metadata
        :rtype: str
        """
        msg = "Summary of metadata types\n\n    Sample metadata:\n"
        for col in self.adata.obs.columns:
            msg += "        " + col + ": " + self.adata.obs[col].dtype.name + "\n"
        msg += "    Feature metadata:\n"
        for col in self.adata.var.columns:
            msg += "        " + col + ": " + self.adata.var[col].dtype.name + "\n"
        return msg
    
    def write_h5ad(self,
                   filename: str,
                   safe_mode: bool = True):
        """Write dataset to .h5ad file.

        :param filename: filepath
        :type filename: str
        :param safe_mode: If overwriting an existing h5ad file of the same name, then the new file is first written to a temporary file before overwriting.
            By doing this, errors during writing do not corrupt the existing h5ad file. Only after the file has successfully been written to the temporary file does the original file get overwritten.
        :type filename: bool, 
        """
        filename = os.path.abspath(filename)
        if os.path.exists(filename) and safe_mode:
            head, tail = os.path.split(filename)
            temp_filename = os.path.join(head, tail + "." + os.urandom(4).hex() + ".tmp")
            
            msg = f"Writing to temporary file {temp_filename}"
            logging.info(msg)
            self.append_to_history(msg)
            self.adata.write_h5ad(temp_filename)

            msg = f"Write completed. Moving to {filename}"
            logging.info(msg)
            self.append_to_history(msg)
            shutil.move(temp_filename, filename)
            logging.info(f"Move completed.")
        else:
            logging.info(f"Writing to {filename}")
            self.append_to_history("Writing to {filename}")
            self.adata.write_h5ad(filename)
            logging.info(f"Write completed.")
    
    def to_df(self,
              normalized: bool = False):
        """Get data matrix as a `pd.DataFrame`

        :param normalized: Set true for TPM normalized output, defaults to False
        :type normalized: bool, optional
        :return: observations × features data matrix
        :rtype: pd.DataFrame
        """
        df = self.adata.to_df()
        if normalized and not self.is_normalized:
            df = df.div(df.sum(axis=1), axis=0) * 1e6  # TPM normalization
        return df

    def remove_unfactorizable_observations(self):
        """Removes observations with all zeros from the data matrix.
        """
        df = self.to_df(normalized=False)
        # Check for observations with all zeros
        obs_zeros = (df.T == 0).all()
        n_obs_zeros = obs_zeros.sum()
        logging.info(f"{n_obs_zeros} of {self.adata.n_obs} observations are all zeros.")
        if n_obs_zeros:
            logging.warning(f"Subsetting observations to those with at least 1 positive value.")
            msg_string = repr(obs_zeros[obs_zeros].index.to_list())
            self.append_to_history(f"Removing observations with only zeros: {msg_string}")
        
        self.adata = self.adata[~obs_zeros,:].copy()


    def remove_unfactorizable_features(self):
        """Removes features with missing values or zero variance from the data matrix.
        """
        df = self.to_df(normalized=False)
        # Check for features with missing values
        genes_with_missingvalues = df.isnull().any()
        
        n_missing = genes_with_missingvalues.sum()
        logging.info(f"{n_missing} of {self.adata.n_vars} features are missing values.")
        if n_missing:
            logging.warning(f"Subsetting features to those with no missing values.")
            msg_string = repr(genes_with_missingvalues[genes_with_missingvalues].index.to_list())
            self.append_to_history(f"Removing features with missing values: {msg_string}")
        # Check for genes with zero variance
        zerovargenes = (df.var() == 0)
        n_zerovar = zerovargenes.sum()
        logging.info(f"{n_zerovar} of {self.adata.n_vars} features have a variance of zero.")
        if n_zerovar:
            logging.warning(f"Subsetting features to those with nonzero variance.")
            msg_string = repr(zerovargenes[zerovargenes].index.to_list())
            self.append_to_history(f"Removing features with zero variance: {msg_string}")
        genes_to_keep = (~genes_with_missingvalues) & (~zerovargenes)
        self.adata = self.adata[:,genes_to_keep].copy()

    def cross_validate_imputation(self, imputer: Union[KNNImputer, SimpleImputer], n_folds: int = 100):
        """Perform k-fold cross validation of imputation on the dataset without modifying the data.

        :param imputer: imputer object from scikit-learn
        :type imputer: Union[KNNImputer, SimpleImputer]
        :param n_folds: number of folds, defaults to 100
        :type n_folds: int, optional
        :return: Datafram with statistics for each gene, including preimputation log-mean and log-variance, and NRMSD mean and variance across all folds.
        :rtype: :class:`pandas.DataFrame`
        """
        logging.info(f"Performing k-fold cross-validation, with {n_folds} folds")
        # k-fold cross-validation
        rmsd_folds = []   # store gene-wise rmsd
        data = self.adata.to_df()
        random_numbers = np.random.random(data.shape)  # used to assign each data point to a fold
        bins = np.linspace(0, 1, n_folds + 1)
        for fold in tqdm(range(n_folds), unit="fold"):
            lower, upper = bins[fold:fold + 2]
            mask = (random_numbers >= lower) & (random_numbers < upper)
            test = data.mask(~mask)
            training = data.mask(mask)
            imputed = imputer.fit_transform(training)
            imputed = pd.DataFrame(imputed, index=data.index, columns=data.columns)
            resid = (imputed - test)
            rmsd = np.sqrt((resid ** 2).sum() / resid.notnull().sum())
            rmsd_folds.append(rmsd)
        rmsd_folds = pd.concat(rmsd_folds, axis=1)
        nrmsd_folds = rmsd_folds.div(data.mean(), axis=0)
        
        # record stats in adata.var
        cvstats = pd.DataFrame({
            "log_mean_preimputation": np.log10(data.mean()),
            "log_variance_preimputation": np.log10(data.var()),
            "imputation_nrmsd_mean": nrmsd_folds.mean(axis=1),
            "imputation_nrmsd_var": nrmsd_folds.mean(axis=1)
        })

        return cvstats

    def impute_knn(self,
                   n_neighbors: int = 5,
                   weights: Literal["distance", "uniform"] = "distance",
                   cross_validate: bool = True,
                   n_folds: int = 100):
        
        """Imputation for completing missing values using k-Nearest Neighbors.
           Each sample's missing values are imputed using the mean value from
           `n_neighbors` nearest neighbors. Two samples are
           close if the features that neither is missing are close. 

        :param n_neighbors: Number of neighboring samples to use for imputation, defaults to 5
        :type n_neighbors: int, optional
        :param weights: Weight function used in prediction, defaults to 'distance'. Possible values:
            - 'uniform' : uniform weights. All points in each neighborhood are
            weighted equally.
            - 'distance' : weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
        :type weights: Literal["distance", "uniform"], optional
        :param cross_validate: perform k-fold cross-validation, defaults to True
        :type cross_validate: bool, optional
        :param n_folds: number of folds for k-fold cross-validation, defaults to 100
        :type n_folds: int, optional
        """

        logging.info(f"Imputing data with k-Nearest Neighbors, k={n_neighbors}, {weights} weights")

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, keep_empty_features=True)
        if n_folds < 2:
            logging.warning(f"n_folds = {n_folds} is less than 2. Skipping cross-validation.")
            cross_validate = False
        if cross_validate:
            cvstats = self.cross_validate_imputation(imputer=imputer, n_folds=n_folds)
            self.adata.var = self.adata.var.join(cvstats, validate="1:1")
            
        self.adata.X = imputer.fit_transform(self.adata.X)
        self.append_to_history(f"KNN Imputation: n_neighbors = {n_neighbors}, weights = {weights}")
        self.adata.uns["imputation"] = {"method": "knn",
                                        "n_neighbors": n_neighbors,
                                        "weights": weights,
                                        "cross_validation": cross_validate}
        if cross_validate:
            self.adata.uns["imputation"]["n_folds"] = n_folds

    def impute_zeros(self,
                     cross_validate: bool = True,
                     n_folds: int = 100):

        """Imputation by filling missing values with zeros.

        :param cross_validate: perform k-fold cross-validation, defaults to True
        :type cross_validate: bool, optional
        :param n_folds: number of folds for k-fold cross-validation, defaults to 100
        :type n_folds: int, optional
        """
        logging.info(f"Imputing data with zeros")
        
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
        if n_folds < 2:
            logging.warning(f"n_folds = {n_folds} is less than 2. Skipping cross-validation.")
            cross_validate = False
        if cross_validate:
            cvstats = self.cross_validate_imputation(imputer=imputer, n_folds=n_folds)
            self.adata.var = self.adata.var.join(cvstats, validate="1:1")
        self.adata.X = imputer.fit_transform(self.adata.X)
        self.append_to_history(f"Zero imputation")
        self.adata.uns["imputation"] = {"method": "zero",
                                        "cross_validation": cross_validate}
        if cross_validate:
            self.adata.uns["imputation"]["n_folds"] = n_folds

    def model_overdispersed_genes(self, odg_default_spline_degree: int = 3, odg_default_dof: int = 8, max_missingness: float = 0.5):
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
        :param max_missingness: Exclude features with high pre-imputation missingness. Value must be between 0 and 1, defaults to 0.5
        :type max_missingness: float, optional
        """
        data_raw = self.to_df()
        data_normalized = self.to_df(normalized=True)
        
        # create dataframe of per-gene statistics
        self.adata.var["mean"] = data_normalized.mean()
        self.adata.var["rank_mean"] = self.adata.var["mean"].rank()
        self.adata.var["variance"] = data_normalized.var()
        self.adata.var["sd"] = data_normalized.std()
        self.adata.var[["log_mean", "log_variance"]] = np.log10(self.adata.var[["mean", "variance"]])
        self.adata.var["mean_counts"] = data_raw.mean()
        self.adata.var["odscore_excluded"] = ((self.adata.var["missingness"] > max_missingness) |
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
        
    def select_overdispersed_genes_from_genelist(self, genes: Collection[str], min_mean=0):
        """Select overdispersed genes/features using a custom list. Genes/features not present in the dataset are automatically filtered out.

        :param genes: gene list
        :type genes: Collection[str]
        :param min_mean: minimum gene expression for genes to be counted as overdispersed, defaults to 0
        :type min_mean: int, optional
        """
        self.adata.var["selected"] = self.adata.var.index.isin(genes) & (self.adata.var["mean_counts"] >= min_mean)
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
                    "Ensure that you call the `Dataset.model_overdispersed_genes()` first."
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
        self.adata.obs["hvg_all_0"] = (self.to_df(normalized=True).loc[:, selected_genes].sum(axis=1) == 0).astype("str").astype("category")
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
        :rtype: :class:`mosaicmpi.cnmf.cNMF`
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

        # Subset out high-variance genes
        norm_counts = self.adata[:, overdispersed_genes].copy()  # copy to prevent overwrite
        ## Scale genes to unit variance
        if sp.issparse(tpm.X):
            raise NotImplementedError("AnnDatas with sparse matrices are not supported by mosaicMPI yet.")
            # sc.pp.scale(norm_counts, zero_center=False)
            # if np.isnan(norm_counts.X.data).sum() > 0:
            #     raise ValueError('NaNs in normalized counts matrix')                       
        else:
            norm_counts.X /= norm_counts.X.std(axis=0, ddof=1)
            if np.isnan(norm_counts.X).sum().sum() > 0:
                raise ValueError('NaNs in normalized counts matrix')                    

        ## Save a \n-delimited list of the high-variance genes used for factorization
        open(cnmf_obj.paths['nmf_genes_list'], 'w').write('\n'.join(overdispersed_genes))

        if norm_counts.X.dtype != np.float64:
            norm_counts.X = norm_counts.X.astype(np.float64)

        norm_counts.write_h5ad(cnmf_obj.paths['normalized_counts'])

        # save parameters for factorization step
        cnmf_obj.save_nmf_iter_params(*cnmf_obj.get_nmf_iter_params(ks=kvals, n_iter=n_iter, random_state_seed=seed, beta_loss=beta_loss))

        # save parameters in AnnData object
        self.adata.uns["cnmf"] = cnmf_obj.get_nmf_iter_params(ks=kvals, n_iter=n_iter, random_state_seed=seed, beta_loss=beta_loss)[1]  # dict of cnmf parameters
        
        # output dataset with new information on overdispersed genes and cNMF parameters
        self.write_h5ad(os.path.join(cnmf_output_dir, cnmf_name, cnmf_name + ".h5ad"))
        self.append_to_history(f"cNMF parameters added. cNMF inputs initialized in {cnmf_output_dir}/{cnmf_name}")
        return cnmf_obj
    
    def add_cnmf_results(self, cnmf_output_dir, cnmf_name, local_density_threshold: float = None, local_neighborhood_size: float = None):
        """
        After factorization, add completed cNMF results in [cnmf_output_dir]/[cnmf_name] to the dataset object.

        :param cnmf_output_dir: Output directory for cNMF results
        :type cnmf_output_dir: str
        :param cnmf_name: Name of the cNMF results. Files will be output to [cnmf_output_dir]/[cnmf_name]/
        :type cnmf_name: str
        :param local_density_threshold: Threshold for the local density filtering prior to program consensus. Acceptable thresholds are > 0 and <= 2 (2.0 is no filtering). Defaults to None.
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
            raise RuntimeError(f"local_density_threshold of {local_density_threshold} does not match what is in the cNMF result directory: {sensed_ldts}")
        self.adata.uns["ldt"] = ldt
        self.adata.uns["lns"] = local_neighborhood_size
            
        # Import programs
        result_types = {
            "gene_spectra_score": "cnmf_gep_score",
            "gene_spectra_tpm": "cnmf_gep_tpm",
            "spectra": "cnmf_gep_raw"
            }
        for matchstr, result_type in result_types.items():
            logging.info(f"Importing programs: {matchstr}")  
            meta_w = []
            for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.{matchstr}.k_*.{ldt_str}.*txt")):
                k = int(os.path.basename(fn).removeprefix(f"{cnmf_name}.{matchstr}.").split(".")[0].replace("k_", ""))
                w = pd.read_table(fn, index_col=0)
                w.index = str(k) + "." + w.index.astype(str)
                meta_w.append(w)
            meta_w = pd.concat(meta_w, axis=0).T.reindex(self.adata.var.index).rename_axis(["k.program"], axis=1)
            self.adata.varm[result_type] = meta_w

        # Import Usages
        logging.info(f"Importing Usages")  
        usage = []
        for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.usages.k_*.{ldt_str}.*txt")):
            k = int(os.path.basename(fn).removeprefix(f"{cnmf_name}.usages.").split(".")[0].replace("k_", ""))
            h = pd.read_table(fn, index_col=0)
            h.columns = str(k) + "." + h.columns.astype(str)
            usage.append(h)
        self.adata.obsm["cnmf_usage"] = pd.concat(usage, axis=1).sort_index(axis=1).rename_axis(["k.program"], axis=1)
        
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
        Generate dataframe of program usage.

        :param k: If an integer or list of integers, returns usages only for specified ranks. Otherwise, returns usage of all programs across ranks. Defaults to None
        :type k: int, optional
        :param discretize: Discretizes the usage matrix such that for each value of k, each sample has usage of only 1 program (the one with the maximum usage). Defaults to False
        :type discretize: bool, optional
        :param normalize: Normalize the program usage matrix such that for each value of k, usage of all programs sums to 1. Defaults to False
        :type normalize: bool, optional
        :return: observation × program matrix
        :rtype: pd.DataFrame
        """
        df = self.adata.obsm["cnmf_usage"].copy()
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(".").to_list())
        df.columns = df.columns.set_levels([l.astype("int") for l in df.columns.levels])
        if normalize:
            normalized = []
            for _, subdf in df.T.groupby(level=0):
                normalized.append(subdf.div(subdf.sum()))
            df = pd.concat(normalized).T
        if discretize:
            discretized = []
            for _, subdf in df.T.groupby(level=0):
                discretized.append(subdf.eq(subdf.max()).astype(int))
            df = pd.concat(discretized).T        
        if k is not None:
            df = df.loc[:, k]
        df = df.sort_index(axis=1)   
        return df

    def get_programs(self,
                 k: Union[int, Iterable] = None,
                 type="cnmf_gep_score"
                 ) -> pd.DataFrame:
        """
        Get feature scores for programs.

        :param k: If an integer or list of integers, returns programs only for specified ranks. Otherwise, returns programs from all ranks. Defaults to None
        :type k: Union[int, Iterable], optional
        :param type: "cnmf_gep_score" or "cnmf_gep_tpm", defaults to "cnmf_gep_score"
        :type type: str, optional
        :return: features × programs matrix
        :rtype: pd.DataFrame
        """
        assert self.has_cnmf_results
        df = self.adata.varm[type].copy()
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(".").to_list())
        df.columns = df.columns.set_levels([l.astype("int") for l in df.columns.levels])
        if isinstance(k, (int, np.integer)):
            df = df.loc[:, k]
            df = df.rename_axis(columns=["program"]).sort_index(axis=1)
        elif isinstance(k, Iterable):
            df = df.loc[:, k]
            df = df.rename_axis(columns=["k", "program"]).sort_index(axis=1)
        return df
    
    def get_approximation(self,
                          k: Optional[int] = None,
                          program_type: Literal["cnmf_gep_tpm", "cnmf_gep_raw"] = "cnmf_gep_tpm"
                         ) -> pd.DataFrame:
        
        """Return the approximated data by multiplying the programs and usage matrices for a given rank (k). Defaults to the highest rank available.

        :param k: rank, defaults to None
        :type k: int, optional
        :return: Approximation of dataset based on programs and usages
        :rtype: pd.DataFrame
        """
        assert self.has_cnmf_results
        if k is None:
            k = self.adata.uns["kvals"].index.max()
        
        approximation = (self.get_usages(k=k)  @ self.get_programs(type=program_type, k=k).T)

        return approximation
        
    def calculate_cnmf_prediction_error(self,
                                       k: Optional[Union[int, Iterable]] = None) -> Union[float, pd.Series]:
        """Calculate cNMF prediction error using the method in the original cNMF package.

        :param k: Specify one or more k values to calculate prediction error. Specify None to calculate for all available k. defaults to None
        :type k: Optional[Union[int, Iterable]], optional
        :raises ValueError: if invalid k is specified
        :return: scalar or vector
        :rtype: _type_
        """

        from sklearn.decomposition import non_negative_factorization


        if k is None:
            kvals = self.adata.uns["kvals"].index
        elif isinstance(k, Iterable):
            kvals = k
        elif isinstance(k, (int, np.integer)):
            kvals = [k]
        else:
            raise ValueError
        
        norm_counts = self.adata.to_df().loc[:, self.adata.var["selected"]]
        norm_counts /= norm_counts.std(axis=0, ddof=1)

        pred_error = pd.Series(index=kvals, name="cNMF prediction_error")
        for kval in kvals:
            # obtain reconstructed normalized counts matrix by re-fitting usage and computing dot product: usage.dot(spectra)
            median_spectra = self.get_programs(k=kval, type="cnmf_gep_raw").dropna()
            rf_usages, rf_spectra, niter = non_negative_factorization(X=norm_counts.values.astype(np.float64),
                                                    alpha_H=0.0, alpha_W=0.0, beta_loss="kullback-leibler", init="random",
                                                    l1_ratio=0.0, max_iter=1000, solver="mu", tol=0.0001, n_components=kval, H=median_spectra.T.values.astype(np.float64), update_H=False)

            rf_usages = pd.DataFrame(rf_usages, index=norm_counts.index, columns=median_spectra.T.index)
            rf_pred_norm_counts = rf_usages.dot(median_spectra.T)
            pred_error[kval] = ((norm_counts - rf_pred_norm_counts)**2).sum().sum()

        if isinstance(k, (int, np.integer)):
            return pred_error[k]
        else:
            return pred_error

    def validate_cnmf_prediction_errors(self, tolerance: float = 1e-4) -> pd.DataFrame:
        """Validate the dataset and cNMF solutions for each rank by comparing the
        prediction error values stored in the object [self.adata.uns.kvals] to those calculated from the
        dataset's data matrices [based on self.adata.X and self.adata.varm['cnmf_gep_raw']]. This can be a quick and sensitive way to assess
        that the dataset and the cNMF solutions have not been altered.

        :param tolerance: maximum relative error for any k when computing the prediction error, defaults to 0.0001
        :type tolerance: float, optional
        :raises ValueError: if the maximum relative error exceeds the tolerance
        :return: DataFrame with stored and computed prediction error, and relative error for each rank
        :rtype: pd.DataFrame
        """
 
        pred_error = pd.Series()
        for k in self.adata.uns['kvals'].index:
            pred_error[k] = self.calculate_cnmf_prediction_error(k=k)
        # validate error of the solutions matches cNMF's stored error values
        df = pd.DataFrame({"stored": self.adata.uns['kvals']["prediction_error"],
                           "computed": pred_error})
        df["relative_error"] = (df["computed"] - df["stored"]).abs() / df["stored"]
        
        if (df["relative_error"] > tolerance).any():
            raise ValueError(f"Dataset validation failed using cNMF prediction error tolerance of {tolerance}. "
                             "This can occur when dataset objects are unintentionally altered and the matrices "
                             "are not consistent with the original ones used for factorization.")

        return df
        
    def validate_feature_stats(self, tolerance: float = 1e-4) -> pd.DataFrame:
        """Validate the dataset and cNMF solutions for each rank by comparing the
        calculated feature statistics (mean, SD, variance) stored in the object [self.adata.var] to those calculated from the
        dataset's data matrices [based on self.adata.X]. This can be a quick and sensitive way to assess
        that the dataset's underlying data has not been altered.

        :param tolerance: maximum relative error for any k when computing the prediction error, defaults to 0.0001
        :type tolerance: float, optional
        :raises ValueError: if the maximum relative error exceeds the tolerance
        :return: DataFrame with relative error for each feature
        :rtype: pd.DataFrame
        """
        X = self.to_df()
        calc_sd = X.std()
        calc_var = X.var()
        calc_mean = X.mean()
        stored_sd = self.adata.var["sd"]
        stored_var = self.adata.var["variance"]
        stored_mean = self.adata.var["mean"]

        df = pd.DataFrame({"sd": (calc_sd - stored_sd) / stored_sd,
                    "variance": (calc_var - stored_var) / stored_var,
                    "mean": (calc_mean - stored_mean) / stored_mean})

        if df.abs().max().max() > tolerance:
            raise ValueError(f"Dataset validation failed using feature-wise SD, variance, and/or mean with a relative error within {tolerance}. "
                            "This can occur when dataset objects are unintentionally altered and the matrices "
                            "are not consistent with the original ones used for factorization.")
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
        if include_categorical:
            for col in df.select_dtypes("category").columns:
                if "nan" in df[col].cat.categories:
                    df[col] = df[col].cat.remove_categories("nan")  # replaces category "nan" with actual np.nan missing values.
        return df
    
    def get_category_overrepresentation(self,
                                        layer: str,
                                        truncate_negative: bool = True,
                                        subset_categories: Collection[str] = None
                                        ) -> pd.DataFrame:
        """Calculate Pearson residual of chi-squared test, associating programs for each rank (k) to categories of samples/observations. By default, truncates negative values.

        :param layer: name of categorical data layer
        :type layer: str
        :param truncate_negative: Truncate negative residuals to 0, defaults to True
        :type truncate_negative: bool, optional
        :param subset_categories: Provide a subset of categories for calculating overrepresentation
        :type subset_categories: Collection[str]
        :return: category × program matrix of overrepresentation values
        :rtype: pd.DataFrame
        """
        usage = self.get_usages(normalize=True).copy()
        sample_to_class = self.get_metadata_df()[layer]
        if subset_categories is not None:
            sample_to_class[~sample_to_class.isin(subset_categories)] = np.nan
        usage.index = usage.index.map(sample_to_class)
        observed = usage.groupby(level=0, observed=True).sum()
        observed = observed[observed.sum(axis=1) > 0]
        n_categories = observed.shape[0]
        if n_categories < 2:
            if layer != "hvg_all_0":
                logging.warning(f"Overrepresentation could not be calculated for layer '{layer}', as only {n_categories} categories were found in the data. "
                                f"Note that empty values in the metadata are not considered a category. "
                                f"Overrepresentation cannot be calculated with fewer than 2 categories for each layer. ")
            return pd.DataFrame(np.nan, index = observed.index, columns=observed.columns)
        expected = []
        for k, obs_k in observed.T.groupby(level=1):
            exp_k = pd.DataFrame(obs_k.sum(axis=1)) @ pd.DataFrame(obs_k.sum(axis=0)).T / obs_k.sum().sum()
            expected.append(exp_k)
        expected = pd.concat(expected).T
        chisq_resid = (observed - expected) / np.sqrt(expected)  # pearson residual of chi-squared test of contingency table
        if truncate_negative:
            chisq_resid = chisq_resid.clip(lower=0)
        return chisq_resid
    
        
    def get_metadata_correlation(self, 
                                 layer: str,
                                 method: str = "pearson"
                                 ) -> pd.Series:
        """Calculate correlation of program usage to numerical metadata across samples/observations.

        :param layer: name of numerical data layer
        :type layer: str
        :param method: Correlation method: "pearson", "spearman", or "kendall". Defaults to "pearson"
        :type method: str, optional
        :return: correlation coefficient of program usage with metadata
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