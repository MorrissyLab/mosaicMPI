
from .dataset import Dataset
from . import cpus_available


from collections.abc import Iterable, Collection, Mapping
from typing import Union, Optional
import logging

import numpy as np
import pandas as pd

class Integration():
    
    def __init__(self,
                 datasets: Union[dict[str, Dataset], Collection[Dataset]],
                 corr_method: str = "pearson",
                 max_median_corr: float =  0,
                 negative_corr_quantile: float = 0.95,
                 k_subset: Union[Collection, dict] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
                 ):
        
        if isinstance(datasets, dict):
            self.datasets = datasets
        elif isinstance(datasets, Collection):
            self.datasets = {dataset.name: dataset for dataset in datasets}
        else:
            raise ValueError
        
        self.corr_method = corr_method
        self.max_median_corr = max_median_corr
        self.negative_corr_quantile = negative_corr_quantile
        self.k_subset = k_subset
        
        # check that all datasets have been factorized
        unfactorized_datasets = [dataset_name for dataset_name, dataset in self.datasets.items() if not dataset.has_cnmf_results]
        if unfactorized_datasets:
            err_str = ", ".join(unfactorized_datasets)
            raise RuntimeError(f"The following datasets are not factorized for any values of k: {err_str}")
        
        # create the k-table
        combined = {}
        for dataset_name, dataset in self.datasets.items():
            kvals = dataset.adata.uns["kvals"].copy()
            kvals["cNMF result"] = True
            combined[dataset_name] = kvals
        self.k_table = pd.concat(combined, axis=1).rename_axis("k", axis=0)
        # compute correlations
        self.compute_corr(method=self.corr_method)
        # rank-reduction for highly autocorrelated GEPs 
        self.filter_geps_rank_reduction(max_median_corr=self.max_median_corr)
        # subset k-values for more sparsely separated k-values to reduce network size
        self.select_k_values(k_subset=self.k_subset)
        # use negative correlation quantile to threshold correlations
        self.compute_pairwise_thresholds(negative_corr_quantile=self.negative_corr_quantile)

    
    @property
    def n_datasets(self):
        return len(self.datasets)        
    
    @property
    def selected_k(self) -> dict:
        by_dataset = {}
        for dataset_name in self.datasets:
            k = self.k_table[dataset_name, "selected_k"]
            k = k[k].index.to_list()
            by_dataset[dataset_name] = k
        return by_dataset
    
    @property
    def sample_to_patient(self) -> dict:
        mapping = {}
        for dataset_name, dataset in self.datasets.items():
            if dataset.patient_id_col is not None:
                for sample_id, patient_id in dataset.adata.obs[dataset.patient_id_col].items():
                    mapping[(dataset_name, sample_id)] = (dataset_name, patient_id)
        if mapping:
            return pd.Series(mapping)
        else:
            return None
    
    
    def get_corr_matrix_lowertriangle(self, max_k_filter=False, selected_k_filter=False, quantile_transformation=False):
        mask = np.tril(np.ones(self.corr_matrix.shape), k=-1).astype(bool)
        tril = self.corr_matrix.where(mask)
        
        # get rank filters
        if max_k_filter:
            maxk_filtered_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if self.k_table.loc[gep[1], (gep[0], "max_k_filter_pass")]])
        if selected_k_filter:
            selected_k_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if self.k_table.loc[gep[1], (gep[0], "selected_k")]])

        if quantile_transformation:
            # create quantile version of tril where correlations are replaced by quantile of intra-and inter-dataset correlations
            for ds1 in self.datasets:
                for ds2 in self.datasets:
                    chunk = tril.loc[tril.index.get_level_values(0) == ds1, tril.index.get_level_values(0) == ds2]
                    flattened_ranks = pd.Series(chunk.values.flatten()).rank() - 1
                    flattened_quantiles = (flattened_ranks / flattened_ranks.max()).values
                    quantile_chunk = pd.DataFrame(data=np.reshape(flattened_quantiles, newshape=chunk.values.shape), index=chunk.index, columns=chunk.columns)
                    tril.loc[tril.index.get_level_values(0) == ds1, tril.index.get_level_values(0) == ds2] = quantile_chunk      
        
        # Filter correlations using dataset-specific max_k thresholds
        if max_k_filter:
            tril = tril.loc[maxk_filtered_index, maxk_filtered_index]
        if selected_k_filter:    
            tril = tril.loc[selected_k_index, selected_k_index]
            
        return tril
    
    def get_geps(self, type="cnmf_gep_score"):
        gep_matrix = {dataset_name: dataset.get_geps(type=type) for dataset_name, dataset in self.datasets.items()}
        gep_matrix = pd.concat(gep_matrix, axis=1).sort_index(axis=0).sort_index(axis=1)
        return gep_matrix

    def get_usages(self, discretize=False, normalize=False):
        usages = {dataset_name: dataset.get_usages(discretize=discretize, normalize=normalize)
                         for dataset_name, dataset in self.datasets.items()}
        for dsname, usage in usages.items():
            usage.index = pd.MultiIndex.from_product([[dsname], usage.index])  # add dataset to sample_name index
        usages = pd.concat(usages, axis=1).sort_index(axis=0).sort_index(axis=1)
        return usages
    
    def compute_corr(self, method="pearson", cpus=cpus_available):
        if method == "pearson":
            try:
                from nancorrmp.nancorrmp import NaNCorrMp
            except ImportError:
                logging.info(f"nancorrmp not installed. To improve computation time, install using `pip install nancorrmp`. Calculating Pearson correlation matrix using 1 CPU.")
                corr = self.get_geps().corr(method)
            else:
                cpu_string = "all" if cpus == -1 else str(cpus)
                logging.info(f"nancorrmp found. Calculating Pearson correlation matrix using {cpu_string} CPUs.")
                corr = NaNCorrMp.calculate(self.get_geps(), n_jobs=cpus)
        elif method == "spearman":
            logging.info(f"Calculating Spearman correlation matrix using 1 CPU.")
            corr = self.get_geps().corr(method)
        else:
            raise ValueError(f"{method} is not a valid correlation method")        
        
        self.corr_matrix = corr

    def filter_geps_rank_reduction(self, max_median_corr=0) -> None:
        # Reduces rank (k) value when correlation distribution is skewed towards 1.
        tril = self.get_corr_matrix_lowertriangle()
        for dataset_name in tril.index.levels[0]:
            dscorr = tril.loc[dataset_name, dataset_name]
            kvals = dscorr.index.levels[0].sort_values(ascending=False)
            max_kval_medians = []
            for max_kval in kvals:
                rankreduced = dscorr.loc[dscorr.index.get_level_values(0) <= max_kval, dscorr.columns.get_level_values(0) <= max_kval]
                median_corr = np.nanmedian(rankreduced.values)
                max_kval_medians.append(median_corr)
            max_kval_medians = pd.Series(max_kval_medians, index=kvals)
            max_k_threshold = None
            for max_k, median_corr in max_kval_medians.items():
                max_k_threshold = max_k
                if median_corr <= max_median_corr:
                    break
            new_columns = pd.DataFrame({"max_k_median_corr": max_kval_medians, "max_k_filter_pass": (max_kval_medians.index.to_series() <= max_k_threshold)})
            new_columns = pd.concat({dataset_name: new_columns}, axis=1)
            self.k_table = self.k_table.merge(new_columns, how="outer", left_index=True, right_index=True)

    def select_k_values(self, k_subset) -> None:
        for dataset_name in self.datasets:
            if isinstance(k_subset, dict):
                ds_k_subset = k_subset[dataset_name]
            elif isinstance(k_subset, Collection):
                ds_k_subset = k_subset
                
            # add column with selected k
            self.k_table[(dataset_name, "selected_k")] = (
                self.k_table[(dataset_name, "cNMF result")] &
                self.k_table[(dataset_name, "max_k_filter_pass")] &
                self.k_table.index.isin(ds_k_subset)
                )
        self.k_table = self.k_table.sort_index(axis=1)

    
    def compute_pairwise_thresholds(self, negative_corr_quantile = 0.95) -> None:
        # Filter correlations using dataset-specific max_k thresholds
        tril = self.get_corr_matrix_lowertriangle(max_k_filter=True, selected_k_filter=False)
        pairwise_thresholds = []
        for row, dataset_row in enumerate(tril.index.levels[0]):
            for col, dataset_col in enumerate(tril.columns.levels[0]):
                distr = tril.loc[dataset_row, dataset_col].values.flatten()

                if not all(np.isnan(distr)):
                    pairwise_thresholds.append({
                        "dataset_row": dataset_row,
                        "dataset_col": dataset_col,
                        "threshold": -np.quantile(distr[distr < 0], q=1-negative_corr_quantile)
                    })

        self.pairwise_thresholds = pd.DataFrame.from_records(pairwise_thresholds).set_index(["dataset_row", "dataset_col"])["threshold"]
        
    def get_node_table(self) -> pd.DataFrame:
        # Table with node stats
        nodetable = {}
        node_filters = {
            "none": self.get_corr_matrix_lowertriangle(),
            "maxk": self.get_corr_matrix_lowertriangle(max_k_filter=True),
            "selectedk": self.get_corr_matrix_lowertriangle(selected_k_filter=True)
        }
        for node_filter, df in node_filters.items():
            for edge_filter, thresholds in (("none", None), ("mincorr", self.pairwise_thresholds)):
                
                if thresholds is not None:
                    df_filt = df.copy(deep=True)
                    # apply pairwise thresholds
                    for dataset_row in df.index.levels[0]:
                        for dataset_col in df.columns.levels[0]:
                            if (dataset_row, dataset_col) in thresholds.index:
                                min_corr = thresholds.loc[(dataset_row, dataset_col)]
                                mask = df_filt.loc[dataset_row, dataset_col] < min_corr
                                df_filt.loc[dataset_row, dataset_col][~mask] = np.NaN
                                df_filt.dropna(axis=0).dropna(axis=1)
                else:
                    df_filt = df

                results = {}
                for dataset_name, subdf in df_filt.groupby(axis=0, level=0):
                    results[dataset_name] = subdf.shape[0]
                nodetable[(node_filter, edge_filter)] = results

        nodetable = pd.DataFrame(nodetable)
        nodetable.columns.rename(["Node filter", "Edge Filter"], inplace=True)
        return nodetable
    
    def get_metadata_df(self,
                        include_categorical: bool = True,
                        include_numerical: bool = True,
                        prepend_dataset_column: bool = False
                        ) -> pd.DataFrame:
        df = {}
        for dataset_name, dataset in self.datasets.items():
            df[dataset_name] = dataset.get_metadata_df(include_categorical=include_categorical,
                                                       include_numerical=include_numerical)
        df = pd.concat(df, axis=0)
        if prepend_dataset_column:
            df.insert(0, "Dataset", df.index.get_level_values(0))
        return df
    
    def get_category_overrepresentation(self,
                                        layer: str,
                                        subset_datasets: Optional[Union[str, Iterable]] = None,
                                        truncate_negative: bool = True) -> pd.DataFrame:
        if subset_datasets is None:
            subset_datasets = self.datasets.keys()
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        else:
            raise ValueError
        
        combined = {}
        for dataset_name in subset_datasets:
            if layer in self.datasets[dataset_name].adata.obs:
                combined[dataset_name] = self.datasets[dataset_name].get_category_overrepresentation(layer=layer, truncate_negative=truncate_negative)
        combined = pd.concat(combined, axis=1)
        return combined
    
    def get_metadata_correlation(self,
                                 layer: str,
                                 subset_datasets = None,
                                 method: str = "pearson") -> pd.Series:
        if subset_datasets is None:
            subset_datasets = self.datasets.keys()
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        else:
            raise ValueError
        
        combined = {}
        for dataset_name in subset_datasets:
            if layer in self.datasets[dataset_name].adata.obs:
                combined[dataset_name] = self.datasets[dataset_name].get_metadata_correlation(layer=layer, method=method)
        combined = pd.concat(combined)
        return combined