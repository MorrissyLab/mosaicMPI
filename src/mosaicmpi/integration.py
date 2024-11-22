
from .dataset import Dataset
from . import cpus_available


from collections.abc import Iterable, Collection, Mapping
from typing import Union, Optional, Dict
import logging

import numpy as np
import pandas as pd
from .nancorrmp import NaNCorrMp

class Integration():
    
    def __init__(self,
                 datasets: dict[str, Dataset],
                 corr_method: str = "pearson",
                 max_median_corr: float =  0,
                 negative_corr_quantile: float = 0.95,
                 k_subset: Union[Collection[int], Dict[str, Collection[int]]] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
                 ):
        """
        Integrate multiple datasets together.

        :param datasets: dictionary of name: Dataset pairs.
        :type datasets: dict[str, :class:`~mosaicmpi.dataset.Dataset`]
        :param corr_method: Correlation method: "pearson", "spearman", or "kendall", defaults to "pearson"
        :type corr_method: str, optional
        :param max_median_corr: Threshold for rank reduction procedure, relevant only for datasets where programs tend to be highly correlated.
            This procedure reduces the maximum rank included for a dataset until the median of the correlation distribution is below the threshold. Defaults to 0
        :type max_median_corr: float, optional
        :param negative_corr_quantile: Threshold for network-based integration, between 0 and 1, with 1 resulting in fewer edges in the network. Defaults to 0.95
        :type negative_corr_quantile: float, optional
        :param k_subset: k-values to use for integration. Either a Collection of integers, or a dict specifying k-values separately for each dataset. Defaults
            to (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
        :type k_subset: Union[Collection[int], Dict[str, Collection[int]]], optional
        """
        if isinstance(datasets, dict):
            self.datasets = datasets
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
        self.k_table = pd.concat(combined, axis=1).rename_axis(index="k")
        # compute correlations
        self.compute_corr(method=self.corr_method)
        # rank-reduction for highly autocorrelated programs 
        self.filter_programs_rank_reduction(max_median_corr=self.max_median_corr)
        # subset k-values for more sparsely separated k-values to reduce network size
        self.select_k_values(k_subset=self.k_subset)
        # use negative correlation quantile to threshold correlations
        self.compute_pairwise_thresholds(negative_corr_quantile=self.negative_corr_quantile)

    
    @property
    def n_datasets(self) -> int:
        """
        Get the number of datasets in the integration

        :return: number of datasets
        :rtype: int
        """
        return len(self.datasets)        
    
    @property
    def selected_k(self) -> dict:
        """
        Gets the values of k selected for integration.

        :return: dictionary of ranks for each dataset
        :rtype: dict
        """
        by_dataset = {}
        for dataset_name in self.datasets:
            k = self.k_table[dataset_name, "selected_k"]
            k = k[k].index.to_list()
            by_dataset[dataset_name] = k
        return by_dataset
    
    @property
    def sample_to_patient(self) -> dict:
        """
        :return: Series with dataset and sample ID index. Values are the patient from which the samples/observations were derived.
        :rtype: pd.Series
        """
        mapping = {}
        for dataset_name, dataset in self.datasets.items():
            if dataset.patient_id_col is not None:
                for sample_id, patient_id in dataset.get_metadata_df()[dataset.patient_id_col].items():
                    mapping[(dataset_name, sample_id)] = (dataset_name, patient_id)
        if mapping:
            return pd.Series(mapping)
        else:
            return None
    
    
    def get_corr_matrix_lowertriangle(self, max_k_filter=False, selected_k_filter=False, quantile_transformation=False) -> pd.DataFrame:
        """
        Get the lower triangular correlation matrix for building the correlation network.

        :param max_k_filter: Apply the max_k_filter, defaults to False
        :type max_k_filter: bool, optional
        :param selected_k_filter: Apply the selected_k_filter, defaults to False
        :type selected_k_filter: bool, optional
        :param quantile_transformation: transform correlations using the quantile transformation, defaults to False
        :type quantile_transformation: bool, optional
        :return: program × program correlation matrix with diagonal and upper triangle set to NaN
        :rtype: pd.DataFrame
        """
        mask = np.tril(np.ones(self.corr_matrix.shape), k=-1).astype(bool)
        tril = self.corr_matrix.where(mask)
        
        # get rank filters
        if max_k_filter:
            maxk_filtered_index = pd.MultiIndex.from_tuples([program for program in tril.index if self.k_table.loc[program[1], (program[0], "max_k_filter_pass")]])
        if selected_k_filter:
            selected_k_index = pd.MultiIndex.from_tuples([program for program in tril.index if self.k_table.loc[program[1], (program[0], "selected_k")]])

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
    
    def get_programs(self, type="cnmf_gep_score") -> pd.DataFrame:
        """
        Get programs.

        :param type: "cnmf_gep_score" or "cnmf_gep_tpm", defaults to "cnmf_gep_score"
        :type type: str, optional
        :return: features × programs matrix
        :rtype: pd.DataFrame
        """
        program_matrix = {dataset_name: dataset.get_programs(type=type) for dataset_name, dataset in self.datasets.items()}
        program_matrix = pd.concat(program_matrix, axis=1).sort_index(axis=0).sort_index(axis=1)
        return program_matrix

    def get_usages(self, discretize=False, normalize=False) -> pd.DataFrame:
        """
        Calculate usage of each program in each dataset and sample/observation.

        :param discretize: Discretizes the usage matrix such that for each value of k, each sample has usage of only 1 program (the one with the maximum usage). Defaults to False
        :type discretize: bool, optional
        :param normalize: Normalize the program usage matrix such that for each value of k, usage of all programs sums to 1. Defaults to False
        :type normalize: bool, optional
        :return: category × programs matrix of overrepresentation values
        :rtype: pd.DataFrame
        """
        usages = {dataset_name: dataset.get_usages(discretize=discretize, normalize=normalize)
                         for dataset_name, dataset in self.datasets.items()}
        for dsname, usage in usages.items():
            usage.index = pd.MultiIndex.from_product([[dsname], usage.index])  # add dataset to sample_name index
        usages = pd.concat(usages, axis=1).sort_index(axis=0).sort_index(axis=1)
        return usages
    
    def compute_corr(self, method="pearson", cpus=cpus_available):
        """Computes correlation matrix of all programs in the integration from all datasets.

        :param method: Correlation method. Values can be "pearson", "spearman", and "kendall". Defaults to "pearson"
        :type method: str, optional
        :param cpus: Number of CPUs to use for nancorrmp (only available for "pearson" method), defaults to all available CPUs
        :type cpus: int, optional
        """
        if method == "pearson":
            cpu_string = "all" if cpus == -1 else str(cpus)
            logging.info(f"Calculating Pearson correlation matrix using {cpu_string} CPUs.")
            corr = NaNCorrMp.calculate(self.get_programs(), n_jobs=cpus)
        elif method == "spearman":
            logging.info(f"Calculating Spearman correlation matrix using 1 CPU.")
            corr = self.get_programs().corr(method)
        else:
            raise ValueError(f"{method} is not a valid correlation method")        
        
        self.corr_matrix = corr

    def filter_programs_rank_reduction(self,
                                   max_median_corr: float = 0.0
                                   ) -> None:
        """
        Filter programs using the rank-reduction procedure, relevant only for datasets where programs tend to be highly correlated.
        This procedure reduces the maximum rank included for a dataset until the median of the correlation distribution is below the max_median_corr threshold.

        :param max_median_corr: Threshold, defaults to 0
        :type max_median_corr: float, optional
        """
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

    def select_k_values(self,
                        k_subset: Union[Collection[int], Dict[str, Collection[int]]] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
                        exclude_unshared_k_values: bool = False,
                        ) -> None:
        """
        Select k-values for integration.

        :param k_subset: k-values to use for integration. Either a Collection of integers, or a dict specifying k-values separately for each dataset.
            Defaults to (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
        :type k_subset: Union[Collection[int], Dict[str, Collection[int]]]
        :param exclude_unshared_k_values: in addition to the k_subset, also exclude k-values that are not shared with all datasets.
        :type exclude_unshared_k_values: bool, optional

        """
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
        if exclude_unshared_k_values:
            shared_selected_k = self.k_table.xs("selected_k", axis=1, level=1).all(axis=1)
            for dataset_name in self.datasets:
                self.k_table[dataset_name, "selected_k"] = shared_selected_k

        self.k_table = self.k_table.sort_index(axis=1)

    
    def compute_pairwise_thresholds(self, negative_corr_quantile: float = 0.95) -> None:
        """
        Compute thresholds for each dataset and dataset pair based on the correlation distribution of programs. This dynamic thresholding enables integration and balances the influence of each dataset in the network.

        :param negative_corr_quantile: Threshold for network-based integration, between 0 and 1, with 1 resulting in fewer edges in the network. Defaults to 0.95
        :type negative_corr_quantile: float, optional
        """
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
        """Get node counts before and after various node and edge filters.

        :return: Summary table of node counts
        :rtype: pd.DataFrame
        """
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
                                mask = pd.concat({dataset_col: pd.concat({dataset_row: mask})}, axis=1)  # adds the dataset levels back for matched indexes
                                df_filt[~mask] = np.nan

                else:
                    df_filt = df

                results = {}
                for dataset_name, subdf in df_filt.groupby(level=0):
                    results[dataset_name] = subdf.shape[0]
                nodetable[(node_filter, edge_filter)] = results

        nodetable = pd.DataFrame(nodetable).rename_axis(columns=["Node filter", "Edge Filter"])
        return nodetable
    
    def get_metadata_df(self,
                        include_categorical: bool = True,
                        include_numerical: bool = True,
                        prepend_dataset_column: bool = False,
                        subset_datasets: Optional[Union[str, Iterable[str]]] = None
                        ) -> pd.DataFrame:
        """Get sample/observation metadata for all datasets.

        :param include_categorical: Include categorical metadata layers, defaults to True
        :type include_categorical: bool, optional
        :param include_numerical: Include numerical metadata layers, defaults to True
        :type include_numerical: bool, optional
        :param prepend_dataset_column: Prepend dataframe with dataset name column, defaults to False
        :type prepend_dataset_column: bool, optional
        :param subset_datasets: dataset name or iterable of dataset names to subset the results, defaults to None
        :type subset_datasets: str or Iterable[str], optional
        :return: observations × metadata matrix
        :rtype: pd.DataFrame
        """

        if subset_datasets is None:
            subset_datasets = self.datasets.keys()
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        elif isinstance(subset_datasets, Iterable):
            pass
        else:
            raise ValueError


        df = {}
        for dataset_name in subset_datasets:
            df[dataset_name] = self.datasets[dataset_name].get_metadata_df(include_categorical=include_categorical,
                                                       include_numerical=include_numerical)
        if df:
            df = pd.concat(df, axis=0)
        
        if prepend_dataset_column:
            df.insert(0, "Dataset", df.index.get_level_values(0))
        return df
    
    def get_features_overlap_table(self):

        all_features = set()
        for ds in self.datasets.values():
            all_features |= set(ds.adata.var_names)
        df = pd.DataFrame(False, index=sorted(list(all_features)), columns=self.datasets.keys())
        for dsname, ds in self.datasets.items():
            df.loc[ds.adata.var_names, dsname] = True
        return df

    def get_overdispersed_features_overlap_table(self):

        all_features = set()
        for ds in self.datasets.values():
            all_features |= set(ds.adata.var_names)
        df = pd.DataFrame(False, index=sorted(list(all_features)), columns=self.datasets.keys())
        for dsname, ds in self.datasets.items():
            df.loc[ds.overdispersed_genes, dsname] = True
        return df
    
    def get_category_overrepresentation(self,
                                        layer: str,
                                        subset_datasets: Optional[Union[str, Iterable[str]]] = None,
                                        truncate_negative: bool = True,
                                        subset_categories: Collection[str] = None
                                        ) -> pd.DataFrame:
        """
        Calculate Pearson residual of chi-squared test, associating programs for each rank (k) to categories of samples/observations. By default, truncates negative values.

        :param layer: name of categorical data layer
        :type layer: str
        :param subset_datasets: dataset name or iterable of dataset names to subset the results, defaults to None
        :type subset_datasets: str or Iterable[str], optional
        :param truncate_negative: Truncate negative residuals to 0, defaults to True
        :type truncate_negative: bool, optional
        :param subset_categories: Provide a subset of categories for calculating overrepresentation
        :type subset_categories: Collection[str]
        :return: category × programs matrix of overrepresentation values
        :rtype: pd.DataFrame
        """

        if subset_datasets is None:
            subset_datasets = self.datasets.keys()
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        elif isinstance(subset_datasets, Iterable):
            pass
        else:
            raise ValueError
        
        combined = {}
        for dataset_name in subset_datasets:
            if layer in self.datasets[dataset_name].get_metadata_df():
                combined[dataset_name] = self.datasets[dataset_name].get_category_overrepresentation(layer=layer,
                                                                                                     truncate_negative=truncate_negative,
                                                                                                     subset_categories=subset_categories)
        combined = pd.concat(combined, axis=1)
        return combined
    
    def get_metadata_correlation(self,
                                 layer: str,
                                 subset_datasets = None,
                                 method: str = "pearson") -> pd.Series:
        
        """Calculate correlation of programs usage to numerical metadata across samples/observations.

        :param layer: name of numerical data layer
        :type layer: str
        :param subset_datasets: dataset name or iterable of dataset names to subset the results, defaults to None
        :type subset_datasets: str or Iterable[str], optional
        :param method: Correlation method: "pearson", "spearman", or "kendall". Defaults to "pearson"
        :type method: str, optional
        :return: correlation of programs to metadata
        :rtype: pd.Series
        """
        if subset_datasets is None:
            subset_datasets = self.datasets.keys()
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        elif isinstance(subset_datasets, Iterable):
            pass
        else:
            raise ValueError
        
        combined = {}
        for dataset_name in subset_datasets:
            if layer in self.datasets[dataset_name].get_metadata_df():
                combined[dataset_name] = self.datasets[dataset_name].get_metadata_correlation(layer=layer, method=method)
        combined = pd.concat(combined)
        return combined