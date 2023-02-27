
import logging
import collections
import typing
import numpy as np
import pandas as pd
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import cnmfsns as cn

class Integration():
    
    def __init__(self,
                 config: cn.Config(),
                 cpus: int = cn.cpus_available):
        self.config = config
        self.datasets = {
            dataset_name: cn.Dataset.from_h5ad(dataset_params["filename"]) for dataset_name, dataset_params in config.datasets.items()
        }
        self.dataset_colors = {
            dataset_name: dataset_params["color"] for dataset_name, dataset_params in config.datasets.items()
        }

        # create the k-table
        combined = {}
        for dataset_name, dataset in self.datasets.items():
            kvals = dataset.adata.uns["kvals"].copy()
            kvals["cNMF result"] = True
            combined[dataset_name] = kvals
        self.k_table = pd.concat(combined, axis=1).rename_axis("k", axis=0)
        
        # compute correlations
        self.compute_corr(method=self.config.integrate["corr_method"])
        
        # rank-reduction for highly autocorrelated GEPs 
        self.filter_geps_rank_reduction(config.integrate["max_median_corr"])
        
        # subset k-values for more sparsely separated k-values to reduce network size
        self.select_k_values()
        
        # use negative correlation quantile to threshold correlations
        self.compute_pairwise_thresholds(negative_corr_quantile=config.integrate["negative_corr_quantile"])
        
    @property
    def n_datasets(self):
        return len(self.datasets)
    
    def get_corr_matrix_lowertriangle(self, max_k_filter=False, selected_k_filter=False):
        mask = np.tril(np.ones(self.corr_matrix.shape), k=-1).astype(bool)
        tril = self.corr_matrix.where(mask)
        
        # Filter correlations using dataset-specific max_k thresholds
        if max_k_filter:
            maxk_filtered_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if self.k_table.loc[gep[1], (gep[0], "max_k_filter_pass")]])
            tril = tril.loc[maxk_filtered_index, maxk_filtered_index]
        if selected_k_filter:    
            selected_k_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if self.k_table.loc[gep[1], (gep[0], "selected_k")]])
            tril = tril.loc[selected_k_index, selected_k_index]
        return tril
    
    def get_geps(self, type="cnmf_gep_score"):
        gep_matrix = {dataset_name: dataset.get_geps(type=type) for dataset_name, dataset in self.datasets.items()}
        gep_matrix = pd.concat(gep_matrix, axis=1).sort_index(axis=0).sort_index(axis=1)
        return gep_matrix
    
    def compute_corr(self, method="pearson", cpus=cn.cpus_available):
        if method == "pearson":
            try:
                from nancorrmp.nancorrmp import NaNCorrMp
            except ImportError:
                logging.info(f"nancorrmp not installed. Calculating Pearson correlation matrix using 1 CPU.")
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

    def filter_geps_rank_reduction(self, max_median_corr=0):
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

    def select_k_values(self):
        for dataset_name, dataset_params in self.config.datasets.items():
            k_param = set()
            for k_entry in dataset_params["selected_k"]:
                if isinstance(k_entry, int):
                    k_param.add(k_entry)
                elif isinstance(k_entry, collections.abc.Collection):
                    assert len(k_entry) == 3
                    for k in range(k_entry[0], k_entry[1]+1, k_entry[2]):
                        k_param.add(k)
            
            self.k_table[(dataset_name, "selected_k")] = (
                self.k_table[(dataset_name, "cNMF result")] &
                self.k_table[(dataset_name, "max_k_filter_pass")] &
                self.k_table.index.isin(k_param)
                )
            final_selected_k = self.k_table[(dataset_name, "selected_k")]
            final_selected_k = sorted(final_selected_k[final_selected_k].index.to_list())
            self.config.datasets[dataset_name]["selected_k"] = final_selected_k
        self.k_table = self.k_table.sort_index(axis=1)
    
    def compute_pairwise_thresholds(self, negative_corr_quantile = 0.95):
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

        self.pairwise_thresholds = pd.DataFrame.from_records(pairwise_thresholds).set_index(["dataset_row", "dataset_col"])
        
    def get_node_table(self):
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
                                min_corr = thresholds.loc[(dataset_row, dataset_col)].values[0]
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