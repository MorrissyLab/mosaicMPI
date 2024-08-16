from collections.abc import Collection, Iterable, Mapping
from typing import Union, Optional
from copy import deepcopy
import os
from types import SimpleNamespace

import tomli
import tomli_w

# Default parameters for the config files are used when missing as input to mosaicmpi integrate

config_defaults = {
    "plot_formats": ["pdf", "png"],
    "corr_method": "pearson",
    "max_median_corr": 0,
    "negative_corr_quantile": 0.95,
    "subset_nodes": "none",
    "edge_weight": "none",
    "community_algorithm": "greedy_modularity",
    "community_algorithm_parameters": {
        "greedy_modularity": {
            "resolution": 2,
            "best_n": "none"},
        "leiden": {
            "resolution": 0.01
        }
    },
    "community_pruning": {
        "min_nodes": 1,
        "min_datasets": 1, 
        "min_nodes_per_dataset": 0
    },
    "layout_algorithm": "community_weighted_spring",   # "neato", "spring", "community_weighted_spring"
    "community_layout_algorithm": "spring",   # "neato", "spring", "community_weighted_spring"
    "layouts": {  # parameters for each layout algorithm
        "neato": {},
        "spring": {},
        "community_weighted_spring": {
            "shared_community_weight": 100,
            "shared_dataset_weight": 1
        }
    },
    "community_layouts": {  # parameters for each layout algorithm
        "centroid": {},
        "neato": {},
        "spring": {
            "k": 8
        }
    },
    "node_size": 30,
    "plot_size_program": [10, 10],
    "pie_size_program": 50,
    "pie_size_community": 50,
    "plot_size_community": [4, 4],
    "max_cells_per_heatmap_dimension": 500,
    "save_network_as_pkl": True,
    "datasets": {}
}
dataset_defaults = {"k_subset": list(range(2, 10)) + list(range(10, 65, 5))}

def recursive_update(d, u):
    d = deepcopy(d)
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class Config(SimpleNamespace):

    def __init__(self, /, **kwargs):
        self.__dict__.update(recursive_update(config_defaults, kwargs))
        new_datasets = {}
        for dataset_name, dataset_parameters in self.datasets.items():
            new_datasets[dataset_name] = recursive_update(dataset_defaults, dataset_parameters)
        self.datasets = new_datasets

    @classmethod
    def from_toml(cls, toml_file):
        with open(toml_file, 'rb') as f:
            return cls(**tomli.load(f))
    
    @classmethod
    def from_h5ad_files(cls, h5ad_files):
        c = {
            "datasets": {os.path.basename(fn).replace(".h5ad",""): {"filename": fn} for fn in h5ad_files}
        }
        return cls(**c)

    def to_toml(self, toml_file, section_subset=None):
        if section_subset is None:
            to_write = self.__dict__
        else:
            to_write = {k:v for k, v in self.__dict__.items() if k in section_subset}
        with open(toml_file, "wb") as f:
            tomli_w.dump(self.__dict__, f)