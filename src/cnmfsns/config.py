from collections.abc import Collection, Iterable, Mapping
from typing import Union, Optional
from copy import deepcopy
import os
from types import SimpleNamespace

import tomli
import tomli_w

# Default parameters for the config files are used when missing as input to cnmfsns integrate

config_defaults = {
    "colormaps": {
        "diverging": "RdBu_r",
        "sequential": "YlOrRd"
    },
    "integrate": {
        "corr_method": "pearson",
        "max_median_corr": 0,
        "negative_corr_quantile": 0.95,
        },
    "sns": {
        "subset_nodes": "none",
        "edge_color": "#bbbbbb20",
        "edge_weight": "none",
        "community_algorithm": "greedy_modularity",
        "communities": {
            "greedy_modularity": {
                "resolution": 2,
                "best_n": "none",
                "resolution_sweep": [0.5, 0.75, 1, 1.25, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0]
            },
            "leiden": {
                "resolution": 0.01,
                "resolution_sweep": [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
            }
        },
        "layout_algorithm": "community_weighted_spring",   # "neato", "spring", "community_weighted_spring"
        "layouts": {  # parameters for each layout algorithm
            "neato": {},
            "spring": {},
            "community_weighted_spring": {
                "within_community": 100,
                "within_dataset": 1
            }
        },
        "node_size": 30,
        "plot_size_gep": [10, 10],
        "pie_size_gep": 50,
        "pie_size_community": 50,
        "plot_size_community": [4, 4],
        },
    "datasets": {},
    "metadata_colors": {"missing_data": "#dddddd"},
    "metadata_colors_group": {}
}
dataset_defaults = {}

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