import tomli
import tomli_w
import pandas as pd
import numpy as np
import sys
import logging
import collections.abc
from copy import deepcopy
from matplotlib import colors
from datetime import datetime
import distinctipy
import os
from types import SimpleNamespace
from anndata import read_h5ad

# Here are default parameters for the config files

config_defaults = {
    "integration": {
        "name": datetime.now().strftime("cnmfsns_%Y%m%d-%H%M%S"),
        "corr_method": "pearson",
        "max_median_corr": 0.01,
        "negative_corr_quantile": 0.95,
        "layout_algorithm": "neato",
        },
    "datasets": {},
    "metadata_colors": {"missing_data": "#dddddd"}
}
dataset_defaults = {
    "k": [[1, 10, 1], [15, 500, 5]],
    }

def recursive_update(d, u):
    d = deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
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
        return cls(c)

    def to_toml(self, toml_file):
        with open(toml_file, "wb") as f:
            tomli_w.dump(self.__dict__, f)

    def add_missing_dataset_colors(self):
        # check provided dataset colors
        invalid_colors = []
        for name, d in self.datasets.items():
            if "color" in d and not colors.is_color_like(d["color"]):
                invalid_colors.append(d["color"])
        if invalid_colors:
            logging.error(f"Datasets were given these invalid colors: {invalid_colors}. Please use valid matplotlib colors in named, hex, or RGB formats.")
            sys.exit(1)

    def add_missing_metadata_colors(self, metadata_df=None):
        """
        Identify missing colors based on metadata. If metadata_df is provided, categorical columns are used; otherwise, metadata_df is derived from the config datasets.
        """

        # get categorical data for which colors should match
        if metadata_df is None:
            # read from h5ad files
            metadata_df = pd.concat({name: read_h5ad(d["filename"]).obs.select_dtypes(include="category") for name, d in self.datasets.items()})

        # check provided metadata colors
        invalid_colors = []
        for layer, layer_colors in self.metadata_colors.items():
            if isinstance(layer_colors, dict):
                for value, color in layer_colors.items():
                    if not colors.is_color_like(color):
                        invalid_colors.append(color)
        if invalid_colors:
            logging.error(f"Metadata colors included these invalid colors: {invalid_colors}. Please use valid matplotlib colors in named, hex, or RGB formats.")
            sys.exit(1)


        # fill in missing values with random colors distinct from existing colors
        for layer, allvalues in metadata_df.items():
            if layer in self.metadata_colors:
                existing_values = set(self.metadata_colors[layer].keys())
                existing_colors = set(colors.to_rgb(self.metadata_colors[layer][val]) for val in existing_values)
            else:
                existing_values = set()
                existing_colors = set()
            if np.NaN in allvalues:
                existing_colors.add(self.metadata_colors["missing_data"])
            missing_values = set(allvalues.dropna().unique()) - existing_values
            if missing_values:
                logging.info(f"Choosing distinct colors for metadata layer {layer}")
                if layer not in self.metadata_colors:
                    self.metadata_colors[layer] = {}
                new_colors = distinctipy.get_colors(len(missing_values), exclude_colors=list(existing_colors))
                new_colors = [colors.to_hex(c) for c in new_colors]
                for value, color in zip(missing_values, new_colors):
                    self.metadata_colors[layer][value] = color
