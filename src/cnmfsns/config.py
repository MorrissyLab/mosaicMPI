import tomli
import tomli_w
import pandas as pd
import numpy as np
import sys
import logging
import collections.abc
from copy import deepcopy
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import distinctipy
import os
from types import SimpleNamespace
from anndata import read_h5ad

# Default parameters for the config files are used when missing as input to cnmfsns integrate

config_defaults = {
    "integrate": {
        "corr_method": "pearson",
        "max_median_corr": 0,
        "negative_corr_quantile": 0.95,
        },
    "sns": {
        "edge_color": "#bbbbbb20",
        "edge_weight": "none",
        "community_algorithm": "greedy_modularity",
        "communities": {
            "greedy_modularity": {
                "resolution": 2,
                "best_n": "none"
            },
            "leiden": {
                "resolution": 0.01
            }
        },
        "layout_algorithm": "neato",   # "neato", "spring", "community_weighted_spring"
        "layouts": {  # parameters for each layout algorithm
            "neato": {},
            "spring": {},
            "community_weighted_spring": {
                "within_community": 100,
                "within_dataset": 1
            }
        },
        "node_size": 30,
        "plot_size": [10, 10]
        },
    "datasets": {},
    "metadata_colors": {"missing_data": "#dddddd"},
    "metadata_colors_group": {}
}
dataset_defaults = {
    "selected_k": [[1, 10, 1], [15, 500, 5]],
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
            metadata_df = pd.concat({name: read_h5ad(d["filename"], backed="r").obs.select_dtypes(include="category") for name, d in self.datasets.items()})
        else:
            metadata_df = metadata_df.select_dtypes(include="category")
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
            layer_colors = self.get_metadata_colors(layer)
            existing_values = set(layer_colors.keys())
            existing_colors = set(layer_colors.values()) | {"#FFFFFF", "#000000"} # add white/back so that it is excluded from new colors.
            if np.NaN in allvalues:
                existing_colors.add(self.metadata_colors["missing_data"])
            colorless_values = set(allvalues.dropna().unique()) - existing_values
            if colorless_values:
                logging.info(f"Choosing distinct colors for metadata layer {layer}")
                if layer not in self.metadata_colors:
                    self.metadata_colors[layer] = {}
                new_colors = distinctipy.get_colors(len(colorless_values), exclude_colors=[colors.to_rgb(c) for c in existing_colors])
                new_colors = [colors.to_hex(c) for c in new_colors]
                for value, color in zip(colorless_values, new_colors):
                    self.metadata_colors[layer][value] = color

    
    def get_metadata_colors(self, layer):
        group_names = [group for group, group_attr in self.metadata_colors_group.items() if layer in group_attr["group"]]
        layer_colors = {}
        if layer in self.metadata_colors:
            layer_colors = {**layer_colors, **self.metadata_colors[layer]}  # updates dict with info from metadata_colors
        if len(group_names) == 1:
            group = group_names[0]
            layer_colors = {**layer_colors, **self.metadata_colors_group[group]['colors']} # updates dict with info from metadata_colors_group
        elif len(group_names) > 1:
            logging.error((
                f"The following column in the metadata matrix (adata.obs) has multiple metadata color groups in the config TOML file:\n"
                f"Metadata column: {layer}\n"
                "Metadata color groups: " + ", ".join(group_names)
            ))
            sys.exit(1)
        return layer_colors

    def plot_metadata_colors_legend(self):
        categorical_columns = [track for track, color_def in self.metadata_colors.items() if isinstance(color_def, dict)]
        categorical_groups = [group for group, group_attr in self.metadata_colors_group.items() if isinstance(group_attr["colors"], dict)]
        n_columns = len(categorical_columns) + len(categorical_groups) + 1
        fig, axes = plt.subplots(1, n_columns, figsize=[3*n_columns, 200], squeeze=False)
        for ax, track in zip(axes, categorical_columns):
            ax = ax[0]
            color_def = self.metadata_colors[track]
            legend_elements = [Patch(label=cat, facecolor=color, edgecolor=None) for cat, color in color_def.items()]
            ax.legend(handles=legend_elements, loc='upper left')
            ax.set_title(track)
            ax.set_axis_off()
        for ax_id, group in enumerate(categorical_groups, len(categorical_columns)):
            ax = axes[ax_id][0]
            color_def = self.metadata_colors_group[group]["colors"]
            legend_elements = [Patch(label=cat, facecolor=color, edgecolor=None) for cat, color in color_def.items()]
            ax.legend(handles=legend_elements, loc='upper left')
            ax.set_title(group)
            ax.set_axis_off()

        # last column is missing data color
        axes[-1][0].legend(handles=[Patch(label="Missing Data", facecolor=self.metadata_colors["missing_data"], edgecolor=None)], loc='upper left')
        axes[-1][0].set_axis_off()

        plt.tight_layout()
        return fig

    def get_usage_matrix(self):
        usage = []
        sample_to_patient = {}
        for dataset_name, dataset in self.datasets.items():
            adata = read_h5ad(dataset["filename"])
            # if "patient_id_column" in dataset:   # this code can be removed once patient-id mapping is implemented elsewhere
            #     for sample, patient in adata.obs[dataset["patient_id_column"]].items():
            #         sample_to_patient[(dataset_name, sample)] = (dataset_name, patient)
            df = adata.obsm["cnmf_usage"]
            df.index = pd.MultiIndex.from_product(([dataset_name], (df.index)))
            df.columns = pd.MultiIndex.from_tuples([(dataset_name, int(col[0]), int(col[1])) for col in df.columns.str.split(".")])
            usage.append(df)
        usage = pd.concat(usage, axis=1).sort_index(axis=0).sort_index(axis=1)
        usage.index.rename(["dataset", "sample"], inplace=True)
        usage.columns.rename(["dataset", "k", "gep"], inplace=True)
        return usage

    def get_sample_patient_mapping(self):
        sample_to_patient = {}
        for dataset_name, dataset in self.datasets.items():
            adata = read_h5ad(dataset["filename"])
            if "patient_id_column" in dataset:   # this code can be removed once patient-id mapping is implemented elsewhere
                for sample, patient in adata.obs[dataset["patient_id_column"]].items():
                    sample_to_patient[(dataset_name, sample)] = (dataset_name, patient)
        return sample_to_patient