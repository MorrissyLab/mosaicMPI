
from .dataset import Dataset
from .integration import Integration
from .config import Config
from .network import Network
from . import utils

import logging
from typing import Optional, Union
from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import distinctipy
import tomli
import tomli_w
import pandas as pd

class Colors():
    
    def __init__(self,
                 metadata_colors: Optional[dict] = {},
                 metadata_colors_group: Optional[dict] = {},
                 dataset_colors: Optional[dict] = {},
                 community_colors: Optional[dict] = {},
                 missing_data_color: str = "#dddddd"):
        self.metadata_colors = metadata_colors
        self.metadata_colors_group = metadata_colors_group
        self.dataset_colors = dataset_colors
        self.community_colors = community_colors
        self.missing_data_color = missing_data_color
        
    @classmethod
    def from_config(cls, config: Config):
        dataset_colors = {}
        for ds_name, ds_param in config.datasets.items():
            if "color" in ds_param:
                dataset_colors[ds_name] = ds_param["color"]
        missing_data_color = config.metadata_colors.pop("missing_data")
        return cls(metadata_colors = config.metadata_colors,
            metadata_colors_group = config.metadata_colors_group,
            dataset_colors = dataset_colors,
            missing_data_color = missing_data_color
            )
    @classmethod
    def from_toml(cls, toml_file: str):
        with open(toml_file, 'rb') as f:
            return cls(**tomli.load(f))

    def to_toml(self, toml_file: str):
        with open(toml_file, "wb") as f:
            tomli_w.dump(self.__dict__, f)

    @classmethod
    def from_dataset(cls, 
                     dataset: Dataset,
                     pastel_factor=0.3,
                     colorblind_type=None):
        colors = cls()
        colors.add_missing_metadata_colors(datasets=dataset, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        return colors
    
    @classmethod
    def from_integration(cls, 
                     integration: Integration,
                     pastel_factor=0.3,
                     colorblind_type=None):
        colors = cls()
        colors.add_missing_dataset_colors(datasets=integration, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        colors.add_missing_metadata_colors(datasets=integration, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        return colors
    
    @classmethod
    def from_named_datasets(cls, 
                     datasets: Mapping[str, Dataset],
                     pastel_factor=0.3,
                     colorblind_type=None):
        colors = cls()
        colors.add_missing_dataset_colors(datasets=datasets.keys(), pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        colors.add_missing_metadata_colors(datasets=datasets, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        return colors
    
    @classmethod
    def from_network(cls, 
                 network: Network,
                 pastel_factor=0.3,
                 colorblind_type=None):
        colors = cls()
        colors.add_missing_dataset_colors(datasets=network.integration, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        colors.add_missing_metadata_colors(datasets=network.integration, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        colors.add_missing_community_colors(network=network, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
        return colors

    @property
    def ordered_community_names(self):
        community_names = sorted(self.community_colors.keys(), key = lambda cstr: [int(lvl) for lvl in cstr.split(".")])
        return community_names

    def add_missing_dataset_colors(self,
                                   datasets: Union[Iterable, Integration],
                                   pastel_factor=0.3,
                                   colorblind_type=None) -> None:
        assert isinstance(datasets, (Integration, Iterable))
        
        if isinstance(datasets, Integration):
            datasets = datasets.datasets.keys()
        # check provided dataset colors
        invalid_colors = [name for name, color in self.dataset_colors.items()
                          if not mpl_colors.is_color_like(color)]
        if invalid_colors:
            raise ValueError(f"Datasets were given these invalid colors: {invalid_colors}. Please use valid matplotlib colors in named, hex, or RGB formats.")

        # fill in missing values with random colors distinct from existing colors
        uncolored_datasets = set(datasets) - self.dataset_colors.keys()
        existing_colors = set(self.dataset_colors.values()) | {"#FFFFFF", "#000000"}  # also exclude white and black
        existing_colors_rgb = [mpl_colors.to_rgb(c) for c in existing_colors]
        
        if uncolored_datasets:
            logging.info(f"Choosing distinct dataset colors")
            new_colors = distinctipy.get_colors(len(uncolored_datasets),
                                                exclude_colors=existing_colors_rgb,
                                                pastel_factor=pastel_factor,
                                                colorblind_type=colorblind_type)
            new_colors = [mpl_colors.to_hex(c) for c in new_colors]
            for name, color in zip(uncolored_datasets, new_colors):
                self.dataset_colors[name] = color
                
    def reset_community_colors(self,
                                     network,
                                     pastel_factor=0.3,
                                     colorblind_type=None) -> None:
        communities = network.communities.keys()
        logging.info(f"Choosing distinct community colors")
        new_colors = distinctipy.get_colors(len(communities),
                                                pastel_factor=pastel_factor,
                                                colorblind_type=colorblind_type)
        new_colors = [mpl_colors.to_hex(c) for c in new_colors]
        for name, color in zip(communities, new_colors):
            self.community_colors[name] = color
    
    def add_missing_community_colors(self,
                                     network,
                                     pastel_factor=0.3,
                                     overwrite_existing = True,
                                     colorblind_type=None) -> None:
        communities = network.communities.keys()
        # check provided community colors
        invalid_colors = [name for name, color in self.community_colors.items()
                          if not mpl_colors.is_color_like(color)]
        if invalid_colors:
            raise ValueError(f"Communities were given these invalid colors: {invalid_colors}. Please use valid matplotlib colors in named, hex, or RGB formats.")

        # fill in missing values with random colors distinct from existing colors
        uncolored_communities = set(communities) - self.community_colors.keys()
        existing_colors = set(self.community_colors.values()) | {"#FFFFFF", "#000000"}  # also exclude white and black
        if overwrite_existing:
            existing_colors_rgb = None
        else:
            existing_colors_rgb = [mpl_colors.to_rgb(c) for c in existing_colors]
        
        if overwrite_existing or uncolored_communities:
            logging.info(f"Choosing distinct community colors")
            new_colors = distinctipy.get_colors(len(uncolored_communities),
                                                exclude_colors=existing_colors_rgb,
                                                pastel_factor=pastel_factor,
                                                colorblind_type=colorblind_type)
            new_colors = [mpl_colors.to_hex(c) for c in new_colors]
            for name, color in zip(uncolored_communities, new_colors):
                self.community_colors[name] = color

    def plot_dataset_colors_legend(self, figsize: Iterable = None, ax = None) -> Figure:
        if ax is None:
            if figsize is None:
                figsize = [3, 1 + 0.25 * len(self.dataset_colors)]
            fig, ax_plot = plt.subplots(figsize=figsize)
        elif figsize is not None:
            raise ValueError
        else:
            ax_plot = ax
        legend_elements = [Patch(label=cat, facecolor=color, edgecolor=None) for cat, color in self.dataset_colors.items()]
        ax_plot.legend(handles=legend_elements, loc='upper center')
        ax_plot.set_title('Dataset')
        ax_plot.set_axis_off()
        plt.tight_layout()
        if ax is None:
            return fig
        
    def plot_community_colors_legend(self, figsize: Iterable = None, ax = None) -> Figure:
        if ax is None:
            if figsize is None:
                figsize = [3, 1 + 0.25 * len(self.dataset_colors)]
            fig, ax_plot = plt.subplots(figsize=figsize)
        elif figsize is not None:
            raise ValueError
        else:
            ax_plot = ax
        
        legend_elements = []
        for community in self.ordered_community_names:
            color = self.community_colors[community]
            legend_elements.append(Patch(label=community, facecolor=color, edgecolor=None))
        ax_plot.legend(handles=legend_elements, loc='upper center')
        ax_plot.set_title('Community')
        ax_plot.set_axis_off()
        plt.tight_layout()
        if ax is None:
            return fig
        
    def add_missing_metadata_colors(self,
                                    datasets: Union[Dataset, Integration, Mapping[str, Dataset]],
                                    pastel_factor=0.3,
                                    colorblind_type=None):
        """
        Identify missing colors based on metadata. If metadata_df is provided, categorical columns are used; otherwise, metadata_df is derived from the config datasets.
        """

        # get categorical data for which colors should match
        if isinstance(datasets, Dataset):
            metadata_df = datasets.get_metadata_df(include_numerical=False)
        elif isinstance(datasets, Mapping):
            metadata_df = pd.concat({dataset_name: dataset.get_metadata_df(include_numerical=False) for dataset_name, dataset in datasets.items()})
        elif isinstance(datasets, Integration):
            metadata_df = datasets.get_metadata_df(include_numerical=False, prepend_dataset_column=False)
        else:
            raise NotImplementedError()
        
        # check provided metadata colors
        invalid_colors = []
        for layer, layer_colors in self.metadata_colors.items():
            for value, color in layer_colors.items():
                if not mpl_colors.is_color_like(color):
                    invalid_colors.append(color)
        if invalid_colors:
            raise ValueError(f"Metadata colors included these invalid colors: {invalid_colors}. "
                             "Please use valid matplotlib colors in named, hex, or RGB formats.")

        # fill in missing values with random colors distinct from existing colors
        for layer, allvalues in metadata_df.items():
            layer_colors = self.get_metadata_colors(layer)
            existing_values = set(layer_colors.keys())
            existing_colors = set(layer_colors.values()) | {"#FFFFFF", "#000000", self.missing_data_color} # add white/black so that it is excluded from new colors.
            existing_colors_rgb = [mpl_colors.to_rgb(c) for c in existing_colors]
            colorless_values = set(allvalues.dropna().unique()) - existing_values
            if colorless_values:
                logging.info(f"Choosing distinct colors for metadata layer {layer}")
                if layer not in self.metadata_colors:
                    self.metadata_colors[layer] = {}
                new_colors = distinctipy.get_colors(len(colorless_values), exclude_colors=existing_colors_rgb, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
                new_colors = [mpl_colors.to_hex(c) for c in new_colors]
                for value, color in zip(colorless_values, new_colors):
                    self.metadata_colors[layer][value] = color
    
    def get_metadata_colors(self, layer):
        group_names = [group for group, group_attr in self.metadata_colors_group.items() if layer in group_attr["group"]]
        layer_colors = {}
        if layer in self.metadata_colors and group_names:
            raise ValueError(f"Metadata column {layer} cannot be within a color group if it is also explicitly defined with its own colors.")
        elif len(group_names) > 1:
            raise ValueError((
                f"The following column in the metadata matrix (adata.obs) has multiple metadata color groups in the config TOML file:\n"
                f"Metadata column: {layer}\n"
                "Metadata color groups: " + ", ".join(group_names)
            ))
        elif layer in self.metadata_colors:
            layer_colors = {**layer_colors, **self.metadata_colors[layer]}  # updates dict with info from metadata_colors
        elif len(group_names) == 1:
            group = group_names[0]
            layer_colors = {**layer_colors, **self.metadata_colors_group[group]['colors']} # updates dict with info from metadata_colors_group
        return layer_colors

    def plot_metadata_colors_legend(self,
                                    layer: str = None,
                                    ax: Axes = None,
                                    char_per_line: int = 20,
                                    subset = None,
                                    figsize: Iterable = None) -> Optional[Figure]:
        
        if layer is not None and isinstance(ax, Axes):
            color_def = self.metadata_colors[layer]
            if subset is not None:
                color_def = {k:v for k, v in color_def.items() if k in subset}
            legend_elements = [Patch(label=utils.newline_wrap(cat, char_per_line), facecolor=color, edgecolor=None) for cat, color in color_def.items()]
            ax.legend(handles=legend_elements, loc='upper center')
            ax.set_title(layer)
            ax.set_axis_off()
                
        elif layer is None and ax is None:
        
            categorical_columns = [track for track, color_def in self.metadata_colors.items() if isinstance(color_def, dict)]
            categorical_groups = [group for group, group_attr in self.metadata_colors_group.items() if isinstance(group_attr["colors"], dict)]
            n_columns = len(categorical_columns) + len(categorical_groups) + 1
            legend_lengths = [
                len(color_def) for color_def in (self.metadata_colors | self.metadata_colors_group).values()
                if isinstance(color_def, dict)
            ]
            if figsize is None:
                figsize = [2*n_columns, 0.3 * max(legend_lengths)]
            fig, axes = plt.subplots(1, n_columns, figsize=figsize, squeeze=False, layout="tight")
            for ax_layer, layer in zip(axes[0], categorical_columns):
                color_def = self.metadata_colors[layer]
                legend_elements = [Patch(label=utils.newline_wrap(cat, char_per_line), facecolor=color, edgecolor=None) for cat, color in color_def.items()]
                ax_layer.legend(handles=legend_elements, loc='upper center')
                ax_layer.set_title(layer)
                ax_layer.set_axis_off()
            for ax_id, group in enumerate(categorical_groups, len(categorical_columns)):
                ax_layer = axes[0][ax_id]
                color_def = self.metadata_colors_group[group]["colors"]
                legend_elements = [Patch(label=utils.newline_wrap(cat, char_per_line), facecolor=color, edgecolor=None) for cat, color in color_def.items()]
                ax_layer.legend(handles=legend_elements, loc='upper center')
                ax_layer.set_title(group)
                ax_layer.set_axis_off()

            # last column is missing data color
            axes[0][-1].legend(handles=[Patch(label="Missing Data", facecolor=self.missing_data_color, edgecolor=None)], loc='upper left')
            axes[0][-1].set_axis_off()
        else:
            raise ValueError("Parameters `layer` and `ax` must both be specified")
        
        if ax is None:
            return fig