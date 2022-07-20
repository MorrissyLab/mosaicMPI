import tomli
import tomli_w
import pandas as pd
import numpy as np
import sys
import logging
from matplotlib import colors
from datetime import datetime
import distinctipy
import os
from types import SimpleNamespace
from anndata import read_h5ad

class Config(SimpleNamespace):

    @classmethod
    def from_toml(cls, toml_file):
        with open(toml_file) as f:
            return cls(**tomli.load(f))
    
    @classmethod
    def from_h5ad_files(cls, h5ad_files):
        c = {
            "name": datetime.now().strftime("cnmfsns_%Y%m%d-%H%M%S"),
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
        Identify missing colors based on attached metadata dataframe.
        """
        # color for missing data
        if not hasattr(self, "metadata_colors"):
            self.metadata_colors = {}
        
        if not "missing_data" in self.metadata_colors:
            self.metadata_colors["missing_data"] = "#dddddd"

        # get categorical data for which colors should match
        if metadata_df is None:
            # read from h5ad files
            metadata_df = pd.concat({name: read_h5ad(d["filename"]).obs for name, d in self.datasets.items()})
        metadata_df = metadata_df.replace({True:"true", False: "false"}) # converts bool to strings
        metadata_df  = metadata_df.select_dtypes(include=["category", "object"]) # excludes int and float metadata, which should use a continuous scale

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
                if layer not in self.metadata_colors:
                    self.metadata_colors[layer] = {}
                new_colors = distinctipy.get_colors(len(missing_values), exclude_colors=list(existing_colors))
                new_colors = [colors.to_hex(c) for c in new_colors]
                for value, color in zip(missing_values, new_colors):
                    self.metadata_colors[layer][value] = color
