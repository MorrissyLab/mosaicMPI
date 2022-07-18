import tomli
import tomli_w
import pandas as pd
import numpy as np
import sys
from matplotlib import colors
from datetime import datetime
import distinctipy
import os
from types import SimpleNamespace

class Config(SimpleNamespace):

    @classmethod
    def from_toml(cls, toml_file):
        with open(toml_file) as f:
            return cls(**tomli.load(f))
    
    @classmethod
    def from_h5mu_files(cls, h5mu_files):
        c = {
            "name": datetime.now().strftime("cnmfsns_%Y%m%d-%H%M%S"),
            "datasets": {os.path.basename(fn).replace(".h5mu",""): {"filename": fn} for fn in h5mu_files}
        }
        return cls(c)

    def to_toml(self, toml_file):
        with open(toml_file, "wb") as f:
            tomli_w.dump(self.__dict__, f)


    def add_missing_colors(self):
        
        # check provided dataset colors
        invalid_colors = []
        for name, d in self.datasets.items():
            if "color" in d and not colors.is_color_like(d["color"]):
                invalid_colors.append(d["color"])
        if invalid_colors:
            print(f"Error: Datasets were given these invalid colors: {invalid_colors}. Please use valid matplotlib colors in named, hex, or RGB formats.")
            sys.exit(1)

        # fill in missing dataset colors
        existing_colors = set(d["color"] for name, d in self.datasets.items() if "color" in d)
        uncolored_datasets = [name for name, d in self.datasets.items() if "color" not in d]
        for d, rgb in zip(uncolored_datasets, distinctipy.get_colors(len(uncolored_datasets))):
            self.datasets[d]["color"] = colors.to_hex(rgb)

        # merge metadata from all datasets
        # TODO: check that colors in config are valid

        metadata = {name: mudata.read_h5mu(d["filename"]).obs for name, d in self.datasets.items()}
        metadata = pd.concat(metadata).iloc[:, 1:]
        metadata = metadata.replace({True:"true", False: "false"}) # converts bool to strings
        metadata = metadata.loc[:,(metadata.dtypes == 'object')] # excludes int and float metadata, which should use a continuous scale
        # fill in missing values with random colors distinct from existing colors
        for layer, allvalues in metadata.items():
            if layer in self.metadata_colors:
                existing_values = set(self.metadata_colors[layer].keys())
                existing_colors = set(colors.to_rgb(self.metadata_colors[layer][val]) for val in existing_values)
            else:
                existing_values = set()
                existing_colors = set()
            missing_values = set(allvalues.fillna("Other").unique()) - existing_values
            if missing_values:
                if layer not in self.metadata_colors:
                    self.metadata_colors[layer] = {}
                new_colors = distinctipy.get_colors(len(missing_values), exclude_colors=list(existing_colors))
                new_colors = [colors.to_hex(c) for c in new_colors]
                for value, color in zip(missing_values, new_colors):
                    if pd.isnull(value):
                        value = "Other"
                    self.metadata_colors[layer][value] = color

    def add_missing_colors_from(self, metadata):
        """
        Identify missing colors based on attached metadata dataframe.
        """
        if not hasattr(self, "metadata_colors"):
            self.metadata_colors = {}
        
        if not "missing_data" in self.metadata_colors:
            self.metadata_colors["missing_data"] = "#dddddd"

        metadata = metadata.select_dtypes(include=["category", "object"]) # excludes int and float metadata, which should use a continuous scale
        # fill in missing values with random colors distinct from existing colors
        for layer, allvalues in metadata.items():
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
