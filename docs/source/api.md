# API


## Datasets

Class to contain datasets, sample/cell metadata, cNMF parameters, and cNMF results:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   mosaicmpi.Dataset
```

These can be made from from pandas data matrices or imported from AnnData files/objects.

```{eval-rst}
.. autosummary::

   mosaicmpi.Dataset.from_df
   mosaicmpi.Dataset.from_h5ad
   mosaicmpi.Dataset.from_anndata
```

## Integration of programs across resolutions and datasets

To identify program anchors, create an Integration object from one or more datasets. Then, create a Network object from the Integration object.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   mosaicmpi.Integration
   mosaicmpi.Network
```

## Visualizing MosaicMPI integrations

Class to enable consistent color palettes for metadata categories, program communities, and datasets.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   mosaicmpi.Colors
```

Visually-distinct colors can be auto-generated from Dataset, Integration, or Network objects:

```{eval-rst}
.. autosummary::
   
   mosaicmpi.Colors.from_dataset
   mosaicmpi.Colors.from_integration
   mosaicmpi.Colors.from_network
```

Functions to create plots from Dataset, Integration, and Network objects are in the plots module:

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :recursive:

   mosaicmpi.plots
```


## Complete API Reference

```{eval-rst}
.. autosummary::
   :toctree: api/
   :recursive:

   mosaicmpi
```