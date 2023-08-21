
![mosaicMPI logo](logo.png)

-----------------

# mosaicMPI: mosaic multi-resolution program integration

![version badge](https://img.shields.io/badge/version-1.9.0-blue)
[![PyPI Latest Release](https://img.shields.io/pypi/v/mosaicmpi.svg)](https://pypi.org/project/mosaicmpi/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/mosaicmpi)](https://anaconda.org/conda-forge/mosaicmpi/)
[![Documentation status](https://readthedocs.org/projects/mosaicmpi/badge/?version=latest&style=flat)]()
[![Downloads](https://static.pepy.tech/badge/cnmfsns)](https://pepy.tech/project/cnmfsns)
[![License](https://img.shields.io/pypi/l/mosaicmpi.svg)](https://github.com/MorrissyLab/mosaicMPI/blob/main/LICENSE)

Authors: [Ted Verhey](https://github.com/verheytb), [Heewon Seo](https://github.com/lootpiz), [Sorana Morrissy](https://github.com/ancasorana)

**mosaicMPI** is a Python package enabling mosaic integration of bulk, single-cell, and spatial expression data through program-level integration.
Programs are first discovered using consensus non-negative matrix factorization and then integrated using a flexible network-based approach to group
similar programs together across resolutions and datasets. Program communities are then interpreted using sample/cell metadata and classical gene
set analyses. Integrative program communities enable metadata transfer across datasets.


## ⚡Main Features

Here are just a few of the things that mosaicMPI does well:

- Identifies interpretable, non-negative programs at multiple resolutions
- Mosaic integration does not require subsetting features/genes to
  a shared or overdispersed subset
- Multi-omics integration does not require shared sample IDs
- Ideal for incremental integration (adding datasets one at a time) since
  deconvolution is performed independently on each dataset
- Integration performs well even when the datasets have mismatched features
  (eg. Microarray, RNA-Seq, Proteomics) or sparsity (eg single-cell vs bulk RNA-Seq and ATAC-Seq)
- Metadata transfer across datasets
- Command-line interface for rapid data exploration and python
  interface for extensibility and flexibility

## 🔧 Install

### ✨ Latest Release

Install the package with conda (in an isolated conda environment):
```bash
conda create -n mosaicmpi -c conda-forge mosaicmpi
conda activate mosaicmpi
```

## 📖 Documentation

### 🗐 Data guidelines

mosaicMPI can factorize a wide variety of datasets, but will work optimally in these conditions:
  - Use untransformed, raw data data where possible, and avoid log-transformed data
  - For single-cell, spatial, or bulk RNA-Seq data, the best data to use is feature counts, then TPM-normalized values, then RPKM/FPKM-normalized values.

### 📓 Python interface

To get started, sample datasets and a Jupyter notebook tutorial is available [here](/tutorial/tutorial.ipynb).

Detailed API reference can be found on [ReadTheDocs](https://mosaicmpi.readthedocs.io/).


### ⌨️ Command line interface

See the [command line interface documentation](/CLI.md).

## 💭 Getting Help

For errors arising during use of mosaicMPI, create and browse issues in the [GitHub "issues" tab](https://github.com/MorrissyLab/mosaicMPI/issues).