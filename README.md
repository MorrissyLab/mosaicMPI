![mosaicMPI logo](https://github.com/MorrissyLab/mosaicMPI/blob/main/docs/source/_static/img/logo.png?raw=True)

# mosaicMPI: mosaic multi-resolution program integration

![version badge](https://img.shields.io/badge/version-2.4.11-blue)
[![PyPI Latest Release](https://img.shields.io/pypi/v/mosaicmpi.svg)](https://pypi.org/project/mosaicmpi/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/mosaicmpi)](https://anaconda.org/conda-forge/mosaicmpi/)
[![Documentation status](https://readthedocs.org/projects/mosaicmpi/badge/?version=latest&style=flat)](https://mosaicmpi.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/mosaicmpi)](https://pepy.tech/project/mosaicmpi)
[![Stars](https://img.shields.io/github/stars/MorrissyLab/mosaicMPI?logo=GitHub&color=yellow)](https://github.com/MorrissyLab/mosaicMPI/stargazers)
[![License](https://img.shields.io/pypi/l/mosaicmpi.svg)](https://github.com/MorrissyLab/mosaicMPI/blob/main/LICENSE)
[![DOI:10.1101/2023.08.18.553919](http://img.shields.io/badge/DOI-10.1101/2023.08.18.553919-B31B1B.svg)](https://doi.org/10.1101/2023.08.18.553919)

Authors: [Ted Verhey](https://github.com/verheytb), [Sorana Morrissy](https://github.com/ancasorana)

Contributors: Hyojin Song, Aaron Gillmor, Gurveer Gill, Courtney Hall

**mosaicMPI** is a Python package enabling mosaic integration of bulk, single-cell, and spatial expression data through program-level integration.
Programs are first discovered using consensus non-negative matrix factorization (cNMF) across multiple-ranks and then integrated using a flexible network-based approach to
group similar programs together across resolutions and datasets. Program communities are then interpreted using sample/cell metadata and gene set analyses. Integrative program communities enable metadata transfer across datasets.

## ⚡Main Features

Here are just a few of the things that mosaicMPI does well:

- Identifies interpretable, non-negative programs at multiple resolutions
- Mosaic integration does not require subsetting features/genes to
  a shared or overdispersed subset
- Multi-omics integration without shared sample IDs
- Ideal for incremental integration (adding datasets one at a time) since
  deconvolution is performed independently on each dataset
- Integration performs well even when the datasets have mismatched features
  (eg. Microarray, RNA-Seq, Proteomics) or sparsity (eg single-cell vs bulk RNA-Seq and ATAC-Seq)
- Metadata transfer across datasets
- Command-line interface for rapid data exploration and python
  interface for extensibility and flexibility

## 🔧 Install

### 🧰 System Requirements

- Compatible with and tested on OS X, Windows and Linux systems
- Memory usage depends on size and number of datasets

### ✨ Latest Release
Install the package with `conda`:
```bash
# create an environment called mosaic and install
conda create -n mosaic -c conda-forge mosaicmpi
conda activate mosaic
```

For ssGSEA analysis, you will also need to install GSEApy into the same environment.

```bash
# if you have conda (MacOS_x86-64 and Linux only)
conda install -c bioconda gseapy
# Windows and MacOS_ARM64 (M1/2-Chip)
pip install gseapy
```

## 📖 Documentation

Read the [documentation](https://mosaicmpi.readthedocs.io/).

## 💭 Getting Help

For errors arising during use of mosaicMPI, create and browse issues in the [GitHub "issues" tab](https://github.com/MorrissyLab/mosaicMPI/issues).