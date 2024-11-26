![mosaicMPI logo](https://github.com/MorrissyLab/mosaicMPI/blob/main/docs/source/_static/img/logo.png?raw=True)

# mosaicMPI: Mosaic Multi-resolution Program Integration

![version badge](https://img.shields.io/badge/version-2.6.6-blue)
[![PyPI Latest Release](https://img.shields.io/pypi/v/mosaicmpi.svg)](https://pypi.org/project/mosaicmpi/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/mosaicmpi)](https://anaconda.org/conda-forge/mosaicmpi/)
[![Documentation status](https://readthedocs.org/projects/mosaicmpi/badge/?version=latest&style=flat)](https://mosaicmpi.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/mosaicmpi)](https://pepy.tech/project/mosaicmpi)
[![Stars](https://img.shields.io/github/stars/MorrissyLab/mosaicMPI?logo=GitHub&color=yellow)](https://github.com/MorrissyLab/mosaicMPI/stargazers)
[![License](https://img.shields.io/pypi/l/mosaicmpi.svg)](https://github.com/MorrissyLab/mosaicMPI/blob/main/LICENSE)
[![DOI:10.1101/2023.08.18.553919](http://img.shields.io/badge/DOI-10.1101/2023.08.18.553919-B31B1B.svg)](https://doi.org/10.1101/2023.08.18.553919)

Authors: [Ted Verhey](https://github.com/verheytb), [Sorana Morrissy](https://github.com/ancasorana)

Contributors: Hyojin Song, Aaron Gillmor, Gurveer Gill, Courtney Hall

**mosaicMPI** is a Python package for enabling mosaic integration of bulk, single-cell, and spatial expression data through program-level integration. Programs are first discovered using unsupervised deconvolution (consensus non-negative matrix factorization, cNMF) across multiple ranks separately for each dataset. A flexible network-based approach groups similar programs together across resolutions and datasets. Program communities are then interpreted using sample/cell metadata and gene set analyses. Integrative program communities enable metadata transfer across datasets.

## ⚡Main Features

Here are just a few of the things that `mosaicMPI` does well:

- Identifies interpretable, non-negative programs at multiple resolutions
- Mosaic integration does not require subsetting features/genes to
  a shared or overdispersed subset
- Multi-omics integration without shared sample IDs
- Incremental integration (adding datasets one at a time) since
  deconvolution is performed independently on each dataset
- High performance integration of datasets with mismatched features
  (eg. Microarray, RNA-Seq, Proteomics) or sparsity (eg. single-cell vs. bulk)
- Metadata transfer across datasets

`mosaicMPI` has two interfaces:
- command-line interface (CLI) with a standardized workflow for rapid data exploration and integration
- python API for greatest flexibility and extensibility

## 🔧 Install

### 🧰 System Requirements

- Compatible with OS X, Windows and Linux systems
- Memory usage depends on size and number of datasets

### ✨ Latest Release
Install the package with `conda`:
```bash
# if using a fresh conda install
conda init

# create an environment called 'mosaicenv' and install
conda create -n mosaicenv -c conda-forge mosaicmpi
conda activate mosaicenv
```

Some analyses require packages from other channels to be installed in the same environment:

- [`g:Profiler`](https://biit.cs.ut.ee/gprofiler/page/apis)

```bash
conda install -c bioconda gprofiler-official
```

- [`GSEApy`](https://github.com/zqfang/GSEApy):

```bash
# if you have conda (MacOS_x86-64 and Linux only)
conda install -c bioconda gseapy
# Windows and macOS (Apple Silicon)
pip install gseapy
```

## 📖 Documentation

Read the [documentation](https://mosaicmpi.readthedocs.io/).

## 💭 Getting Help

For questions arising during use of `mosaicMPI`, create and browse issues in the [GitHub "issues" tab](https://github.com/MorrissyLab/mosaicMPI/issues).
