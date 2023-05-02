
![cNMF-SNS logo](logo.png)

-----------------

# cNMF-SNS: powerful factorization-based multi-omics integration toolkit

![version badge](https://img.shields.io/badge/version-1.2.0-blue)
[![PyPI Latest Release](https://img.shields.io/pypi/v/cnmfsns.svg)](https://pypi.org/project/cnmfsns/)
[![Conda Latest Release](https://anaconda.org/conda-forge/cnmfsns/badges/version.svg)](https://anaconda.org/anaconda/cnmfsns/)
[![Package Status](https://img.shields.io/pypi/status/cnmfsns.svg)](https://pypi.org/project/cnmfsns/)
[![Downloads](https://static.pepy.tech/personalized-badge/cnmfsns?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cnmfsns)
[![License](https://img.shields.io/pypi/l/cnmfsns.svg)](https://github.com/MorrissyLab/cNMF-SNS/blob/main/LICENSE)


<details>
  <summary> </summary>

```md

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3509134.svg)](https://doi.org/10.5281/zenodo.3509134)

```

</details>

**cNMF-SNS** (consensus Non-negative Matrix Factorization Solution Network Space) is a Python package enabling integration of bulk, single-cell, and
spatial expression data between and within datasets. cNMF provides a **robust, 
unsupervised** deconvolution of each dataset into gene expression programs (GEPs).
**Network-based integration** of GEPs enables flexible integration of many datasets
across assays (eg. Protein, RNA-Seq) and patient cohorts.

Communities with GEPs from multiple datasets can be annotated with dataset-specific
annotations to facilitate interpretation.

## ⚡Main Features

Here are just a few of the things that cNMF-SNS does well:

- Integration of expression data does not require subsetting features/genes to
  a shared or 'overdispersed' subset
- Ideal for incremental integration (adding datasets one at a time) since
  deconvolution is performed independently on each dataset generating invariant GEPs
- Does not assume the same level of sparsity/depth (single-cell, bulk)
- Identifies interpretable, additive non-negative gene expression programs
- Two interfaces: command-line interface for rapid data exploration and python
  interface for extensibility and flexibility

## 🔧 Install

### ☁️ Public Release

Install the package with conda:
```bash
conda install -c conda-forge cnmfsns
```

### ✨ Latest version from GitHub

Before installing cNMF-SNS using pip, it is recommended to first set up a separate conda environment and have conda manage as many dependencies as possible.

```bash
conda create --name cnmfsns -c conda-forge python=3.10 anndata pandas numpy scipy matplotlib upsetplot httplib2 tomli tomli-w click pygraphviz python-igraph semantic_version yaml scikit-learn fastcluster scanpy pyyaml gseapy=1.0.3
conda activate cnmfsns
pip install git+https://github.com/MorrissyLab/cNMF-SNS.git
```

## 📖 Documentation

### 📓 Python interface tutorial

To get started, sample proteomics datasets and a Jupyter notebook tutorial is available [here](/tutorial).

### ⌨️ Command line interface

See the [command line interface documentation](/CLI.md).

## 💭 Getting Help

For errors arising during use of cNMF-SNS, create and browse issues in the [GitHub "issues" tab](https://github.com/MorrissyLab/cNMF-SNS/issues).