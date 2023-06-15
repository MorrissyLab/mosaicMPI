
![cNMF-SNS logo](logo.png)

-----------------

# cNMF-SNS: powerful factorization-based multi-omics integration toolkit

![version badge](https://img.shields.io/badge/version-1.4.0-blue)
[![PyPI Latest Release](https://img.shields.io/pypi/v/cnmfsns.svg)](https://pypi.org/project/cnmfsns/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/cnmfsns)](https://anaconda.org/conda-forge/cnmfsns/)
[![Documentation status](https://readthedocs.org/projects/cnmf-sns/badge/?version=latest&style=flat)]()
[![Downloads](https://static.pepy.tech/badge/cnmfsns)](https://pepy.tech/project/cnmfsns)
[![License](https://img.shields.io/pypi/l/cnmfsns.svg)](https://github.com/MorrissyLab/cNMF-SNS/blob/main/LICENSE)

Authors: [Ted Verhey](https://github.com/verheytb), [Heewon Seo](https://github.com/lootpiz), [Sorana Morrissy](https://github.com/ancasorana)

**cNMF-SNS** (consensus Non-negative Matrix Factorization Solution Network Space) is a Python package enabling mosaic integration of bulk, single-cell, and
spatial expression data between and within datasets. Datasets can have partially overlapping features (eg. genes) as well as non-overlapping features. cNMF provides a **robust, 
unsupervised** deconvolution of each dataset into gene expression programs (GEPs).
**Network-based integration** of GEPs enables flexible integration of many datasets
across assays (eg. Protein, RNA-Seq, scRNA-Seq, spatial expression) and patient cohorts.

Communities with GEPs from multiple datasets can be annotated with dataset-specific
annotations to facilitate interpretation.

## ⚡Main Features

Here are just a few of the things that cNMF-SNS does well:

- Identifies interpretable, additive non-negative gene expression programs
- Mosaic integration does not require subsetting features/genes to
  a shared or overdispersed subset
- Ideal for incremental integration (adding datasets one at a time) since
  deconvolution is performed independently on each dataset generating invariant GEPs
- Integration does not require comparable data sparsity or sequencing depth (single-cell, microarray, bulk)
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
conda create --name cnmfsns -c conda-forge python=3.10 anndata pandas numpy scipy matplotlib upsetplot httplib2 tomli tomli-w click pygraphviz python-igraph semantic_version pyyaml scikit-learn fastcluster scanpy pyyaml
conda activate cnmfsns
pip install git+https://github.com/MorrissyLab/cNMF-SNS.git
```

## 📖 Documentation

### 📓 Python interface

To get started, sample proteomics datasets and a Jupyter notebook tutorial is available [here](/tutorial/tutorial.ipynb).

Detailed API reference can be found on [ReadTheDocs](https://cnmf-sns.readthedocs.io/).


### ⌨️ Command line interface

See the [command line interface documentation](/CLI.md).

## 💭 Getting Help

For errors arising during use of cNMF-SNS, create and browse issues in the [GitHub "issues" tab](https://github.com/MorrissyLab/cNMF-SNS/issues).