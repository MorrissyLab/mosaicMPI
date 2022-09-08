# cNMF-SNS

cNMF Solution Neighborhood Space

![](https://img.shields.io/badge/version-0.2.18-blue)

## Installation

### 1. Using `pip` to install the latest version from GitHub:

Before installing cNMF-SNS from `pip`, it is recommended to first set up a separate conda environment and have conda manage as many dependencies as possible.
```
conda create --name py39 python=3.9 anndata pandas numpy scipy matplotlib upsetplot httplib2 tomli tomli-w click pygraphviz
conda activate py39
```

If you use SSH authentication for GitHub, use the following:
```
pip install git+ssh://git@github.com/MorrissyLab/cNMF-SNS.git
```

If you have installed a [personal access token from GitHub](https://github.com/settings/tokens), you can use:
```
pip install git+https://<token>@github.com/MorrissyLab/cNMF-SNS.git
```

### 2. Using `pip` (PyPI version)

> Note: This will work only when cNMF-SNS has been published to PyPI.

```
pip install cnmfsns
```

### 3. Using `conda`

> Note: This will work only when cNMF-SNS has been published to conda-forge.

```
conda install -c conda-forge cnmfsns
```

## Workflow 1A: Factorization for individual datasets

cNMF-SNS is a command line tool for deconvoluting and integrating gene expression and other high-dimensional data.

Each step of a workflow is run as a separate command within cNMF-SNS. You can see which subcommands are available using:
```
cnmfsns --help
```

Easily get help for each subcommand using, for example:

```
cnmfsns model-odg --help
```

### 1. Create AnnData object from text files with gene expression and metadata

If expression and annotation data is in text files, this utility can combine them into a .h5ad file for downstream tools. If you have normalized and count data as text files, use the following command:

```
cnmfsns txt-to-h5ad --normalized normalized.txt --counts counts.txt --metadata metadata.txt -o dataset.h5ad
```

Only one of the `--normalized` and `--counts` options are required. When only count data is provided, TPM normalization is automatically performed and this is used for overdispersed gene selection. If only normalized data is provided, then the normalized data is used both for factorization and for overdispersed gene selection.

#### Input semantics

Expression (normalized and count) data must be indexed as follows:
  - Rows are samples/cells/spots; first column must be sample/cell/spot IDs
  - Columns are genes/features; the first row must be feature names

Metadata must be indexed as follows:
  - The first column must be sample/cell/spot IDs
  - Other columns are metadata 'layers' and must be labelled. Values can be numerical, boolean, or categorical types.
  > Note: if any values in a column are not numerical, the entire column will be treated as categorical. This can have implications for annotated heatmaps where numerical data is usually presented as a continuous color scale, not a set of distinct colors. If a column is numerical with missing values, then these should be empty values (not "NA", "NaN", etc.)
  - Missing values are acceptable. For categorical data, these will be plotted in an "Other" category. For numerical data, these will be ignored.

### 2. Check existing h5ad files for minimum requirements for cNMF.

Check h5ad objects for rows or columns which have missing values, negative values, or variance of 0.

cNMF  supports input data that is sparse (i.e. with zeros), but not with missing values. When missing values are present (eg. from concatenation of datasets with partially overlapping features), the default behaviour is to subset the input matrix to shared features/genes only, but it is recommended to either run each dataset separately or use a dense, imputed data matrix. cNMF will warn the user if missing data is present.

```
cnmfsns check-h5ad -i file.h5ad -o file_filtered.h5ad
```

### 3. Model gene overdispersion to select genes for factorization.

Deconvolution of a gene expression dataset using cNMF requires a set of overdispersed genes/features which will be used for factorization. GEPs will include all genes after a re-fitting step, but cNMF will only optimize the fit for overdispersed genes, providing the user the opportunity to decide which genes are most informative.

Since cNMF performs variance scaling on the input matrix, it is important to remove genes whose variance could be attributable to noise. cNMF-SNS supports two methods for overdispersed gene selection:
  - `cnmf`: v-score and minimum expression threshold (cNMF method: Kotliar, et al. eLife, 2019). This method is only suitable for count data.
  - `default`: residual standard deviation after modeling mean-variance dependence. (STdeconvolve method: Miller, et al. Nat. Comm. 2022) This method makes fewer assumptions about the input data but requires a visual check since the optimal threshold depends on the data type.

To produce plots to guide selection of overdispersed genes, run the following command:

```
cnmfsns model-odg --name example_run --input file_filtered.h5ad
```

This command will create a directory with the name of the run inside the output directory (defaults to current working directory).

### 4. Select overdispersed genes and parameters for factorization

Once you have decided on a method for selecting overdispersed genes and are ready for factorization, you can easily use the default parameters and select values of k as follows:

```
cnmfsns set-parameters --name example_run -k 2 -k 3
```

A more complex range of k values can also be set up using the `--k_range` parameter. For example, to perform cNMF using a range of k values from 5 to 60, with a step of 5 (ie.: 5, 10, 15, ... 60), you would specify `--k_range 5 60 5`.

The default behaviour is to select overdispersed genes using an od-score > 1.0. You can also choose from a number of different methods for selecting overdispersed_genes, for example:
  - `cnmfsns set-parameters --name example_run -m cnmf_topn -p 2000`: select top 2000 genes using cNMF's model and minimal mean threshold (Kotliar et al., 2019, eLife)
  - `cnmfsns set-parameters --name example_run -m default_quantile -p 0.8`: Select top 20% of genes when ranked by od-score
  - `cnmfsns set-parameters --name example_run -m genes_file -p path/to/genesfile.txt`: use a custom list of genes

### 5. Perform cNMF factorization

Factorize the input data. While parameters can be provided which allow for custom parallelization, by default cnmfsns uses a single CPU:

```
cnmfsns factorize --name example_run
```

For submitting jobs to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/cNMF-SNS/tree/main/scripts/slurm.sh).

After editing the script to ensure it is suitable for your HPC cluster, cNMF-SNS will submit jobs using SLURM's `sbatch` command to parallelize factorization.
```
cnmfsns factorize --name example_run --slurm_script /path/to/slurm.sh
```

### 6. Postprocessing

This step will check to ensure that all factorizations completed successfully, and then will create consensus GEPs and usages, updating the `.h5ad` file with the cNMF solution.

```
cnmfsns postprocess --name example_run
```

For downstream analyses, the output AnnData object is in `./example_run/example_run.h5ad`.

### 7. Created annotated heatmaps of GEP usages

This step will create annotated heatmaps of GEP usages from cNMF-SNS outputs:

```
cnmfsns annotated-heatmap --output_dir ./example_run/ -i ./example_run/example_run.h5ad
```

To provide custom colors for the metadata layers, you can specify a TOML-formatted file with a `metadata_colors` section (see `scripts/example_config.toml`) 

## Workflow 1B (for cNMF results generated outside of cNMF-SNS)

If you want to generate annotated heatmaps for usage matrices from the standard cNMF tool. You will need to do the following:
  1. Ensure that you have run up to and including the `cnmf factorize` step. Take note of the `output_dir` and `name` parameters that you used with cNMF. Then, you can run the following cNMF-SNS commands, using the same data you used to input into cNMF.
  2. run `cnmfsns txt-to-h5ad` to create a file with input data/metadata and output it to `<output_dir>/<name>/name.h5ad`.
  3. run `cnmfsns model-odg --output_dir output_dir --name name -i <output_dir>/<name>/name.h5ad`.
  4. run `cnmfsns set-parameters --output_dir output_dir --name name -m genes_file -p <output_dir>/<name>/name.overdispersed_genes.txt` as well as the same parameters used to run cNMF, including:
    - _k_ values using `-k` or `--k_range` parameters
    - `--n_iter`
    - `--seed`
    - `--beta_loss`
  5. run `cnmfsns postprocess --output_dir output_dir --name name` to run cNMF-SNS postprocessing steps and amend the h5ad file with pre-computed cNMF results.
  6. run `cnmfsns annotated-heatmap --input_h5ad `<output_dir>/<name>/name.h5ad` --output_dir output_dir/name/` to create annotated heatmaps within the cnmf output directory.


## Workflow Part 2: Integration of multiple datasets
>> Note: the following workflow is under active development and may change.

### 1. Identify datasets for integration

Specify an output directory for your integration.

A [TOML](https://toml.io/en/) configuration file is the most flexible way to configure cNMF-SNS. An example is found in `scripts/example_config.toml`. The minimal data to include in a TOML file is the paths to each of the datasets. Default values for other parameters will be used if not specified. 

```
cnmfsns integrate -c config.toml -o output_directory
```

Alternatively, you can can also initialize a cNMF-SNS integration by providing a set of h5ad files from [`cnmfsns postprocess`](#6.-postprocessing) to integrate:
```
cnmfsns integrate -i file1.h5ad -i file2.h5ad -i file3.h5ad -o output_directory
```

Once the command has completed, outputs are located in `<output_directory>/integrate/`.
- UpSet plot of OD Genes between datasets
- plot correlation between cohorts
- rank reduction plots to exclude high ranks with highly correlated GEPs

### 2. Network-based analyses

```
cnmfsns create-network -o output_directory
```

Outputs from this command are located in `<output_directory>/sns_networks/<timestamp>/`