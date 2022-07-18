# cNMF-SNS

cNMF Solution Neighborhood Space

![](https://img.shields.io/badge/version-0.2.0-blue)

## Installation

### 1. Using `pip` to install the latest version from GitHub:

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

## Workflow

cNMF-SNS is a command line tool for deconvoluting and integrating gene expression and other high-dimensional data.

Each step of a workflow is run as a separate command within cNMF-SNS. You can see which subcommands are available using:
```
cnmfsns --help
```

Easily get help for each subcommand using, for example:

```
cnmfsns model-odg --help
```

### 1. Create AnnData object from text files with expression and annotations.

If expression and annotation data is in text files, this utility can combine them into a .h5ad file for downstream tools.

```
cnmfsns txt-to-h5ad --tpm tpm.txt --counts counts.txt --metadata metadata.txt -o dataset.h5ad
```

#### Input semantics

Expression (normalized and count) data must be indexed as follows:
  - The first column must be sample/cell/spot IDs
  - The first row must be genes or other features.

Metadata must be indexed as follows:
  - The first column must be sample/cell/spot IDs
  - Other columns are metadata 'layers' and must be labelled. Values can be numerical, boolean, or categorical types.
  - Missing values are acceptable. For categorical data, these will be plotted as an "Other" category. For numerical data, these will be ignored.

### 2. Check existing h5ad files for minimum requirements for cNMF.

> Warning: Not completely implemented yet!

Check h5ad objects for cells, spots, samples, or genes which have missing values, negative values, or sum of 0.

cNMF  supports input data that is sparse (i.e. with zeros), but not with missing values. When missing values are present (eg. from concatenation of datasets with partially overlapping features), the default behaviour is to subset the input matrix to shared features/genes only, but it is recommended to either run each dataset separately or use a dense, imputed data matrix. cNMF will warn the user if missing data is present.


```
cnmfsns check_h5ad file.h5ad file_filtered.h5ad
```

### 3. Model gene overdispersion to select genes for factorization.

Deconvolution of a gene expression dataset using cNMF requires a set of overdispersed genes/features which will be used for factorization. GEPs will include all genes after a re-fitting step, but cNMF will only optimize the fit for overdispersed genes, providing the user the opportunity to decide which genes are most informative.

Since cNMF performs variance scaling on the input matrix, it is important to remove genes whose variance could be attributable to noise. cNMF-SNS supports two methods for overdispersed gene selection:
    Model gene overdispersion and plot calibration plots for selection of overdispersed genes, using two methods:
    
    - `cnmf`: v-score and minimum expression threshold (cNMF method: Kotliar, et al. eLife, 2019). This method is only suitable for count data.
    - `default`: residual standard deviation after modeling mean-variance dependence. (STdeconvolve method: Miller, et al. Nat. Comm. 2022) This method makes fewer assumptions about the input data but requires a visual check since the optimal threshold depends on the data type.

To produce plots to guide selection of overdispersed genes, run the following command:

```
cnmfsns model-odg --name example_run --input file_filtered.h5ad
```

### 4. Select overdispersed genes and parameters for factorization

A simple command sets the parameters for the factorization using default parameters:
  - overdispersed genes: od-score >= 1
  - k = 2 - 10 inclusive
  - beta-loss metric: kullback-leibler

```
cnmfsns set_parameters --name example_run
```

### 5. Perform cNMF factorization

Factorize the input data. While parameters can be provided which allow for custom parallelization, by default cnmfsns uses a single CPU:

```
cnmfsns factorize --name example_run
```
For submitting jobs to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/cNMF-SNS/tree/main/scripts/slurm.sh).

After editing the script to ensure it is suitable for your compute cluster, you can maximally parallelize your run using:
```
cnmfsns factorize --name example_run --slurm-script /path/to/slurm.sh
```

### 6. Postprocessing

This step will check to ensure that all factorizations completed successfully, and then will create consensus GEPs and usages, updating the `.h5ad` file with the cNMF solution.

```
cnmfsns postprocess --name example_run
```
### 7. Created annotated heatmaps of GEP usages

This step will create annotated heatmaps of GEP usages from cNMF-SNS outputs:

```
cnmfsns create-annotated_heatmaps --name example_run
```

To provide custom colors for the metadata layers, you can specify a TOML-formatted file with a `metadata_colors` mapping (see `scripts/example_config.toml`) 

> ### Working with cNMF results generated outside of cNMF-SNS:
> 
> If you want to generate annotated heatmaps for usage matrices from the standard cNMF tool, you will need to do the following:
>   1. run up to and including the `cnmf factorize` step.
> Take note of the `output_dir` and `name` parameters that you used with cNMF. Then, you can run the following cNMF commands, using the same data you used to input into cNMF:
>   2. run `cnmfsns txt-to-h5ad` to create a file with input data/metadata and output it to `<output_dir>/<name>/name.h5ad`.
>   3. run `cnmfsns postprocess --output_dir output_dir --name name` to run cNMF-SNS postprocessing steps
>   4. run `cnmfsns create_annotated_heatmaps --output_dir output_dir --name name` to create annotated heatmaps.

## Incomplete documentation >>>


### Initialize the cNMF-SNS integration using configuration

A [TOML](https://toml.io/en/) configuration file is the most flexible way to configure cNMF-SNS. An example is found in `scripts/example_config.toml`.

```
cnmfsns initialize -c config.toml -o output_directory
```

cNMF-SNS can also initializae an integration by providing a set of h5mu files to integrate:
```
cnmfsns initialize -o output_directory -i file1.h5mu file2.h5mu file3.h5mu file4.h5mu
```

After this step, several plots are generated which can help guide parameter selection for the next step. Parameters that are required for the integration include:
    1. range(s) of k (decide based on # of samples?)
    2. Specific values of k (eg. 5, 10, 15, 20)
    3. Community resolution
    4. 

```
cnmfsns create-sns -o output_directory
```

### TOML configuration file

Parameters for SNS integration are specified in the TOML configuration file. If none are chosen, default values for each parameter will be used. 

### `cnmfsns initialize`

- import multiple cnmf outputs for integration
    - requires h5ad files from previous runs, or
    - a config file (spec in progress, see example in 'scripts/example_config.toml')
- specify an output directory for the integration being performed (eg. "gbm_proteomics")
- UpSet plot of OD Genes between datasets
- plot correlation between cohorts

### `cnmfsns optimize-integration`

- plot to compare integration using spearman and pearson
- plot to decide range of k?
- we need to prioritize communities that represent more than just 1 sample
- collapse GEPs with similar
- metrics for integration / shannon index?
- metrics for communities 

### `cnmfsns create-sns`

- uses config information from 
- creates SNS map and all plots without metadata (fast step)

### `cnmfsns annotate-sns`

- creates spike plots and related data outputs (slower steps)
