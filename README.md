# cNMF-SNS
cNMF Solution Neighborhood Space

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

Each step of a workflow is run as a separate command within cNMF-SNS. You can see which subcommands are available using:
```
cnmfsns --help
```

Easily get help for each subcommand using, for example:

```
cnmfsns model-odg --help
```

### 1. (Optional) Create AnnData object from text files with expression and annotations.

If expression and annotation data is in text files, this utility can combine and check them into a .h5ad file for convenience.

TPM and count matrices must have identical axes (ie., row names and column names). The metadata file must be 
```
cnmfsns txt-to-h5ad --tpm tpm.txt --counts counts.txt --metadata metadata.txt
```

### 2. (Optional) Check existing h5ad files for minimum requirements for cNMF.

Check h5ad objects for cells, spots, samples, or genes which have missing values, negative values, or sum of 0.

cNMF  supports input data that is sparse (i.e. with zeros), but not with missing values. When missing values are present (eg. from concatenation of datasets with partially overlapping features), the default behaviour is to subset the input matrix to shared features/genes only, but it is recommended to either run each dataset separately or use a dense, imputed data matrix. cNMF will warn the user if missing data is present.

```
cnmfsns check_h5ad file.h5ad file_filtered.h5ad
```

### 3. Model gene overdispersion to select genes for factorization.

The first step of cNMF-SNS is to check inputs for completeness and integrity, as well as to guide the selection of parameters for running cNMF on a particular dataset.


Deconvolution of a gene expression dataset using cNMF requires a set of overdispersed genes which will be used for factorization. GEPs will include all genes after a re-fitting step, but error will only be calculated on overdispersed genes, providing the user the opportunity to decide which genes are most informative.

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

Use `cnmf`'s parallelization, which is adaptable for any cluster configuration. This command defaults to single CPU run so a small test dataset can be run like this:
```
cnmfsns factorize --name example_run
```
For submitting jobs to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/cNMF-SNS/tree/main/scripts/slurm.sh).

After editing the script to ensure it is suitable for your compute cluster, you can massively parallelize your run using:
```
cnmfsns factorize --name example_run --slurm-script /path/to/slurm.sh
```

### 6. Postprocessing

This step will check to ensure that all factorizations completed successfully, and then will create consensus GEPs and usages

- will check to ensure all factorizations completed successfully
- upon completion, `cnmf combine` and `cnmf consensus` steps to get consensus GEPs and usages
- call marker genes
- default local_density_threshold = 2.0
- create output plots from cnmf, including k selection plot
- compress cnmf output into h5ad file for exporting to python or R environments
- optionally deletes cnmf working directory

## Workflow for starting with outputs from cNMF

> Warning: This method does not enforce common cNMF parameters/methods between datasets/runs. Proceed at your own risk!

### 1. Package cNMF runs into `.h5mu` file
In the case of integrated already completed cNMF runs, a quick command will generate the h5mu file, which encodes the cNMF results in a cross-platform object readable in R and Python.

```
cnmfsns create-h5mu -d cnmf_result_dir -o cnmf_run.h5mu
``` 

### 2. Initialize the cNMF-SNS integration using configuration

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

## TOML configuration file

Parameters for SNS integration are specified in the TOML configuration file. If none are chosen, default values for each parameter will be used. 


### 4B. `cnmfsns create-h5mu`
> Note: Use of this is for backwards compatibility with cNMF runs that were started outside of the cnmfsns framework, and thus, cNMF results are not guaranteed to be complete, or the parameters correct.
- Packages cNMF results into a MuData .h5mu file within the parent directory for exporting to python or R environments
- Optionally deletes cnmf working directory

### 5. `cnmfsns annotate-usages`

- create annotated heatmaps from h5ad
- _this step can be run any time after_ `cnmfsns postprocess`_, not necessarily in this order_ 

### 6. `cnmfsns initialize`

- import multiple cnmf outputs for integration
    - requires h5mu files from previous runs, or
    - a config file (spec in progress, see example in 'scripts/example_config.toml')
- specify an output directory for the integration being performed (eg. "gbm_proteomics")
- UpSet plot of OD Genes between datasets
- plot correlation between cohorts

### 7. `cnmfsns optimize-integration`

- plot to compare integration using spearman and pearson
- plot to decide range of k?
- we need to prioritize communities that represent more than just 1 sample
- collapse GEPs with similar
- metrics for integration / shannon index?
- metrics for communities 

### 7. `cnmfsns create-sns`

- uses config information from 
- creates SNS map and all plots without metadata (fast step)

### 8. `cnmfsns annotate-sns`

- creates spike plots and related data outputs (slower steps)
