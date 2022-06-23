# cNMF-SNS
cNMF Solution Neighborhood Space


## Installation

### 1. Using `pip` to install the latest version from GitHub:

If you use SSH authentication for GitHub, use the following:
```
pip install git+ssh://git@github.com/MorrissyLab/cNMF-SNS.git
```

If you have a personal access token from GitHub installed and properly configured, you can use:
```
pip install git+https://github.com/MorrissyLab/cNMF-SNS.git
```

### 2. Using `pip` (PyPI version)

>> Note: This will work only when cNMF-SNS has been published to PyPI.

`pip install cnmfsns`


## Workflows

Each step of a workflow is run as a separate command within cnmfsns.

### 1. Starting with expression matrix
```
cnmfsns 
```

### 2. Starting with completed cNMF runs
In the case of integrated already completed cNMF runs, a quick command will generate the h5mu file which will enable importing the data into 
```
cnmfsns create-h5mu -d cnmf_result_dir -o cnmf_run.h5mu
```
```
cnmfsns initialize -c config.toml -o output_directory
```
After this step, several plots are generated which can help guide parameter selection for the next step:
```
cnmfsns create-sns -o output_directory
```


## Metadata


## TOML configuration file

Parameters for SNS integration are specified in the TOML configuration file. If none are chosen, default values for each parameter will be used. 


## Overview of each cnmfsns command

### 1.`cnmfsns inspect-inputs`

- Preempt errors by doing QC on the input data matrices
- (?) Check sample names matching to metadata key columns
- Check missing data
- create odgenes plots to guide threshold selection, but do not select genes yet
- 
### 2. `cnmfsns select-genes`

- Select marker genes
- wrap `cnmf prepare` step which creates directory for cnmf outputs
- prepare data for factorization
- 
### 3A. `cnmfsns factorize`

- Use `cnmf`'s methods for parallelization, which is adaptable for any cluster configuration, it defaults to single CPU run so a small test dataset will have very simple commands.

### 3B. `cnmfsns factorize --config morrissylab`

- Use an optimized one-step script for use by our lab on ARC
- Will automatically assess number of jobs and submit jobs to scheduler
- 
### 4A. `cnmfsns postprocess`

- will check to ensure all factorizations completed successfully
- upon completion, `cnmf combine` and `cnmf consensus` steps to get consensus GEPs and usages
- default local_density_threshold = 2.0
- create output plots from cnmf, including k selection plot
- compress cnmf output into h5ad file for exporting to python or R environments
- optionally deletes cnmf working directory

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
- plot to compare integration using spearman and pearson
- plot to decide range of k?

### 7. `cnmfsns create-sns`

- uses config information from 
- creates SNS map and all plots without metadata (fast step)

### 8. `cnmfsns annotate-sns`

- creates spike plots and related data outputs (slower steps)
