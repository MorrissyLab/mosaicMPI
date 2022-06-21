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



## Workflow

Each step of the workflow is run as separate commands within cnmfsns

### 1. `cnmfsns inspect-inputs`

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

### 6. `cnmfsns integrate`

- import multiple cnmf outputs for integration
- specify an output directory for the integration being performed (eg. "gbm_proteomics")
- plot correlation between cohorts
- plot to decide range of k?
- plot to compare integration using spearman and pearson

### 7. `cnmfsns create-sns`

- outputs to a subdirectory for this particular set of parameters
- creates SNS map and all plots without metadata

### 8. `cnmfsns annotate-sns`

- creates spike plots and related data outputs
- patient 
