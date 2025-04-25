# Command-line interface: Factorization workflow

Each step of a workflow can be run as a subcommand within mosaicMPI. You can see which subcommands are available using:

```bash
mosaicmpi --help
```

Easily get help for each subcommand using like this:

```bash
mosaicmpi select-hvf --help
```

## Part I: Using mosaicMPI to factorize datasets

For this part of the tutorial, subsampled single-nuclei RNA-Seq data is used from the [tutorial data](https://github.com/MorrissyLab/mosaicMPI/tree/main/tutorial_data).

### 1. Create AnnData object from text files with gene expression and metadata

If gene expression and annotation data is in text files, this utility can combine them into a .h5ad file for downstream tools. Import the data and metadata into a
combined AnnData h5 file. For count data, simply use:

```bash
mosaicmpi txt-to-h5ad --data cptac_snRNA_subsampled.txt --metadata cptac_snRNA_subsampled.metadata.txt -o cptac_snrna.h5ad
```

If no counts data is available and/or the dataset is normalized (eg., for non-count based assays), you can also specify `--is_normalized` to prevent mosaicMPI from
performing a TPM normalization step for the purposes of overdispersed gene selection. If `--is_normalized` is specified, the input matrix is used both for overdispersed
gene selection and for factorization. 

By default, text files are tab-separated, although other characters can be specified using `--data-delimiter` and `--metadata-delimiter`.

By default, Expression data must be indexed as follows.
  - Rows are observations (eg., samples/cells/spots); first column must be unique observation identifiers
  - Columns are features (eg. genes); the first row must be unique feature namesk_selection
If your data is in the opposite orientation, specify `--transpose`.

Values must be numerical, but missing values are permitted. For text inputs, these must be 'empty' cells, rather than "NA" or "NaN" text values. When missing
values are included in the inputs, you must run `mosaicmpi impute-knn` or `mosaicmpi impute-zeros` prior to feature selection and factorization.

Metadata must be indexed as follows:

  - The first column must be sample/cell/spot IDs, and should be unique
  - Other columns are metadata 'layers' and must be labelled. Values can be numerical, boolean, or categorical types.
      > Note: if any values in a column are not numerical, the entire column will be treated as categorical. This can have implications for annotated heatmaps
      where numerical data is usually presented as a continuous color scale, not a set of distinct colors. If a column is numerical with missing values, then these should be empty values (not "NA", "NaN", etc.)
  - Missing values are acceptable. For categorical data, these will be plotted in an "Other" category. For numerical data, these will be ignored.

### 2. Check existing h5ad files for unfactorizable features

Check h5ad objects for rows or columns which have missing values, negative values, or variance of 0.

cNMF  supports input data that is sparse (i.e. with zeros), but not with missing values. When missing values are present (eg. from concatenation of datasets with partially overlapping features), the default behaviour is to subset the input matrix to shared features only, but it is recommended to either run each dataset separately or use a dense, imputed data matrix.

```bash
mosaicmpi check-h5ad -i cptac_snrna.h5ad -o cptac_snrna.h5ad
```

### 3. Select highly-variable features (HVFs) for factorization

Deconvolution of a gene expression dataset using cNMF requires a list of HVFs which will be used for factorization. Programs will include all features after a re-fitting step, but cNMF will primarily optimize the fit based on the HVFs, providing the user the opportunity to decide which features are most informative. Since cNMF scales the features to unit variance prior to NMF, it is important to remove features whose variance could be attributable to noise.

In most scenarios, automatic HVF selection can be accomplished on the dataset. Normalized data is used to model mean-variance dependence, and then a p-value is calculated from the model's positive residuals to identify significantly overdispersed features using p-values from the F-distribution.

```bash
mosaicmpi select-hvf default -i cptac_snrna.h5ad -o results -n cptac_snrna --alpha 0.05
```

Alternatively, input a custom list of features using a whitespace-delimited file of features:
```bash
mosaicmpi select-hvf custom -i cptac_snrna.h5ad -o results -n cptac_snrna --feature_list features.txt
```

While `mosaicmpi select-hvf default` is the most flexible means to select HVFs, other subcommands for `select-hvf` include previously published methods, reimplemented in mosaicMPI:
  - `stdeconvolve` method (Miller, et al. Nat. Comm. 2022)
  - `cnmf` method (Kotliar et. al., 2019)

`select-hvf` will create a directory with the name of the run inside the output directory, in this case `results/cptac_snrna/`.
Inside this directory it will create a file called `feature_mean_var.pdf` which shows the relationship between feature mean and variance, as well as the model (black line) and selected features in red. Additionally, a histogram of the overdispersion score will be created (either `feature_vscore.pdf` or `feature_odscore.pdf`, depending on the parameters).

Sometimes it is desirable to stratify the dataset and conduct HVF selection separately for each partition of the data. In this tutorial, let's say we want to make sure that we capture all the genes that might capture the differences within each patient's scRNA-Seq data. Thus, we can identify HVFs separately for each patient and then take the union:

```bash
mosaicmpi select-hvf default -i cptac_snrna.h5ad -o results -n cptac_snrna --alpha 0.05 --stratify_by patient
```

When running in stratified mode, the two plots will display not just the selected/unselected features for the whole dataset, but for each strata separately.

### 4. Initialize cNMF

Once highly-variable features (HVFs) have been selected, you can set up the factorization for certain values of k as follows:

```bash
mosaicmpi initialize-cnmf -o results -n cptac_snrna -k 2 -k 3 -k 4 -k 5
```

In many cases, you will want to factorize a wider range of values using the `--k_range` parameter. We recommend the following range for initial exploration of bulk and single-cell datasets:

```bash
mosaicmpi initialize-cnmf -o results -n cptac_snrna --k_range 2 60 1
```
This will factorize from k = 2 - 60. A more complex range of k values can also be set up. For example, to perform cNMF using a range of k values from 5 to 100, with a step of 5 (ie.: 5, 10, 15, ... 100), you would specify `--k_range 5 100 5`. Note that you can also factorize over all ranks and choose a subset at the integration step later, if this is not known in advance.

### 5. Factorize

Factorize the input data. While parameters can be provided which allow for custom parallelization, by default mosaicmpi uses a single CPU:

```bash
mosaicmpi factorize -o results -n cptac_snrna
```

For submitting jobs to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/mosaicMPI/tree/main/scripts/slurm_factorize.sh).

After editing the script to ensure it is suitable for your HPC cluster, mosaicMPI will submit jobs using SLURM's `sbatch` command to parallelize factorization.

```bash
mosaicmpi factorize -o results -n cptac_snrna --slurm_script /path/to/slurm_factorize.sh
```

### 6. Postprocess

This step will check to ensure that all factorizations completed successfully, and then will create consensus programs and usages, updating the `.h5ad` file with the cNMF solution.

```bash
mosaicmpi postprocess -o results -n cptac_snrna
```

To submit as a job to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/mosaicMPI/tree/main/scripts/slurm_postprocess.sh).

After editing the script to ensure it is suitable for your HPC cluster, mosaicMPI will submit jobs using SLURM's `sbatch` command to parallelize factorization.

```bash
mosaicmpi postprocess -o results -n cptac_snrna --slurm_script /path/to/slurm_postprocess.sh
```

For downstream analyses, the input data and cNMF programs are all contained in `results/cptac_snrna/cptac_snrna.h5ad`.

### *7. [Optional] Create annotated heatmaps of program usage*

This step will create annotated heatmaps of program usages from mosaicMPI outputs:

```bash
mosaicmpi usage-heatmap -o results -i results/cptac_snrna/cptac_snrna.h5ad
```

To provide custom colors for the metadata layers, you can specify a `metadata_colors.toml` file. You can also gain finer control over the plots including subsampling cells, showing only some metadata fields, and other parameters. To learn more, run `mosaicmpi usage-heatmap -h`.


### *8. [Optional] Calculate overrepresentation for categorical metadata*

This step will create overrepresentation heatmaps from mosaicMPI outputs:

```bash
mosaicmpi overrepresentation -o results/overrepresentation -i results/cptac_snrna/cptac_snrna.h5ad -c celltype
```

To provide custom colors for the metadata layers, you can specify a `metadata_colors.toml` file. Learn about other parameters by running `mosaicmpi overrepresentation -h`.


### *9. [Optional] Update the metadata for a dataset's .h5ad file*

Metadata can be updated for a dataset using the following command:

```bash
mosaicmpi update-h5ad-metadata -i results/cptac_snrna/cptac_snrna.h5ad -m cptac_snRNA_subsampled.metadata_updated.txt
```

