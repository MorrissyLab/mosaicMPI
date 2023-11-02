# Command-line interface and workflow

Each step of a workflow can be run as a subcommand within mosaicMPI. You can see which subcommands are available using:

```bash
mosaicmpi --help
```

Easily get help for each subcommand using like this:

```bash
mosaicmpi model-odg --help
```

## Part I: Factorizing individual datasets

### 1. Create AnnData object from text files with gene expression and metadata

If expression and annotation data is in text files, this utility can combine them into a .h5ad file for downstream tools. If you have counts data, then use the following command:

```bash
mosaicmpi txt-to-h5ad --data_file counts.txt --metadata metadata.txt -o dataset.h5ad
```

If no counts data is available and the dataset is normalized (eg., for non-count based assays), you can also specify `--is_normalized` to prevent mosaicMPI from performing a TPM normalization step for the purposes of overdispersed gene selection. If `--is_normalized` is specified, the input matrix is used both for overdispersed gene selection and for factorization.

By default, text files are tab-separated, although other characters can be specified using `--data-delimiter` and `--metadata-delimiter`.

By default, Expression data must be indexed as follows.
  - Rows are samples/cells/spots; first column must be sample/cell/spot IDs
  - Columns are genes/features; the first row must be feature names
If your data is in the opposite orientation, specify `--transpose`.

Values must be numerical, but missing values are permitted. For text inputs, these must be 'empty' cells, rather than "NA" or "NaN" text values. When missing values are included in the inputs, you must run `mosaicmpi impute-knn` or `mosaicmpi impute-zeros` prior to feature selection and factorization.

Metadata must be indexed as follows:

  - The first column must be sample/cell/spot IDs
  - Other columns are metadata 'layers' and must be labelled. Values can be numerical, boolean, or categorical types.
      > Note: if any values in a column are not numerical, the entire column will be treated as categorical. This can have implications for annotated heatmaps where numerical data is usually presented as a continuous color scale, not a set of distinct colors. If a column is numerical with missing values, then these should be empty values (not "NA", "NaN", etc.)
  - Missing values are acceptable. For categorical data, these will be plotted in an "Other" category. For numerical data, these will be ignored.

Additionally 

### 2. Check existing h5ad files for unfactorizable genes.

Check h5ad objects for rows or columns which have missing values, negative values, or variance of 0.

cNMF  supports input data that is sparse (i.e. with zeros), but not with missing values. When missing values are present (eg. from concatenation of datasets with partially overlapping features), the default behaviour is to subset the input matrix to shared features/genes only, but it is recommended to either run each dataset separately or use a dense, imputed data matrix. cNMF will warn the user if missing data is present.

```bash
mosaicmpi check-h5ad -i file.h5ad -o file_filtered.h5ad
```

### 3. Model gene overdispersion to select genes for factorization.

Deconvolution of a gene expression dataset using cNMF requires a set of overdispersed genes/features which will be used for factorization. programs will include all genes after a re-fitting step, but cNMF will only optimize the fit for overdispersed genes, providing the user the opportunity to decide which genes are most informative.

Since cNMF performs variance scaling on the input matrix, it is important to remove genes whose variance could be attributable to noise. mosaicMPI supports two methods for overdispersed gene selection:

  - `cnmf`: v-score and minimum expression threshold (cNMF method: Kotliar, et al. eLife, 2019). This method is only suitable for count data.
  - `default`: residual standard deviation after modeling mean-variance dependence. (STdeconvolve method: Miller, et al. Nat. Comm. 2022) This method makes fewer assumptions about the input data but requires a visual check since the optimal threshold depends on the data type. Differently than all other methods, this method does not make assumptions about mean/sum of expression across samples, permitting the discovery of rare cell types.

To produce plots to guide selection of overdispersed genes, run the following command:

```bash
mosaicmpi model-odg --name example_run --input file_filtered.h5ad
```

This command will create a directory with the name of the run inside the output directory (defaults to current working directory).

### 4. Select overdispersed genes and parameters for factorization

Once you have decided on a method for selecting overdispersed genes and are ready for factorization, you can easily use the default parameters and select values of k as follows:

```bash
mosaicmpi set-parameters --name example_run -k 2 -k 3
```

In many cases, you will want to factorize a wider range of values using the `--k_range` parameter. We recommend the following range for initial exploration of bulk and single-cell datasets:

```bash
mosaicmpi set-parameters --name example_run --k_range 2 60 1
```
This will factorize from k = 2 - 60. A more complex range of k values can also be set up. For example, to perform cNMF using a range of k values from 5 to 100, with a step of 5 (ie.: 5, 10, 15, ... 100), you would specify `--k_range 5 100 5`.

Although programs discovered using cNMF contain all features, factorization is directed by a set of overdispersed genes. The default behaviour is to select overdispersed genes using an od-score > 1.0. You can also choose from a number of different methods for selecting overdispersed_genes using the `-m` and `-p` parameters:

  - `-m cnmf_topn -p 2000`: select top 2000 genes using cNMF's model and minimal mean threshold (Kotliar et al., 2019, eLife)
  - `-m default_quantile -p 0.8`: Select top 20% of genes when ranked by od-score
  - `-m genes_file -p path/to/genesfile.txt`: use a custom list of genes

### 5. Perform cNMF factorization

Factorize the input data. While parameters can be provided which allow for custom parallelization, by default mosaicmpi uses a single CPU:

```bash
mosaicmpi factorize --name example_run
```

For submitting jobs to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/mosaicMPI/tree/main/scripts/slurm_factorize.sh).

After editing the script to ensure it is suitable for your HPC cluster, mosaicMPI will submit jobs using SLURM's `sbatch` command to parallelize factorization.

```bash
mosaicmpi factorize --name example_run --slurm_script /path/to/slurm_factorize.sh
```

### 6. Postprocessing

This step will check to ensure that all factorizations completed successfully, and then will create consensus programs and usages, updating the `.h5ad` file with the cNMF solution.

```bash
mosaicmpi postprocess --name example_run
```

To submit as a job to the SLURM job scheduler, you can download a sample job submission script [here](https://github.com/MorrissyLab/mosaicMPI/tree/main/scripts/slurm_postprocess.sh).

After editing the script to ensure it is suitable for your HPC cluster, mosaicMPI will submit jobs using SLURM's `sbatch` command to parallelize factorization.

```bash
mosaicmpi postprocess --name example_run --slurm_script /path/to/slurm_postprocess.sh
```


For downstream analyses, the input data and cNMF programs are all contained in `example_run/example_run.h5ad`.

### 7. Created annotated heatmaps of program usages

This step will create annotated heatmaps of program usages from mosaicMPI outputs:

```bash
mosaicmpi annotated-heatmap --output_dir example_run -i example_run/example_run.h5ad
```

To provide custom colors for the metadata layers, you can specify a `metadata_colors.toml` file.

## Part II: Integration of multiple datasets

One you have run `mosaicmpi postprocess` on each of your datasets, they can be used as input for integration.

### 1. Create a configuration file with input datasets

A [TOML](https://toml.io/en/) configuration file is the most flexible way to configure mosaicMPI. This is generated using the command:

```bash
mosaicmpi create-config -i dataset1.h5ad -i dataset2.h5ad -i dataset3.h5ad -o config.toml
```
This will output a config.toml file in the current directory, which contains the datasets to be integrated as well as default parameters for the integration, which can be modified prior to running `mosaicmpi integrate`. For more information, see the [tutorial on editing config.toml files](configtoml.md).

### 2. Integrating datasets

To integrate one or more datasets using mosaicMPI, run:

```bash
mosaicmpi integrate -c config.toml -o output_directory
```

Additional customization can be achieved using:
  - `-m communities.toml`: a TOML file with communities (eg., from a previous run), just to re-make figures
  - `-l colors.toml`: a color scheme for datasets, sample/cell labels (metadata), and communities

Once the command has completed, outputs in the output directory will include:

- files enabling downstream analysis:
  - `network_integration.pkl.gz`: a file that contains all datasets and the network integration information, including correlations, metadata, and input data. This can be used for downstream analysis of a mosaicMPI integration using the [mosaicMPI API](getting-started.ipynb).
  - `pearson.df.npz`: pearson correlation matrix of all programs within and between datasets, compatible with [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
  - `program_graph.graphml`: GraphML-compatible file containing the nodes and edges of the program graph
  - `communities.toml`: TOML file containing the communities from the program graph
  - `program_communities.txt`: a text file with communities from the program graph. Nodes from a single community can be used for the `subset_nodes` parameter of the TOML configuration file to subcluster a community.
  - `colors.toml`: TOML file containing the visually-distinct colors calculated for plots that require color mappings for datasets, communities, and metadata labels. This can be re-used in subsequent runs for consistency.

- representative programs: one program per dataset and community that is nearest to the median of programs (and thus representative of that community)
  - `representative_programs.txt`: genes × programs matrix
  - `representative_program_usages.txt`: samples/cells × programs matrix

- legends that are relevant to multiple plots:
  - `dataset_colors_legend.pdf`: legend with dataset colors
  - `metadata_colors_legend.pdf`: legend with colors for metadata labels

- plots and tables providing an overview of the integration process and results:
  - `pairwise_corr.pdf`: correlation distributions between datasets
  - `pairwise_corr_overlaid.pdf`: correlation distributions between datasets overlaid to emphasize thresholding strategy
  - `pairwise_corr_thresholds.txt`: thresholds for each dataset and dataset pair to calibrate for systematic differences between datasets.
  - `k_values.txt`: information and statistics for each cNMF run included in the integration. For each dataset and rank, there is information about whether cNMF results were found (`cNMF results`), whether that rank passed the `max_k_filter` based on the `max_k_median_corr`, as well as the `prediction_error` and `stability` of the cNMF run. Finally, `selected_k` indicates whether programs from this rank were included in the network.
  - `rank_reduction.pdf`: plot of the median correlation of all programs up to each rank in the x-axis. Ranks below which the median correlation rises above the value set in the TOML configuration file will be excluded.
  - `features.txt`: table showing features present in different datasets.
  - `features_upsetplot.pdf`: UpSet plot showing overlaps of features between datasets.
  - `overdispersed_features.txt`: table showing features that are overdispersed between datasets
  - `overdispersed_features_upsetplot.pdf`: UpSet plot showing overlaps of overdispersed features between datasets.
  - `node_stats.txt`: number of nodes before and after node and edge filters, per dataset.
  - `community_contribution.pdf`: contribution of datasets and ranks to each community

- program network plots:
  - `network_communities.pdf`: Colored by community
  - `network_datasets.pdf`: Colored by dataset
  - `network_rank.pdf`: Colored by dataset, size inversely proportional to rank (k)
  - `network_n_patients.pdf`: Node size is proportional to the number of patients with this program as its maximum
  - `network_n_samples.pdf`: Node size is proportional to the number of samples with this program as its maximum
  - `annotated_programs/overrepresentation_network/<dataset>/<annotation_layer>.pdf`: plots overrepresentation of each category on the network for categorical metadata
  - `annotated_programs/correlation_network/<dataset>/<annotation_layer>.pdf`: plots correlation of numerical metadata to program usage across samples on the network

- program bar plots:
  - `annotated_programs/overrepresentation_bar/<dataset>/<annotation_layer>.pdf`: plots overrepresentation of each category for each program for categorical metadata
  - `annotated_programs/correlation_bar/<dataset>/<annotation_layer>.pdf`: plots correlation of numerical metadata to program usage across samples for each program

- community network plots:
  - `community_network_summary.pdf`: Each community is represented as a single node located in the centroid of the community in the program network plots. Nodes are colored by community, and the edges are proportional to the number of edges in the program networks.
  - `annotated_communities/overrepresentation_network/<dataset>/<annotation_layer>.pdf`: plots overrepresentation of each category on the network for categorical metadata
  - `annotated_communities/correlation_network/<dataset>/<annotation_layer>.pdf`: plots correlation of numerical metadata to program usage across samples on the network
  - `annotated_communities/patient_network/<annotation_layer>/<n>samplesperpatient.pdf`: plots the community usage for each sample within a patient, averaged and colored by sample category for categorical metadata.

- community bar plots:
  - `annotated_communities/overrepresentation_bar/<dataset>/<annotation_layer>.pdf`: plots overrepresentation of each category for each community for categorical metadata
  - `annotated_communities/correlation_bar/<dataset>/<annotation_layer>.pdf`: plots correlation of numerical metadata to program usage across samples for each community