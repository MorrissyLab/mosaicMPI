# Command-line interface: Integration workflow

Each step of a workflow can be run as a subcommand within mosaicMPI. You can see which subcommands are available using:

```bash
mosaicmpi --help
```

Easily get help for each subcommand using like this:

```bash
mosaicmpi select-hvf --help
```

## Part II: Using mosaicMPI to integrate multiple datasets

Once you have run `mosaicmpi postprocess` on each of your datasets, they can be used as input for integration. For this tutorial, run the [CLI Factorization tutorial](cli-factorization.md) using the three data types in the [tutorial data](https://github.com/MorrissyLab/mosaicMPI/tree/main/tutorial_data) to generate three factorized datasets: snRNA, RNA, and protein.

### *1. [Optional] Mapping datasets to a common set of identifiers*

Before integration, it is important to have feature overlap between datasets. This requires converting individual datasets (before or after factorization) to a shared feature space such as Ensembl IDs for genes. Also, it is important to perform the integration with identifiers for the same species. If you wish to do cross species integration, then it is important to convert features to a common species. To facilitate this, mosaicMPI can convert common gene identifiers using the Ensembl database, from and to either gene names or ensemble gene IDs, within or between species.

For example, to convert human gene names to mouse Ensembl IDs (ie., identifiers beginning ENSMUSG):

```bash
mosaicmpi map-gene-ids -i dataset1.h5ad -o dataset1_mouse.h5ad --source_species hsapiens --dest_species mmusculus --source_ids gene_name --dest_ids ensembl_gene
```

### 2. Create a configuration file with input datasets

A [TOML](https://toml.io/en/) configuration file is the most flexible way to configure mosaicMPI through the command-line. This is generated using the command:

```bash
mosaicmpi create-config -i dataset1.h5ad -i dataset2.h5ad -i dataset3.h5ad -o config.toml
```
This will output a config.toml file in the current directory, which contains the datasets to be integrated as well as default parameters for the integration, which can be modified prior to running `mosaicmpi integrate`. For more information, see the [tutorial on editing config.toml files](cli-integration-configuration.md).

### 3. Integrating datasets

To integrate one or more datasets using mosaicMPI, run:

```bash
mosaicmpi integrate -c config.toml -o output_directory
```

Additional customization can be achieved using:
  - `-m communities.toml`: a TOML file with communities (eg., from a previous run), just to re-make figures
  - `-l colors.toml`: a color scheme for datasets, sample/cell labels (metadata), and communities

Once the command has completed, outputs in the output directory will include:

- files enabling downstream analysis:
  - `network_integration.pkl.gz`: a file that contains all datasets and the network integration information, including correlations, metadata, and input data. This can be used for downstream analysis of a mosaicMPI integration using the [python API](python_integration.ipynb), as well as with some CLI commands.
  - `pearson.df.npz`: pearson correlation matrix of all programs within and between datasets, compatible with [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
  - `program_graph.graphml`: GraphML-compatible file containing the nodes and edges of the program graph
  - `community_graph.graphml`: GraphML-compatible file containing the nodes and edges of the community graph
  - `communities.toml`: TOML file containing the communities from the program graph. This can be edited
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
  - `hvfeatures.txt`: table showing features that are overdispersed between datasets
  - `hvfeatures_upsetplot.pdf`: UpSet plot showing overlaps of overdispersed features between datasets.
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

- sample information
  - `categories/<dataset>/<annotation_layer>.pdf: bar plots showing the number of samples for categorical metadata

### 4. Label transfer between datasets

After integration, sample types can be predicted using `mosaicmpi transfer-labels`. Using an existing integration (`.pkl.gz` file), specify the source and destination datasets, as well as the variable you want to predict (`layer_name`). You can optionally provide the name of a data layer from the destination dataset as an annotation for the heatmap.

```bash
mosaicmpi transfer-labels -o output_directory -n network_integration.pkl.gz -l layer_name -s Dataset1 -d Dataset2 -a annotation_layer
```

Additional customization can be achieved using:
  - `-m colors.toml`: a color scheme sample/cell labels (metadata) that provides consistency across plots

Once the command has completed, outputs in the output directory will include:

- files enabling downstream analysis:
  - `transfer_score.txt`: tab-separated file with the transfer scores for all label transfers.
  
- transfer score for each sample in the destination dataset
  - `s.<source>_d.<dest>_l.<layer>.pdf`: clustered heatmap of the transfer scores for each sample in the destination dataset. Multiple figures if multiple datasets or layers are specified.
  - `legend.png`, `legend.pdf`: legend for heatmap annotation tracks
