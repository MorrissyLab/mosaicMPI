# Command-line interface: Integration workflow

Each step of a workflow can be run as a subcommand within mosaicMPI. You can see which subcommands are available using:

```bash
mosaicmpi --help
```

Easily get help for each subcommand using like this:

```bash
mosaicmpi ssgsea --help
```

## Part III: Gene set analysis of programs

mosaicMPI programs are factors whose variation explains expression datasets well. Whether discovered at single-cell, spatial, or bulk datasets, a useful tool for characterization is gene set analysis. mosaicMPI enables gene set analysis using ssGSEA or g:Profiler.

### ssGSEA

Compute and plot ssGSEA Normalized Enrichment Scores (NES) for mosaicMPI programs. If a network_integration.pkl.gz file is provided, ssGSEA is performed on reprepresentative programs from an integration. If a dataset .h5ad file is provided, ssGSEA is performed on all programs for that dataset.

To run ssGSEA on all cNMF programs for a dataset, using GO:BP terms (the default):

```bash
mosaicmpi ssgsea -o ssgsea_results -n cptac_snrna.h5ad -g GO_Biological_Process_2023
```

Alternatively, a custom GMT file can be used with the `-g` parameter to control the gene sets more directly.

### G:Profiler

Compute gene set enrichments for marker genes from mosaicMPI programs. If a network_integration.pkl.gz file is provided, gProfiler is run for reprepresentative programs from an integration. If a dataset .h5ad file is provided, g:Profiler is performed on all programs for that dataset.

To run ssGSEA on all cNMF programs for a dataset, using GO:BP terms (by default, all G:Profiler sources are used):

```bash
mosaicmpi gprofiler -o gprofiler_results -n cptac_snrna.h5ad -g GO:BP
```

Alternatively, a custom GMT file can be used with the `-g` parameter to control the gene sets more directly.
