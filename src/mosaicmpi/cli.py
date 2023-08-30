
from .dataset import Dataset
from .integration import Integration
from .config import Config
from .colors import Colors
from .network import Network
from .cnmf import cNMF
from .plots import *
from . import utils, __version__, cpus_available

import os
import logging
import subprocess
import collections
import sys
from datetime import datetime
from typing import Optional, Mapping

import click
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class _OrderedGroup(click.Group):
    """
    Overwrites Groups in click to allow ordered commands.
    """
    def __init__(self, name: Optional[str] = None, commands: Optional[Mapping[str, click.Command]] = None, **kwargs):
        super(_OrderedGroup, self).__init__(name, commands, **kwargs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@click.group(cls=_OrderedGroup)
@click.version_option(version=__version__)
def cli():
    """
    mosaicMPI is a tool for deconvolution and integration of multiple datasets based on consensus Non-Negative Matrix Factorization (cNMF).
    """

@click.command(name="txt-to-h5ad")
@click.option(
    "-d", "--data_file", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input counts or normalized matrix as delimited text file. Rows are samples/cells and columns are genes/features (unless --transpose is specified).")
@click.option(
    "--is_normalized", is_flag=True, help="Specify if input data is normalized instead of count data.")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Optional delimited text file with metadata for samples/cells with one row each. Columns are annotation layers.")
@click.option(
    "--sparsify", is_flag=True,
    help="Save resulting data in sparse format. Recommended to increase performance for sparse datasets such as scRNA-Seq, scATAC-Seq, and 10X Visium, but not for bulk expression data.")
@click.option(
    "--transpose", is_flag=True,
    help="Transpose an input data matrix where rows are genes/features and columns are samples/cells into the correct orientation.")
@click.option(
    "--data_delimiter", type=str, default="\t",
    help="Delimiter for data file, defaults to tab-delimited.")
@click.option(
    "--metadata_delimiter", type=str, default="\t",
    help="Delimiter for metadata file, defaults to tab-delimited.")
@click.option(
    "-o", '--output', type=click.Path(dir_okay=False, exists=False), required=True,
    help="Path to output .h5ad file.")
def cmd_txt_to_h5ad(data_file, is_normalized, metadata, output, transpose, sparsify, data_delimiter, metadata_delimiter):
    """
    Create .h5ad file with data and metadata (`adata.obs`).
    """
    utils.start_logging()
    df = pd.read_table(data_file, index_col=0, sep=data_delimiter)
    if transpose:
        df = df.T
    if metadata:
        metadata_df = pd.read_table(metadata, index_col=0, sep=metadata_delimiter).dropna(axis=1, how="all")
    else:
        metadata_df = None
    dataset = Dataset.from_df(data=df, obs=metadata_df, sparsify=sparsify, is_normalized=is_normalized)
    logging.info("Data types for non-missing values in each layer of metadata:\n"
                 + dataset.get_metadata_type_summary())
    dataset.write_h5ad(output)

@click.command(name="update-h5ad-metadata")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Tab-separated text file with metadata for samples/cells/spots with one row each. Columns are annotation layers.")
@click.option(
    "-i", '--input_h5ad', type=click.Path(dir_okay=False, exists=True), required=True,
    help="Path to input .h5ad file.")
def cmd_update_h5ad_metadata(input_h5ad, metadata):
    """
    Update metadata in a .h5ad file at any point in the mosaicMPI workflow. New metadata will overwrite (`adata.obs`).
    """
    utils.start_logging()
    dataset = Dataset.from_h5ad(input_h5ad)
    metadata_df = pd.read_table(metadata, index_col=0).dropna(axis=1, how="all")
    dataset.update_obs(metadata_df)
    logging.info("Data types for non-missing values in each layer of metadata:\n"
                 + dataset.get_metadata_type_summary())
    dataset.write_h5ad(input_h5ad)

@click.command(name="check-h5ad")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input .h5ad file.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, exists=False), required=False,
    help="Output .h5ad file. If not specified, no output file will be written.")
def cmd_check_h5ad(input, output):
    """
    Removes unfactorizable genes from a .h5ad file.
    """
    utils.start_logging()
    dataset = Dataset.from_h5ad(input)
    dataset.remove_unfactorizable_genes()
    
    # Save output to new h5ad file
    if output is not None:
        dataset.write_h5ad(output)


@click.command(name="model-odg")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="h5ad file containing expression data (adata.X=normalized and adata.raw.X = count) as well as any cell/sample metadata (adata.obs).")
@click.option(
    "--default_spline_degree", type=int, default=3, show_default=True,
    help="Degree for BSplines for the Generalized Additive Model (default method). For example, a constant spline would be 0, linear would be 1, and cubic would be 3.")
@click.option(
    "--default_dof", type=int, default=20, show_default=True,
    help="Degrees of Freedom (number of components) for the Generalized Additive Model (default method).")
def cmd_model_odg(name, output_dir, input, default_spline_degree, default_dof):
    """
    Model gene overdispersion and plot calibration plots for selection of overdispersed genes, using two methods:
    
    - `cnmf`: v-score and minimum expression threshold for count data (cNMF method: Kotliar, et al. eLife, 2019) 
    - `default`: residual standard deviation after modeling mean-variance dependence. (STdeconvolve method: Miller, et al. Nat. Comm. 2022)
    
    Examples:

        # use default parameters, suitable for most datasets
        mosaicmpi model-odg -n test -i test.h5ad

        # Explicitly use a linear model instead of a BSpline Generalized Additive Model
        mosaicmpi model-odg -n test -i test.h5ad --default_spline_degree 0 --default_dof 1
    """
    cNMF(output_dir=output_dir, name=name)  # creates directories for cNMF
    utils.start_logging(os.path.join(output_dir, name, "logfile.txt"))
    dataset = Dataset.from_h5ad(input)
    
    # Create gene stats table and save h5ad file
    dataset.compute_gene_stats(odg_default_spline_degree=default_spline_degree,
                               odg_default_dof=default_dof)
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    gene_stats = dataset.adata.var
    os.makedirs(os.path.normpath(os.path.join(output_dir, name, "odgenes")), exist_ok=True)
    gene_stats.to_csv(os.path.join(output_dir, name, "odgenes", "genestats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = plot_feature_dispersion(dataset, show_selected=False)
    fig.savefig(os.path.join(output_dir, name, "odgenes.pdf"), facecolor='white')
    fig.savefig(os.path.join(output_dir, name, "odgenes.png"), dpi=400, facecolor='white')


@click.command(name="set-parameters")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-m", "--odg_method",
    type=click.Choice([
        "default_topn",
        "default_minscore",
        "default_quantile",
        "cnmf_topn",
        "cnmf_minscore",
        "cnmf_quantile",
        "genes_file"
        ]), default="default_minscore", show_default=True,
    help="Select the model and method of overdispersed gene selection.")
@click.option(
    "-p", '--odg_param', default="1.0", show_default=True,
    help="Parameter for odg_method.")
@click.option(
    "--min_mean", default=0.0, show_default=True,
    help="Exclude genes from overdispersed gene lists if mean of counts data (or normalized data, if no count data exists) is less than this threshold.")
@click.option(
    '--k_range', type=int, nargs=3,
    help="Specify a range of components for factorization, using three numbers: first, last, step_size. Eg. '4 23 4' means `k`=4,8,12,16,20")
@click.option(
    "-k", type=int, multiple=True,
    help="Specify individual components for factorization. Multiple may be selected like this: -k 2 -k 3")
@click.option(
    '--n_iter', type=int, show_default=True, default=100,
    help="Number of iterations for factorization. If several `k` are specified, this many iterations will be run for each value of `k`")
@click.option(
    '--seed', type=int,
    help="Seed for sklearn random state.")
@click.option(
    '--beta_loss', type=click.Choice(["frobenius", "kullback-leibler"]), default="kullback-leibler", show_default=True,
    help="Measure of Beta divergence to be minimized.")

def cmd_set_parameters(name, output_dir, odg_method, odg_param, min_mean, k_range, k, n_iter, seed, beta_loss):
    """
    Set parameters for factorization, including selecting overdispersed genes.
    
    Overdispersed genes can be modelled using the `default` or `cnmf` models, and thresholds
    can be specified based on the top N, score threshold, or quantile. Alternatively, a text file with one gene per line can be used to specify the genes manually.
    
    For `top_n` methods, select an integer number of genes. For `min_score`, specify a score threshold. For `quantile` methods, specify the quantile of 
    genes to include (eg., the top 25% would be 0.75).
    
    Examples:

        # default behaviour does this
        mosaicmpi set_parameters -n test -m default_minscore -p 1.0

        # to reproduce cNMF default behaviour (Kotliar et al., 2019, eLife)
        mosaicmpi set_parameters -n test -m cnmf_topn -p 2000          

        # select top 20% of genes when ranked by od-score
        mosaicmpi set_parameters -n test -m default_quantile -p 0.8

        # input a gene list from text file
        mosaicmpi set_parameters -n test -m genes_file -p path/to/genesfile.txt
    """
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    utils.start_logging(os.path.join(output_dir, name, "logfile.txt"))
    dataset = Dataset.from_h5ad(os.path.join(output_dir, name, name + ".h5ad"))

    if odg_method == "genes_file":
        odg_param = click.Path(exists=True, dir_okay=False)(odg_param)
        genes = open(odg_param).read().rstrip().split(os.linesep)
        dataset.select_overdispersed_genes_from_genelist(genes)
    else:
        overdispersion_metric = odg_method.split("_")[0]
        if overdispersion_metric == "default":
            metric_str = "odscore"
        elif overdispersion_metric == "cnmf":
            metric_str = "vscore"
        else:
            raise RuntimeError
        
        cli_method_str = odg_method.split("_")[1]
        if cli_method_str == "top_n":
            method_str = "top_n"
            threshold = int(odg_param)
        elif cli_method_str == "minscore":
            method_str = "min_score"
            threshold = float(odg_param)
        else:
            method_str = cli_method_str
            threshold = float(odg_param)
        method = {method_str: threshold}
        dataset.select_overdispersed_genes(overdispersion_metric=metric_str, min_mean=min_mean,
                                           **method)

    # create mean vs variance plot, updated with selected genes
    fig = plot_feature_dispersion(dataset, show_selected=True)
    fig.savefig(os.path.join(output_dir, name, "odgenes.pdf"), facecolor='white')
    fig.savefig(os.path.join(output_dir, name, "odgenes.png"), dpi=400, facecolor='white')

    # output table with gene overdispersion measures
    dataset.adata.var.to_csv(os.path.join(output_dir, name, "odgenes", "genestats.tsv"), sep="\t")
    
    # process k-value selection inputs
    kvals = set(k)
    if k_range is not None:
        kvals |= set(range(k_range[0], k_range[1] + 1, k_range[2]))
    kvals = sorted(list(kvals))
    # prepare cNMF directory for factorization
    dataset.initialize_cnmf(cnmf_output_dir = output_dir, cnmf_name=name, kvals=kvals, n_iter=n_iter, beta_loss=beta_loss, seed=seed)
    
    # output dataset with new information on overdispersed genes and cNMF parameters
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    

@click.command(name="factorize")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    '--worker_index', type=int, default=0, show_default=True,
    help="Index of current worker (the first worker should have index 0) if --slurm_script is not specified.")
@click.option(
    '--total_workers', type=int, default=1, show_default=True,
    help="Total number of workers to distribute jobs to if --slurm_script is not specified.")
@click.option(
    '--slurm_script', type=click.Path(dir_okay=False, exists=True),
    help="Submit jobs to SLURM scheduler using this job submission script. Sample scripts are located in `scripts/slurm.sh`.")

def cmd_factorize(name, output_dir, worker_index, total_workers, slurm_script):
    """
    Performs factorization according to parameters specified using `mosaicmpi set-parameters`.
    """
    cnmf_obj = cNMF(output_dir=output_dir, name=name)
    utils.start_logging(os.path.join(output_dir, name, "logfile.txt"))
    
    run_params = utils.load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    if run_params.shape[0] == 0:
        logging.error("No factorization to do: either no values of k were selected using `mosaicmpi set-parameters` or iterations were set to 0.")

    if slurm_script is None:
        cnmf_obj.factorize(worker_i=worker_index, total_workers=total_workers)
    else:
        subprocess.Popen(['sbatch', slurm_script, os.getcwd(), output_dir, name])
    
@click.command(name="postprocess")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option('--cpus', type=int, default=1, show_default=True, help="Number of CPUs to use. Note that multi-CPU runs can use large amounts of memory and stall silently.")
@click.option(
    '--local_density_threshold', type=float, default=2.0, show_default=True,
    help="Threshold for the local density filtering prior to program consensus. Acceptable thresholds are > 0 and <= 2 (2.0 is no filtering).")
@click.option(
    '--local_neighborhood_size', type=float, default=0.3, show_default=True,
    help="Fraction of the number of replicates to use as nearest neighbors for local density filtering.")
@click.option(
    '--skip_missing_iterations', is_flag=True,
    help="If specified, consensus programs and usages will be calculated even though individual iterations are missing.")
@click.option(
    '--force_h5ad_update', is_flag=True,
    help="If specified, overwrites cNMF results already saved to the .h5ad file.")
def cmd_postprocess(name, output_dir, cpus, local_density_threshold, local_neighborhood_size, skip_missing_iterations, force_h5ad_update):
    """
    Perform post-processing routines on cNMF after factorization. This includes checking factorization outputs for completeness, combining individual
    iterations, calculating consensus programs and usage matrices, and creating the k-selection and annotated usage plots.
    """
    cnmf_obj = cNMF(output_dir=output_dir, name=name)
    utils.start_logging(os.path.join(output_dir, name, "logfile.txt"))
    cnmf_obj.postprocess(cpus=cpus,
                         local_density_threshold=local_density_threshold,
                         local_neighborhood_size=local_neighborhood_size,
                         skip_missing_iterations=skip_missing_iterations)
    h5ad_path = os.path.join(output_dir, name, name + ".h5ad")
    dataset = Dataset.from_h5ad(h5ad_path)
    
    cnmf_data_loaded =  "cnmf_usage" in dataset.adata.obsm or\
                        "cnmf_gep_score" in dataset.adata.varm or\
                        "cnmf_gep_tpm" in dataset.adata.varm or\
                        "cnmf_gep_raw" in dataset.adata.varm
    if cnmf_data_loaded and not force_h5ad_update:
        logging.Error(f"Error: AnnData already contains cNMF results. Use --force_h5ad_update to overwrite.")
        sys.exit(1)

    dataset.add_cnmf_results(cnmf_output_dir=output_dir,
                             cnmf_name=name,
                             local_density_threshold=local_density_threshold,
                             local_neighborhood_size=local_neighborhood_size
                             )
    dataset.write_h5ad(h5ad_path)
    
@click.command("annotated-heatmap")
@click.option(
    "-i", "--input_h5ad", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to AnnData (.h5ad) file containing cNMF results.")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False), default=os.getcwd(), show_default=True,
    help="Output directory for annotated heatmaps.")
@click.option(
    '-m', '--metadata_colors_toml', type=click.Path(dir_okay=False, exists=True),
    help="TOML file with metadata_colors specification. See README for more information. If not provided, visually distinct colors will be chosen automatically.")
@click.option(
    '--max_categories_per_layer', type=int,
    help="Filter metadata layers by the number of categories. This parameter is useful to simplify heatmaps with too many annotations.")
@click.option(
    '--hide_sample_labels', is_flag=True,
    help="Hide sample labels on usage heatmap")

def cmd_annotated_heatmap(input_h5ad, output_dir, metadata_colors_toml, max_categories_per_layer, hide_sample_labels):
    """
    Create heatmaps of usages with annotation tracks.
    """
    utils.start_logging()
    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_h5ad(input_h5ad)
    
    # get metadata colors
    if metadata_colors_toml:
        colors = Colors.from_toml(metadata_colors_toml)
    else:
        colors = Colors()
    colors.add_missing_metadata_colors(dataset)
    colors.to_toml(os.path.join(output_dir, "metadata_colors.toml"))
    
    # plot legend
    fig = colors.plot_metadata_colors_legend()
    fig.savefig(os.path.join(output_dir, f"metadata_legend.pdf"))
    
    # filter metadata layers with too many categories
    if max_categories_per_layer is not None:
        subset_columns = dataset.get_metadata_df(include_numerical=False).apply(lambda x: len(x.cat.categories)) <= max_categories_per_layer
        subset_columns = subset_columns[subset_columns].index.to_list()
        subset_columns.extend(dataset.get_metadata_df(include_categorical=False).columns.to_list())
    else:
        subset_columns = None
    if not dataset.has_cnmf_results:
        logging.error("cNMF results have not been merged into .h5ad file. Ensure that you have run `mosaicmpi postprocess` before creating annotated usage heatmaps.")
        sys.exit(1)

    # create annotated plots for each k
    for k in dataset.adata.uns["kvals"].index:
        logging.info(f"Creating annotated usage heatmap for k={k}")
        cnmf_name = dataset.adata.uns["cnmf_name"]
        title = f"{cnmf_name} k={k}"
        filename = os.path.join(output_dir, f"{cnmf_name}.usages.k{k:03}.pdf")
        fig = plot_usage_heatmap(
            dataset=dataset, k=k, subset_metadata=subset_columns, colors=colors, title=title,
            cluster_samples=True, cluster_programs=False, show_sample_labels=(not hide_sample_labels))
        fig.savefig(filename, transparent=False, bbox_inches = "tight")


@click.command(name="create-config")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), multiple=True, help=".h5ad file with cNMF results. Can be used multiple times to specify one or more datasets from which to create a config.toml file.")
@click.option('-o', '--output_toml', type=click.Path(file_okay=False), required=False, help="Output .toml file for configuring integration.")
def cmd_create_config(input_h5ad, output_toml):
    """Creates a TOML config file with default parameters to be used as input for `mosaicmpi integrate`.

    :param input_h5ad: _description_
    :type input_h5ad: _type_
    :param output_toml: _description_
    :type output_toml: _type_
    """
    utils.start_logging()
    if output_toml is None:
        output_toml = "config.toml"
    elif not output_toml.lower().endswith(".toml"):
        logging.warning(f"{output_toml} does not have a `.toml` extension.")
        output_toml = output_toml + ".toml"
    if os.path.exists(output_toml):
        logging.warning(f"{output_toml} already exists. Overwriting.")
    
    config = Config.from_h5ad_files(input_h5ad)
    config.to_toml(output_toml)
    logging.info(f"Output TOML file to: {output_toml}")


@click.command(name="integrate")
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="Output directory for mosaicMPI results")
@click.option('-c', '--config_toml', type=click.Path(exists=True, dir_okay=False), required=True, help="TOML config file")
@click.option('-m', '--communities_toml', type=click.Path(exists=True, dir_okay=False), required=False, 
    help="Use custom communities instead of performing unsupervised community discovery, such as those from a previous run.")
@click.option(
    '-l', '--colors_toml', type=click.Path(dir_okay=False, exists=True), required=False,
    help="TOML file with metadata_colors specification. If not provided, visually distinct colors will be chosen automatically.")
@click.option('--cpus', type=int, default=cpus_available, show_default=True, help="Number of CPUs to use for MP-enable operations")
def cmd_integrate(output_dir, config_toml, communities_toml, colors_toml, cpus):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5ad files directly, it is recommended to use a single TOML file instead for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    

    utils.start_logging()  # allows warning messages to be printed even though logfile hasn't been made yet

    # set CPU count for MP-enabled tasks
    global cpus_available
    cpus_available = cpus

    config = Config.from_toml(config_toml)

    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        logging.warning(f"{output_dir} is not empty. Files may be overwritten.")

    # write to log file
    utils.start_logging(os.path.join(output_dir, "logfile.txt"))

    # Create dataset objects from config
    datasets = {}
    for dsname, dsparams in config.datasets.items():
        
        dataset = Dataset.from_h5ad(dsparams["filename"], force_migrate = False)
            
        if "patient_id_col" in dsparams:
            dataset.patient_id_col = dsparams["patient_id_col"]
        datasets[dsname] = dataset
        
    # gets k-values from config
    has_k_subset = {dsname: ("k_subset" in dsparams) for dsname, dsparams in config.datasets.items()}
    if all(list(has_k_subset.values())):
        k_subset = {dsname: dsparams["k_subset"] for dsname, dsparams in config.datasets.items()}
    elif any(list(has_k_subset.values())):
        no_k_subset = ", ".join([dsname for dsname, has in has_k_subset.items() if not has])
        raise ValueError('"k_subset" parameter must be set for all datasets or none of them. '
                         f'These datasets are missing selected_k parameter: {no_k_subset}')
    else:
        k_subset = list(range(2, 10)) + list(range(10, 65, 5))

    # creates integration object
    integration = Integration(
        datasets=datasets,
        corr_method = config.integrate["corr_method"],
        max_median_corr = config.integrate["max_median_corr"],
        negative_corr_quantile = config.integrate["negative_corr_quantile"],
        k_subset = k_subset
        )

    # save correlation matrix
    corr_path = os.path.join(output_dir, config.integrate["corr_method"] + ".df.npz")
    utils.save_df_to_npz(integration.corr_matrix, corr_path)
    
    # output k-value and pairwise correlation thresholds
    integration.k_table.to_csv(os.path.join(output_dir, "k_values.txt"), sep="\t")
    integration.pairwise_thresholds.to_csv(os.path.join(output_dir, "pairwise_corr_thresholds.txt"), sep="\t")

    # Rank Reduction Plots
    fig = plot_rank_reduction(integration)
    fig.savefig(os.path.join(output_dir, f"rank_reduction.pdf"))
    fig.savefig(os.path.join(output_dir, f"rank_reduction.png"))

    logging.info("Plotting pairwise correlation distributions")
    # plot pairwise corr of all k
    fig = plot_pairwise_corr(integration)
    fig.savefig(os.path.join(output_dir, "pairwise_corr.pdf"))
    fig.savefig(os.path.join(output_dir, "pairwise_corr.png"), dpi=400)

    # plot mirrored distributions with thresholds (which are computed on max-k filtered data only)
    fig = plot_pairwise_corr_overlaid(integration)
    fig.savefig(os.path.join(output_dir, "pairwise_corr_overlaid.pdf"))
    fig.savefig(os.path.join(output_dir, "pairwise_corr_overlaid.png"), dpi=400)

    logging.info("Analyzing feature overlaps")
    # Features and overdispersed features tables and UpSet plots
    df = integration.get_overdispersed_features_overlap_table()
    df.to_csv(os.path.join(output_dir, "overdispersed_features.txt"), sep="\t")

    df = integration.get_features_overlap_table()
    df.to_csv(os.path.join(output_dir, "features.txt"), sep="\t")
    
    if integration.n_datasets > 1:
        fig = plot_overdispersed_features_upset(integration)
        fig.savefig(os.path.join(output_dir, "overdispersed_features_upsetplot.pdf"))
        fig.savefig(os.path.join(output_dir, "overdispersed_features_upsetplot.png"), dpi=200)
        
        fig = plot_features_upset(integration)
        fig.savefig(os.path.join(output_dir, "all_features_upsetplot.pdf"))

    nodetable = integration.get_node_table()
    nodetable.to_csv(os.path.join(output_dir, "node_stats.txt"), sep="\t")

    # creates Network object from Integration
    logging.info("Creating integration network")
    if config.integrate["subset_nodes"] == "none":
        subset_nodes = None
    else:
        subset_nodes = config.integrate["subset_nodes"]
        
    network = Network(integration=integration, subset_nodes=subset_nodes)
    

    if communities_toml is None:
        community_algorithm = config.integrate["community_algorithm"]
        network.community_search(algorithm=community_algorithm,
                                resolution=config.integrate["communities"][community_algorithm]["resolution"])
    else:
        network.read_communities_from_toml(communities_toml)
    nx.write_graphml(network.program_graph, os.path.join(output_dir, "program_graph.graphml"))

    network.compute_layout(
        algorithm=config.integrate["layout_algorithm"],
        shared_community_weight = config.integrate["layouts"]["community_weighted_spring"]["within_community"],
        shared_dataset_weight = config.integrate["layouts"]["community_weighted_spring"]["within_dataset"]
    )
    community_layout_algorithm = config.integrate["community_layout_algorithm"]
    network.compute_community_network_layout(
        algorithm=community_layout_algorithm,
        **config.integrate["community_layouts"][community_layout_algorithm]
    )

    # persist Network object to file
    logging.info("Writing network_integration.pkl.gz file")
    if config.integrate["write_integration_pkl"]:
        network.to_pkl(os.path.join(output_dir, "network_integration.pkl.gz"))

    # make figure legends for metadata
    if colors_toml is None:
        logging.info("Creating visually distinct color palettes")
        colors = Colors.from_network(network)
    else:
        logging.info("Using provided .toml file for color palettes")
        colors = Colors.from_toml(colors_toml)
        colors.add_missing_dataset_colors(datasets=integration)
        colors.add_missing_metadata_colors(datasets=integration)
        colors.add_missing_community_colors(network=network)
    colors.to_toml(os.path.join(output_dir, "colors.toml"))
    
    fig = colors.plot_metadata_colors_legend()
    fig.savefig(os.path.join(output_dir, "metadata_colors_legend.pdf"))
    plt.close(fig)
    
    fig = colors.plot_dataset_colors_legend()
    fig.savefig(os.path.join(output_dir, "dataset_colors_legend.pdf"))
    plt.close(fig)
    
    network.write_communities_toml( os.path.join(output_dir, "communities.toml"))
    pd.DataFrame.from_dict(data=network.program_communities, orient='index').to_csv(os.path.join(output_dir, 'program_communities.txt'), sep="\t", header=False)
    
    # Write representative programs and usages
    logging.info("Writing representative programs")
    central_program_ids = network.get_representative_programs()
    central_programs = network.integration.get_programs()[central_program_ids.index]
    central_programs.columns = pd.MultiIndex.from_tuples([[community] + list(program_id) for community, program_id in zip(central_program_ids, central_program_ids.index)], names=["Community", "dataset", "k", "Program"])
    central_programs.to_csv(os.path.join(output_dir, "representative_programs.txt"), sep="\t") # outputs the Programs to text file
    central_program_usage = network.integration.get_usages()[central_program_ids.index]
    central_program_usage.columns = pd.MultiIndex.from_tuples([[community] + list(program_id) for community, program_id in zip(central_program_ids, central_program_ids.index)], names=["Community", "dataset", "k", "Program"])
    central_program_usage.to_csv(os.path.join(output_dir, "representative_program_usageks.txt"), sep="\t") # outputs the Programs to text file
    
    logging.info("Creating network plots")

    # plot membership of datasets and ranks for each community
    fig = plot_community_by_dataset_rank(network, colors)
    fig.savefig(os.path.join(output_dir, "communities_by_dataset_rank.pdf"))
    fig.savefig(os.path.join(output_dir, "communities_by_dataset_rank.png"), dpi=600)
    plt.close(fig)
    
    # summary community network
    fig = plot_community_network_summary(network, colors)
    fig.savefig(os.path.join(output_dir, "community_network_summary.pdf"))
    fig.savefig(os.path.join(output_dir, "community_network_summary.png"), dpi=600)
    plt.close(fig)


    # Plot network colored by dataset
    fig = plot_program_network_datasets(network, colors)
    fig.savefig(os.path.join(output_dir, "network_datasets.pdf"))
    fig.savefig(os.path.join(output_dir, "network_datasets.png"), dpi=600)
    plt.close(fig)
    
    # Plot network colored by dataset and size by rank
    fig = plot_program_network_datasets(network, colors, node_size_kval=True)
    fig.savefig(os.path.join(output_dir, "network_rank.pdf"))
    fig.savefig(os.path.join(output_dir, "network_rank.png"), dpi=600)
    plt.close(fig)
    
    # Plot network colored by community
    fig = plot_program_network_communities(network, colors)
    fig.savefig(os.path.join(output_dir, "network_communities.pdf"))
    fig.savefig(os.path.join(output_dir, "network_communities.png"), dpi=600)
    plt.close(fig)
    
    # Cumulative proportion of samples contributing to each Program
    fig = plot_program_network_nsamples(network, colors)
    fig.savefig(os.path.join(output_dir, "network_n_samples.pdf"))
    fig.savefig(os.path.join(output_dir, "network_n_samples.png"), dpi=600)
    plt.close(fig)
    
    # Cumulative proportion of patients contributing to each Program
    if network.integration.sample_to_patient is not None:
        fig = plot_program_network_npatients(network, colors)
        fig.savefig(os.path.join(output_dir, "network_n_patients.pdf"))
        fig.savefig(os.path.join(output_dir, "network_n_patients.png"), dpi=600)
        plt.close(fig)
    
    
    logging.info("Creating community usage heatmap...")

    # integrated community usage
    ic_usage = network.get_community_usage()
    ic_usage.to_csv(os.path.join(output_dir, "community_usage.txt"), sep="\t")
    fig = plot_community_usage_heatmap(network, colors)
    fig.savefig(os.path.join(output_dir, "community_usage.pdf"))
    plt.close(fig)
   

    logging.info("Computing community-level associations")

    # Community-level, categorical metadata, overrepresentation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "overrepresentation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            df = network.get_community_category_overrepresentation(layer=layer, subset_datasets=dataset_name, truncate_negative=False)
            df.to_csv(os.path.join(output_dir, "annotated_communities", "overrepresentation", dataset_name, layer + ".txt"), sep='\t')
            
    # Community-level, numerical metadata, correlation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "correlation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            df = network.get_community_metadata_correlation(layer=layer, subset_datasets=dataset_name, method="pearson")
            df.to_csv(os.path.join(output_dir, "annotated_communities", "correlation", dataset_name, layer + ".txt"), sep='\t')

    logging.info("Creating community-level bar plots")

    # Community-level, categorical data, overrepresentation bar plots
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "overrepresentation_bar", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            fig = plot_overrepresentation_community_bar(network, colors, layer=layer, subset_datasets=dataset_name)
            os.makedirs(os.path.join(output_dir, "annotated_communities", "overrepresentation_bar"), exist_ok=True)
            fig.savefig(os.path.join(output_dir, "annotated_communities", "overrepresentation_bar", dataset_name, layer + ".pdf"))
            plt.close(fig)
        
    # Community-level, numerical metadata, correlation bar plots
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "correlation_bar", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            fig = plot_metadata_correlation_community_bar(network, colors, layer=layer, subset_datasets=dataset_name)
            os.makedirs(os.path.join(output_dir, "annotated_communities", "correlation_bar"), exist_ok=True)
            fig.savefig(os.path.join(output_dir, "annotated_communities", "correlation_bar", dataset_name, layer + ".pdf"))
            plt.close(fig)


    logging.info("Creating community-level network plots")

    # Community-level, categorical data, overrepresentation network
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "overrepresentation_network", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            fig = plot_overrepresentation_community_network(network, colors, layer=layer, subset_datasets=dataset_name)   
            fig.savefig(os.path.join(output_dir, "annotated_communities", "overrepresentation_network", dataset_name, layer + ".pdf"))
            fig.savefig(os.path.join(output_dir, "annotated_communities", "overrepresentation_network", dataset_name, layer + ".png"), dpi=600)
            plt.close(fig)
 
    # Community-level, numerical data, correlation network
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "correlation_network", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            fig = plot_metadata_correlation_community_network(network, colors, layer=layer, subset_datasets=dataset_name)   
            fig.savefig(os.path.join(output_dir, "annotated_communities", "correlation_network", dataset_name, layer + ".pdf"))
            fig.savefig(os.path.join(output_dir, "annotated_communities", "correlation_network", dataset_name, layer + ".png"), dpi=600)
            plt.close(fig)    

    logging.info("Computing program-level associations")

    # Program-level, categorical metadata, overrepresentation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_programs", "overrepresentation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            df = dataset.get_category_overrepresentation(layer)
            df.to_csv(os.path.join(output_dir, "annotated_programs", "overrepresentation", dataset_name, layer + ".txt"), sep='\t')
            
    # Program-level, numerical metadata, correlation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_programs", "correlation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            df = dataset.get_metadata_correlation(layer, method="pearson")
            df.to_csv(os.path.join(output_dir, "annotated_programs", "correlation", dataset_name, layer + ".txt"), sep='\t')

    logging.info("Creating program-level bar plots")

    # Program-level, categorical data, overrepresentation bar plots
    for dataset_name in integration.datasets:
        fig = plot_overrepresentation_program_bar(network, colors, dataset_name=dataset_name)
        os.makedirs(os.path.join(output_dir, "annotated_programs", "overrepresentation_bar"), exist_ok=True)
        fig.savefig(os.path.join(output_dir, "annotated_programs", "overrepresentation_bar", dataset_name + ".pdf"))
        plt.close(fig)
        
    # Program-level, numerical metadata, correlation bar plots
    for dataset_name in integration.datasets:
        fig = plot_metadata_correlation_program_bar(network, colors, dataset_name=dataset_name)
        os.makedirs(os.path.join(output_dir, "annotated_programs", "correlation_bar"), exist_ok=True)
        fig.savefig(os.path.join(output_dir, "annotated_programs", "correlation_bar", dataset_name + ".pdf"))
        plt.close(fig)

    logging.info("Creating program-level network plots")

    # Program-level, categorical data, overrepresentation network
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_programs", "overrepresentation_network", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            fig = plot_overrepresentation_program_network(network, colors, layer=layer, subset_datasets=dataset_name)   
            fig.savefig(os.path.join(output_dir, "annotated_programs", "overrepresentation_network", dataset_name, layer + ".pdf"))
            fig.savefig(os.path.join(output_dir, "annotated_programs", "overrepresentation_network", dataset_name, layer + ".png"), dpi=600)
            plt.close(fig)
 
    # Program-level, numerical data, correlation network
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_programs", "correlation_network", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            fig = plot_metadata_correlation_program_network(network, colors, layer=layer, subset_datasets=dataset_name)   
            fig.savefig(os.path.join(output_dir, "annotated_programs", "correlation_network", dataset_name, layer + ".pdf"))
            fig.savefig(os.path.join(output_dir, "annotated_programs", "correlation_network", dataset_name, layer + ".png"), dpi=600)
            plt.close(fig)       


    # for dataset_name, programs in selected_programs.droplevel(axis=1, level=[2,3]).groupby(axis=1, level=1):
    #     programs = programs.droplevel(axis=1, level=1)
    #     programs.columns = programs.columns.astype("int")
    #     for feature in config.features["features_of_interest"]:
    #         if feature in programs.index:
    #             positive_scores = programs.loc[feature].groupby(axis=0, level=0).mean().reindex(communities.keys()).fillna(0).clip(lower=0)
    #             if positive_scores.sum() == 0:
    #                 continue
    #             fig = plot_community_network(
    #                 graph = Gcomm,
    #                 layout = community_layout,
    #                 plot_size = config.integrate["plot_size_community"],
    #                 title = f"Dataset: {dataset_name}\nFeature: {feature}",
    #                 node_sizes=positive_scores,
    #                 edge_weights="n_edges",
    #                 config=config,
    #                 community_colors=community_colors
    #             )
    #             os.makedirs(os.path.join(output_dir, "annotated_communities", "cNMF_score"), exist_ok=True)
    #             fig.savefig(os.path.join(output_dir, "annotated_communities", "cNMF_score", f"{feature}_{dataset_name}.pdf"))
    #             plt.close(fig)
        
    # # per patient community level plots
    # if any(["patient_id_column" in d for d in config.datasets.values()]):
    #     patient_to_samples = {patient: [] for sample, patient in sample_to_patient.items()}
    #     for sample, patient in sample_to_patient.items():
    #         patient_to_samples[patient].append(sample)
    #     patient_to_samples = pd.Series(patient_to_samples).explode()

    #     n_cols = 4
    #     for annotation_layer in merged_metadata.select_dtypes("category").columns:
    #         if annotation_layer == "Dataset":
    #             colordict = {dsname: dsparam["color"] for dsname, dsparam in config.datasets.items()}
    #         else:
    #             colordict = config.get_metadata_colors(annotation_layer)
    #             colordict[""] = config.metadata_colors["missing_data"]
    #         for min_samples_per_patient in [1,2]:
    #             n_plots = (patient_to_samples.groupby(axis=0, level=[0,1]).count() >= min_samples_per_patient).sum() + 1  # one plot per patient as well as an extra for the legend.
    #             n_rows = 1 + n_plots // n_cols
    #             fig, axes = plt.subplots(n_rows, n_cols, figsize = [config.integrate["plot_size_community"][0]*n_cols, config.integrate["plot_size_community"][1]*n_rows], squeeze=False, layout="constrained")
    #             for row in range(n_rows):
    #                 for col in range(n_cols):
    #                     axes[row,col].set_axis_off()

    #             # Add legend
    #             ax = axes[0, 0]  # get lower right axes
    #             legend_elements = []
    #             for name, color in colordict.items():
    #                 legend_elements.append(Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=10))
    #             ax.legend(handles=legend_elements, loc='center')
    #             ax.set_title("Legend", fontdict={'fontsize': 26})

    #             plot_count = 1
    #             for (dataset, patient), patient_samples in patient_to_samples.groupby(axis=0, level=[0,1]):
    #                 if patient_samples.shape[0] >= min_samples_per_patient:
    #                     ax = axes[plot_count // n_cols, plot_count % n_cols]
    #                     bar_data = ic_usage.loc[patient_samples].fillna(0)
    #                     bar_data.index = merged_metadata.loc[bar_data.index][annotation_layer].cat.add_categories("").fillna("")
    #                     bar_data = bar_data.groupby(axis=0, level=0).mean().dropna(how="any")
    #                     plot_overrepresentation_network(Gcomm, community_layout, f"{dataset}\n{patient}", overrepresentation=bar_data, colordict=colordict, pie_size=config.integrate["pie_size_community"], ax=ax, edge_weights=None, show_legends=False)
    #                     plot_count += 1
    #             os.makedirs(os.path.join(output_dir, "annotated_communities", "patient_network", annotation_layer), exist_ok=True)
    #             fig.savefig(os.path.join(output_dir, "annotated_communities", "patient_network", annotation_layer, f"{min_samples_per_patient}samplesperpatient.pdf"))
    #             fig.savefig(os.path.join(output_dir, "annotated_communities", "patient_network", annotation_layer, f"{min_samples_per_patient}samplesperpatient.png"), dpi=100)
    #             plt.close("all")
        
    # # Plot pairwise correlation heatmaps of community usage across samples
    # def plot_icusage_correlation(ic_usage_corr, title=None):
    #     mask = np.triu(np.ones_like(ic_usage_corr), 1)
    #     fig, ax = plt.subplots(figsize=[8,6])
    #     sns.heatmap(ic_usage_corr, center=0, vmin=-1, vmax=1, cmap=config.colormaps["diverging"], mask=mask, ax=ax)
    #     ax.set_title(title)
    #     return fig

    # plots = {"All Datasets": plot_icusage_correlation(ic_usage.dropna(axis=1).corr("spearman"), title="All Datasets")}
    # plots.update({
    #     dataset_name: plot_icusage_correlation(df.corr("spearman"), title=dataset_name)
    #     for dataset_name, df in ic_usage.dropna(axis=1).groupby(axis=0, level=0)
    #     })

    # os.makedirs(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_shared"), exist_ok=True)
    # for plot_name, fig in plots.items():
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_shared", plot_name + ".pdf"))
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_shared", plot_name + ".png"), dpi=600)
        
    # plots = {"All Datasets": plot_icusage_correlation(ic_usage.corr("spearman"), title="All Datasets")}
    # plots.update({
    #     dataset_name: plot_icusage_correlation(df.corr("spearman"), title=dataset_name)
    #     for dataset_name, df in ic_usage.groupby(axis=0, level=0)
    #     })

    # os.makedirs(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_all"), exist_ok=True)
    # for plot_name, fig in plots.items():
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_all", plot_name + ".pdf"))
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_all", plot_name + ".png"), dpi=600)
    
    
    # # Diversity analysis (shannon entropy of community-level usage)

    # diversity = ic_usage.apply(lambda x: entropy(x.dropna()), axis=1)
    # diversity.to_csv(os.path.join(output_dir, "integrated_community_usage", "diversity.txt"), sep="\t")
    # os.makedirs(os.path.join(output_dir, "integrated_community_usage", "diversity"), exist_ok=True)

    # for col in merged_metadata.select_dtypes("float").columns:  # association of diversity with numerical metadata
    #     df = pd.DataFrame({"diversity": diversity, col: merged_metadata[col]})
    #     df["Dataset"] = df.index.get_level_values(0)
    #     fig, ax = plt.subplots(figsize=[4,4])
    #     sns.scatterplot(data=df, x=col, y="diversity", hue="Dataset", ax=ax)
    #     correlation = merged_metadata[col].corr(diversity, method="spearman")
    #     ax.text(s=f"Spearman ρ = {correlation:.3f}", x=0.01, y=0.01, va="bottom", transform=ax.transAxes)
    #     ax.set_title(col)
    #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", f"{col}.pdf"))
    #     plt.close('all')

    # sample_groups = merged_metadata["Dataset"]  # association of diversity with dataset
    # fig = plot_icu_diversity(sample_groups, diversity, config, title=f"Datasets")
    # fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", f"datasets.pdf"), bbox_inches="tight")
    # fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", f"datasets.png"), bbox_inches="tight", dpi=400)
    # plt.close('all')

    # for dataset in config.datasets: # association of diversity with categorical data by dataset
    #     os.makedirs(os.path.join(output_dir, "integrated_community_usage", "diversity", dataset), exist_ok=True)   
    #     for annotation_layer in merged_metadata.select_dtypes("category").columns:
    #         sample_groups = merged_metadata.loc[dataset, annotation_layer]
    #         if 0 < sample_groups.nunique() < 20:
    #             fig = plot_icu_diversity(sample_groups, diversity.loc[dataset], config, title=f"{dataset}\n{annotation_layer}")
    #             fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", dataset, f"{annotation_layer}.pdf"), bbox_inches="tight")
    #             fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", dataset, f"{annotation_layer}.png"), bbox_inches="tight", dpi=400)
    #             plt.close('all')
    
    
cli.add_command(cmd_txt_to_h5ad)
cli.add_command(cmd_update_h5ad_metadata)
cli.add_command(cmd_check_h5ad)
cli.add_command(cmd_model_odg)
cli.add_command(cmd_set_parameters)
cli.add_command(cmd_factorize)
cli.add_command(cmd_postprocess)
cli.add_command(cmd_annotated_heatmap)
cli.add_command(cmd_create_config)
cli.add_command(cmd_integrate)
