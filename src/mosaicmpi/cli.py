
from .dataset import Dataset
from .integration import Integration
from .config import Config
from .colors import Colors
from .network import Network, compare_community_jaccard_similarity
from .cnmf import cNMF
from .plots import *
from . import utils, __version__, cpus_available

import os
import json
import logging
import subprocess
import collections
import sys
from itertools import product
from typing import Optional, Mapping

from tqdm import tqdm
import click
import pandas as pd
import matplotlib

# For CLI, use the Agg background which doesn't support plt.show() and is faster
matplotlib.use('Agg')

# adds both -h and --help as options
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}

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


@click.group(cls=_OrderedGroup, context_settings=CONTEXT_SETTINGS)
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
    "-f", "--feature_metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Optional delimited text file with metadata for features with one row each. Columns are annotation layers.")
@click.option(
    "--sparsify", is_flag=True,
    help="Save resulting data in sparse format. Recommended to increase performance for sparse datasets such as scRNA-Seq, scATAC-Seq, and 10X Visium"
         ", but not for bulk expression data. [Note that this feature is experimental]")
@click.option(
    "--transpose", is_flag=True,
    help="Transpose an input data matrix where rows are genes/features and columns are samples/cells into the correct orientation.")
@click.option(
    "--data_delimiter", type=str, default="\t",
    help="Delimiter for data file, defaults to tab-delimited.")
@click.option(
    "--metadata_delimiter", type=str, default="\t",
    help="Delimiter for metadata files, defaults to tab-delimited.")
@click.option(
    "-o", '--output', type=click.Path(dir_okay=False, exists=False), required=True,
    help="Path to output .h5ad file.")
def cmd_txt_to_h5ad(data_file, is_normalized, metadata, feature_metadata, output, transpose, sparsify, data_delimiter, metadata_delimiter):
    """
    Create .h5ad file with data and metadata (`adata.obs`).
    """
    utils.start_logging()
    df = pd.read_table(data_file, index_col=0, sep=data_delimiter)
    if transpose:
        df = df.T
    if metadata:
        sample_metadata_df = pd.read_table(metadata, index_col=0, sep=metadata_delimiter).dropna(axis=1, how="all")
    else:
        sample_metadata_df = None
    if feature_metadata:
        feature_metadata_df = pd.read_table(feature_metadata, index_col=0, sep=metadata_delimiter).dropna(axis=1, how="all")
    else:
        feature_metadata_df = None
    dataset = Dataset.from_df(data=df, obs=sample_metadata_df, var=feature_metadata_df, sparsify=sparsify, is_normalized=is_normalized)
    logging.info(dataset.get_printable_metadata_type_summary())
    dataset.write_h5ad(output)
    logging.info("All tasks completed successfully.")

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
    logging.info(dataset.get_printable_metadata_type_summary())
    dataset.write_h5ad(input_h5ad)
    logging.info("All tasks completed successfully.")

@click.command(name="impute-zeros")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input .h5ad file.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, exists=False), required=True,
    help="Output .h5ad file.")
@click.option(
    "-n", "--n_folds", type=int, default=100,
    help="Number of folds for k-fold cross-validation. 0 disables cross-validation.")

def cmd_impute_zeros(input, output, n_folds):
    """
    Impute missing values with zeros.
    """
    utils.start_logging()
    dataset = Dataset.from_h5ad(input)
    dataset.impute_zeros(n_folds=n_folds)
    # Save output to new h5ad file
    dataset.write_h5ad(output)
    
    logging.info("All tasks completed successfully.")

@click.command(name="impute-knn")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input .h5ad file.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, exists=False), required=True,
    help="Output .h5ad file.")
@click.option(
    "--n_neighbors", default=5, type=int, show_default=True,
    help="Number of neighboring samples to use for imputation.")
@click.option(
    "--weights", default = "distance", type=str, show_default=True,
    help="""Weight function used in prediction, defaults to 'distance'. Possible values:
            
            - 'uniform' : uniform weights. All points in each neighborhood are
            weighted equally.

            - 'distance' : weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
            
            .
            """
)
@click.option(
    "-n", "--n_folds", type=int, default=100,
    help="Number of folds for k-fold cross-validation. 0 disables cross-validation.")
def cmd_impute_knn(input, output, n_neighbors, weights, n_folds):
    """
    k-Nearest Neighbour (KNN) imputation of missing values.
    """
    utils.start_logging()
    dataset = Dataset.from_h5ad(input)
    dataset.impute_knn(n_neighbors = n_neighbors, weights = weights, n_folds=n_folds)
    # Save output to new h5ad file
    dataset.write_h5ad(output)
    
    logging.info("All tasks completed successfully.")


@click.command(name="check-h5ad")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input .h5ad file.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, exists=False), required=False,
    help="Output .h5ad file. If not specified, no output file will be written.")
def cmd_check_h5ad(input, output):
    """
    Removes unfactorizable features (features with zero variance and features with missing values).
    This step prevents errors during factorization. If missing values are present in a larger
    number of features, it is recommended to run `mosaicmpi impute_knn` first to impute missing
    values. Then, this command will remove only those features that have zero variance or were
    not able to be imputed for other reasons.
    """
    utils.start_logging()
    dataset = Dataset.from_h5ad(input)
    dataset.remove_unfactorizable_observations()
    dataset.remove_unfactorizable_features()
    
    # Save output to new h5ad file
    if output is not None:
        dataset.write_h5ad(output)

    logging.info("All tasks completed successfully.")


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
@click.option(
    "--max_missingness", type=float, default=0.0, show_default=True,
    help="""Maximum proportion of missing values allowed for each feature prior to modelling the mean-variance relationship.
            This parameter is helpful for reducing the tendency of kNN- and zero-imputed features to have higher variance 
            relative to unimputed genes.
            """
)
def cmd_model_odg(name, output_dir, input, default_spline_degree, default_dof, max_missingness):
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
    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
    dataset = Dataset.from_h5ad(input)
    
    # Create gene stats table and save h5ad file
    dataset.model_overdispersed_genes(odg_default_spline_degree=default_spline_degree,
                               odg_default_dof=default_dof,
                               max_missingness=max_missingness)
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    dataset.adata.var.to_csv(os.path.join(output_dir, name, "feature_stats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = plot_feature_dispersion(dataset, show_selected=False)
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_meanvar"), formats=("pdf", "png"), target_dpi=400, facecolor='white')
    
    if dataset.is_imputed:
        logging.info("Creating plots for imputed data")

        fig = plot_feature_missingness(dataset, proportion=True)
        utils.save_fig(fig, os.path.join(output_dir, name, "missingness_histogram"), formats=("pdf", "png"), target_dpi=400, facecolor='white')

    logging.info("All tasks completed successfully.")


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
    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
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
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_meanvar"), formats=("pdf", "png"), target_dpi=400, facecolor='white')

    # output table with gene overdispersion measures
    dataset.adata.var.to_csv(os.path.join(output_dir, name, "feature_stats.tsv"), sep="\t")
    
    # process k-value selection inputs
    kvals = set(k)
    if k_range is not None:
        kvals |= set(range(k_range[0], k_range[1] + 1, k_range[2]))
    kvals = sorted(list(kvals))
    
    # prepare cNMF directory for factorization
    dataset.initialize_cnmf(cnmf_output_dir = output_dir, cnmf_name=name, kvals=kvals, n_iter=n_iter, beta_loss=beta_loss, seed=seed)
    
    logging.info("All tasks completed successfully.")
    

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
    os.makedirs(os.path.join(output_dir, name, "logs", "factorize"), exist_ok=True)
    utils.start_logging(os.path.join(output_dir, name, "logs", "factorize", f"logfile_{worker_index}.txt"))
    
    run_params = utils.load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    if run_params.shape[0] == 0:
        logging.error("No factorization to do: either no values of k were selected using `mosaicmpi set-parameters` or iterations were set to 0.")
        sys.exit(1)
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
@click.option(
    '--slurm_script', type=click.Path(dir_okay=False, exists=True),
    help="Submit jobs to SLURM scheduler using this job submission script. Sample script is located in `scripts/slurm_postprocess.sh`.")
def cmd_postprocess(name, output_dir, cpus, local_density_threshold, local_neighborhood_size, skip_missing_iterations, force_h5ad_update, slurm_script):
    """
    Perform post-processing routines on cNMF after factorization. This includes checking factorization outputs for completeness, combining individual
    iterations, calculating consensus programs and usage matrices, and creating the k-selection plot.
    """

    if slurm_script is not None:
        flags = ""
        if skip_missing_iterations:
            flags += "--skip_missing_iterations "
        if force_h5ad_update:
            flags += "--force_h5ad_update "

        subprocess.Popen(['sbatch', slurm_script,
                          os.getcwd(),
                          output_dir,
                          name,
                          str(cpus),
                          str(local_density_threshold),
                          str(local_neighborhood_size),
                          flags])
    else:
        cnmf_obj = cNMF(output_dir=output_dir, name=name)
        os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)
        utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
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
            logging.error(f"Error: AnnData already contains cNMF results. Use --force_h5ad_update to overwrite.")
            sys.exit(1)

        dataset.add_cnmf_results(cnmf_output_dir=output_dir,
                                cnmf_name=name,
                                local_density_threshold=local_density_threshold,
                                local_neighborhood_size=local_neighborhood_size
                                )
        dataset.write_h5ad(h5ad_path)
        
        logging.info("All tasks completed successfully.")

    
@click.command("annotated-heatmap")
@click.option(
    "-i", "--input_h5ad", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to AnnData (.h5ad) file containing cNMF results.")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False), default=os.getcwd(), show_default=True,
    help="Output directory for annotated heatmaps.")
@click.option(
    '-m', '--metadata_colors_toml', type=click.Path(dir_okay=False, exists=True),
    help="TOML file with metadata_colors specification. If not provided, visually distinct colors will be chosen automatically.")
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
    utils.save_fig(fig, os.path.join(output_dir, "metadata_legend"),)

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
        plt.close(fig)
    
    logging.info("All tasks completed successfully.")


@click.command(name="map-gene-ids")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), required=True, help="Input .h5ad file")
@click.option('-o', '--output_h5ad', type=click.Path(exists=False, dir_okay=False), required=True, help="Output .h5ad file with mapped gene identifiers.")
@click.option('--source_ids', type=click.Choice(["ensembl_gene", "gene_name"]), default="gene_name",
              help="Whether the source feature IDs are gene names (eg., EGFR), or Ensembl genes (eg., ENSG00000146648)")
@click.option('--dest_ids', type=click.Choice(["ensembl_gene", "gene_name"]), default="gene_name",
              help="Whether the dest feature IDs are gene names (eg., EGFR), or Ensembl genes (eg., ENSG00000146648)")
@click.option('--source_species', type=click.Choice(["hsapiens", "mmusculus", "rnorvegicus", "sscrofa", "dmelanogaster", "drerio", "celegans"]), required=True,
              help="Species of source feature IDs")
@click.option('--dest_species', type=click.Choice(["hsapiens", "mmusculus", "rnorvegicus", "sscrofa", "dmelanogaster", "drerio", "celegans"]), required=True,
              help="Species of dest feature IDs")
@click.option('--unmapped_prefix', type=str, default="unmapped_", help="String to prefix unmapped feature/gene IDs")
@click.option('--one_to_many', is_flag=True, help="Map one-to-many relationships, duplicating features to accommodate.")
def cmd_map_gene_ids(input_h5ad, output_h5ad, source_ids, dest_ids, source_species, dest_species, unmapped_prefix, one_to_many):
    """
    Map gene IDs for a dataset, keeping both mapped and unmapped IDs. By default, only one-to-one relationships are mapped.
    """
    utils.start_logging()
    dataset = Dataset.from_h5ad(input_h5ad)
    dataset.map_gene_ids(source_species=source_species,
                         dest_species=dest_species,
                         source_ids=source_ids,
                         dest_ids=dest_ids,
                         unmapped_prefix=unmapped_prefix,
                         one_to_many=("duplicate" if one_to_many else False)
                         )
    message = "Feature counts by mapping relationship:"
    for mapping_type, count in dataset.adata.var["mapping_relationship"].value_counts().items():
        message += f"\n\t{mapping_type}: {count}"
    logging.info(message)
    dataset.write_h5ad(output_h5ad)
    logging.info("All tasks completed successfully.")

@click.command(name="create-config")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), multiple=True, help=".h5ad file with cNMF results. Can be used multiple times to specify one or more datasets from which to create a config.toml file.")
@click.option('-o', '--output_toml', type=click.Path(exists=False, dir_okay=False), required=False, help="Output .toml file for configuring integration.")
def cmd_create_config(input_h5ad, output_toml):
    """
    Creates a TOML config file with default parameters to be used as input for `mosaicmpi integrate`.
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
@click.option('--cpus', type=int, default=cpus_available, show_default=True, help="Number of CPUs for MP-enabled tasks")
def cmd_integrate(output_dir, config_toml, communities_toml, colors_toml, cpus):
    """
    Integrate one or more datasets across ranks using a TOML configuration file.
    """
    import networkx as nx
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

    # check datasets for duplicated features
    datasets_dupfeatures = []
    for dsname, dataset in datasets.items():
        if not dataset.adata.var_names.unique:
            datasets_dupfeatures.append(dsname)
    if datasets_dupfeatures:
        logging.error("Datasets with duplicated features cannot be used for integration with other datasets. "
                      f"These datasets have duplicated features: {(', '.join(datasets_dupfeatures))}")
        sys.exit(1)

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
        corr_method = config.corr_method,
        max_median_corr = config.max_median_corr,
        negative_corr_quantile = config.negative_corr_quantile,
        k_subset = k_subset
        )

    logging.info("Analyzing feature overlaps")
    # Features and overdispersed features tables and UpSet plots
    df = integration.get_overdispersed_features_overlap_table()
    df.to_csv(os.path.join(output_dir, "overdispersed_features.txt"), sep="\t")

    df = integration.get_features_overlap_table()
    df.to_csv(os.path.join(output_dir, "features.txt"), sep="\t")
    
    # save correlation matrix
    corr_path = os.path.join(output_dir, config.corr_method + ".df.npz")
    utils.save_df_to_npz(integration.corr_matrix, corr_path)
    
    # output k-value and pairwise correlation thresholds
    integration.k_table.to_csv(os.path.join(output_dir, "k_values.txt"), sep="\t")
    integration.pairwise_thresholds.to_csv(os.path.join(output_dir, "pairwise_corr_thresholds.txt"), sep="\t")

    # Rank Reduction Plots
    fig = plot_rank_reduction(integration)
    utils.save_fig(fig, os.path.join(output_dir, f"rank_reduction"), target_dpi=300, formats=config.plot_formats)

    logging.info("Plotting pairwise correlation distributions")
    # plot pairwise corr of all k
    fig = plot_pairwise_corr(integration)
    utils.save_fig(fig, os.path.join(output_dir, f"pairwise_corr"), target_dpi=300, formats=config.plot_formats)

    # plot mirrored distributions with thresholds (which are computed on max-k filtered data only)
    fig = plot_pairwise_corr_overlaid(integration)
    utils.save_fig(fig, os.path.join(output_dir, f"pairwise_corr_overlaid"), target_dpi=300, formats=config.plot_formats)

    if integration.n_datasets > 1:
        fig = plot_overdispersed_features_upset(integration)
        utils.save_fig(fig, os.path.join(output_dir, "overdispersed_features_upsetplot"), target_dpi=200, formats=config.plot_formats)
        
        fig = plot_features_upset(integration)
        utils.save_fig(fig, os.path.join(output_dir, "features_upsetplot"), target_dpi=200, formats=config.plot_formats)


    # creates Network object from Integration
    logging.info("Creating integration network")
    if config.subset_nodes == "none":
        subset_nodes = None
    else:
        subset_nodes = config.subset_node
        
    network = Network(integration=integration, subset_nodes=subset_nodes)
    
    # community discovery
    if communities_toml is None:
        community_algorithm = config.community_algorithm
        network.community_search(algorithm=community_algorithm,
                                resolution=config.community_algorithm_parameters[community_algorithm]["resolution"])
        network.prune_communities(renumber=True, **config.community_pruning)
    else:
        network.read_communities_from_toml(communities_toml)

    nodetable = network.get_node_table()
    nodetable.to_csv(os.path.join(output_dir, "node_stats.txt"), sep="\t")

    nx.write_graphml(network.program_graph, os.path.join(output_dir, "program_graph.graphml"))
    nx.write_graphml(network.comm_graph, os.path.join(output_dir, "community_graph.graphml"))

    layout_algorithm = config.layout_algorithm
    network.compute_layout(algorithm=layout_algorithm, **config.layouts[layout_algorithm])
    
    community_layout_algorithm = config.community_layout_algorithm
    network.compute_community_network_layout(
        algorithm=community_layout_algorithm,
        **config.community_layouts[community_layout_algorithm]
    )

    # persist Network object to file
    if config.save_network_as_pkl:
        logging.info("Writing network_integration.pkl.gz file")
        network.to_pkl(os.path.join(output_dir, "network_integration.pkl.gz"))

    network.write_communities_toml( os.path.join(output_dir, "communities.toml"))
    pd.DataFrame.from_dict(data=network.program_communities, orient='index').to_csv(os.path.join(output_dir, 'program_communities.txt'), sep="\t", header=False)
    
    # Write representative programs and usages
    logging.info("Writing representative programs")
    rep_programs_ids = network.get_representative_program_ids()
    rep_programs = network.integration.get_programs()[rep_programs_ids.index]
    rep_programs.columns = pd.MultiIndex.from_tuples([[community] + list(program_id) for community, program_id in zip(rep_programs_ids, rep_programs_ids.index)], names=["Community", "dataset", "k", "Program"])
    rep_programs.to_csv(os.path.join(output_dir, "representative_programs.txt"), sep="\t") # outputs the programs to text file
    rep_programs_usage = network.integration.get_usages()[rep_programs_ids.index]
    rep_programs_usage.columns = pd.MultiIndex.from_tuples([[community] + list(program_id) for community, program_id in zip(rep_programs_ids, rep_programs_ids.index)], names=["Community", "dataset", "k", "Program"])
    rep_programs_usage.to_csv(os.path.join(output_dir, "representative_program_usages.txt"), sep="\t") # outputs the program usages to text file
    

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
    utils.save_fig(fig, os.path.join(output_dir, f"metadata_colors_legend"), target_dpi=300, formats=config.plot_formats)
    
    fig = colors.plot_dataset_colors_legend()
    utils.save_fig(fig, os.path.join(output_dir, f"dataset_colors_legend"), target_dpi=300, formats=config.plot_formats)

    logging.info("Creating network plots")

    # plot membership of datasets and ranks for each community
    fig = plot_community_contribution(network, colors, orientation="horizontal")
    utils.save_fig(fig, os.path.join(output_dir, f"community_contribution"), target_dpi=600, formats=config.plot_formats)

    fig = plot_community_contribution(network, colors, orientation="vertical")
    utils.save_fig(fig, os.path.join(output_dir, f"community_contribution_vertical"), target_dpi=600, formats=config.plot_formats)
    
    # summary community network
    fig = plot_community_network_summary(network, colors)
    utils.save_fig(fig, os.path.join(output_dir, f"community_network_summary"), target_dpi=600, formats=config.plot_formats)


    # Plot network colored by dataset
    fig = plot_program_network_datasets(network, colors)
    utils.save_fig(fig, os.path.join(output_dir, f"network_datasets"), target_dpi=600, formats=config.plot_formats)
    
    # Plot network colored by dataset and size by rank
    fig = plot_program_network_datasets(network, colors, node_size_kval=True)
    utils.save_fig(fig, os.path.join(output_dir, f"network_rank"), target_dpi=600, formats=config.plot_formats)
    
    # Plot network colored by community
    fig = plot_program_network_communities(network, colors)
    utils.save_fig(fig, os.path.join(output_dir, f"network_communities"), target_dpi=600, formats=config.plot_formats)

    # Cumulative proportion of samples contributing to each Program
    fig = plot_program_network_nsamples(network, colors)
    utils.save_fig(fig, os.path.join(output_dir, f"network_n_samples"), target_dpi=600, formats=config.plot_formats)
    
    # Cumulative proportion of patients contributing to each Program
    if network.integration.sample_to_patient is not None:
        fig = plot_program_network_npatients(network, colors)
        utils.save_fig(fig, os.path.join(output_dir, f"network_n_patients"), target_dpi=600, formats=config.plot_formats)
    

    # integrated community usage
    ic_usage = network.get_community_usage()
    ic_usage.to_csv(os.path.join(output_dir, "community_usage.txt"), sep="\t")
    n_samples = ic_usage.shape[0]
    if n_samples <= config.max_cells_per_heatmap_dimension:
        logging.info("Creating community usage heatmap...")
        fig = plot_community_usage_heatmap(network, colors)
        utils.save_fig(fig, os.path.join(output_dir, f"community_usage"), target_dpi=200, formats=config.plot_formats)
    else:
        logging.info(f"Skipped community usage heatmap. n_samples = {n_samples} is greater than the 'max_cells_per_heatmap_dimension' parameter.")

    logging.info("Computing community-level associations")

    # category counts for each layer
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_numerical=False):
            fig = plot_sample_numbers(dataset=dataset, layer=layer)
            layer_str = layer.replace("\\", "_").replace("/", "_")
            utils.save_fig(fig, os.path.join(output_dir, "categories", dataset_name, layer_str), target_dpi=300, formats=config.plot_formats)

    # Community-level, categorical metadata, overrepresentation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "overrepresentation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            df = network.get_community_category_overrepresentation(layer=layer, subset_datasets=dataset_name, truncate_negative=False)
            df.to_csv(os.path.join(output_dir, "annotated_communities", "overrepresentation", dataset_name, layer_str + ".txt"), sep='\t')
            
    # Community-level, numerical metadata, correlation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_communities", "correlation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            df = network.get_community_metadata_correlation(layer=layer, subset_datasets=dataset_name, method="pearson")
            df.to_csv(os.path.join(output_dir, "annotated_communities", "correlation", dataset_name, layer_str + ".txt"), sep='\t')

    logging.info("Creating community-level bar plots")

    # Community-level, categorical data, overrepresentation bar plots
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_numerical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            fig = plot_overrepresentation_community_bar(network, colors, layer=layer, subset_datasets=dataset_name)
            utils.save_fig(fig, os.path.join(output_dir, "annotated_communities", "overrepresentation_bar", dataset_name, layer_str), target_dpi=300, formats=config.plot_formats)
        
    # Community-level, numerical metadata, correlation bar plots
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_categorical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            fig = plot_metadata_correlation_community_bar(network, colors, layer=layer, subset_datasets=dataset_name)
            utils.save_fig(fig, os.path.join(output_dir, "annotated_communities", "correlation_bar", dataset_name, layer_str), target_dpi=300, formats=config.plot_formats)


    logging.info("Creating community-level network plots")

    # Community-level, categorical data, overrepresentation network
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_numerical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            fig = plot_overrepresentation_community_network(network, colors, layer=layer, subset_datasets=dataset_name)
            utils.save_fig(fig, os.path.join(output_dir, "annotated_communities", "overrepresentation_network", dataset_name, layer_str), target_dpi=600, formats=config.plot_formats)
 
    # Community-level, numerical data, correlation network
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_categorical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            fig = plot_metadata_correlation_community_network(network, colors, layer=layer, subset_datasets=dataset_name)   
            utils.save_fig(fig, os.path.join(output_dir, "annotated_communities", "correlation_network", dataset_name, layer_str), target_dpi=600, formats=config.plot_formats)

    logging.info("Computing program-level associations")

    # Program-level, categorical metadata, overrepresentation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_programs", "overrepresentation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_numerical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            df = dataset.get_category_overrepresentation(layer)
            df.to_csv(os.path.join(output_dir, "annotated_programs", "overrepresentation", dataset_name, layer_str + ".txt"), sep='\t')
            
    # Program-level, numerical metadata, correlation
    for dataset_name, dataset in integration.datasets.items():
        os.makedirs(os.path.join(output_dir, "annotated_programs", "correlation", dataset_name), exist_ok=True)
        for layer in dataset.get_metadata_df(include_categorical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            df = dataset.get_metadata_correlation(layer, method="pearson")
            df.to_csv(os.path.join(output_dir, "annotated_programs", "correlation", dataset_name, layer_str + ".txt"), sep='\t')

    logging.info("Creating program-level bar plots")

    # Program-level, categorical data, overrepresentation bar plots
    for dataset_name in integration.datasets:
        fig = plot_overrepresentation_program_bar(network, colors, dataset_name=dataset_name)
        utils.save_fig(fig, os.path.join(output_dir, "annotated_programs", "overrepresentation_bar", dataset_name), target_dpi=300, formats=config.plot_formats)
        
    # Program-level, numerical metadata, correlation bar plots
    for dataset_name in integration.datasets:
        fig = plot_metadata_correlation_program_bar(network, colors, dataset_name=dataset_name)
        utils.save_fig(fig, os.path.join(output_dir, "annotated_programs", "correlation_bar", dataset_name), target_dpi=300, formats=config.plot_formats)

    logging.info("Creating program-level network plots")

    # Program-level, categorical data, overrepresentation network
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_numerical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            fig = plot_overrepresentation_program_network(network, colors, layer=layer, subset_datasets=dataset_name)  
            utils.save_fig(fig, os.path.join(output_dir, "annotated_programs", "overrepresentation_network", dataset_name, layer_str), target_dpi=600, formats=config.plot_formats)
 
    # Program-level, numerical data, correlation network
    for dataset_name, dataset in integration.datasets.items():
        for layer in dataset.get_metadata_df(include_categorical=False):
            layer_str = layer.replace("\\", "_").replace("/", "_")
            fig = plot_metadata_correlation_program_network(network, colors, layer=layer, subset_datasets=dataset_name)   
            utils.save_fig(fig, os.path.join(output_dir, "annotated_programs", "correlation_network", dataset_name, layer_str), target_dpi=600, formats=config.plot_formats)

    logging.info("All tasks completed successfully.")
    
@click.command(name="ssgsea")
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, show_default=True,
    help="Output directory for ssgsea results")
@click.option('-n', '--pkl_file', type=click.Path(exists=True, dir_okay=False),
    help="Path to network_integration.pkl.gz file from `mosaicmpi integrate` step")
@click.option('-i', '--h5ad_file', type=click.Path(exists=True, dir_okay=False),
    help="Path to .h5ad file from `mosaicmpi postprocess`")
@click.option('-g', '--gene_sets', type=str, required=True, 
    help="Path to GMT file with gene sets or Enrichr Library name.")
@click.option("--min_intersection", type=int, default=5, show_default=True, 
    help="Minimum intersection size for gene sets")
@click.option("--max_intersection", type=int, default=500, show_default=True, 
    help="Minimum intersection size for gene sets")
@click.option('--cpus', type=int, default=cpus_available, show_default=True,
    help="Number of CPUs for MP-enabled tasks")
def cmd_ssgsea(output_dir, pkl_file, h5ad_file, gene_sets, min_intersection, max_intersection, cpus):
    """
    Compute and plot ssGSEA Normalized Enrichment Scores (NES) for mosaicMPI programs. If a network_integration.pkl file
    is provided, ssGSEA is performed on reprepresentative programs from an integration. If a .h5ad file is provided, ssGSEA
    is performed on all programs.
    """
    utils.start_logging()  # allows warning messages to be printed even though logfile hasn't been made yet
    try:
        import gseapy
    except ImportError:
        
        logging.error("gseapy is not installed. Please install using:\n\n\t"
                      "# if you have conda (MacOS_x86-64 and Linux only)\n\t"
                      "conda install -c bioconda gseapy\n\n\t"
                      "# Windows and MacOS_ARM64(M1/2-Chip)\n\t"
                      "pip install gseapy\nl"
                      )
        sys.exit(1)

    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        logging.warning(f"{output_dir} is not empty. Files may be overwritten.")

    # write to log file
    utils.start_logging(os.path.join(output_dir, "logfile.txt"))

    if h5ad_file and not pkl_file:
        # run ssGSEA on all programs from a .h5ad file
        dataset = Dataset.from_h5ad(h5ad_file)
        programs = dataset.get_programs()
        result = gseapy.ssgsea(data=programs, gene_sets=gene_sets, min_size=min_intersection, max_size=max_intersection, threads=cpus)
        prog_nes = result.res2d.pivot(index="Term", columns="Name", values="NES")
        prog_nes.columns = pd.MultiIndex.from_tuples(prog_nes.columns, names=("k", "program"))
        prog_nes.to_csv(os.path.join(output_dir, "program_nes.txt"), sep="\t")
    elif pkl_file and not h5ad_file:
        # run ssGSEA on representative programs for each dataset in a network_integration.pkl file
        network = Network.from_pkl(pkl_file)
        rep_programs = network.get_representative_programs()
        result = gseapy.ssgsea(data=rep_programs, gene_sets=gene_sets, min_size=min_intersection, max_size=max_intersection, threads=cpus)
        rep_nes = result.res2d.pivot(index="Term", columns="Name", values="NES")
        rep_nes.columns = pd.MultiIndex.from_tuples(rep_nes.columns, names=("community", "dataset", "k", "program"))
        sorter = rep_nes.astype(float).idxmax(axis=1)
        sorter = sorter.str[0].sort_values()
        rep_nes = rep_nes.loc[sorter.index]
        rep_nes.to_csv(os.path.join(output_dir, "representative_program_nes.txt"), sep="\t")

        # plot ssGSEA NES scores by community and dataset
        fig, figlegend = plot_representative_program_nes(network=network, rep_nes=rep_nes)
        utils.save_fig(fig, os.path.join(output_dir, "representative_program_nes"))
        utils.save_fig(figlegend, os.path.join(output_dir, "representative_program_nes.legend"))
        
    else:
        logging.error("mosaicmpi ssgsea requires either a factorized dataset (.h5ad) file or a network_integration.pkl file to run ssGSEA on programs.")
        sys.exit(1)
    logging.info("All tasks completed successfully.")
    
    
@click.command(name="gprofiler")
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, show_default=True,
    help="Output directory for gprofiler results")
@click.option('-n', '--pkl_file', type=click.Path(exists=True, dir_okay=False),
    help="Path to network_integration.pkl.gz file from `mosaicmpi integrate` step")
@click.option('-i', '--h5ad_file', type=click.Path(exists=True, dir_okay=False),
    help="Path to .h5ad file from `mosaicmpi postprocess`")
@click.option('-g', '--gene_sets',
              type = click.Choice(["GO:MF", "GO:CC", "GO:BP", "KEGG", "REAC", "WP", "TF", "MIRNA", "HPA", "CORUM", "HP"
                                   ]),
              multiple=True, default=[],
    help="Source for gene sets from g:Profiler. Defaults to all sources. "
    "Use multiple times to specify multiple sources, e.g.: -g GO:CC -g GO:BP")
@click.option('-s', '--species', type=click.Choice(["hsapiens", "mmusculus"]), required=True,
    help="Species for gene name IDs")
@click.option("--min_intersection", type=int, default=5, show_default=True, 
    help="Minimum intersection size for gene sets")
@click.option("--max_intersection", type=int, default=500, show_default=True, 
    help="Minimum intersection size for gene sets")
@click.option("--n_hsg", type=int, default=1000, show_default=True)
@click.option("--cmap", type=str, default="Blues",
              help="matplotlib colormap name for heatmap plots.")
@click.option("--vmin", type=float, default=0,
              help="minimum -log10(pval) for heatmap plots")
@click.option("--vmax", type=float, default=10,
              help="maximum -log10(pval) for heatmap plots")
@click.option("--no_plot", is_flag=True,
              help="Skip plotting geneset significance heatmaps")

def cmd_gprofiler(output_dir, pkl_file, h5ad_file, gene_sets, species, min_intersection, max_intersection, n_hsg, cmap, vmin, vmax, no_plot):
    """
    Perform gProfiler gene set analysis of highly-scoring genes from mosaicMPI programs. If a network_integration.pkl file
    is provided, gProfiler is performed on reprepresentative programs from an integration. If a .h5ad file is provided, gProfiler
    is performed on all programs.
    """
    utils.start_logging()  # allows warning messages to be printed even though logfile hasn't been made yet

    try:
        from .gprofiler import program_gprofiler, order_genesets
    except ImportError:
        logging.error("gprofiler-official is not installed. Please install using:\n\n\t"
                      "conda install -c bioconda gprofiler-official"
                      )
        sys.exit(1)

    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        logging.warning(f"{output_dir} is not empty. Files may be overwritten.")

    # write to log file
    utils.start_logging(os.path.join(output_dir, "logfile.txt"))

    if h5ad_file and not pkl_file:
        # run gProfiler on all programs from a .h5ad file
        dataset = Dataset.from_h5ad(h5ad_file)
        programs = dataset.get_programs()
        
        logging.info(f"Querying g:Profiler...")
        result = program_gprofiler(program_df=programs, species=species, n_hsg=n_hsg,
                                   gene_sets=gene_sets, min_intersection=min_intersection, max_intersection=max_intersection)

        # write background genes
        with open(os.path.join(output_dir, f"background.txt"), "w") as f:  # gProfiler input format for web version
            f.write(f">background\n")
            f.write(" ".join(result.background) + "\n")

        # write query genes
        with open(os.path.join(output_dir, "query.txt"), "w") as f:   # gProfiler input format for web version
            for name, querylist in result.query.items():
                f.write(f">{name}\n")
                f.write(" ".join(querylist) + "\n")

        # write gProfiler output
        result.gprofiler_output.to_csv(os.path.join(output_dir, "result.full.txt"), sep="\t")
        result.summary.to_csv(os.path.join(output_dir, "result.txt"), sep="\t")

        # create p-value heatmaps separately for each k
        max_k = dataset.adata.uns["kvals"].index.max()
        for k in tqdm(dataset.adata.uns["kvals"].index, total=max_k, unit="k", desc="Creating heatmaps"):
            df = result.summary["-log10pval"][k].dropna(how="all").fillna(0)
            df = order_genesets(df)
            df.to_csv(os.path.join(output_dir, f"ordered_genesets_k{k}.txt"), sep="\t")
            if not no_plot:
                fig, figlegend = plot_geneset_pval_heatmap(df=df)
                ax = fig.axes[0]
                ax.set_title("g:Profiler pathways\n" + ",".join(gene_sets))
                ax.set_xlabel("")
                ax.set_ylabel("")
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color('#aaaaaa') 
                ax.set_xlabel(f"Program (k={k})")
                figlegend.savefig(os.path.join(output_dir, f"ordered_genesets_k{k}.legend.pdf"))
                fig.savefig(os.path.join(output_dir, f"ordered_genesets_k{k}.pdf"))
                plt.close(fig)
                plt.close(figlegend)


    elif pkl_file and not h5ad_file:
        # run gProfiler on representative programs for each dataset in a network_integration.pkl file
        network = Network.from_pkl(pkl_file)
        rep_programs = network.get_representative_programs()
        for dataset_name in network.integration.datasets.keys():
            programs = rep_programs.xs(dataset_name, axis=1, level=1).dropna()
            programs = programs.sort_index(axis=1, level=0, key=network.get_vectorized_community_sort_key).droplevel(axis=1, level=[1, 2])

            logging.info(f"Querying g:Profiler...")
            result = program_gprofiler(program_df=programs, species=species, n_hsg=n_hsg,
                                    gene_sets=gene_sets, min_intersection=min_intersection, max_intersection=max_intersection)

            # write background genes
            os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
            with open(os.path.join(output_dir, dataset_name, f"background.txt"), "w") as f:  # gProfiler input format for web version
                f.write(f">background\n")
                f.write(" ".join(result.background) + "\n")

            # write query genes
            with open(os.path.join(output_dir, dataset_name, "query.txt"), "w") as f:   # gProfiler input format for web version
                for name, querylist in result.query.items():
                    f.write(f">{name}\n")
                    f.write(" ".join(querylist) + "\n")

            # write gProfiler output
            result.gprofiler_output.to_csv(os.path.join(output_dir, dataset_name, "result.full.txt"), sep="\t")
            result.summary.to_csv(os.path.join(output_dir, dataset_name, "result.txt"), sep="\t")
            
            # ordered pathways
            df = result.summary["-log10pval"].dropna(how="all").fillna(0)
            df = order_genesets(df)
            df.to_csv(os.path.join(output_dir, dataset_name, f"ordered_genesets.txt"), sep="\t")

            if not no_plot:
                fig, figlegend = plot_geneset_pval_heatmap(df=df)
                ax = fig.axes[0]
                ax.set_title("g:Profiler pathways\n" + ",".join(gene_sets))
                ax.set_xlabel("")
                ax.set_ylabel("")
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color('#aaaaaa') 
                ax.set_xlabel(f"Community\n(Representative Program)")
                figlegend.savefig(os.path.join(output_dir, dataset_name, f"ordered_genesets.legend.pdf"))
                fig.savefig(os.path.join(output_dir, dataset_name, f"ordered_genesets.pdf"))
                plt.close(fig)
                plt.close(figlegend)
        
    else:
        logging.error("mosaicmpi gprofiler requires either a factorized dataset (.h5ad) file or "
                      "a network_integration.pkl.gz file to run g:Profiler on programs.")
        sys.exit(1)
    logging.info("All tasks completed successfully.")


@click.command(name="compare-integrations")
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True,
    help="Output directory for results")
@click.option('--name1', type=str, default="Network 1",
    help="Name of first network (plotted on the y-axis)")
@click.option('--pkl1', type=click.Path(exists=True, dir_okay=False), required=True,
    help="Path to first network_integration.pkl.gz file")
@click.option('--name2', type=str, default="Network 2",
    help="Name of second network (plotted on the x-axis)")
@click.option('--pkl2', type=click.Path(exists=True, dir_okay=False), required=True,
    help="Path to second network_integration.pkl.gz file")
@click.option('-l', '--colors_toml', type=click.Path(dir_okay=False, exists=True), required=False,
    help="TOML file with dataset_colors specification. If not provided, visually distinct colors will be chosen automatically.")
def cmd_compare_integrations(output_dir, name1, pkl1, name2, pkl2, colors_toml):
    """Compare communities from two network_integration.pkl files generated using `mosaicmpi integrate`.

    """
    utils.start_logging()  # allows warning messages to be printed even though logfile hasn't been made yet
    
    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        logging.warning(f"{output_dir} is not empty. Files may be overwritten.")

    # write to log file
    utils.start_logging(os.path.join(output_dir, "logfile.txt"))
    
    network1 = Network.from_pkl(pkl1)
    network2 = Network.from_pkl(pkl2)

    jaccard = compare_community_jaccard_similarity(name1=name1, name2=name2, network1=network1, network2=network2)
    jaccard.to_csv(os.path.join(output_dir, "jaccard.txt"), sep="\t")

    # create or import a color for each dataset
    if colors_toml is None:
        colors = Colors()        
    else:
        logging.info("Using provided TOML file for color palettes")
        colors = Colors.from_toml(colors_toml)

    dataset_names = set(network1.integration.datasets) | set(network2.integration.datasets)
    colors.add_missing_dataset_colors(datasets=dataset_names)
    colors.to_toml(os.path.join(output_dir, "colors.toml"))

    fig = plot_compare_integrations(name1, network1, name2, network2, colors=colors)
    utils.save_fig(fig, os.path.join(output_dir, "jaccard_heatmap"))
    logging.info("All tasks completed successfully.")

@click.command(name="transfer-labels")
@click.option("-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory for results.")
@click.option('-n', '--pkl_file', type=click.Path(exists=True, dir_okay=False), required=True,
    help="Path to network_integration.pkl.gz file from `mosaicmpi integrate` step")
@click.option('-s', '--source', type=str, multiple=True,
    help="Only calculate Name of source dataset for sample labels.")
@click.option('-d', '--dest', type=str, multiple=True,
    help="Name of destination dataset")
@click.option('-l', '--layer', type=str, multiple=True,
    help="Name of categorical data layer to transfer")
@click.option('-a', '--annotate', type=str, required=False, multiple=True,
    help="Annotate transfer heatmap using categorical data layer(s) from destination dataset")
@click.option('-m', '--metadata_colors_toml', type=click.Path(dir_okay=False, exists=True), required=False,
    help="TOML file with metadata_colors specification. If not provided, visually distinct colors will be chosen automatically.")
def cmd_transfer_labels(output_dir, pkl_file, source, dest, layer, annotate, metadata_colors_toml):
    """Transfer labels from a source to a destination dataset.
    """ 
    network = Network.from_pkl(pkl_file)
    os.makedirs(output_dir, exist_ok=True)

    transfer_df = network.transfer_labels(source=(source if source else None),
                                          dest=(dest if dest else None),
                                          layer=(layer if layer else None))
    transfer_df.to_csv(os.path.join(output_dir, "transfer_score.txt"), sep="\t")

    if annotate:
        if metadata_colors_toml:
            colors = Colors.from_toml(metadata_colors_toml)
        else:
            colors = Colors.from_network(network)
    else:
        colors = None
        annotate = None

    if not source:
        source = list(network.integration.datasets.keys())
    if not dest:
        dest = list(network.integration.datasets.keys())
    if not layer:
        layer = network.integration.get_metadata_df(subset_datasets=source, include_numerical=False).columns
    
    total = len(source) * len(dest) * len(layer)

    for s, d, l in tqdm(product(source, dest, layer), total=total, unit="plot", desc="Creating plots"):
        if len(network.integration.datasets[s].get_metadata_df()[l].unique()) > 1:
            cgrid = plot_metadata_transfer(network=network, source=s, dest=d, layer=l, annotate=annotate, colors=colors)
            cgrid.fig.suptitle(f"source: {s}, dest: {d}, layer: {l}")
            cgrid.savefig(os.path.join(output_dir, f"s.{s}_d.{d}_l.{l}.pdf"))

    if annotate:
        # create legends for annotation tracks
        width = 3 * len(annotate)
        height = max([1 + len(colors.get_metadata_colors(a)) / 4 for a in annotate])
        fig, axes = plt.subplots(1, len(annotate), figsize=[width, height], layout="constrained", squeeze=False)
        for layer, ax in zip(annotate, axes[0]):
            colors.plot_metadata_colors_legend(layer=layer, ax=ax)
        utils.save_fig(fig, os.path.join(output_dir, "legend"), formats=("pdf", "png"), target_dpi=400, facecolor='white')


    logging.info("All tasks completed successfully.")

cli.add_command(cmd_txt_to_h5ad)
cli.add_command(cmd_update_h5ad_metadata)
cli.add_command(cmd_impute_knn)
cli.add_command(cmd_impute_zeros)
cli.add_command(cmd_check_h5ad)
cli.add_command(cmd_model_odg)
cli.add_command(cmd_set_parameters)
cli.add_command(cmd_factorize)
cli.add_command(cmd_postprocess)
cli.add_command(cmd_annotated_heatmap)
cli.add_command(cmd_map_gene_ids)
cli.add_command(cmd_create_config)
cli.add_command(cmd_integrate)
cli.add_command(cmd_ssgsea)
cli.add_command(cmd_gprofiler)
cli.add_command(cmd_compare_integrations)
cli.add_command(cmd_transfer_labels)
