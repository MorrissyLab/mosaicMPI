import os
import logging
import subprocess
import collections
import sys
from datetime import datetime
from typing import Optional, Mapping

import click
import numpy as np
import pandas as pd
import cnmfsns as cn
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import tomli_w
import distinctipy
from matplotlib.lines import Line2D
from scipy.stats import entropy

if hasattr(os, "sched_getaffinity"):
    cpus_available = len(os.sched_getaffinity(0))
else:
    cpus_available = os.cpu_count()


#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(6.0,4.0)    #(6.0,4.0)
mpl.rcParams['font.size']=10                #10 
mpl.rcParams['savefig.dpi']=72             #72 
mpl.rcParams['figure.subplot.bottom']=.125    #.125


def start_logging(output_path=None):
    if output_path is None:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
            handlers=[
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
            handlers=[
                logging.FileHandler(output_path, mode="a"),
                logging.StreamHandler()
            ]
        )
    return


class OrderedGroup(click.Group):
    """
    Overwrites Groups in click to allow ordered commands.
    """
    def __init__(self, name: Optional[str] = None, commands: Optional[Mapping[str, click.Command]] = None, **kwargs):
        super(OrderedGroup, self).__init__(name, commands, **kwargs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@click.group(cls=OrderedGroup)
@click.version_option(version=cn.__version__)
def cli():
    """
    cNMF-SNS is a tool for deconvolution and integration of multiple datasets based on consensus Non-Negative Matrix Factorization (cNMF).
    """

@click.command(name="txt-to-h5ad")
@click.option(
    "-d", "--data_file", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input counts or normalized matrix as tab-delimited text file. Columns are samples/cells and rows are genes/features.")
@click.option(
    "--is_normalized", is_flag=True, help="Specify if input data is normalized instead of count data.")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Optional tab-separated text file with metadata for samples/cells with one row each. Columns are annotation layers.")
@click.option(
    "--sparsify", is_flag=True,
    help="Save resulting data in sparse format. Recommended to increase performance for sparse datasets such as scRNA-Seq, scATAC-Seq, and 10X Visium, but not for bulk expression data.")
@click.option(
    "-o", '--output', type=click.Path(dir_okay=False, exists=False), required=True,
    help="Path to output .h5ad file.")
def cmd_txt_to_h5ad(data_file, is_normalized, metadata, output, sparsify):
    """
    Create .h5ad file with data and metadata (`adata.obs`).
    """
    start_logging()
    df = pd.read_table(data_file, index_col=0)
    metadata_df = pd.read_table(metadata, index_col=0).dropna(axis=1, how="all")
    dataset = cn.Dataset.from_df(data=df, obs=metadata_df, sparsify=sparsify, is_normalized=is_normalized)
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
    Update metadata in a .h5ad file at any point in the cNMF-SNS workflow. New metadata will overwrite (`adata.obs`).
    """
    start_logging()
    dataset = cn.Dataset.from_h5ad(input_h5ad)
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
    start_logging()
    dataset = cn.Dataset.from_h5ad(input)
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
        cnmfsns model-odg -n test -i test.h5ad

        # Explicitly use a linear model instead of a BSpline Generalized Additive Model
        cnmfsns model-odg -n test -i test.h5ad --odg_default_spline_degree 0 --odg_default_dof 1
    """
    cn.cnmf.cNMF(output_dir=output_dir, name=name)  # creates directories for cNMF
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    dataset = cn.Dataset.from_h5ad(input)
    
    # Create gene stats table and save h5ad file
    dataset.compute_gene_stats()
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    gene_stats = dataset.adata.var
    os.makedirs(os.path.normpath(os.path.join(output_dir, name, "odgenes")), exist_ok=True)
    gene_stats.to_csv(os.path.join(output_dir, name, "odgenes", "genestats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = cn.plots.plot_feature_dispersion(gene_stats, show_selected=False)["default"]
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
        cnmfsns set_parameters -n test -m default_minscore -p 1.0

        # to reproduce cNMF default behaviour (Kotliar et al., 2019, eLife)
        cnmfsns set_parameters -n test -m cnmf_topn -p 2000          

        # select top 20% of genes when ranked by od-score
        cnmfsns set_parameters -n test -m default_quantile -p 0.8

        # input a gene list from text file
        cnmfsns set_parameters -n test -m genes_file -p path/to/genesfile.txt
    """
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    dataset = cn.Dataset.from_h5ad(os.path.join(output_dir, name, name + ".h5ad"))

    if odg_method == "genes_file":
        odg_param = click.Path(exists=True, dir_okay=False)(odg_param)
        genes = open(odg_param).read().rstrip().split(os.linesep)
        dataset.select_overdispersed_genes_from_genelist(genes)
    else:
        overdispersion_metric = odg_method.split("_")[0]
        
        method = {odg_method.split("_")[1]: odg_param}
        dataset.select_overdispersed_genes(overdispersion_metric=overdispersion_metric, min_mean=min_mean,
                                           **method)

    # create mean vs variance plots, updated with selected genes
    fig = cn.plots.plot_feature_dispersion(df=dataset.adata.var, show_selected=True)["default"]
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
    dataset.initialize_cnmf(output_dir = output_dir, name=name, kvals=kvals, n_iter=n_iter, beta_loss=beta_loss, seed=seed)
    
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
    Performs factorization according to parameters specified using `cnmfsns set-parameters`.
    """
    cnmf_obj = cn.cnmf.cNMF(output_dir=output_dir, name=name)
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    
    run_params = cn.io.load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    if run_params.shape[0] == 0:
        logging.error("No factorization to do: either no values of k were selected using `cnmfsns set-parameters` or iterations were set to 0.")

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
    help="Threshold for the local density filtering prior to GEP consensus. Acceptable thresholds are > 0 and <= 2 (2.0 is no filtering).")
@click.option(
    '--local_neighborhood_size', type=float, default=0.3, show_default=True,
    help="Fraction of the number of replicates to use as nearest neighbors for local density filtering.")
@click.option(
    '--skip_missing_iterations', is_flag=True,
    help="If specified, consensus GEPs and usages will be calculated even though individual iterations are missing.")
@click.option(
    '--force_h5ad_update', is_flag=True,
    help="If specified, overwrites cNMF results already saved to the .h5ad file.")
def cmd_postprocess(name, output_dir, cpus, local_density_threshold, local_neighborhood_size, skip_missing_iterations, force_h5ad_update):
    """
    Perform post-processing routines on cNMF after factorization. This includes checking factorization outputs for completeness, combining individual
    iterations, calculating consensus GEPs and usage matrices, and creating the k-selection and annotated usage plots.
    """
    cnmf_obj = cn.cnmf.cNMF(output_dir=output_dir, name=name)
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    cnmf_obj.postprocess(cpus=cpus,
                         local_density_threshold=local_density_threshold,
                         local_neighborhood_size=local_neighborhood_size,
                         skip_missing_iterations=skip_missing_iterations)
    h5ad_path = os.path.join(output_dir, name, name + ".h5ad")
    dataset = cn.Dataset.from_h5ad(h5ad_path)
    
    cnmf_data_loaded =  "cnmf_usage" in dataset.adata.obsm or\
                        "cnmf_gep_score" in dataset.adata.varm or\
                        "cnmf_gep_tpm" in dataset.adata.varm or\
                        "cnmf_gep_raw" in dataset.adata.varm
    if cnmf_data_loaded and not force_h5ad_update:
        logging.Error(f"Error: AnnData already contains cNMF results. Use --force_h5ad_update to overwrite.")
        sys.exit(1)

    dataset.add_cnmf_results(cnmf_output_dir=output_dir,
                             cnmf_name=name,
                             h5ad_path=h5ad_path,
                             local_density_threshold=local_density_threshold,
                             local_neighborhood_size=local_neighborhood_size,
                             skip_missing_iterations=skip_missing_iterations
                             )
    logging.info(f"Writing h5ad file to {h5ad_path}")
    dataset.to_h5ad(h5ad_path)
    
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
    start_logging()
    os.makedirs(output_dir, exist_ok=True)
    dataset = cn.Dataset.from_h5ad(input_h5ad)
    
    # get metadata colors
    if metadata_colors_toml:
        cfg = cn.Config.from_toml(metadata_colors_toml)
    else:
        cfg = cn.Config()
    cfg.add_missing_metadata_colors(dataset)
    cfg.to_toml(os.path.join(output_dir, "metadata_colors.toml"), section_subset=["metadata_colors"])
    # plot legend
    fig = cfg.plot_metadata_colors_legend()
    fig.savefig(os.path.join(output_dir, f"metadata_legend.pdf"))
    
    # filter metadata layers with too many categories
    exclude_maxcat = dataset.adata.obs.select_dtypes(include="category").apply(lambda x: len(x.cat.categories)) > max_categories_per_layer
    if dataset.adata.obs.shape[1] > 0:
        metadata = dataset.adata.obs.drop(columns=exclude_maxcat[exclude_maxcat].index).dropna(axis=1, how="all")
    else:
        metadata = dataset.adata.obs.dropna(axis=1, how="all")
    
    if not dataset.has_cnmf_results:
        logging.error("cNMF results have not been merged into .h5ad file. Ensure that you have run `cnmfsns postprocess` before creating annotated usage heatmaps.")
        sys.exit(1)

    # create annotated plots for each k
    metadata_colors = {col: cfg.get_metadata_colors(col) for col in dataset.adata.obs.columns}
    for k in dataset.adata.uns["kvals"].index:
        logging.info(f"Creating annotated usage heatmap for k={k}")
        cnmf_name = dataset.adata.uns["cnmf_name"]
        title = f"{cnmf_name} k={k}"
        filename = os.path.join(output_dir, f"{cnmf_name}.usages.k{k:03}.pdf")
        fig = cn.plots.plot_annotated_usages(
            dataset=dataset, k=k, metadata=metadata, metadata_colors=metadata_colors, missing_data_color=cfg.metadata_colors["missing_data"], title=title, filename=filename,
            cluster_samples=True, cluster_geps=False, show_sample_labels=(not hide_sample_labels), ylabel="GEP")
        fig.savefig(filename, transparent=False, bbox_inches = "tight")

@click.command(name="integrate")
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="Output directory for cNMF-SNS results")
@click.option('-c', '--config_toml', type=click.Path(exists=True, dir_okay=False), required=False, help="TOML config file")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), multiple=True, help="h5ad file with cNMF results. Can be used to specify multiple datasets for integration instead of a TOML config file.")
@click.option('--cpus', type=int, default=cpus_available, show_default=True, help="Number of CPUs to use for calculating correlation matrix")
def cmd_integrate(output_dir, config_toml, cpus, input_h5ad):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5ad files directly, it is recommended to use a single TOML file instead for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    start_logging()
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        logging.warning(f"Integration directory {output_dir} is not empty. Files may be overwritten.")
    start_logging(os.path.join(output_dir, "logfile.txt"))
    os.makedirs(os.path.join(output_dir, "integrate"), exist_ok=True)
    if config_toml is None and len(input_h5ad) == 0:
        logging.error("Datasets for integration must be specified in a config TOML file using `-c` or using `-i` for individual .h5ad files. ")
        sys.exit(1)
    if config_toml is not None and input_h5ad:
        logging.error("A TOML config file can be specified, or 1 or more .h5ad files can be specified, but not both.")
        sys.exit(1)
    if not all(fn.endswith(".h5ad") for fn in input_h5ad):
        logging.error("Input files must be AnnData .h5ad files.")
        sys.exit(1)

    # create config
    if config_toml is not None:
        config = Config.from_toml(config_toml)
    elif input_h5ad:
        config = Config.from_h5ad_files(input_h5ad)

    # add missing colors to config
    logging.info("Checking metadata colors for completeness...")
    config.add_missing_dataset_colors()
    config.add_missing_metadata_colors()

    k_table = {}
    geps = {}
    for dataset_name, dataset in config.datasets.items():
        adata = read_h5ad(dataset["filename"], backed="r")
        df = adata.varm["cnmf_gep_score"]
        df.columns = pd.MultiIndex.from_tuples([(int(gep[0]), int(gep[1])) for gep in df.columns.str.split(".")])
        geps[dataset_name] = df
        kvals = adata.uns["kvals"].copy()
        kvals["cNMF result"] = True
        k_table[dataset_name] = kvals
        
    geps = pd.concat(geps, axis=1).sort_index(axis=1)
    k_table = pd.concat(k_table, axis=1)

    corr_path = os.path.join(output_dir, "integrate", config.integrate["corr_method"] + ".df.npz")
    try:
        corr = load_df_from_npz(corr_path)
        logging.info(f"Loaded previously calculated correlation matrix from {corr_path}")
    except FileNotFoundError:
        logging.info(f"Calculating correlation matrix")
        if config.integrate["corr_method"] == "pearson":
            try:
                from nancorrmp.nancorrmp import NaNCorrMp
            except ImportError:
                logging.info(f"nancorrmp not installed. Calculating Pearson correlation matrix using 1 CPU.")
                corr = geps.corr(config.integrate["corr_method"])
            else:
                cpu_string = "all" if cpus == -1 else str(cpus)
                logging.info(f"nancorrmp found. Calculating Pearson correlation matrix using {cpu_string} CPUs.")
                corr = NaNCorrMp.calculate(geps, n_jobs=cpus)
        else:
            logging.info(f"Calculating Spearman correlation matrix using 1 CPU.")
            corr = geps.corr(config.integrate["corr_method"])
        save_df_to_npz(corr, corr_path)
    # Check that rows and columns of correlation matrix are identical
    assert (corr.index == corr.columns).all()

    # Lower triangular matrix contains each edge only once and removes diagonal
    tril = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))

    # Reduces rank (k) value when correlation distribution is skewed towards 1.
    for dataset_name in tril.index.levels[0]:
        dscorr = tril.loc[dataset_name, dataset_name]
        kvals = dscorr.index.levels[0].sort_values(ascending=False)
        max_kval_medians = []
        for max_kval in kvals:
            rankreduced = dscorr.loc[dscorr.index.get_level_values(0) <= max_kval, dscorr.columns.get_level_values(0) <= max_kval]
            median_corr = np.nanmedian(rankreduced.values)
            max_kval_medians.append(median_corr)
        max_kval_medians = pd.Series(max_kval_medians, index=kvals)
        max_k_threshold = None
        for max_k, median_corr in max_kval_medians.items():
            max_k_threshold = max_k
            if median_corr <= config.integrate["max_median_corr"]:
                break
        new_columns = pd.DataFrame([max_kval_medians, (max_kval_medians.index.to_series() <= max_k_threshold)]).T #, columns=pd.MultiIndex.from_product([[dataset_name],["max_k_median_corr", "max_k_filter_pass"]]))
        new_columns = pd.DataFrame({"max_k_median_corr": max_kval_medians, "max_k_filter_pass": (max_kval_medians.index.to_series() <= max_k_threshold)})
        new_columns = pd.concat({dataset_name: new_columns}, axis=1)
        k_table = k_table.merge(new_columns, how="outer", left_index=True, right_index=True)

    # output updated TOML with default k value selections based on filters etc
    for dataset_name, dataset_params in config.datasets.items():
        k_param = set()
        for k_entry in dataset_params["selected_k"]:
            if isinstance(k_entry, int):
                k_param.add(k_entry)
            elif isinstance(k_entry, collections.abc.Collection):
                assert len(k_entry) == 3
                for k in range(k_entry[0], k_entry[1]+1, k_entry[2]):
                    k_param.add(k)
        k_table[(dataset_name, "selected_k")] = k_table[(dataset_name, "cNMF result")] & k_table[(dataset_name, "max_k_filter_pass")] & k_table.index.isin(k_param)
        config.datasets[dataset_name]["selected_k"] = sorted(list(k_table[(dataset_name, "selected_k")][k_table[(dataset_name, "selected_k")]].index))
    k_table = k_table.sort_index(axis=1)
    k_table.to_csv(os.path.join(output_dir, "integrate", "k_filters.txt"), sep="\t")
    output_toml = os.path.join(output_dir, "integrate", "config.toml")
    config.to_toml(output_toml)
    logging.info(f"Output updated TOML file to: {output_toml}")

    # Rank Reduction Plots
    for dataset_name in config.datasets:
        fig = plot_rank_reduction(k_table.loc[:, dataset_name], config.integrate["max_median_corr"])
        fig.savefig(os.path.join(output_dir, "integrate", f"{dataset_name}.rank_reduction.pdf"))
        fig.savefig(os.path.join(output_dir, "integrate", f"{dataset_name}.rank_reduction.png"))

    # Filter correlations using dataset-specific max_k thresholds
    maxk_filtered_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if k_table.loc[gep[1], (gep[0], "max_k_filter_pass")]])
    selected_k_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if k_table.loc[gep[1], (gep[0], "selected_k")]])
    maxk_filtered_tril = tril.loc[maxk_filtered_index, maxk_filtered_index]
    selected_k_tril = tril.loc[selected_k_index, selected_k_index]

    # Pairwise correlation thresholds from unfiltered correlation matrix 
    pairwise_thresholds = []
    for row, dataset_row in enumerate(maxk_filtered_tril.index.levels[0]):
        for col, dataset_col in enumerate(maxk_filtered_tril.columns.levels[0]):
            distr = maxk_filtered_tril.loc[dataset_row, dataset_col].values.flatten()

            if not all(np.isnan(distr)):
                pairwise_thresholds.append({
                    "dataset_row": dataset_row,
                    "dataset_col": dataset_col,
                    "threshold": -np.quantile(distr[distr < 0], q=1-config.integrate["negative_corr_quantile"])
                })

    pairwise_thresholds = pd.DataFrame.from_records(pairwise_thresholds).set_index(["dataset_row", "dataset_col"])
    pairwise_thresholds.to_csv(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr_thresholds.txt"), sep="\t")


    # plot pairwise corr of all k
    fig = plot_pairwise_corr(tril=tril, thresholds=pairwise_thresholds)
    fig.savefig(os.path.join(output_dir, "integrate", "all.pairwise_corr.pdf"))
    fig.savefig(os.path.join(output_dir, "integrate", "all.pairwise_corr.png"), dpi=600)

    # plot pairwise corr with max_k_thresholds
    fig = plot_pairwise_corr(tril=maxk_filtered_tril, thresholds=pairwise_thresholds)
    fig.savefig(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr.pdf"))
    fig.savefig(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr.png"), dpi=600)

    # plot pairwise corr with max_k_thresholds
    fig = plot_pairwise_corr(tril=selected_k_tril, thresholds=pairwise_thresholds)
    fig.savefig(os.path.join(output_dir, "integrate", "selected_k.pairwise_corr.pdf"))
    fig.savefig(os.path.join(output_dir, "integrate", "selected_k.pairwise_corr.png"), dpi=600)

    # plot mirrored distributions with thresholds (which are computed on max-k filtered data only)
    fig = plot_pairwise_corr_overlaid(tril=maxk_filtered_tril, thresholds=pairwise_thresholds)
    fig.savefig(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr_thresholds.pdf"))
    fig.savefig(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr_thresholds.png"), dpi=600)

    # UpSet plot of odgenes and all genes in each dataset
    if len(config.datasets) > 1:
        for fig_name, fig in plot_genelist_upsets(config).items():
            fig.savefig(os.path.join(output_dir, "integrate", fig_name + ".pdf"))
            fig.savefig(os.path.join(output_dir, "integrate", fig_name + ".png"), dpi=600)

    # Table with node stats
    nodetable = {}
    node_filters = {
        "none": tril,
        "maxk": maxk_filtered_tril,
        "selectedk": selected_k_tril
    }
    for node_filter, df in node_filters.items():
        for edge_filter, thresholds in (("none", None), ("mincorr", pairwise_thresholds)):
            
            if thresholds is not None:
                df_filt = df.copy(deep=True)
                # apply pairwise thresholds
                for dataset_row in df.index.levels[0]:
                    for dataset_col in df.columns.levels[0]:
                        if (dataset_row, dataset_col) in thresholds.index:
                            min_corr = thresholds.loc[(dataset_row, dataset_col)].values[0]
                            mask = df_filt.loc[dataset_row, dataset_col] < min_corr
                            df_filt.loc[dataset_row, dataset_col][~mask] = np.NaN
                            df_filt.dropna(axis=0).dropna(axis=1)
            else:
                df_filt = df

            results = {}
            for dataset_name, subdf in df_filt.groupby(axis=0, level=0):
                results[dataset_name] = subdf.shape[0]
            nodetable[(node_filter, edge_filter)] = results

    nodetable = pd.DataFrame(nodetable)
    nodetable.columns.rename(["Node filter", "Edge Filter"], inplace=True)
    nodetable.to_csv(os.path.join(output_dir, "integrate", "node_stats.txt"), sep="\t")

        

@click.command(name="create-network")
@click.option(
    '-o', '--output_dir', type=click.Path(file_okay=False, exists=True), required=True,
    help="Output directory for cNMF-SNS results generated using `cnmfsns integrate`")
@click.option(
    '-n', '--name', type=str, default=datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S"),
    help="Name for this network. Output from this step will be in [output_dir]/sns_networks/[name]/...")
@click.option(
    '-c', '--config_toml', type=click.Path(exists=True, dir_okay=False), 
    help="TOML config file. Defaults to file output from `cnmfsns integrate` step: [output_dir]/integrate/config.toml")
def cmd_create_network(output_dir, name, config_toml):
    """
    Create network integration.
    """
    start_logging(os.path.join(output_dir, "logfile.txt"))

    if config_toml is None:
        config = Config.from_toml(os.path.join(output_dir, "integrate", "config.toml"))
    else:
        config = Config.from_toml(config_toml)

    sns_output_dir = os.path.join(output_dir, "sns_networks", name)
    os.makedirs(sns_output_dir, exist_ok=True)

    fig = config.plot_metadata_colors_legend()
    fig.savefig(os.path.join(sns_output_dir, "annotation_legend.pdf"))
    plt.close(fig)

    # write current configuration to config.toml file in the SNS output directory
    config.to_toml(os.path.join(sns_output_dir, "config.toml"))

    logging.info("Creating GEP network")
    G = create_graph(output_dir, config)
    nx.write_graphml(G, os.path.join(sns_output_dir, "gep_network.graphml"))
    communities_resolution_sweep, selected_resolution = sweep_community_resolution(G, config)
    communities = communities_resolution_sweep[selected_resolution]
    write_communities_toml(communities, os.path.join(sns_output_dir, "communities.toml"))
    gep_communities = {gep: community for community, geps in communities.items() for gep in geps}
    pd.DataFrame.from_dict(data=gep_communities, orient='index').to_csv(os.path.join(sns_output_dir, 'gep_communities.txt'), sep="\t", header=False)
        
    add_community_weights_to_graph(G, gep_communities, config)

    geps = {}
    for dataset_name, dataset in config.datasets.items():
        adata = read_h5ad(dataset["filename"], backed="r")
        df = adata.varm["cnmf_gep_score"]
        df.columns = pd.MultiIndex.from_tuples([(int(gep[0]), int(gep[1])) for gep in df.columns.str.split(".")])
        geps[dataset_name] = df
    geps = pd.concat(geps, axis=1).sort_index(axis=1)
    
    # get minimum k GEPs for each community/dataset combination.
    selected_geps_labels = []
    for community, nodes in communities.items():
        nodes = pd.DataFrame([n.split("|") for n in nodes], columns=["dataset", "k", "GEP"])
        for dataset_name in config.datasets:
            dataset_nodes = nodes[nodes["dataset"] == dataset_name].copy()
            if dataset_nodes.shape[0]:
                block = dataset_nodes[dataset_nodes["k"].astype(int) == dataset_nodes["k"].astype(int).min()].copy()
                block["Community"] = community
                selected_geps_labels.append(block)
    selected_geps_labels = pd.concat(selected_geps_labels).set_index("Community")
    selected_geps_labels.to_csv(os.path.join(sns_output_dir, "selected_geps_labels.txt"), sep="\t") # outputs the GEP identities

    selected_geps = []
    for community, gep in selected_geps_labels.iterrows():
        selected_geps.append(geps[gep['dataset'], int(gep['k']), int(gep['GEP'])].rename(f"{community}|{gep['dataset']}|{gep['k']}|{gep['GEP']}"))
    selected_geps = pd.concat(selected_geps, axis=1)
    selected_geps.to_csv(os.path.join(sns_output_dir, "selected_geps_scores.txt"), sep="\t") # outputs the GEPs themselves (ie., gene profiles in z-score units)
    selected_geps.columns = pd.MultiIndex.from_tuples((tuple(x) for x in selected_geps.columns.str.split("|")))

    # plot membership of datasets and ranks for each community
    fig = plot_community_by_dataset_rank(communities, config)
    fig.savefig(os.path.join(sns_output_dir, "communities_by_dataset_rank.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "communities_by_dataset_rank.png"), dpi=600)


    # Define community colors
    logging.info("Identifying distinct colors for each community")
    community_colors = {community: color for community, color in zip(communities, distinctipy.get_colors(n_colors=len(communities), pastel_factor=0.2))}
    with open(os.path.join(sns_output_dir, "community_colors.toml"), "wb") as f:
        tomli_w.dump({str(comm): col for comm, col in community_colors.items()}, f)

    # Graph layout
    layout = get_graph_layout(G, config)
    with open(os.path.join(sns_output_dir, "layout.toml"), "wb") as f:
        tomli_w.dump({"layout": layout}, f)

    ### Plot network layout ###
    fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
    ax.set_aspect(1)
    nx.draw(G, pos=layout, node_color="#444444", node_size=config.sns["node_size"], linewidths=0, width=0.2, edge_color=config.sns["edge_color"], with_labels=True, font_size=2)
    plt.tight_layout()
    fig.savefig(os.path.join(sns_output_dir, "gep_network.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "gep_network.png"), dpi=600)
    plt.close(fig)
    
    ### Plot network colored by dataset ###

    # create legend
    dataset_colors = {ds: ds_attr["color"] for ds, ds_attr in config.datasets.items()}
    dataset_legend = []
    for dataset, color in dataset_colors.items():
        dataset_legend.append(Line2D([0], [0], marker='o', color='w', label=dataset, markerfacecolor=color, markersize=8))

    colors = []
    for node in G:
        colors.append(dataset_colors[node.split("|")[0]])

    # Labels without dataset names
    labels = {}
    for node in G:
        labels[node] = node.partition("|")[2]

    # Node sizes inversely proportional to k
    sizes = {}
    min_rank = min([k for ds_attr in config.datasets.values() for k in ds_attr["selected_k"]])

    for node in G:
        sizes[node] = 200 / (int(node.split("|")[1]) + 0.5 - min_rank)
    node_sizes = [(sizes[n] if n in sizes else 0) for n in G]

    # Plot nodes colored by dataset
    fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
    nx.draw(G, pos=layout,
            with_labels=False, node_color=colors, labels=labels, node_size=30, linewidths=0, width=0.2, edge_color=config.sns["edge_color"], font_size=4)
    ax.legend(handles=dataset_legend)
    ax.set_title("Datasets")
    plt.tight_layout()
    # Save Figure
    fig.savefig(os.path.join(sns_output_dir, "gep_network_datasets.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "gep_network_datasets.png"), dpi=600)
    plt.close(fig)
    
    # Plot the network with labelled nodes and radius inversely proportional to k
    fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
    nx.draw(G, pos=layout,
            with_labels=True, node_color=colors, labels=labels, node_size=node_sizes, linewidths=0, width=0.2, edge_color=config.sns["edge_color"], font_size=4)
    ax.legend(handles=dataset_legend)
    ax.set_title("Datasets")
    plt.tight_layout()
    # Save Figure
    fig.savefig(os.path.join(sns_output_dir, "gep_network_rank.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "gep_network_rank.png"), dpi=600)
    plt.close(fig)
    
    ### Plot network colored by community ###
    colors = [community_colors[gep_communities[node]] for node in G]

    # Plot the network
    fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
    nx.draw(G, pos=layout,
            with_labels=False, node_color=colors, node_size=config.sns["node_size"], linewidths=0, width=0.2, edge_color=config.sns["edge_color"])

    # Add legend
    legend_elements = []
    for name, color in community_colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=10))
    ax.set_title("GEP Communities")
    ax.legend(handles=legend_elements)
    # Save Figure
    fig.savefig(os.path.join(sns_output_dir, "gep_network_communities.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "gep_network_communities.png"), dpi=600)
    plt.close(fig)
    
    # plot community resolution sweep results
    with mpl.backends.backend_pdf.PdfPages(os.path.join(sns_output_dir, "community_resolution_sweep.pdf")) as pdf:
        for resolution, res_communities in communities_resolution_sweep.items():
            ### Plot network colored by community ###
            community_colors = {community: color for community, color in zip(res_communities, distinctipy.get_colors(n_colors=len(res_communities), pastel_factor=0.2))}
            res_gep_communities = {gep: community for community, geps in res_communities.items() for gep in geps}
            colors = [community_colors[res_gep_communities[node]] for node in G]

            # Plot the network
            fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
            nx.draw(G, pos=layout,
                    with_labels=False, node_color=colors, node_size=config.sns["node_size"], linewidths=0, width=0.2, edge_color=config.sns["edge_color"])

            # Add legend
            legend_elements = []
            for name, color in community_colors.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=10))
            community_algorithm = config.sns["community_algorithm"]
            ax.set_title(f"GEP Communities\n{community_algorithm}, res={resolution}")
            ax.legend(handles=legend_elements)
            # Save Figure
            pdf.savefig(fig)
            plt.close(fig)

    ### Maximum Correlation between Datasets and Communities
    max_corr_communities = get_max_corr_communities(communities, output_dir, config)
    max_corr_communities = max_corr_communities.astype("float").dropna(how="all", axis=0).dropna(how="all", axis=1).reorder_levels([1,0], axis=0).reorder_levels([1,0], axis=1)
    fig, ax = plt.subplots(figsize=[16,16])
    sns.heatmap(max_corr_communities, xticklabels=True, yticklabels=True, cmap=config.colormaps["diverging"], center=0, vmin=-1, vmax=1, ax=ax)
    fig.suptitle("Maximum correlation between GEPs grouped by dataset and community")
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_communities.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_communities.png"), dpi=600)
    plt.close(fig)

    max_corr_communities = max_corr_communities.sort_index(axis=0).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=[16,16])
    sns.heatmap(max_corr_communities, xticklabels=True, yticklabels=True, cmap=config.colormaps["diverging"], center=0, vmin=-1, vmax=1, ax=ax)
    fig.suptitle("Maximum correlation between GEPs grouped by dataset and community")
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_datasets.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_datasets.png"), dpi=600)
    plt.close(fig)

    # get usage matrix
    usage = config.get_usage_matrix()
    sample_to_patient = config.get_sample_patient_mapping()

    # Number of samples, patients per GEP
    if any(["patient_id_column" in d for d in config.datasets.values()]):
        figs = plot_number_of_patients(usage, sample_to_patient, G, layout, config)
        for method, fig in figs.items():
            fig.savefig(os.path.join(sns_output_dir, f"{method}.pdf"))
            plt.close(fig)

    # integrated community usage matrix (samples X communities)
    
    logging.info("Plotting integrated community usage")
    ic_usage = []
    for dataset_name in config.datasets:
        data = []
        for community, nodes in communities.items():
            geps = []
            for node in nodes:
                gep = node.split("|")
                if gep[0] == dataset_name:
                    geps.append((gep[0], int(gep[1]), int(gep[2])))

            gep_comm = usage[geps]
            gep_comm = gep_comm / gep_comm.median()
            data.append(gep_comm.median(axis=1).rename((community, dataset_name)))
        data = pd.concat(data, axis=1).droplevel(axis=1, level=1).dropna(how="all")
        data.columns.rename("Community", inplace=True)
        ic_usage.append(data.sort_index(axis=0))
    ic_usage = pd.concat(ic_usage)
    ic_usage.to_csv(os.path.join(sns_output_dir, "integrated_community_usage.txt"), sep="\t")
    
    # plot integrated community usage as an annotated heatmap
    merged_metadata = []
    for dataset_name, dataset_params in config.datasets.items():
        metadata = read_h5ad(dataset_params["filename"], backed="r").obs.dropna(axis=1, how="all").copy()
        metadata["Dataset"] = dataset_name  # adds a column for displaying dataset as a metadata layer
        metadata = metadata[ ['Dataset'] + [col for col in metadata.columns if col != 'Dataset'] ]  # Move to front
        metadata.index = pd.MultiIndex.from_product([[dataset_name], metadata.index])  # adds dataset name to index to disambiguate samples with the same name but different datasets
        merged_metadata.append(metadata)
    merged_metadata = pd.concat(merged_metadata)
    for col in merged_metadata.select_dtypes(include="object").columns:  # required to fix 'object' dtypes created during concatenation
        merged_metadata[col] = merged_metadata[col].astype("category")
    merged_metadata.index.rename(["dataset", "sample"], inplace=True)
    metadata_colors = {col: config.get_metadata_colors(col) for col in merged_metadata.columns}
    metadata_colors["Dataset"] = {dsname: dsparam["color"] for dsname, dsparam in config.datasets.items()}
    plot_annotated_usages(
        df=ic_usage,
        metadata=merged_metadata,
        metadata_colors=metadata_colors,
        missing_data_color=config.metadata_colors["missing_data"],
        title="Community Usage",
        filename=os.path.join(sns_output_dir, "integrated_community_usage.pdf"),
        cluster_samples=True, cluster_geps=False, show_sample_labels=False,
        ylabel="Community")

    # GEP level, categorical data, overrepresentation plots
    
    logging.info("Creating GEP network plots for categorical metadata")
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(include="category").dropna(axis=1, how="all")  # only use categorical data
        if metadata.shape[1] == 0:
            continue
        
        # bar charts
        dataset_is_in_nodes = any([node.split("|")[0] == dataset_name for community in communities.values() for node in community])
        if not dataset_is_in_nodes:
            continue
        
        fig = plot_overrepresentation_geps_bar(usage, metadata, communities, dataset_name, config)
        os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_bar_by_community"), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_bar_by_community", dataset_name + ".pdf"))
        plt.close(fig)

        for annotation_layer, sample_to_class in metadata.items():
            colordict = config.get_metadata_colors(annotation_layer)
            if sample_to_class.isnull().any(): # add 
                sample_to_class = sample_to_class.cat.add_categories("").fillna("")
                colordict[""] = config.metadata_colors["missing_data"]
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            overrepresentation = get_category_overrepresentation(ds_usage, sample_to_class)
            os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation", dataset_name), exist_ok=True)
            overrepresentation.to_csv(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation", dataset_name, annotation_layer + ".txt"), sep='\t')
            overrepresentation.columns = pd.Index([f"{c[0]}|{c[1]}|{c[2]}" for c in overrepresentation.columns])
            fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
            plot_overrepresentation_network(
                graph=G,
                layout=layout,
                title=f"Dataset: {dataset_name}\nMetadata Layer: {annotation_layer}",
                overrepresentation=overrepresentation,
                colordict=colordict,
                pie_size=config.sns["pie_size_gep"],
                ax=ax
                )
                
            os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_network", dataset_name), exist_ok=True)
            fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_network", dataset_name, annotation_layer + ".pdf"))
            fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_network", dataset_name, annotation_layer + ".png"), dpi=600)
            plt.close(fig)

    # GEP level, numerical data, correlation plots
    logging.info("Creating GEP network plots for numerical metadata")
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(exclude="category").dropna(axis=1, how="all")  # exclude categorical data
        if metadata.shape[1] == 0:
            continue
        
        dataset_is_in_nodes = any([node.split("|")[0] == dataset_name for community in communities.values() for node in community])
        if not dataset_is_in_nodes:
            continue
        
        # bar charts
        fig = plot_metadata_correlation_geps_bar(usage, metadata, communities, dataset_name, config)
        os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "correlation_bar_by_community"), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "correlation_bar_by_community", dataset_name + ".pdf"))
        plt.close(fig)

        for annotation_layer, sample_to_numeric in metadata.items():
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            correlation = ds_usage.corrwith(sample_to_numeric, method="spearman")
            os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "correlation", dataset_name), exist_ok=True)
            correlation.to_csv(os.path.join(sns_output_dir, "annotated_geps", "correlation", dataset_name, annotation_layer + ".txt"), sep='\t')
            correlation.index = pd.Index([f"{c[0]}|{c[1]}|{c[2]}" for c in correlation.index])
            fig = plot_metadata_correlation_network(
                graph=G,
                layout=layout,
                title=f"Dataset: {dataset_name}\nMetadata Layer: {annotation_layer}",
                correlation=correlation,
                plot_size=config.sns["plot_size_gep"],
                node_size=config.sns["node_size"],
                config=config
            )
            os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "correlation_network", dataset_name), exist_ok=True)
            fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "correlation_network", dataset_name, annotation_layer + ".pdf"))
            fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "correlation_network", dataset_name, annotation_layer + ".png"), dpi=600)
            plt.close(fig)

    # Community-level Network
    logging.info("Creating community network")
    edge_list = []
    for c1, n1 in communities.items():
        for c2, n2 in communities.items():
            if c1 != c2:  # no self-loops
                n_edges = len(list(nx.edge_boundary(G, n1, n2)))
                edge_list.append((c1, c2, n_edges))

    edge_list = pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges"))
    Gcomm = nx.from_pandas_edgelist(pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges")), "comm1", "comm2", "n_edges")
    Gcomm.add_nodes_from(communities.keys())
    # Centroid method for community layout
    community_layout = {}
    for community_name, nodes in communities.items():
        points = np.array([layout[node] for node in nodes])
        centroid = (np.median(points[:, 0]), np.median(points[:, 1]))
        community_layout[community_name] = centroid

    ### Plot network layout ###
    fig, ax = plt.subplots(figsize=config.sns["plot_size_gep"])
    ax.set_aspect(1)
    if Gcomm.edges:
        width = np.array(list(nx.get_edge_attributes(Gcomm, "n_edges").values()))
        width = 20 * width / np.max(width)
    else:
        width = None
    sizes = np.array([len(communities[node]) for node in Gcomm.nodes])
    sizes = 20 * config.sns["node_size"] * sizes / np.max(sizes)
    node_colors = [community_colors[node] for node in Gcomm]
    nx.draw(Gcomm, pos=community_layout, node_color=node_colors, node_size=sizes, linewidths=0, width=width, edge_color=config.sns["edge_color"], with_labels=True, font_size=20)
    plt.tight_layout()
    fig.savefig(os.path.join(sns_output_dir, "community_network.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "community_network.png"), dpi=600)
    plt.close(fig)
            
    # Community-level overrepresentation plots
    logging.info("Creating community network plots for categorical metadata")
    plot_data = {}
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(include="category").dropna(axis=1, how="all")  # only use categorical data
        for row, (annotation_layer, sample_to_class) in enumerate(metadata.items()):
            # usage subset to dataset
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            overrepresentation = get_category_overrepresentation(ds_usage, sample_to_class)
            result_df = []
            for col, community in enumerate(sorted(list(communities))):
                geps = []
                for node in communities[community]:
                    dataset_str, k_str, gep_str = node.split("|")
                    if dataset_str == dataset_name:
                        geps.append((dataset_str, int(k_str), int(gep_str)))
                geps = sorted(geps)
                if geps:
                    com_es = overrepresentation[geps].mean(axis=1)
                    com_es[com_es < 0] = 0
                else:
                    com_es = pd.Series(0, index=overrepresentation.index)
                result_df.append(com_es.rename(community))
            result_df = pd.concat(result_df, axis=1)
            plot_data[(dataset_name, annotation_layer)] = result_df

    for (dataset_name, annotation_layer), community_es in plot_data.items():
        # bar plots
        fig, ax = plt.subplots()
        community_es.T.plot.bar(stacked=True, width=0.9, ax=ax, legend=None, rot=0, color=config.get_metadata_colors(annotation_layer))
        os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_bar", dataset_name), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_bar", dataset_name, annotation_layer + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_bar", dataset_name, annotation_layer + ".png"), dpi=600)
        plt.close(fig)

        # network plots
        fig, ax = plt.subplots(figsize=config.sns["plot_size_community"])
        plot_overrepresentation_network(
            graph=Gcomm,
            layout=community_layout,
            title=f"Dataset: {dataset_name}\nAnnotations: {annotation_layer}",
            overrepresentation=community_es,
            colordict=config.get_metadata_colors(annotation_layer),
            pie_size=np.array(config.sns["pie_size_community"]),
            edge_weights="n_edges",
            ax=ax
        )
        os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_network", dataset_name), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_network", dataset_name, annotation_layer + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_network", dataset_name, annotation_layer + ".png"), dpi=600)
        plt.close(fig)


    # Community-level correlation plots
    logging.info("Creating community network plots for numerical metadata")
    plot_data = {}
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(exclude="category").dropna(axis=1, how="all")  # only use categorical data
        for row, (annotation_layer, sample_to_numeric) in enumerate(metadata.items()):
            # usage subset to dataset
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            correlation = ds_usage.corrwith(sample_to_numeric)
            result = {}
            for col, community in enumerate(sorted(list(communities))):
                geps = []
                for node in communities[community]:
                    dataset_str, k_str, gep_str = node.split("|")
                    if dataset_str == dataset_name:
                        geps.append((dataset_str, int(k_str), int(gep_str)))
                geps = sorted(geps)
                if geps:
                    com_meancorr = correlation[geps].mean()
                else:
                    com_meancorr = 0
                result[community] = com_meancorr
            plot_data[(dataset_name, annotation_layer)] = pd.Series(result).sort_index()

    for (dataset_name, annotation_layer), community_corr in plot_data.items():
        # bar plots
        fig, ax = plt.subplots()
        community_corr.plot.bar(width=0.9, ax=ax, legend=None, rot=0, ylim=[-1, 1])
        ax.set_ylabel("Pearson correlation")
        os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "correlation_bar", dataset_name), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "correlation_bar", dataset_name, annotation_layer + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "correlation_bar", dataset_name, annotation_layer + ".png"), dpi=600)
        plt.close(fig)

        # network plots
        fig = plot_metadata_correlation_network(
            graph=Gcomm,
            layout=community_layout,
            title=f"Dataset: {dataset_name}\nMetadata Layer: {annotation_layer}",
            correlation=community_corr,
            plot_size=config.sns["plot_size_community"],
            node_size=np.array(config.sns["node_size"]),
            edge_weights="n_edges",
            config=config
        )
        os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "correlation_network", dataset_name), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "correlation_network", dataset_name, annotation_layer + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "correlation_network", dataset_name, annotation_layer + ".png"), dpi=600)
        plt.close(fig)
        
    for dataset_name, geps in selected_geps.droplevel(axis=1, level=[2,3]).groupby(axis=1, level=1):
        geps = geps.droplevel(axis=1, level=1)
        geps.columns = geps.columns.astype("int")
        for feature in config.features["features_of_interest"]:
            if feature in geps.index:
                positive_scores = geps.loc[feature].groupby(axis=0, level=0).mean().reindex(communities.keys()).fillna(0).clip(lower=0)
                if positive_scores.sum() == 0:
                    continue
                fig = plot_community_network(
                    graph = Gcomm,
                    layout = community_layout,
                    plot_size = config.sns["plot_size_community"],
                    title = f"Dataset: {dataset_name}\nFeature: {feature}",
                    node_sizes=positive_scores,
                    edge_weights="n_edges",
                    config=config,
                    community_colors=community_colors
                )
                os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "cNMF_score"), exist_ok=True)
                fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "cNMF_score", f"{feature}_{dataset_name}.pdf"))
                plt.close(fig)
        
    # per patient community level plots
    if any(["patient_id_column" in d for d in config.datasets.values()]):
        patient_to_samples = {patient: [] for sample, patient in sample_to_patient.items()}
        for sample, patient in sample_to_patient.items():
            patient_to_samples[patient].append(sample)
        patient_to_samples = pd.Series(patient_to_samples).explode()

        n_cols = 4
        for annotation_layer in merged_metadata.select_dtypes("category").columns:
            if annotation_layer == "Dataset":
                colordict = {dsname: dsparam["color"] for dsname, dsparam in config.datasets.items()}
            else:
                colordict = config.get_metadata_colors(annotation_layer)
                colordict[""] = config.metadata_colors["missing_data"]
            for min_samples_per_patient in [1,2]:
                n_plots = (patient_to_samples.groupby(axis=0, level=[0,1]).count() >= min_samples_per_patient).sum() + 1  # one plot per patient as well as an extra for the legend.
                n_rows = 1 + n_plots // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize = [config.sns["plot_size_community"][0]*n_cols, config.sns["plot_size_community"][1]*n_rows], squeeze=False, layout="constrained")
                for row in range(n_rows):
                    for col in range(n_cols):
                        axes[row,col].set_axis_off()

                # Add legend
                ax = axes[0, 0]  # get lower right axes
                legend_elements = []
                for name, color in colordict.items():
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=10))
                ax.legend(handles=legend_elements, loc='center')
                ax.set_title("Legend", fontdict={'fontsize': 26})

                plot_count = 1
                for (dataset, patient), patient_samples in patient_to_samples.groupby(axis=0, level=[0,1]):
                    if patient_samples.shape[0] >= min_samples_per_patient:
                        ax = axes[plot_count // n_cols, plot_count % n_cols]
                        bar_data = ic_usage.loc[patient_samples].fillna(0)
                        bar_data.index = merged_metadata.loc[bar_data.index][annotation_layer].cat.add_categories("").fillna("")
                        bar_data = bar_data.groupby(axis=0, level=0).mean().dropna(how="any")
                        plot_overrepresentation_network(Gcomm, community_layout, f"{dataset}\n{patient}", overrepresentation=bar_data, colordict=colordict, pie_size=config.sns["pie_size_community"], ax=ax, edge_weights=None, show_legends=False)
                        plot_count += 1
                os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "patient_network", annotation_layer), exist_ok=True)
                fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "patient_network", annotation_layer, f"{min_samples_per_patient}samplesperpatient.pdf"))
                fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "patient_network", annotation_layer, f"{min_samples_per_patient}samplesperpatient.png"), dpi=100)
                plt.close("all")
        
    # Plot pairwise correlation heatmaps of community usage across samples
    def plot_icusage_correlation(ic_usage_corr, title=None):
        mask = np.triu(np.ones_like(ic_usage_corr), 1)
        fig, ax = plt.subplots(figsize=[8,6])
        sns.heatmap(ic_usage_corr, center=0, vmin=-1, vmax=1, cmap=config.colormaps["diverging"], mask=mask, ax=ax)
        ax.set_title(title)
        return fig

    plots = {"All Datasets": plot_icusage_correlation(ic_usage.dropna(axis=1).corr("spearman"), title="All Datasets")}
    plots.update({
        dataset_name: plot_icusage_correlation(df.corr("spearman"), title=dataset_name)
        for dataset_name, df in ic_usage.dropna(axis=1).groupby(axis=0, level=0)
        })

    os.makedirs(os.path.join(sns_output_dir, "integrated_community_usage", "correlation_heatmaps_shared"), exist_ok=True)
    for plot_name, fig in plots.items():
        fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "correlation_heatmaps_shared", plot_name + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "correlation_heatmaps_shared", plot_name + ".png"), dpi=600)
        
    plots = {"All Datasets": plot_icusage_correlation(ic_usage.corr("spearman"), title="All Datasets")}
    plots.update({
        dataset_name: plot_icusage_correlation(df.corr("spearman"), title=dataset_name)
        for dataset_name, df in ic_usage.groupby(axis=0, level=0)
        })

    os.makedirs(os.path.join(sns_output_dir, "integrated_community_usage", "correlation_heatmaps_all"), exist_ok=True)
    for plot_name, fig in plots.items():
        fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "correlation_heatmaps_all", plot_name + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "correlation_heatmaps_all", plot_name + ".png"), dpi=600)
    
    
    # Diversity analysis (shannon entropy of community-level usage)

    diversity = ic_usage.apply(lambda x: entropy(x.dropna()), axis=1)
    diversity.to_csv(os.path.join(sns_output_dir, "integrated_community_usage", "diversity.txt"), sep="\t")
    os.makedirs(os.path.join(sns_output_dir, "integrated_community_usage", "diversity"), exist_ok=True)

    for col in merged_metadata.select_dtypes("float").columns:  # association of diversity with numerical metadata
        df = pd.DataFrame({"diversity": diversity, col: merged_metadata[col]})
        df["Dataset"] = df.index.get_level_values(0)
        fig, ax = plt.subplots(figsize=[4,4])
        sns.scatterplot(data=df, x=col, y="diversity", hue="Dataset", ax=ax)
        correlation = merged_metadata[col].corr(diversity, method="spearman")
        ax.text(s=f"Spearman ρ = {correlation:.3f}", x=0.01, y=0.01, va="bottom", transform=ax.transAxes)
        ax.set_title(col)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "diversity", f"{col}.pdf"))
        plt.close('all')

    sample_groups = merged_metadata["Dataset"]  # association of diversity with dataset
    fig = plot_icu_diversity(sample_groups, diversity, config, title=f"Datasets")
    fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "diversity", f"datasets.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "diversity", f"datasets.png"), bbox_inches="tight", dpi=400)
    plt.close('all')

    for dataset in config.datasets: # association of diversity with categorical data by dataset
        os.makedirs(os.path.join(sns_output_dir, "integrated_community_usage", "diversity", dataset), exist_ok=True)   
        for annotation_layer in merged_metadata.select_dtypes("category").columns:
            sample_groups = merged_metadata.loc[dataset, annotation_layer]
            if 0 < sample_groups.nunique() < 20:
                fig = plot_icu_diversity(sample_groups, diversity.loc[dataset], config, title=f"{dataset}\n{annotation_layer}")
                fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "diversity", dataset, f"{annotation_layer}.pdf"), bbox_inches="tight")
                fig.savefig(os.path.join(sns_output_dir, "integrated_community_usage", "diversity", dataset, f"{annotation_layer}.png"), bbox_inches="tight", dpi=400)
                plt.close('all')
    
    
cli.add_command(cmd_txt_to_h5ad)
cli.add_command(cmd_update_h5ad_metadata)
cli.add_command(cmd_check_h5ad)
cli.add_command(cmd_model_odg)
cli.add_command(cmd_set_parameters)
cli.add_command(cmd_factorize)
cli.add_command(cmd_postprocess)
cli.add_command(cmd_annotated_heatmap)
cli.add_command(cmd_integrate)
cli.add_command(cmd_create_network)
