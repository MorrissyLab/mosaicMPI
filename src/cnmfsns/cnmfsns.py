import os
import logging
import shutil
import subprocess
import collections
import click
import cnmf
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from functools import partial
from multiprocessing.pool import Pool
from datetime import datetime
from typing import Optional, Mapping
from anndata import AnnData, read_h5ad
from cnmfsns.containers import add_cnmf_results_to_h5ad
from cnmfsns.config import Config
from cnmfsns.odg import model_overdispersion, odg_plots, fetch_hgnc_protein_coding_genes
from cnmfsns.plots import (
    plot_annotated_geps_by_community,
    plot_annotated_usages,
    plot_rank_reduction,
    plot_pairwise_corr,
    plot_pairwise_corr_overlaid,
    plot_genelist_upsets,
    plot_community_by_dataset_rank,
    plot_overrepresentation_network)

from cnmfsns.sns import (
    add_community_weights_to_graph, 
    save_df_to_npz, 
    load_df_from_npz, 
    create_graph, 
    community_search, 
    get_graph_layout, 
    get_max_corr_communities)
from cnmfsns import __version__


import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import tomli_w
import distinctipy
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


def get_and_check_consensus(k, cnmf_obj, local_density_threshold, local_neighborhood_size):
    logging.info(f"Creating consensus GEPs and usages for k={k}")
    cnmf_obj.consensus(k, density_threshold=local_density_threshold,
        local_neighborhood_size=local_neighborhood_size,
        show_clustering=True,
        close_clustergram_fig=True)
    density_threshold_repl = str(local_density_threshold).replace(".", "_")
    filenames = [
        cnmf_obj.paths['consensus_spectra']%(k, density_threshold_repl),
        cnmf_obj.paths['consensus_spectra']%(k, density_threshold_repl),
        cnmf_obj.paths['consensus_usages']%(k, density_threshold_repl),
        cnmf_obj.paths['consensus_stats']%(k, density_threshold_repl),
        cnmf_obj.paths['consensus_spectra__txt']%(k, density_threshold_repl),
        cnmf_obj.paths['consensus_usages__txt']%(k, density_threshold_repl),
        cnmf_obj.paths['gene_spectra_tpm']%(k, density_threshold_repl),
        cnmf_obj.paths['gene_spectra_tpm__txt']%(k, density_threshold_repl),
        cnmf_obj.paths['gene_spectra_score']%(k, density_threshold_repl),
        cnmf_obj.paths['gene_spectra_score__txt']%(k, density_threshold_repl)
        ]
    for filename in filenames:
        if not os.path.exists(filename):
            logging.error(f"cNMF postprocessing could not find output file {filename}. This can arise in low memory conditions.")
            sys.exit(1)

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
@click.version_option(version=__version__)
def cli():
    """
    cNMF-SNS is a tool for deconvolution and integration of multiple datasets based on consensus Non-Negative Matrix Factorization (cNMF).
    """

@click.command()
@click.option(
    "-c", "--counts", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Input (cell/sample x gene) counts matrix as tab-delimited text file. "
         "This is the matrix which will be variance normalized and used for factorization. "
         "If not provided, the normalized matrix will be used instead.")
@click.option(
    "-n", "--normalized", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Pre-computed (cell/sample x gene) normalized matrix as tab delimited text file. "
         "This is the matrix which is used to select overdispersed genes and to which final GEPs will be scaled. "
         "If not provided, TPM normalization will be calculated from the count matrix.")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Optional tab-separated text file with metadata for samples/cells/spots with one row each. Columns are annotation layers.")
@click.option(
    "--sparsify", is_flag=True,
    help="Save resulting data in sparse format. Recommended for sparse datasets such as scRNA-Seq, scATAC-Seq, and 10X Visium, but not for bulk expression data.")
@click.option(
    "-o", '--output', type=click.Path(dir_okay=False, exists=False), required=True,
    help="Path to output .h5ad file.")
def txt_to_h5ad(counts, normalized, metadata, output, sparsify):
    """
    Create .h5ad file with normalized and raw expression data, as well as metadata.
    """
    start_logging()
    if counts is None and normalized is None:
        logging.error("Either a counts matrix or normalized matrix of gene expression must be supplied.")
        sys.exit(1)
    elif counts is not None and normalized is None:
        counts = pd.read_table(counts, index_col=0)
        normalized = counts.div(counts.sum(axis=1), axis=0) * 1e6 # compute TPM
    elif normalized and counts is None:
        normalized = pd.read_table(normalized, index_col=0)
        counts = normalized
    elif normalized and counts:
        counts = pd.read_table(counts, index_col=0)
        normalized = pd.read_table(normalized, index_col=0)
    if (counts.index != normalized.index).all() or (counts.columns != normalized.columns).all():
        logging.error("Index and Columns of counts and normalized matrices are not the same")
        sys.exit(1)
    if metadata is not None:
        metadata = pd.read_table(metadata, index_col=0)
        # convert 'object' dtype to categorical, converting bool values to strings as these are not supported by AnnData on-disk format
        for col in metadata.select_dtypes(include="object").columns:
            metadata[col] = metadata[col].replace({True: "True", False: "False"}).astype("category")
        
        # print final summary for review before saving to *.h5ad file]
        msg = ""
        for col in metadata.columns:
            msg += "Column: " + col + "\n"
            for value_type, count in metadata[col].dropna().map(type).value_counts().items():
                msg += f"   {value_type}: {count}\n"
        
        logging.info("Data types for non-missing values in each layer of metadata:\n" + msg)
        missing_samples_in_X = metadata.index.difference(normalized.index).astype(str).to_list()
        if missing_samples_in_X:
            logging.warning("The following samples in the metadata were not present in the data (`adata.X`):\n  - " + "\n  - ".join(missing_samples_in_X))
        missing_samples_in_md = normalized.index.difference(metadata.index).astype(str).to_list()
        if missing_samples_in_md:
            logging.warning("The following samples in the data (`adata.X`) were absent in the metadata:\n  - " + "\n  - ".join(missing_samples_in_md))
        
        metadata = metadata.reindex(normalized.index)

    if sparsify:
        adata = AnnData(X=sp.csr_matrix(normalized.values), raw=AnnData(X=sp.csr_matrix(counts)), obs=metadata)
    else:
        adata = AnnData(X=normalized, dtype=normalized.values.dtype, raw=AnnData(X=counts, dtype=counts.values.dtype), obs=metadata)
    adata.write_h5ad(output)

@click.command()
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Tab-separated text file with metadata for samples/cells/spots with one row each. Columns are annotation layers.")
@click.option(
    "-i", '--input_h5ad', type=click.Path(dir_okay=False, exists=True), required=True,
    help="Path to input .h5ad file.")
def update_h5ad_metadata(input_h5ad, metadata):
    """
    Update metadata in a .h5ad file at any point in the cNMF-SNS workflow. New metadata will overwrite (`adata.obs`).
    """
    start_logging()

    metadata = pd.read_table(metadata, index_col=0)
    # convert 'object' dtype to categorical, converting bool values to strings as these are not supported by AnnData on-disk format
    for col in metadata.select_dtypes(include="object").columns:
        metadata[col] = metadata[col].replace({True: "True", False: "False"}).astype("category")
    
    # print final summary for review before saving to *.h5ad file]
    logging.info("Data types for non-missing values in each layer of metadata: ")
    for col in metadata.columns:
        print("Column:", col)
        for value_type, count in metadata[col].dropna().map(type).value_counts().items():
            print(f"   {value_type}:", count)

    adata = read_h5ad(input_h5ad)
    adata.obs = metadata.reindex(index=adata.obs.index)
    adata.write(input_h5ad)


@click.command()
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input .h5ad file.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, exists=False), required=False,
    help="Output .h5ad file. If not specified, no output file will be written.")
def check_h5ad(input, output):
    start_logging()
    adata = read_h5ad(input)
    if adata.raw is None:
        logging.error(f".h5ad file is missing count data (`adata.raw.X`).")
        sys.exit(1)
    if adata.X is None:
        logging.error(f".h5ad file is missing normalized data (`adata.X`).")
        sys.exit(1)

    # convert sparse to dense matrices
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    if sp.issparse(adata.raw.X):
        raw = adata.raw.X.toarray()
    else:
        raw = adata.raw.X
    normalized = pd.DataFrame(data=X, index=adata.obs.index, columns=adata.var.index)
    counts = pd.DataFrame(data=raw, index=adata.obs.index, columns=adata.var.index)
    if adata.X is None and adata.raw.X is not None:
        logging.warning("Normalized data matrix (`adata.X`) is empty.")
        if output:
            logging.warning("Normalized data matrix being generated from count matrix (`adata.raw.X`) using TPM normalization.")
            normalized = counts * 1e6 / counts.sum(axis=1) # compute TPM
    elif adata.raw.X is None and adata.X is not None:
        logging.warning("Count data matrix (`adata.raw.X`) is empty.")
        if output:
            logging.warning("Normalized data matrix (`adata.X`) will be used instead.")
            counts = normalized.copy()
    elif adata.raw.X is None and adata.X is None:
        logging.error(".h5ad file must contain a counts matrix (`adata.raw.X`) and/or a normalized matrix (`adata.X`).")
        sys.exit(1)
    
    # Check counts for variables with missing values
    genes_with_missingvalues = counts.isnull().any().sum()
    if genes_with_missingvalues:
        logging.warning(f"{genes_with_missingvalues} of {adata.n_vars} variables are missing values in counts data (`adata.raw.X`).")
        if output:
            logging.warning(f"Subsetting variables to those with no missing values.")
            counts = counts.dropna(how="any", axis=1)
    # Check normalized for variables with missing values
    genes_with_missingvalues = normalized.isnull().any().sum()
    if genes_with_missingvalues:
        logging.warning(f"{genes_with_missingvalues} of {adata.n_vars} variables are missing values in normalized data (`adata.X`).")
        if output:
            logging.warning(f"Subsetting variables to those with no missing values.")
            normalized = normalized.dropna(how="any", axis=1)
    # Check for genes with zero variance
    zerovargenes = (counts.var() == 0).sum()
    if zerovargenes:
        logging.warning(f"{zerovargenes} of {adata.n_vars} variables have a variance of zero in counts data (`adata.raw.X`).")
        if output:
            logging.warning(f"Subsetting variables to those with nonzero variance.")
            counts = counts.loc[:, counts.var() > 0]
    # Check for genes with zero variance
    zerovargenes = (normalized.var() == 0).sum()
    if zerovargenes:
        logging.warning(f"{zerovargenes} of {adata.n_vars} variables have a variance of zero in normalized data (`adata.X`).")
        if output:
            logging.warning(f"Subsetting variables to those with nonzero variance.")
            normalized = normalized.loc[:, normalized.var() > 0]
    
    # check for linear scaling of counts to normalized matrix (eg. TPM) for cNMF
    is_nonlinear_scaling = (np.abs(normalized.corrwith(counts, axis=1, method="pearson") - 1) > 0.01).any()  # Uses pearson correlation to detect non-linear relationships
    if is_nonlinear_scaling:
        logging.warning(f"Normalized data (`adata.X`) does not appear to be a linear scaling of counts (`adata.raw.X`) data. Linear scaling such as TPM is recommended for cNMF.")
    
    # Save output to new h5ad file
    if output is not None:
        adata = adata[:, normalized.columns]
        adata.X = normalized
        adata.raw = AnnData(counts, dtype=counts.values.dtype)
        adata.write(output)


@click.command()
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
    "--cnmf_mean_threshold", type=float, default=0.5, show_default=True,
    help="Minimum mean for overdispersed genes (cnmf method).")
@click.option(
    "--annotate_hgnc_protein_coding", is_flag=True,
    help="Annotate whether features have a protein-coding locus type from HGNC, assuming that features are HGNC symbols"
)
def model_odg(name, output_dir, input, default_spline_degree, default_dof, cnmf_mean_threshold, annotate_hgnc_protein_coding):
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
    cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)  # creates directories for cNMF
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    adata = read_h5ad(input)
    
    # Create gene stats table
    df = model_overdispersion(
            adata=adata,
            odg_default_spline_degree=default_spline_degree,
            odg_default_dof=default_dof,
            odg_cnmf_mean_threshold=cnmf_mean_threshold
            )
    if annotate_hgnc_protein_coding:
        protein_coding_genes = fetch_hgnc_protein_coding_genes()
        df["HGNC protein-coding"] = df.index.isin(protein_coding_genes)
    os.makedirs(os.path.normpath(os.path.join(output_dir, name, "odgenes")), exist_ok=True)
    df.to_csv(os.path.join(output_dir, name, "odgenes", "genestats.tsv"), sep="\t")

    # create od genes plots
    for fig_id, fig in odg_plots(df, show_selected=False).items():
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".pdf"), facecolor='white')
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".png"), dpi=400, facecolor='white')

    # update/copy h5ad
    uns_odg = {
        "default_spline_degree": default_spline_degree,
        "default_dof": default_dof,
        "cnmf_mean_threshold": cnmf_mean_threshold,
        "gene_stats": df
    }
    adata.uns["odg"] = uns_odg
    adata.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))


@click.command()
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

def set_parameters(name, output_dir, odg_method, odg_param, k_range, k, n_iter, seed, beta_loss):
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
    cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    adata = read_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    df = adata.uns["odg"]["gene_stats"]

    # Convert parameter to expected type
    if odg_method.endswith("topn"):
        odg_param = int(odg_param)
    elif odg_method.endswith("minscore"):
        odg_param = float(odg_param)
    elif odg_method.endswith("quantile"):
        odg_param = float(odg_param)
    elif odg_method == "genes_file":
        odg_param = click.Path(exists=True, dir_okay=False)(odg_param)
    else:
        raise RuntimeError

    if odg_method == "default_topn":
        # top N genes ranked by od-score
        genes = df["odscore"].sort_values(ascending=False).head(odg_param).index
    elif odg_method == "default_minscore":
        # filters genes by od-score
        genes = df[(df["odscore"] >= odg_param)]["odscore"].sort_values(ascending=False).index
    elif odg_method == "default_quantile":
        # takes the specified quantile of genes after removing NaNs
        genes = df["odscore"].sort_values(ascending=False).head(int(odg_param * df["odscore"].notnull().sum())).index
    elif odg_method == "cnmf_topn":
        # top N genes ranked by v-score
        genes = df["vscore"].sort_values(ascending=False).head(odg_param).index
    elif odg_method == "cnmf_minscore":
        # filters genes by v-score
        genes = df[(df["vscore"] >= odg_param)]["vscore"].sort_values(ascending=False).index
    elif odg_method == "cnmf_quantile":
        # takes the specified quantile of genes after removing NaNs
        genes = df["vscore"].sort_values(ascending=False).head(int(odg_param * df["vscore"].notnull().sum())).index
    elif odg_method == "genes_file":
        genes = open(odg_param).read().rstrip().split(os.linesep)

    logging.info(f"{len(genes)} genes selected for factorization")
    df["selected"] = df.index.isin(genes)

    # update plots with threshold information
    for fig_id, fig in odg_plots(df, show_selected=True).items():
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".pdf"), facecolor='white')
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".png"), dpi=400, facecolor='white')

    # output table with gene overdispersion measures
    df.to_csv(os.path.join(output_dir, name, "odgenes", "genestats.tsv"), sep="\t")

    # write TPM (normalized) data
    input_counts = AnnData(X=adata.raw.X, obs=adata.obs, var=adata.var, dtype=np.float64)
    tpm = AnnData(X=adata.X, obs=adata.obs, var=adata.var, dtype=np.float64)
    tpm.write_h5ad(cnmf_obj.paths["tpm"])

    gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
    gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)
    input_tpm_stats = pd.DataFrame([gene_tpm_mean, gene_tpm_stddev], index = ['__mean', '__std']).T
    cnmf.cnmf.save_df_to_npz(input_tpm_stats, cnmf_obj.paths['tpm_stats'])
    norm_counts = cnmf_obj.get_norm_counts(input_counts, tpm, high_variance_genes_filter=genes)
    if norm_counts.X.dtype != np.float64:
        norm_counts.X = norm_counts.X.astype(np.float64)
    cnmf_obj.save_norm_counts(norm_counts)

    kvals = set(k)
    if k_range is not None:
        kvals |= set(range(k_range[0], k_range[1] + 1, k_range[2]))
    kvals = sorted(list(kvals))
    # save parameters for factorization step
    cnmf_obj.save_nmf_iter_params(*cnmf_obj.get_nmf_iter_params(ks=kvals, n_iter=n_iter, random_state_seed=seed, beta_loss=beta_loss))

    # save parameters in AnnData object
    adata.uns["odg"]["genestats"] = df
    adata.uns["odg"]["method"] = odg_method
    adata.uns["odg"]["param"] = odg_param
    adata.uns["cnmf"] = cnmf_obj.get_nmf_iter_params(ks=kvals, n_iter=n_iter, random_state_seed=seed, beta_loss=beta_loss)[1]  # dict of cnmf parameters
    adata.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))

@click.command()
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

def factorize(name, output_dir, worker_index, total_workers, slurm_script):
    """
    Performs factorization according to parameters specified using `cnmfsns set-parameters`.
    """
    cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    if slurm_script is None:
        cnmf_obj.factorize(worker_i=worker_index, total_workers=total_workers)
    else:
        subprocess.Popen(['sbatch', slurm_script, os.getcwd(), output_dir, name])
    
@click.command()
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
def postprocess(name, output_dir, cpus, local_density_threshold, local_neighborhood_size, skip_missing_iterations, force_h5ad_update):
    """
    Perform post-processing routines on cNMF after factorization. This includes checking factorization outputs for completeness, combining individual
    iterations, calculating consensus GEPs and usage matrices, and creating the k-selection and annotated usage plots.
    """
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)
    run_params = cnmf.cnmf.load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    # first check for combined outputs:
    missing_combined = []
    for k in sorted(set(run_params.n_components)):
        merged_result = cnmf_obj.paths['merged_spectra'] % k
        if not os.path.exists(merged_result) or os.path.getsize(merged_result) == 0:
            missing_combined.append(merged_result)
    if missing_combined:
        failed = []
        # Check if all output files and iterations exist
        for _, row in run_params.iterrows():
            iter_result = cnmf_obj.paths['iter_spectra'] % (row['n_components'], row['iter'])
            if not os.path.exists(iter_result) or os.path.getsize(iter_result) == 0:
                failed.append(iter_result)
        if failed:
            logging.error(
                f"{(len(failed))} files from the factorization step are missing or empty:\n  - " + 
                "\n  - ".join(failed)
            )
        if failed and not skip_missing_iterations:
            logging.error(
                f"Postprocessing could not proceed. To skip missing iterations, use --skip_missing_iterations."
            )
            sys.exit(1)
        elif failed and skip_missing_iterations:
            logging.warning("Missing files will be skipped")
        else:
            logging.info(f"Factorization outputs (individual iterations) were found for all values of k. No missing files were detected.")

        # combine individual iterations
        for k in sorted(set(run_params.n_components)):
            logging.info(f"Merging iterations for k={k}")
            cnmf_obj.combine_nmf(k, skip_missing_files=skip_missing_iterations)
    else:
        logging.info(f"Factorization outputs (merged iterations) were found for all values of k.")
    # calculate consensus GEPs and usages
    logging.info(f"Creating consensus GEPs and usages using {cpus} CPUs")
    call_consensus = partial(
        get_and_check_consensus,
        cnmf_obj=cnmf_obj,
        local_density_threshold=local_density_threshold,
        local_neighborhood_size=local_neighborhood_size)
    
    if cpus > 1:
        Pool(processes=cpus).map(call_consensus, sorted(set(run_params.n_components)))
    elif cpus == 1:
        for k in sorted(set(run_params.n_components)):
            call_consensus(k)
    else:
        logging.error(f"{cpus} is an invalid number of cpus. Please specify a positive integer.")

    # create k-selection plot
    cnmf_obj.k_selection_plot(close_fig=True)
   
    # update h5ad file with cnmf results
    logging.info(f"Updating h5ad file with cNMF results")
    input_h5ad = os.path.join(output_dir, name, name + ".h5ad")
    add_cnmf_results_to_h5ad(output_dir, name, input_h5ad, local_density_threshold, local_neighborhood_size, force=force_h5ad_update)

@click.command()
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

def annotated_heatmap(input_h5ad, output_dir, metadata_colors_toml, max_categories_per_layer, hide_sample_labels):
    """
    Create heatmaps of usages with annotation tracks.
    """
    start_logging()
    os.makedirs(output_dir, exist_ok=True)
    adata = read_h5ad(input_h5ad, backed="r")
    # annotate usage plots
    if metadata_colors_toml:
        cfg = Config.from_toml(metadata_colors_toml)
    else:
        cfg = Config()
    cfg.add_missing_metadata_colors(metadata_df=adata.obs)
    cfg.to_toml(os.path.join(output_dir, "metadata_colors.toml"))
    exclude_maxcat = adata.obs.select_dtypes(include=["object", "category"]).apply(lambda x: len(x.cat.categories)) > max_categories_per_layer
    if adata.obs.shape[1] > 0:
        metadata = adata.obs.drop(columns=exclude_maxcat[exclude_maxcat].index)
    else:
        metadata = adata.obs
    if "cnmf_usage" not in adata.obsm:
        logging.error("cNMF results have not been merged into .h5ad file. Ensure that you have run `cnmfsns postprocess` before creating annotated usage heatmaps.")
        sys.exit(1)
    usage = adata.obsm["cnmf_usage"]
    usage.columns=pd.MultiIndex.from_tuples(usage.columns.str.split(".").to_list())

    # create annotated plots
    for k in usage.columns.levels[0].astype(int).sort_values():
        logging.info(f"Creating annotated usage heatmap for k={k}")
        k_usage = usage.loc[:, str(k)]
        cnmf_name = adata.uns["cnmf_name"]
        title = f"{cnmf_name} k={k}"
        filename = os.path.join(output_dir, f"{cnmf_name}.usages.k{k:03}.pdf")
        plot_annotated_usages(
            df=k_usage, metadata=metadata, metadata_colors=cfg.metadata_colors, title=title, filename=filename,
            cluster_samples=True, cluster_geps=False, show_sample_labels=(not hide_sample_labels))
            

@click.command()
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="Output directory for cNMF-SNS results")
@click.option('-c', '--config_toml', type=click.Path(exists=True, dir_okay=False), help="TOML config file")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), multiple=True, help="h5ad file with cNMF results. Can be used to specify multiple files instead of a TOML config file.")
@click.option('--cpus', type=int, default=len(os.sched_getaffinity(0)), show_default=True, help="Number of CPUs to use for calculating correlation matrix")
def integrate(output_dir, config_toml, cpus, input_h5ad):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5ad files directly, it is recommended to use a single TOML file instead for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    start_logging(os.path.join(output_dir, "logfile.txt"))
    if os.listdir(output_dir):
        logging.warning(f"Integration directory {output_dir} is not empty. Files may be overwritten.")

    os.makedirs(os.path.join(output_dir, "input", "datasets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "integrate"), exist_ok=True)

    if config_toml is not None and input_h5ad:
        logging.error("A TOML config file can be specified, or 1 or more h5mu files can be specified, but not both.")
        sys.exit(1)
    if not all(fn.endswith(".h5ad") for fn in input_h5ad):
        logging.error("Input files must be AnnData .h5ad files.")
        sys.exit(1)

    # create config
    if config_toml is not None:
        config = Config.from_toml(config_toml)
    elif input_h5ad:
        config = Config.from_h5ad_files(input_h5ad)
        print(config)
    logging.info("Copying data to output directory...")
    # # copy files to output directory
    # for name, d in config.datasets.items():
    #     shutil.copy(d["filename"], os.path.join(output_dir, "input", "datasets", name + ".h5ad"))

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

    corr_path = os.path.join(output_dir, "integrate", config.integration["corr_method"] + ".df.npz")
    try:
        corr = load_df_from_npz(corr_path)
        logging.info(f"Loaded previously calculated correlation matrix from {corr_path}")
    except FileNotFoundError:
        logging.info(f"Calculating correlation matrix")
        if config.integration["corr_method"] == "pearson":
            try:
                from nancorrmp.nancorrmp import NaNCorrMp
            except ImportError:
                logging.info(f"nancorrmp not installed. Calculating Pearson correlation matrix using 1 CPU.")
                corr = geps.corr(config.integration["corr_method"])
            else:
                cpu_string = "all" if cpus == -1 else str(cpus)
                logging.info(f"nancorrmp found. Calculating Pearson correlation matrix using {cpu_string} CPUs.")
                corr = NaNCorrMp.calculate(geps, n_jobs=cpus)
        else:
            logging.info(f"Calculating Spearman correlation matrix using 1 CPU.")
            corr = geps.corr(config.integration["corr_method"])
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
            if median_corr <= config.integration["max_median_corr"]:
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
    output_toml = os.path.join(output_dir, "config.toml")
    config.to_toml(output_toml)
    logging.info(f"Output updated TOML file to: {output_toml}")

    # Rank Reduction Plots
    for dataset_name in config.datasets:
        fig = plot_rank_reduction(k_table.loc[:, dataset_name], config.integration["max_median_corr"])
        fig.savefig(os.path.join(output_dir, "integrate", f"{dataset_name}.rank_reduction.pdf"))
        fig.savefig(os.path.join(output_dir, "integrate", f"{dataset_name}.rank_reduction.png"))

    # Pairwise correlation thresholds from unfiltered correlation matrix 
    pairwise_thresholds = []
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            distr = tril.loc[dataset_row, dataset_col].values.flatten()

            #TODO Currently calculates it on all correlations, not max-k filtered

            if not all(np.isnan(distr)):
                pairwise_thresholds.append({
                    "dataset_row": dataset_row,
                    "dataset_col": dataset_col,
                    "threshold": -np.quantile(distr[distr < 0], q=1-config.integration["negative_corr_quantile"])
                })

    pairwise_thresholds = pd.DataFrame.from_records(pairwise_thresholds).set_index(["dataset_row", "dataset_col"])
    pairwise_thresholds.to_csv(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr_thresholds.txt"), sep="\t")


    # Filter correlations using dataset-specific max_k thresholds
    maxk_filtered_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if k_table.loc[gep[1], (gep[0], "max_k_filter_pass")]])
    selected_k_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if k_table.loc[gep[1], (gep[0], "selected_k")]])
    maxk_filtered_tril = tril.loc[maxk_filtered_index, maxk_filtered_index]
    selected_k_tril = tril.loc[selected_k_index, selected_k_index]

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

        

@click.command()
@click.option(
    '-o', '--output_dir', type=click.Path(file_okay=False, exists=True), required=True,
    help="Output directory for cNMF-SNS results generated using `cnmfsns integrate`")
@click.option(
    '-n', '--name', type=str, default=datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S"),
    help="Name for specific integration. Output from this step will be in [output_dir]/sns_networks/[name]/...")
@click.option(
    '-c', '--config_toml', type=click.Path(exists=True, dir_okay=False), 
    help="TOML config file. Defaults to file output from `cnmfsns integrate` step: [output_dir]/config.toml")
def create_network(output_dir, name, config_toml):
    start_logging(os.path.join(output_dir, "logfile.txt"))

    if config_toml is None:
        config = Config.from_toml(os.path.join(output_dir, "config.toml"))
    else:
        config = Config.from_toml(config_toml)

    fig = config.plot_metadata_colors_legend()
    fig.savefig(os.path.join(output_dir, "annotation_legend.pdf"))
    plt.close(fig)

    sns_output_dir = os.path.join(output_dir, "sns_networks", name)
    os.makedirs(sns_output_dir, exist_ok=True)

    G = create_graph(output_dir, config)
    communities = community_search(G, config)
    gep_communities = {gep: community for community, geps in communities.items() for gep in geps}
    with open(os.path.join(sns_output_dir, "gep_communities.toml"), "wb") as f:
        tomli_w.dump(gep_communities, f)
    add_community_weights_to_graph(G, gep_communities, config)

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


    # TODO: simplify below code
    ### Plot network layout ###
    fig, ax = plt.subplots(figsize=config.sns["plot_size"])
    ax.set_aspect(1)
    ax.set_title("Network Layout")
    nx.draw(G, pos=layout, node_color="#444444", node_size=config.sns["node_size"], linewidths=0, width=0.2, edge_color=config.sns["edge_color"], font_size=2)
    plt.tight_layout()
    fig.savefig(os.path.join(sns_output_dir, "layout.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "layout.png"), dpi=600)

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
    fig, ax = plt.subplots(figsize=config.sns["plot_size"])
    nx.draw(G, pos=layout,
            with_labels=False, node_color=colors, labels=labels, node_size=30, linewidths=0, width=0.2, edge_color=config.sns["edge_color"], font_size=4)
    ax.legend(handles=dataset_legend)
    ax.set_title("Datasets")
    plt.tight_layout()
    # Save Figure
    fig.savefig(os.path.join(sns_output_dir, "datasets.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "datasets.png"), dpi=600)

    # Plot the network with labelled nodes and radius inversely proportional to k
    fig, ax = plt.subplots(figsize=config.sns["plot_size"])
    nx.draw(G, pos=layout,
            with_labels=True, node_color=colors, labels=labels, node_size=node_sizes, linewidths=0, width=0.2, edge_color=config.sns["edge_color"], font_size=4)
    ax.legend(handles=dataset_legend)
    ax.set_title("Datasets")
    plt.tight_layout()
    # Save Figure
    fig.savefig(os.path.join(sns_output_dir, "gep_labels.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "gep_labels.png"), dpi=600)

    ### Plot network colored by community ###
    colors = [community_colors[gep_communities[node]] for node in G]

    # Plot the network
    fig, ax = plt.subplots(figsize=config.sns["plot_size"])
    nx.draw(G, pos=layout,
            with_labels=False, node_color=colors, node_size=config.sns["node_size"], linewidths=0, width=0.2, edge_color=config.sns["edge_color"])

    # Add legend
    legend_elements = []
    for name, color in community_colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=10))
    ax.set_title("Network Communities")
    ax.legend(handles=legend_elements)
    # Save Figure
    fig.savefig(os.path.join(sns_output_dir, "communities.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "communities.png"), dpi=600)

    ### Maximum Correlation between Datasets and Communities ### TODO: test whether max or mean is more informative
    max_corr_communities = get_max_corr_communities(communities, output_dir, config)
    max_corr_communities = max_corr_communities.astype("float").dropna(how="all", axis=0).dropna(how="all", axis=1).reorder_levels([1,0], axis=0).reorder_levels([1,0], axis=1)
    fig, ax = plt.subplots(figsize=[16,16])
    sns.heatmap(max_corr_communities, xticklabels=True, yticklabels=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    fig.suptitle("Maximum correlation between GEPs grouped by dataset and community")
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_communities.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_communities.png"), dpi=600)

    max_corr_communities = max_corr_communities.sort_index(axis=0).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=[16,16])
    sns.heatmap(max_corr_communities, xticklabels=True, yticklabels=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    fig.suptitle("Maximum correlation between GEPs grouped by dataset and community")
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_datasets.pdf"))
    fig.savefig(os.path.join(sns_output_dir, "community_maxcorr_datasets.png"), dpi=600)

    ### Number of samples, patients ###
    usage = []
    sample_to_patient = {}
    for dataset_name, dataset in config.datasets.items():
        adata = read_h5ad(dataset["filename"])
        if "patient_id_column" in dataset:
            for sample, patient in adata.obs[dataset["patient_id_column"]].items():
                sample_to_patient[(dataset_name, sample)] = (dataset_name, patient)
        df = adata.obsm["cnmf_usage"]
        df.index = pd.MultiIndex.from_product(([dataset_name], (df.index)))
        df.columns = pd.MultiIndex.from_tuples([(dataset_name, int(col[0]), int(col[1])) for col in df.columns.str.split(".")])
        usage.append(df)
    usage = pd.concat(usage, axis=1).sort_index(axis=0).sort_index(axis=1)
    usage.index.rename(["dataset", "sample"], inplace=True)
    usage.columns.rename(["dataset", "k", "gep"], inplace=True)
    usage

    # normalized usage (usages sum to 1 for each value of k)
    normalized_usage = []
    for k, subdf in usage.groupby(axis=1, level=[0,1]):
        normalized_usage.append(subdf.div(subdf.sum(axis=1), axis=0))
    normalized_usage = pd.concat(normalized_usage, axis=1)
    normalized_usage

    # discrete usage (Samples are assigned to GEPs with the highest usage)
    discrete_usage = []
    for k, subdf in usage.groupby(axis=1, level=[0,1]):
        discrete_usage.append(subdf.eq(subdf.max(axis=1), axis=0).astype(float))
    discrete_usage = pd.concat(discrete_usage, axis=1)
    discrete_usage[usage.isnull()] = np.NaN

    patients_to_geps = discrete_usage.loc[sample_to_patient.keys()].copy(deep=True)
    patients_to_geps.index = patients_to_geps.index.map(sample_to_patient)
    patients_to_geps = patients_to_geps.groupby(axis=0, level=[0,1]).any()

    nodes = []
    for node in G.nodes:
        dataset_name, k_str, gep_str = node.split("|")
        nodes.append((dataset_name, int(k_str), int(gep_str)))

    for method in ('nsamples_continuous', 'nsamples_discrete', "npatients_discrete"):
        if method == 'nsamples_continuous':
            labels = normalized_usage[nodes].sum().apply(lambda x: "{:.1f}".format(x)).to_dict() # Label is number of samples
            sizes = (normalized_usage[nodes].sum() * 5).to_dict() # Size is proportional to number of samples
        elif method == 'nsamples_discrete':
            labels = discrete_usage[nodes].sum().apply(lambda x: int(x)).to_dict() # Label is number of samples
            sizes = (discrete_usage[nodes].sum() * 5).to_dict() # Size is proportional to number of samples
        elif method == "npatients_discrete":
            labels = patients_to_geps[nodes].sum().to_dict()
            sizes = (patients_to_geps.sum() * 5).to_dict()
        labels = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in labels.items()}
        sizes = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in sizes.items()}
        colors = [node.partition("|")[2] for node in G]
        node_sizes = [(sizes[n] if n in sizes else 0) for n in G]
        colors = [dataset_colors[node.split("|")[0]] for node in G]
        fig, ax = plt.subplots(figsize=config.sns["plot_size"])
        nx.draw(G, pos=layout,
        with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, linewidths=0, width=0.2, edge_color=config.sns["edge_color"], font_size=3, ax=ax)
        ax.legend(handles=dataset_legend)
        ax.set_title(method)
        plt.tight_layout()
        fig.savefig(os.path.join(sns_output_dir, f"{method}.pdf"))

    # Overrepresentation bars per GEP
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(include="category")  # only use categorical data
        # number of bars in each community for this dataset
        community_gep_counts = [len([node for node in communities[c] if node.split("|")[0] == dataset_name]) for c in sorted(list(communities))]
        width_ratios = [1 + gep_counts for gep_counts in community_gep_counts]
        fig, axes = plt.subplots(metadata.shape[1], len(communities), figsize=[len(communities) + 0.1 * sum(community_gep_counts), metadata.shape[1] * 3], sharey='row', gridspec_kw={"width_ratios": community_gep_counts})
        for row, (annotation_layer, sample_to_class) in enumerate(metadata.items()):
            # usage subset to dataset
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            ds_usage.index = ds_usage.index.map(sample_to_class)
            ds_usage = ds_usage[ds_usage.index.notnull()]
            category_sum_null = pd.Series(1, index=ds_usage.index).groupby(axis=0, level=0).sum()
            category_prop_null = category_sum_null / category_sum_null.sum()
            category_sum = ds_usage.groupby(axis=0, level=0).sum()
            category_prop = category_sum / category_sum.sum()
            overrepresentation = category_prop.div(category_prop_null, axis=0)
            overrepresentation[overrepresentation < 1] = 1
            overrepresentation = np.log2(overrepresentation)
            for col, community in enumerate(sorted(list(communities))):
                ax = axes[row, col]
                geps = []
                for node in communities[community]:
                    dataset_str, k_str, gep_str = node.split("|")
                    if dataset_str == dataset_name:
                        geps.append((dataset_str, int(k_str), int(gep_str)))
                geps = sorted(geps)
                if geps:
                    overrepresentation[geps].T.plot.bar(stacked=True, width=0.9, ax=ax, legend=None, color=config.metadata_colors[annotation_layer])
                ax.set_xlabel("")
                ax.set_xticks([])
                if col == 0:
                    ax.set_ylabel(annotation_layer)
                if row == 0:
                    ax.set_title(community, size=14)

        fig.supxlabel("GEP")
        fig.supylabel("Overrepresentation")
        fig.suptitle("Community")
        os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_bar_by_community"), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_bar_by_community", dataset_name + ".pdf"))
        plt.close(fig)


    ### Categorical data overlay using spike plots ###
    
    edge_list = []
    for c1, n1 in communities.items():
        for c2, n2 in communities.items():
            if c1 != c2:  # no self-loops
                n_edges = len(list(nx.edge_boundary(G, n1, n2)))
                edge_list.append((c1, c2, n_edges))

    edge_list = pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges"))
    Gcomm = nx.from_pandas_edgelist(pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges")), "comm1", "comm2", "n_edges")


    # Centroid method for community layout

    community_layout = {}
    for community_name, nodes in communities.items():
        points = np.array([layout[node] for node in nodes])
        centroid = (np.mean(points[:, 0]), np.mean(points[:, 1]))
        community_layout[community_name] = centroid

    # # Neato layout
    # community_layout = nx.nx_agraph.graphviz_layout(Gcomm, prog="neato", args='-Goverlap=true')

    # community-level overrepresentation plots

    plot_data = {}
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(include="category")  # only use categorical data
        for row, (annotation_layer, sample_to_class) in enumerate(metadata.items()):
            # usage subset to dataset
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            ds_usage.index = ds_usage.index.map(sample_to_class)
            ds_usage = ds_usage[ds_usage.index.notnull()]
            category_sum_null = pd.Series(1, index=ds_usage.index).groupby(axis=0, level=0).sum()
            category_prop_null = category_sum_null / category_sum_null.sum()
            category_sum = ds_usage.groupby(axis=0, level=0).sum()
            category_prop = category_sum / category_sum.sum()
            # enrichment = np.log2(category_prop.div(category_prop_null, axis=0))
            enrichment = category_prop.div(category_prop_null, axis=0)
            enrichment[enrichment < 1] = 1
            enrichment = np.log2(enrichment)
            result_df = []
            for col, community in enumerate(sorted(list(communities))):
                geps = []
                for node in communities[community]:
                    dataset_str, k_str, gep_str = node.split("|")
                    if dataset_str == dataset_name:
                        geps.append((dataset_str, int(k_str), int(gep_str)))
                geps = sorted(geps)
                if geps:
                    com_es = enrichment[geps].mean(axis=1)
                    com_es[com_es < 0] = 0
                else:
                    com_es = pd.Series(0, index=enrichment.index)
                result_df.append(com_es.rename(community))
            result_df = pd.concat(result_df, axis=1)
            plot_data[(dataset_name, annotation_layer)] = result_df

    for (dataset_name, annotation_layer), community_es in plot_data.items():
        # bar plots
        fig, ax = plt.subplots()
        community_es.T.plot.bar(stacked=True, width=0.9, ax=ax, legend=None, rot=0, color=config.metadata_colors[annotation_layer])
        os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_bar", dataset_name), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_bar", dataset_name, annotation_layer + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_bar", dataset_name, annotation_layer + ".png"), dpi=600)
        plt.close(fig)

        # community network plots
        fig = plot_overrepresentation_network(
            graph=Gcomm,
            layout=community_layout,
            title=f"Dataset: {dataset_name}\nAnnotations: {annotation_layer}",
            overrepresentation=community_es,
            colordict=config.metadata_colors[annotation_layer],
            plot_size=config.sns["plot_size"],
            node_size=np.array(config.sns["node_size"]) * 8,
            edge_weights="n_edges"
        )
        os.makedirs(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_network", dataset_name), exist_ok=True)
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_network", dataset_name, annotation_layer + ".pdf"))
        fig.savefig(os.path.join(sns_output_dir, "annotated_communities", "overrepresentation_network", dataset_name, annotation_layer + ".png"), dpi=600)
        plt.close(fig)
    
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(include="category")  # only use categorical data
        for annotation_layer, sample_to_class in metadata.items():
            colordict = config.metadata_colors[annotation_layer]
            if sample_to_class.isnull().any(): # add 
                sample_to_class = sample_to_class.cat.add_categories("").fillna("")
                colordict[""] = config.metadata_colors["missing_data"]
            # if len(sample_to_class.cat.categories) > 30:
            #     # too many categories to be useful?
            #     continue
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            ds_usage.index = ds_usage.index.map(sample_to_class)
            category_sum_null = pd.Series(1, index=ds_usage.index).groupby(axis=0, level=0).sum()
            category_prop_null = category_sum_null / category_sum_null.sum()
            category_sum = ds_usage.groupby(axis=0, level=0).sum()
            category_prop = category_sum / category_sum.sum()
            overrepresentation = category_prop.div(category_prop_null, axis=0)
            overrepresentation[overrepresentation < 1] = 1
            overrepresentation = np.log2(overrepresentation)
            overrepresentation.columns = pd.Index([f"{c[0]}|{c[1]}|{c[2]}" for c in overrepresentation.columns])
            fig = plot_overrepresentation_network(
                graph=G,
                layout=layout,
                title=f"Dataset: {dataset_name}\nAnnotations: {annotation_layer}",
                overrepresentation=overrepresentation,
                colordict=colordict,
                plot_size=config.sns["plot_size"],
                node_size=config.sns["node_size"]
                )
                
            os.makedirs(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_network", dataset_name), exist_ok=True)
            fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_network", dataset_name, annotation_layer + ".pdf"))
            fig.savefig(os.path.join(sns_output_dir, "annotated_geps", "overrepresentation_network", dataset_name, annotation_layer + ".png"), dpi=600)
            plt.close(fig)


cli.add_command(txt_to_h5ad)
cli.add_command(update_h5ad_metadata)
cli.add_command(check_h5ad)
cli.add_command(model_odg)
cli.add_command(set_parameters)
cli.add_command(factorize)
cli.add_command(postprocess)
cli.add_command(annotated_heatmap)
cli.add_command(integrate)
cli.add_command(create_network)

if __name__ == "__main__":
    cli()