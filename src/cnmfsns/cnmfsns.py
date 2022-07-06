import os
import logging
import shutil
import click
import cnmf
import sys
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Optional, Mapping
from anndata import AnnData, read_h5ad
from cnmfsns.containers import CnmfResult, Integration
from cnmfsns.config import Config
from cnmfsns.odg import model_overdispersion

def start_logging(output_dir):
    logging.captureWarnings(True)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "logfile.txt"), mode="a"),
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
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@click.group(cls=OrderedGroup)
def cli():
    pass

@click.command()
@click.option(
    "-c", "--counts", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Input (cell/sample x gene) counts matrix as .df.npz or tab-delimited text file. "
         "This is the matrix which will be variance normalized and used for factorization. "
         "If not provided, the TPM matrix will be used instead.")
@click.option(
    "-t", "--tpm", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Pre-computed (cell/sample x gene) TPM matrix as .df.npz or tab delimited text file. "
         "This is the matrix which is used to select overdispersed genes and to which final GEPs will be scaled. "
         "If not provided, TPM normalization will be calculated from the count matrix.")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Pre-computed (cell/sample x gene) TPM matrix as .df.npz or tab delimited text file. "
         "This is the matrix which is used to select overdispersed genes and to which final GEPs will be scaled. "
         "If not provided, TPM normalization will be calculated from the count matrix.")
@click.option(
    "-o", '--output', type=click.Path(dir_okay=False, exists=False), required=True,
    help="Path to output .h5ad file.")
def txt_to_h5ad(counts, tpm, metadata, output):
    if counts is None and tpm is None:
        logging.error("Either a counts matrix or normalized (TPM) matrix of gene expression must be supplied.")
        sys.exit(1)
    elif counts is not None and tpm is None:
        counts = pd.read_table(counts, index_col=0)
        tpm = counts * 1e6 / counts.sum(axis=1) # compute TPM
    elif tpm and counts is None:
        tpm = pd.read_table(tpm, index_col=0)
        counts = tpm
    elif tpm and counts:
        counts = pd.read_table(counts, index_col=0)
        tpm = pd.read_table(tpm, index_col=0)
    if (counts.index != tpm.index).all() or (counts.columns != tpm.columns).all():
        logging.error("Index and Columns of counts and tpm matrices are not the same")
        sys.exit(1)
    if metadata is not None:
        metadata = pd.read_table(metadata, index_col=0)
        for col in metadata.select_dtypes(include="object").columns:
            metadata[col] = metadata[col].astype("str").astype("category")
    adata = AnnData(X=tpm, raw=AnnData(X=counts), obs=metadata)
    adata.write_h5ad(output)

@click.command()
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=False), required=True,
    help="Input .h5ad file.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, exists=False), required=True,
    help="Output .h5ad file.")
def check_h5ad(input, output):
    
    ## TODO: implement
    adata = read_h5ad(input)
    adata.write(output)
    # Check for missing values in counts matrix.
    if counts.isnull().sum().sum() > 0:
        logging.warning("Counts matrix contains missing (NaN) data .")
    # Check for missing values in counts matrix.
    if tpm.isnull().sum().sum() > 0:
        logging.warning("TPM matrix contains missing (NaN) data.")
    # if counts is None and tpm is None:
    #     logging.error("Either a counts matrix or normalized (TPM) matrix of gene expression must be supplied.")
    #     sys.exit(1)
    # elif counts and tpm is None:
    #     counts = pd.read_table(counts, index_col=0)
    #     tpm = counts * 1e6 / counts.sum(axis=1) # compute TPM
    # elif tpm and counts is None:
    #     tpm = pd.read_table(tpm, index_col=0)
    #     counts = tpm
    # elif tpm and counts:
    #     counts = pd.read_table(counts, index_col=0)
    #     tpm = pd.read_table(tpm, index_col=0)
    pass

@click.command()
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(),
    help="Output directory. Defaults to current directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=False,
    help="h5ad file containing expression data (adata.X=normalized (TPM) and adata.raw.X = count) as well as any cell/sample metadata (adata.obs).")
@click.option(
    "--odg_default_spline_degree", type=int, default=3,
    help="Degree for BSplines for the Generalized Additive Model (default method). For example, a constant spline would be 0, linear would be 1, and cubic would be 3.")
@click.option(
    "--odg_default_dof", type=int, default=8,
    help="Degrees of Freedom (number of components) for the Generalized Additive Model (default method).")
@click.option(
    "--odg_cnmf_mean_threshold", type=float, default=0.5,
    help="Minimum mean for overdispersed genes (cnmf method).")
def model_odg(name, output_dir, input, odg_default_spline_degree, odg_default_dof, odg_cnmf_mean_threshold):
    """
    Model gene overdispersion and plot calibration plots for selection of overdispersed genes, using two methods:
    
    - `cnmf`: v-score and minimum expression threshold for count data (cNMF method: Kotliar, et al. eLife, 2019) 
    - `default`: residual standard deviation after modeling mean-variance dependence. (STdeconvolve method: Miller, et al. Nat. Comm. 2022)

    """
    cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)
    adata = read_h5ad(input)
    os.makedirs(os.path.normpath(os.path.join(output_dir, name, "odgenes")), exist_ok=True)
    shutil.copy(input, os.path.join(output_dir, name, "input.h5ad"))
    # Create diagnostic plots
    df, figs = model_overdispersion(
            adata=adata,
            odg_default_spline_degree=odg_default_spline_degree,
            odg_default_dof=odg_default_dof,
            odg_cnmf_mean_threshold=odg_cnmf_mean_threshold
            )
    for fig_id, fig in figs.items():
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".pdf"), facecolor='white')
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".png"), dpi=400, facecolor='white')

    # output table with gene overdispersion measures
    df.to_csv(os.path.join(output_dir, name, "odgenes", "all.tsv"), sep="\t")


@click.command()
@click.option(
    "-m", "--method",
    type=click.Choice([
        "default_topn",
        "default_minscore",
        "default_quantile",
        "cnmf_topn",
        "cnmf_minscore",
        "cnmf_quantile",
        "genes_file"
        ]), default="default_minscore",
    help="Select the model and method of overdispersed gene selection.")
@click.argument("parameter", default=1.0)
def select_odg(name, output_dir, method, parameter):
    """
    Select overdispersed genes for factorization. Overdispersed genes can be modelled using the `default` or `cnmf` models, and thresholds
    can be specified based on the top N, score threshold, or quantile. Alternatively, a text file with one gene per line can be used to specify the genes manually.
    The default arguments are equivalent to:
        -m default_minscore 1.0
    
    For `top_n` methods, select an integer number of genes. For `min_score`, specify a score threshold. For `quantile` methods, specify the quantile of 
    genes to include (eg., the top 25% would be 0.75).
    
    """
    df = pd.read_table(os.path.join(output_dir, name, "odgenes", "all.tsv"), sep="\t")
    print(df)
    if method == "default_topn":
        pass
    elif method == "default_minscore":
        pass
    elif method == "default_quantile":
        pass
    elif method == "cnmf_topn":
        pass
    elif method == "cnmf_minscore":
        pass
    elif method == "cnmf_quantile":
        pass
    elif method == "genes_file":
        pass

@click.command()
def factorize():
    pass

@click.command()
def postprocess():
    pass

@click.command()
def annotate_usages():
    pass

@click.command()
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="Output directory for cNMF-SNS")
@click.option('-c', '--config_file', type=click.Path(exists=True, dir_okay=False), help="TOML config file")
@click.option('-i', '--input_h5mu', type=click.Path(exists=True, dir_okay=False), multiple=True, help="h5mu input file")
def initialize(output_dir, config_file, input_h5mu):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5mu files directly, it is recommended to use a .toml file which allows for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    start_logging(output_dir)
    logging.info("cnmfsns initialize")

    if config_file is not None and input_h5mu:
        logging.error("A TOML config file can be specified, or 1 or more h5mu files can be specified, but not both.")
    if not all(fn.endswith(".h5mu") for fn in input_h5mu):
        logging.error("Input files must be h5mu mudata files.")
    
    # create directory structure, warn if overwriting
    output_dir = os.path.normpath(output_dir)
    if os.path.isdir(output_dir):
        warnings.warn(f"Integration directory {output_dir} already exists. Files may be overwritten.")
    os.makedirs(os.path.join(output_dir, "input", "datasets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output", "correlation_distributions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output", "overdispersed_genes"), exist_ok=True)
    
    # create config
    if config_file is not None:
        config = Config.from_toml(config_file)
    elif input_h5mu:
        config = Config.from_h5mu_files(input_h5mu)
    logging.info("Copying data to output directory...")
    # copy files to output directory
    for name, d in config.datasets.items():
        shutil.copy(d["filename"], os.path.join(output_dir, "input", "datasets", name + ".h5mu"))
        shutil.copy(d["metadata"], os.path.join(output_dir, "input", "datasets", name + ".metadata.txt"))

    # add missing colors to config
    config.add_missing_colors()

    # test if variables already exist for SNS steps, otherwise create defaults that can be edited
    pass

    # save config file to output directory
    config.to_toml(os.path.join(output_dir, "config.toml"))

    # Import data from h5mu files
    data_pearson = Integration(config, output_dir, corr_method="pearson", min_corr=-1)
    data_spearman = Integration(config, output_dir, corr_method="spearman", min_corr=-1)

    # overdispersed gene UpSet plots
    data_pearson.plot_genelist_upset()

    # Create plot of correlation distributions for pearson and spearman
    data_pearson.plot_pairwise_corr(show_threshold=False)
    data_spearman.plot_pairwise_corr(show_threshold=False)

@click.command()
def create_sns():
    pass

@click.command()
def annotate_sns():
    pass

@click.command()
@click.option('-d', '--cnmf_result_dir', type=click.Path(exists=True, file_okay=False), required=True, help="cNMF result directory")
@click.option('-l', '--local_density_threshold', default=None, type=float, show_default=True,
              help='Choose this local density threshold from those that were used to run cNMF. If unspecified, it is inferred from filenames.')
@click.option('-o', '--output_file', type=click.Path(exists=False, dir_okay=False), default=None, help="Path to output file ending with .h5mu")
@click.option('--delete', is_flag=True, help='Delete cNMF result directory after completion.')
def create_h5mu(cnmf_result_dir, local_density_threshold, output_file, delete):
    cnmf_result_dir = os.path.normpath(cnmf_result_dir)
    if output_file is None:
        output_file = os.path.normpath(cnmf_result_dir) + ".h5mu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CnmfResult.from_dir(cnmf_result_dir, local_density_threshold).to_mudata().write(output_file)
    if delete:
        shutil.rmtree(cnmf_result_dir)

cli.add_command(txt_to_h5ad)
cli.add_command(check_h5ad)
cli.add_command(model_odg)
cli.add_command(select_odg)
cli.add_command(factorize)
cli.add_command(postprocess)
cli.add_command(annotate_usages)
cli.add_command(initialize)
cli.add_command(create_sns)
cli.add_command(annotate_sns)
cli.add_command(create_h5mu)


if __name__ == "__main__":
    cli()