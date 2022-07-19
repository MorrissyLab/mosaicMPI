import os
import logging
import shutil
import subprocess
import click
import cnmf
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import OrderedDict
from typing import Optional, Mapping
from anndata import AnnData, read_h5ad
from cnmfsns.containers import Integration, add_cnmf_results_to_h5ad
from cnmfsns.config import Config
from cnmfsns.odg import model_overdispersion, create_diagnostic_plots, fetch_hgnc_protein_coding_genes
from cnmfsns.plots import create_annotated_heatmaps, plot_annotated_usages

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
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@click.group(cls=OrderedGroup)
def cli():
    pass

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
    help="Tab-separated text file with metadata for samples/cells/spots with one row each. Columns are annotation layers.")
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
        adata = AnnData(X=normalized, raw=AnnData(X=counts), obs=metadata)
    adata.write_h5ad(output)

@click.command()
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Tab-separated text file with metadata for samples/cells/spots with one row each. Columns are annotation layers.")
@click.option(
    "-i", '--input_h5ad', type=click.Path(dir_okay=False, exists=True), required=True,
    help="Path to input .h5ad file.")
@click.option(
    "-o", '--output_h5ad', type=click.Path(dir_okay=False, exists=False), required=False,
    help="Path to output .h5ad file. If none are provided, input file will be overwritten.")
def update_h5ad_metadata(input_h5ad, metadata, output_h5ad):
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

    if output_h5ad is None:
        output_h5ad = input_h5ad
    adata = read_h5ad(input_h5ad)
    adata.obs = metadata
    adata.write_h5ad(output_h5ad)


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
            counts = normalized
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
            logging.warning(f"Subsetting variables to those with nonzero data.")
            counts = counts[counts.var() > 0]
    # Check for genes with zero variance
    zerovargenes = (normalized.var() == 0).sum()
    if zerovargenes:
        logging.warning(f"{zerovargenes} of {adata.n_vars} variables have a variance of zero in normalized data (`adata.X`).")
        if output:
            logging.warning(f"Subsetting variables to those with nonzero data.")
            normalized = normalized[normalized.var() > 0]
    
    # check for linear scaling of counts to normalized matrix (eg. TPM) for cNMF
    is_nonlinear_scaling = (np.abs(normalized.corrwith(counts, axis=1, method="pearson") - 1) > 0.01).any()  # Uses pearson correlation to detect non-linear relationships
    if is_nonlinear_scaling:
        logging.warning(f"Normalized data (`adata.X`) does not appear to be a linear scaling of counts (`adata.raw.X`) data. Linear scaling such as TPM is recommended for cNMF.")
    
    # Save output to new h5ad file
    if output is not None:
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
    help="h5ad file containing expression data (adata.X=normalized (TPM) and adata.raw.X = count) as well as any cell/sample metadata (adata.obs).")
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
    for fig_id, fig in create_diagnostic_plots(df, show_selected=False).items():
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
    '--beta_loss', type=click.Choice(["frobenius", "kullback-leibler"]), default="kullback-leibler",
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

    df["selected"] = df.index.isin(genes)

    # update plots with threshold information
    for fig_id, fig in create_diagnostic_plots(df, show_selected=True).items():
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".pdf"), facecolor='white')
        fig.savefig(os.path.join(output_dir, name, "odgenes", ".".join(fig_id) + ".png"), dpi=400, facecolor='white')

    # output table with gene overdispersion measures
    df.to_csv(os.path.join(output_dir, name, "odgenes", "genestats.tsv"), sep="\t")

    # write TPM (normalized) data
    adata = read_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
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
    start_logging(os.path.join(output_dir, name, "logfile.txt"))
    cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=name)
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
@click.option(
    '--local_density_threshold', type=float, default=2.0, show_default=True,
    help="Threshold for the local density filtering prior to GEP consensus. Acceptable thresholds are > 0 and <= 2 (2.0 is no filtering).")
@click.option(
    '--local_neighborhood_size', type=float, default=0.3, show_default=True,
    help="Fraction of the number of replicates to use as nearest neighbors for local density filtering.")
@click.option(
    '--keep_individual_iterations', is_flag=True,
    help="If specified, individual iteration files will be retained even after merging.")
@click.option(
    '--force_h5ad_update', is_flag=True,
    help="If specified, overwrites cNMF results already saved to the .h5ad file.")
def postprocess(name, output_dir, local_density_threshold, local_neighborhood_size, keep_individual_iterations, force_h5ad_update):
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
                f"Postprocessing could not proceed because {(len(failed))} files from the factorization step are missing or empty:\n  - " + 
                "\n  - ".join(failed)
            )
            sys.exit(1)
        else:
            # combine individual iterations
            logging.info(f"Factorization outputs (individual iterations) were found for all values of k.")
            for k in sorted(set(run_params.n_components)):
                cnmf_obj.combine_nmf(k, remove_individual_iterations=(not keep_individual_iterations))
    else:
        logging.info(f"Factorization outputs (merged iterations) were found for all values of k.")
    # calculate consensus GEPs and usages
    for k in sorted(set(run_params.n_components)):
        cnmf_obj.consensus(k, local_density_threshold, local_neighborhood_size, True,
                            close_clustergram_fig=True)
    # create k-selection plot
    cnmf_obj.k_selection_plot(close_fig=True)
   
    # update h5ad file with cnmf results
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
def create_annotated_heatmaps(input_h5ad, output_dir, metadata_colors_toml, max_categories_per_layer):
    """
    Create heatmaps of usages with annotation tracks.
    """
    start_logging()
    os.makedirs(output_dir, exist_ok=True)
    adata = read_h5ad(input_h5ad)
    # annotate usage plots
    if metadata_colors_toml:
        cfg = Config.from_toml(metadata_colors_toml)
    else:
        cfg = Config()
        
    cfg.add_missing_colors_from(adata.obs)
    cfg.to_toml(os.path.join(output_dir, "metadata_colors.toml"))
    exclude_maxcat = adata.obs.select_dtypes(include=["object", "category"]).apply(lambda x: len(x.cat.categories)) > max_categories_per_layer
    metadata = adata.obs.drop(columns=exclude_maxcat[exclude_maxcat].index)
    if "cnmf_usage" not in adata.obsm:
        logging.error("cNMF results have not been merged into .h5ad file. Ensure that you have run `cnmfsns postprocess` before creating annotated usage heatmaps.")
        sys.exit(1)
    usage = adata.obsm["cnmf_usage"]
    usage.columns=pd.MultiIndex.from_tuples(usage.columns.str.split(".").to_list())

    # create annotated plots
    for k in usage.columns.levels[0]:
        k_usage = usage.loc[:, k]
        cnmf_name = adata.uns["cnmf_name"]
        ldt = adata.uns["ldt"]
        title = f"{cnmf_name} k={k} ldt={ldt}"
        filename = os.path.join(output_dir, f"{cnmf_name}.usages.k{(int(k)):03}.pdf")
        plot_annotated_usages(df=k_usage, metadata=metadata, metadata_colors=cfg.metadata_colors, title=title, filename=filename)

@click.command()
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="Output directory for cNMF-SNS results")
@click.option('-c', '--config_toml', type=click.Path(exists=True, dir_okay=False), help="TOML config file")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), multiple=True, help="h5ad input file containing")
def initialize(output_dir, config_toml, input_h5ad):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5ad files directly, it is recommended to use a .toml file which allows for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    start_logging(os.path.join(output_dir, "logfile.txt"))
    logging.info("cnmfsns initialize")

    if config_toml is not None and input_h5ad:
        logging.error("A TOML config file can be specified, or 1 or more h5mu files can be specified, but not both.")
        sys.exit(1)
    if not all(fn.endswith(".h5ad") for fn in input_h5ad):
        logging.error("Input files must be AnnData .h5ad files.")
        sys.exit(1)
    
    # create directory structure, warn if overwriting
    output_dir = os.path.normpath(output_dir)
    if os.path.isdir(output_dir):
        warnings.warn(f"Integration directory {output_dir} already exists. Files may be overwritten.")
    os.makedirs(os.path.join(output_dir, "input", "datasets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output", "correlation_distributions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output", "overdispersed_genes"), exist_ok=True)
    
    # create config
    if config_toml is not None:
        config = Config.from_toml(config_toml)
    elif input_h5ad:
        config = Config.from_h5ad_files(input_h5ad)
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

cli.add_command(txt_to_h5ad)
cli.add_command(update_h5ad_metadata)
cli.add_command(check_h5ad)
cli.add_command(model_odg)
cli.add_command(set_parameters)
cli.add_command(factorize)
cli.add_command(postprocess)
cli.add_command(create_annotated_heatmaps)
cli.add_command(initialize)
cli.add_command(create_sns)
cli.add_command(annotate_sns)

if __name__ == "__main__":
    cli()