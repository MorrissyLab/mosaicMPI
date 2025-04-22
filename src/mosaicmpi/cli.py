
from .dataset import Dataset
from .integration import Integration
from .config import Config
from .colors import Colors
from .network import Network, compare_community_jaccard_similarity
from .cnmf import cNMF
from .plots import *
from . import utils, __version__, cpus_available

import os
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

# Orders the commands in the help menus to match the workflow
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


@click.group(cls=_OrderedGroup, context_settings={'help_option_names': ['-h', '--help'], 'max_content_width': 160})
@click.version_option(version=__version__)
def cli():
    """
    mosaicMPI is a tool for deconvolution and integration of multiple datasets based on consensus Non-Negative Matrix Factorization (cNMF).
    """


    # start logging to stdout before any subcommands are run
    utils.start_logging()

    # log the original command
    logging.info(" ".join(sys.argv))

@cli.result_callback()
def post_cli(result, **kwargs):
    
    # log command completion
    logging.info("All tasks completed successfully.")


@click.command(name="txt-to-h5ad")
@click.option(
    "-d", "--data", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Input counts or normalized matrix as delimited text file. Rows are observations (eg. samples/cells) and columns are features (eg. genes) (unless --transpose is specified).")
@click.option(
    "--is_normalized", is_flag=True, help="Specify if input data is normalized instead of count data.")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Optional delimited text file with metadata for samples/cells with one row each. Columns are annotation layers.")
@click.option(
    "-f", "--feature_metadata", type=click.Path(dir_okay=False, exists=True), required=False,
    help="Optional delimited text file with metadata for features with one row each. Columns are annotation layers.")
@click.option(
    "--transpose", is_flag=True,
    help="Transpose an input data matrix where rows are features and columns are observations.")
@click.option(
    "--data_delimiter", type=str, default="\t",
    help="Delimiter for data file, defaults to tab-delimited.")
@click.option(
    "--metadata_delimiter", type=str, default="\t",
    help="Delimiter for metadata files, defaults to tab-delimited.")
@click.option(
    "-o", '--output', type=click.Path(dir_okay=False, exists=False), required=True,
    help="Path to output .h5ad file.")
@click.option(
    "--sparsify", is_flag=True,
    help="[Experimental feature] Save resulting data in sparse format. Recommended to increase performance for sparse datasets such as scRNA-Seq, scATAC-Seq, and 10X Visium"
         ", but not for bulk expression data.")
def cmd_txt_to_h5ad(data, is_normalized, metadata, feature_metadata, output, transpose, data_delimiter, metadata_delimiter, sparsify):
    """
    Create .h5ad file with data and metadata (`adata.obs`).
    """
    df = pd.read_table(data, index_col=0, sep=data_delimiter)
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


@click.command(name="update-h5ad-metadata")
@click.option(
    "-m", "--metadata", type=click.Path(dir_okay=False, exists=True), required=True,
    help="Optional delimited text file with metadata for samples/cells with one row each. Columns are annotation layers.")
@click.option(
    "--metadata_delimiter", type=str, default="\t",
    help="Delimiter for metadata files, defaults to tab-delimited.")
@click.option(
    "-i", '--input_h5ad', type=click.Path(dir_okay=False, exists=True), required=True,
    help="Path to input .h5ad file.")
def cmd_update_h5ad_metadata(input_h5ad, metadata, metadata_delimiter):
    """
    Update metadata in a .h5ad file at any point in the mosaicMPI workflow. New metadata will overwrite (`adata.obs`).
    """
    dataset = Dataset.from_h5ad(input_h5ad)
    sample_metadata_df = pd.read_table(metadata, index_col=0, sep=metadata_delimiter).dropna(axis=1, how="all")
    dataset.update_obs(sample_metadata_df)
    logging.info(dataset.get_printable_metadata_type_summary())
    dataset.write_h5ad(input_h5ad)


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
    dataset = Dataset.from_h5ad(input)
    dataset.impute_zeros(n_folds=n_folds)
    # Save output to new h5ad file
    dataset.write_h5ad(output)


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
    dataset = Dataset.from_h5ad(input)
    dataset.impute_knn(n_neighbors = n_neighbors, weights = weights, n_folds=n_folds)
    # Save output to new h5ad file
    dataset.write_h5ad(output)


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
    dataset = Dataset.from_h5ad(input)
    dataset.remove_unfactorizable_observations()
    dataset.remove_unfactorizable_features()
    
    # Save output to new h5ad file
    if output is not None:
        dataset.write_h5ad(output)


@click.group(name="select-hvf", cls=_OrderedGroup)
def select_hvf():
    """
    Select highly variable features for factorization.
    """


@click.command(name="stdeconvolve")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="h5ad file containing expression data as well as any cell/sample metadata (adata.obs).")
@click.option(
    "--stratify_by", type=str, default=None, show_default=True,
    help="Model gene-variance relationship separately for each class of samples/cells based on the provided metadata field. For example, you "
         "could stratify by Sample ID for single-cell datasets.")
@click.option(
    "--stratify_mode", type=click.Choice(["union", "intersection"]), default="union", show_default=True,
    help="Select the union or intersection of gene lists identified from dataset strata.")
@click.option(
    "--max_cells_proportion", type=float, default=1.0, show_default=True,
    help="Exclude features with greater than this proportion of positive values.")
@click.option(
    "--min_cells_proportion", type=float, default=0.05, show_default=True,
    help="Exclude features with less than this proportion of positive values.")
@click.option(
    "--min_features", type=int, default=0, show_default=True,
    help="Exclude samples/cells with fewer than this number of positive features.")
@click.option(
    "--min_raw_sum", type=float, default=0.0, show_default=True,
    help="Exclude samples/cells with a summed signal less than this threshold.")
@click.option(
    "--n_splines", type=int, default=5, show_default=True,
    help="Number of splines to use for fitting the Linear GAM, must be greater than spline_order.")
@click.option(
    "--spline_order", type=int, default=3, show_default=True,
    help="spline order (constant = 0, linear = 1, quadratic = 2, and cubic = 3).")
@click.option(
    "--top_n", type=int, default=1000, show_default=True,
    help="Number of features to select after ranking features by score.")
@click.option(
    "--alpha",
    type=float, default=0.05, show_default=True, required=True,
    help="Alpha (p-value) threshold for selection of HVFs.")
@click.option(
    "--use_unadjusted_pvals", is_flag=True, show_default=True,
    help="Threshold based on unadjusted, rather than Benjamini-Hochberg adjusted p-values.")
def cmd_select_hvf_stdeconvolve(name, output_dir, input, stratify_by, stratify_mode, max_cells_proportion, min_cells_proportion,
                           min_features, min_raw_sum, n_splines, spline_order, top_n, alpha, use_unadjusted_pvals):
    """
    STdeconvolve method (Miller, et al. Nat. Comm. 2022) to select highly-variable features. 

    First, the dataset is preprocessed. If --stratify_by is provided, the dataset is split into strata of samples/cells based on the provided metadata. --stratify_mode
    allows either the union (the default) or intersection of HVFs identified separately for each stratum. Cells are filtered based on
    --min_cells_proportion, and --max_cells_proportion parameters, to remove features which are present in too many or too few samples/cells. These parameters are identical
    to the --removeAbove and --removeBelow parameters in STdeconvolve, and by default are disabled.

    Second, an overdispersion score is calculated A Linear Generalized Additive Model (Linear GAM) models the relationship of log10(mean) to log10(variance), with the --n_splines
    and --spline_degree parameters. This curve is used to identify the relative overdispersion of each gene relative to other features with similar expression levels. Once this
    score is calculated, it can be quantile normalized and its significance can be assessed using p-values from the F-distribution.

    Thirdly, one or more thresholding methods are applied. --alpha allows thresholding on the p-value, which is by default Benjamini-Hochberg adjusted, unless --use_unadjusted_pvals
    is specified. If --top_n is specified, only the intersection of these two filters is selected. 

    Example:
        
        \b
        # use default parameters, suitable for most datasets
        mosaicmpi select-hvf default -n test -i test.h5ad --alpha 0.05

        \b
        # specify a custom gene list in a whitespace-delimited text file
        mosaicmpi select-hvf default -n test -i test.h5ad --feature_list genelist.txt

        \b
        # identify HVF features separately for single-cells from each patient, then get the union
        mosaicmpi select-hvf default -n test -i test.h5ad --alpha 0.05 --stratify_by patientID --stratify_mode union

    """

    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)  # creates output directory
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
    dataset = Dataset.from_h5ad(input)
    
    # get features from feature list file
    if feature_list is not None:
        feature_list = open(feature_list).read().split()

    # Create gene stats table and save h5ad file
    dataset.select_hvf_stdeconvolve(
        stratify_by = stratify_by,
        stratify_mode = stratify_mode,
        max_cells_proportion = max_cells_proportion,
        min_cells_proportion = min_cells_proportion,
        min_features = min_features,
        min_raw_sum = min_raw_sum,
        n_splines = n_splines,
        spline_order = spline_order,
        top_n = top_n,
        alpha = alpha,
        adjust_pvals = not use_unadjusted_pvals)
    
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    dataset.hvf_stats.to_csv(os.path.join(output_dir, name, "hvf_stats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = plot_feature_mean_variance(dataset)
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_mean_var"), formats=("pdf", "png"), target_dpi=400, facecolor='white')
    
    # create overdispersion histogram plot
    fig = plot_feature_statistic_histogram(dataset, show_selected = True, y_unit="odscore")
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_mean_odscore"), formats=("pdf", "png"), target_dpi=400, facecolor='white')

    if dataset.is_imputed:
        logging.info("Creating plots for imputed data")

        fig = plot_feature_missingness(dataset, proportion=True)
        utils.save_fig(fig, os.path.join(output_dir, name, "missingness_histogram"), formats=("pdf", "png"), target_dpi=400, facecolor='white')


@click.command(name="custom")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="h5ad file containing expression data as well as any cell/sample metadata (adata.obs).")
@click.option(
    "--feature_list", type=click.Path(exists=True, dir_okay=False), show_default=True, required=True,
    help="Text file with feature names separated by newlines.")
def cmd_select_hvf_custom(name, output_dir, input, feature_list):
    """
    Provide a text file of feature names to select highly variable features.

    \b
    Example:
        # specify a custom gene list in a whitespace-delimited text file
        mosaicmpi select-hvf custom -n test -i test.h5ad --feature_list genelist.txt

    """
    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)  # creates output directory
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
    dataset = Dataset.from_h5ad(input)
    # get features from feature list file
    feature_list = open(feature_list).read().split()

    # Create gene stats table and save h5ad file
    dataset.select_hvf(feature_list = feature_list)
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    dataset.hvf_stats.to_csv(os.path.join(output_dir, name, "hvf_stats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = plot_feature_mean_variance(dataset)
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_meanvar"), formats=("pdf", "png"), target_dpi=400, facecolor='white')
    
    # create odscore histogram plot
    fig = plot_feature_statistic_histogram(dataset, hue="selected", statistic="odscore", log_scale=[10, False])
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_odscore"), formats=("pdf", "png"), target_dpi=400, facecolor='white')

    if dataset.is_imputed:
        logging.info("Creating plots for imputed data")

        fig = plot_feature_missingness(dataset, proportion=True)
        utils.save_fig(fig, os.path.join(output_dir, name, "missingness_histogram"), formats=("pdf", "png"), target_dpi=400, facecolor='white')


@click.command(name="default")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="h5ad file containing expression data as well as any cell/sample metadata (adata.obs).")
@click.option(
    "--stratify_by", type=str, default=None, show_default=True,
    help="Model gene-variance relationship separately for each class of samples/cells based on the provided metadata field. For example, you "
         "could stratify by Sample ID for single-cell datasets.")
@click.option(
    "--stratify_mode", type=click.Choice(["union", "intersection"]), default="union", show_default=True,
    help="Select the union or intersection of gene lists identified from dataset strata.")
@click.option(
    "--use_counts", is_flag=True, show_default=True,
    help="model mean and variance of the count data (rather than normalized data), if it exists.")
@click.option(
    "--max_missingness", type=float, default=0.0, show_default=True,
    help="For datasets imputed using mosaicMPI, exclude features with greater than this proportion of imputed values.")
@click.option(
    "--max_cells_proportion", type=float, default=1.0, show_default=True,
    help="Exclude features with greater than this proportion of positive values.")
@click.option(
    "--min_cells_proportion", type=float, default=0.0, show_default=True,
    help="Exclude features with less than this proportion of positive values.")
@click.option(
    "--min_cells_mean", type=float, default=0.0, show_default=True,
    help="Exclude features with less than this mean.")
@click.option(
    "--min_features", type=int, default=0, show_default=True,
    help="Exclude samples/cells with fewer than this number of positive features.")
@click.option(
    "--min_raw_sum", type=float, default=0.0, show_default=True,
    help="Exclude samples/cells with a summed signal less than this threshold.")
@click.option(
    "--n_splines", type=int, default=5, show_default=True,
    help="Number of splines to use for fitting the Linear GAM, must be greater than spline_order.")
@click.option(
    "--spline_order", type=int, default=3, show_default=True,
    help="spline order (constant = 0, linear = 1, quadratic = 2, and cubic = 3).")
@click.option(
    "--score_type", type=click.Choice(["vscore", "odscore"]), default="odscore", show_default=True,
    help="Type of score for calculating overdispersion.")
@click.option(
    "--min_score",
    type=float, default=None, show_default=True,
    help="Minimum score threshold for feature selection.")
@click.option(
    "--top_n",
    type=int, default=None, show_default=True,
    help="Number of features to select after ranking features by score.")
@click.option(
    "--top_quantile",
    type=float, default=None, show_default=True,
    help="Proportion of top features to select after ranking the score.")
@click.option(
    "--alpha",
    type=float, default=None, show_default=True,
    help="Alpha (p-value) threshold for selection of HVFs.")
@click.option(
    "--use_unadjusted_pvals", is_flag=True, show_default=True,
    help="Threshold based on unadjusted, rather than Benjamini-Hochberg adjusted p-values.")
@click.option(
    "--feature_list", type=click.Path(exists=True, dir_okay=False), show_default=True,
    help="Text file with feature names separated by newlines.")
@click.option(
    "--multiple_threshold_mode", type=click.Choice(["union", "intersection"]), default="intersection", show_default=True,
    help="Method to combine multiple thresholds.")
def cmd_select_hvf_default(name, output_dir, input, stratify_by, stratify_mode, use_counts, max_missingness, max_cells_proportion, min_cells_proportion, min_cells_mean,
                           min_features, min_raw_sum, n_splines, spline_order, score_type, min_score, top_n, top_quantile, alpha, use_unadjusted_pvals, feature_list, multiple_threshold_mode):
    """
    Flexible way to select highly-variable features. The process is controlled by several groups of parameters.

    First, the dataset is preprocessed. If --stratify_by is provided, the dataset is split into strata of samples/cells based on the provided metadata. --stratify_mode
    allows either the union (the default) or intersection of HVFs identified separately for each stratum. Either for the whole dataset, or separately in a stratified dataset,
    normalized data is used by default, unless --use_counts is specified. If the .h5ad file was created from a single counts matrix, then this means that the TPM normalization
    will be performed. Cells are then removed based on the --max_missingness, which only applies to datasets imputed by mosaicMPI. Cells are further filtered based on
    --min_cells_proportion, and --max_cells_proportion parameters, to remove features which are present in too many or too few samples/cells. These parameters are identical
    to the --removeAbove and --removeBelow parameters in STdeconvolve, and by default are disabled.

    Second, an overdispersion score is calculated using one of two methods specified using --score_type. If vscore, the scoring system is the same as cNMF (Kotliar et. al., 2019). By default,
    the score_type is odscore, which is the same as STdeconvolve (Miller, et al. Nat. Comm. 2022). To calculate the odscore, a Linear Generalized Additive Model (Linear GAM)
    is used to model the relationship of log10(mean) to log10(variance), with the --n_splines and --spline_degree parameters. This curve is used to identify the relative
    overdispersion of each gene relative to other features with similar expression levels. Once this score is calculated, it can be quantile normalized and its significance can be
    assessed using p-values from the F-distribution.

    Thirdly, one or more thresholding methods are applied. --alpha allows thresholding on the p-value, which is by default Benjamini-Hochberg adjusted, unless --use_unadjusted_pvals
    is specified. --min_score thresholds on the score value, --top_n thresholds based on the number of features after ranking, and --top_quantile thresholds the features on the
    provided quantile (eg. 0.10) of the scores. --feature_list allows the option to provide a file with whitespace-delimited feature names
    to either restrict or expand the selected features to a provided list.

    If multiple thresholding methods are selected, --multiple_threshold_mode controls whether to select the union or the intersection of these methods.

    Examples:

        \b
        # use default parameters, suitable for most datasets
        mosaicmpi select-hvf default -n test -i test.h5ad --alpha 0.05

        \b
        # specify a custom gene list in a whitespace-delimited text file
        mosaicmpi select-hvf default -n test -i test.h5ad --feature_list genelist.txt

        \b
        # identify HVF features separately for single-cells from each patient, then get the union
        mosaicmpi select-hvf default -n test -i test.h5ad --alpha 0.05 --stratify_by patientID --stratify_mode union

    """

    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)  # creates output directory
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
    dataset = Dataset.from_h5ad(input)
    
    # get features from feature list file
    if feature_list is not None:
        feature_list = open(feature_list).read().split()

    # Create gene stats table and save h5ad file
    dataset.select_hvf(
        stratify_by = stratify_by,
        stratify_mode = stratify_mode,
        use_normalized = not use_counts,
        max_missingness = max_missingness,
        max_cells_proportion = max_cells_proportion,
        min_cells_proportion = min_cells_proportion,
        min_cells_mean = min_cells_mean,
        min_features = min_features,
        min_raw_sum = min_raw_sum,
        n_splines = n_splines,
        spline_order = spline_order,
        score_type = score_type,
        min_score = min_score,
        top_n = top_n,
        top_quantile = top_quantile,
        alpha = alpha,
        adjust_pvals = not use_unadjusted_pvals,
        feature_list = feature_list,
        multiple_threshold_mode = multiple_threshold_mode)
    
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    dataset.hvf_stats.to_csv(os.path.join(output_dir, name, "hvf_stats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = plot_feature_mean_variance(dataset)
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_mean_var"), formats=("pdf", "png"), target_dpi=400, facecolor='white')
    
    # create overdispersion histogram plot
    hue = "selection_overlap" if stratify_by is not None else "selected"
    fig = plot_feature_statistic_histogram(dataset, hue=hue, statistic=score_type, log_scale=[10, False])
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_" + score_type), formats=("pdf", "png"), target_dpi=400, facecolor='white')

    if dataset.is_imputed:
        logging.info("Creating plots for imputed data")

        fig = plot_feature_missingness(dataset, proportion=True)
        utils.save_fig(fig, os.path.join(output_dir, name, "missingness_histogram"), formats=("pdf", "png"), target_dpi=400, facecolor='white')


@click.command(name="cnmf")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    "-i", "--input", type=click.Path(dir_okay=False, exists=True), required=True,
    help="h5ad file containing expression data as well as any cell/sample metadata (adata.obs).")
@click.option(
    "--stratify_by", type=str, default=None, show_default=True,
    help="Model gene-variance relationship separately for each class of samples/cells based on the provided metadata field. For example, you "
         "could stratify by Sample ID for single-cell datasets.")
@click.option(
    "--stratify_mode", type=click.Choice(["union", "intersection"]), default="union", show_default=True,
    help="Select the union or intersection of gene lists identified from dataset strata.")
@click.option(
    "--min_cells_mean", type=float, default=0.5, show_default=True,
    help="Exclude features with less than this mean.")
@click.option(
    "--top_n",
    type=int, default=2000, show_default=True,
    help="Number of features to select after ranking features by score.")
def cmd_select_hvf_cnmf(name, output_dir, input, stratify_by, stratify_mode, min_cells_mean, top_n):
    """
    cNMF method (Kotliar et. al., 2019) to select highly-variable features.

    First, the dataset is preprocessed. If --stratify_by is provided, the dataset is split into strata of samples/cells based on the provided metadata. --stratify_mode
    allows either the union (the default) or intersection of HVFs identified separately for each stratum. Second, the vscore is calculated from the normalized matrix to identify
    feature overdispersion. Features with a mean less than the --min_cells_mean parameter are excluded. Then, the --top_n features are selected.

    Example:
        \b
        # use default parameters, same as cNMF's default parameters
        mosaicmpi select-hvf cnmf -n test -i test.h5ad

    """

    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)  # creates output directory
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
    dataset = Dataset.from_h5ad(input)
    # Create gene stats table and save h5ad file
    dataset.select_hvf_cnmf(
        stratify_by = stratify_by,
        stratify_mode = stratify_mode,
        min_cells_mean = min_cells_mean,
        top_n = top_n)
    
    dataset.write_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    
    # output text file
    dataset.hvf_stats.to_csv(os.path.join(output_dir, name, "hvf_stats.tsv"), sep="\t")

    # create mean vs variance plots
    fig = plot_feature_mean_variance(dataset)
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_meanvar"), formats=("pdf", "png"), target_dpi=400, facecolor='white')
    
    # create overdispersion histogram plot
    hue = "selection_overlap" if stratify_by is not None else "selected"
    fig = plot_feature_statistic_histogram(dataset, hue=hue, statistic="vscore", log_scale=[10, False])
    utils.save_fig(fig, os.path.join(output_dir, name, "feature_vscore"), formats=("pdf", "png"), target_dpi=400, facecolor='white')

    if dataset.is_imputed:
        logging.info("Creating plots for imputed data")

        fig = plot_feature_missingness(dataset, proportion=True)
        utils.save_fig(fig, os.path.join(output_dir, name, "missingness_histogram"), formats=("pdf", "png"), target_dpi=400, facecolor='white')


@click.command(name="initialize-cnmf")
@click.option(
    "-n", "--name", type=str, required=True, 
    help="Name for cNMF analysis. All output will be placed in [output_dir]/[name]/...")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory. All output will be placed in [output_dir]/[name]/... ")
@click.option(
    '--k_range', type=int, nargs=3, multiple=True,
    help="Specify a range of components for factorization, using three numbers: first, last, step_size. Eg. '4 23 4' means k=4,8,12,16,20. "
    "Note that this argument can be supplied multiple times for multiple ranges")
@click.option(
    "-k", type=int, multiple=True,
    help="Specify individual components for factorization. Multiple may be selected like this: -k 2 -k 3")
@click.option(
    '--n_iter', type=int, show_default=True, default=100,
    help="Number of iterations for factorization. If several `k` are specified, this many iterations will be run for each value of `k`")
@click.option(
    '--seed', type=int, help="Seed for scikit-learn random state.")
@click.option(
    '--beta_loss', type=click.Choice(["frobenius", "kullback-leibler"]), default="kullback-leibler", show_default=True,
    help="Measure of beta-divergence to be minimized.")
def cmd_initialize_cnmf(name, output_dir, k_range, k, n_iter, seed, beta_loss):
    """
    
    Initialize cNMF with inputs for factorization.
    
    Examples:
        \b
        # specify k with ranges: 10-100 (by 10s), and 100-500 (by 100s)
        mosaicmpi initialize-cnmf --k_range 10 100 10 --k_range 100 500 100
    """
    os.makedirs(os.path.join(output_dir, name, "logs"), exist_ok=True)
    utils.start_logging(os.path.join(output_dir, name, "logs", "logfile.txt"))
    
    # process k-value selection inputs
    kvals = set(k)
    for start, stop, step in k_range:
        kvals |= set(range(start, stop + 1, step))
    kvals = sorted(list(kvals))

    if not kvals:
        logging.error("Please specify rank(s) for factorization using -k and/or --k_range parameters.")
        sys.exit(1)
    logging.info("Setting up factorization for the following ranks: " + ", ".join([str(k) for k in kvals]))
    
    # prepare cNMF directory for factorization
    dataset = Dataset.from_h5ad(os.path.join(output_dir, name, name + ".h5ad"))
    dataset.initialize_cnmf(cnmf_output_dir = output_dir, cnmf_name=name, kvals=kvals, n_iter=n_iter, beta_loss=beta_loss, seed=seed)


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
            logging.error(f"AnnData already contains cNMF results. Use --force_h5ad_update to overwrite.")
            sys.exit(1)

        dataset.add_cnmf_results(cnmf_output_dir=output_dir,
                                cnmf_name=name,
                                local_density_threshold=local_density_threshold,
                                local_neighborhood_size=local_neighborhood_size
                                )
        dataset.write_h5ad(h5ad_path)
        fig = plot_stability_error(dataset=dataset)
        utils.save_fig(fig, os.path.join(output_dir, name, name + '.k_selection'))
    
@click.command("usage-heatmap")
@click.option(
    "-i", "--input_h5ad", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to AnnData (.h5ad) file containing cNMF results.")
@click.option(
    "-o", '--output_dir', type=click.Path(file_okay=False), default=os.getcwd(), show_default=True,
    help="Output directory for annotated heatmaps.")
@click.option(
    '-m', '--metadata_colors_toml', type=click.Path(dir_okay=False, exists=True),
    help="TOML file with metadata_colors specification. If not provided, visually distinct colors will be chosen automatically.")
@click.option(
    '--show_sample_labels', is_flag=True,
    help="Show sample labels on usage heatmap")
@click.option(
    '--subsample', type=int, default=None,
    help="Randomly subsample this many samples/cells (without replacement) for usage heatmaps. Defaults to no subsetting.")
@click.option(
    "-k", type=int, multiple=True, default=[],
    help="Specify individual ranks (k values) to analyze. Multiple may be selected like this: -k 2 -k 3. Defaults to all k.")
@click.option(
    "-s", "--subset_metadata", multiple=True,
    help="Specify metadata fields to show as annotation tracks. Multiple may be selected like this: -s layer1 -s layer2. Defaults to all."
)
def cmd_usage_heatmap(input_h5ad, output_dir, metadata_colors_toml, show_sample_labels, subsample, k, subset_metadata):
    """
    Create a heatmap of program usage with annotated samples
    """
    try:
        import PyComplexHeatmap
    except ImportError:
        
        logging.error("PyComplexHeatmap is not installed. Please install using:\n\n\t"
                      "pip install PyComplexHeatmap\n"
                      )
        sys.exit(1)


    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_h5ad(input_h5ad)
    
    # Subsample the data
    if subsample is not None:
        if subsample < dataset.adata.n_obs:
            i = np.random.RandomState(seed=1).choice(dataset.adata.n_obs, subsample, replace=False)
            dataset.adata = dataset.adata[i]
        else:
            logging.warning(f"The number of observations in the dataset is {dataset.adata.n_obs}, but subsample is set to {subsample}. Subsampling disabled.")

    if not dataset.has_cnmf_results:
        logging.error("cNMF results have not been imported into .h5ad file. Ensure that you have run `mosaicmpi postprocess` before annotating programs.")
        sys.exit(1)

    # get and check k values
    kval_options = dataset.adata.uns["kvals"].index.to_list()
    kval_option_str = ", ".join([str(k) for k in kval_options])
    if k:
        kvals = k
        bad_kvals = [kval for kval in kvals if kval not in kval_options]
        bad_kval_str = ", ".join([str(b) for b in bad_kvals])
        if bad_kvals:
            logging.error(f"The following k values were specified but not available as factorization solutions in the dataset: {bad_kval_str}. "
                          f"Please choose from the following options: {kval_option_str}")
            sys.exit(1)
    else:
        kvals = kval_options

    # subset metadata
    if subset_metadata:
        metadata_columns = dataset.get_metadata_df().columns
        bad_subset = [m for m in subset_metadata if m not in metadata_columns]
        bad_subset_str = ", ".join(bad_subset)
        metadata_columns_str = ", ".join(dataset.get_metadata_df().columns)
        if bad_subset:
            logging.error(f"The following metadata fields were selected using --subset_metadata but are not available in the dataset: {bad_subset_str}. "
                          f"Please choose from the following options: {metadata_columns_str}")
    else:
        # change empty list [] to None
        subset_metadata = None
    
    # get metadata colors
    if metadata_colors_toml:
        colors = Colors.from_toml(metadata_colors_toml)
    else:
        colors = Colors()
    colors.add_missing_metadata_colors(dataset)
    colors.to_toml(os.path.join(output_dir, "metadata_colors.toml"))
    
    # plot legend
    fig = colors.plot_metadata_colors_legend()
    utils.save_fig(fig, os.path.join(output_dir, "metadata_legend"))

    # create annotated plots for each k
    for kval in kvals:
        logging.info(f"Creating annotated usage heatmap for k={kval}")
        cnmf_name = dataset.adata.uns["cnmf_name"]
        title = f"{cnmf_name} k={kval}"
        if subsample:
            title += f"\nsubsampled to {subsample} observations"
        filename = os.path.join(output_dir, f"{cnmf_name}.usages.k{kval:03}")
        fig = plot_usage_heatmap(
            dataset=dataset, k=kval, subset_metadata=subset_metadata, colors=colors, title=title,
            cluster_samples=True, cluster_programs=False, show_sample_labels=show_sample_labels)
        utils.save_fig(fig, filename, target_dpi=200, formats="pdf")
        plt.close(fig)


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

@click.command(name="create-config")
@click.option('-i', '--input_h5ad', type=click.Path(exists=True, dir_okay=False), multiple=True, help=".h5ad file with cNMF results. Can be used multiple times to specify one or more datasets from which to create a config.toml file.")
@click.option('-o', '--output_toml', type=click.Path(exists=False, dir_okay=False), required=False, help="Output .toml file for configuring integration.")
def cmd_create_config(input_h5ad, output_toml):
    """
    Creates a TOML config file with default parameters to be used as input for `mosaicmpi integrate`.
    """
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
    # Features and HVF subsets: tables and UpSet plots #TODO: Upset plots
    df = integration.get_hvf_overlap_table()
    df.to_csv(os.path.join(output_dir, "hvfeatures.txt"), sep="\t")

    df = integration.get_features_overlap_table()
    df.to_csv(os.path.join(output_dir, "features.txt"), sep="\t")
    
    if integration.n_datasets > 1:
        fig = plot_integration_hvf_upset(integration)
        utils.save_fig(fig, os.path.join(output_dir, "hvfeatures_upsetplot"), target_dpi=200, formats=config.plot_formats)
        
        fig = plot_features_upset(integration)
        utils.save_fig(fig, os.path.join(output_dir, "features_upsetplot"), target_dpi=200, formats=config.plot_formats)

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
    
@click.command(name="ssgsea")
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, show_default=True,
    help="Output directory for ssgsea results")
@click.option('-n', '--pkl_file', type=click.Path(exists=True, dir_okay=False), multiple=False,
    help="Path to network_integration.pkl.gz file from `mosaicmpi integrate` step")
@click.option('-i', '--h5ad_file', type=click.Path(exists=True, dir_okay=False), multiple=False,
    help="Path to .h5ad file from `mosaicmpi postprocess`")
@click.option('-g', '--gene_sets', type=str, required=True, default = "GO_Biological_Process_2023", show_default=True,
    help="Path to GMT file with gene sets or Enrichr Library name.")
@click.option("--min_intersection", type=int, default=5, show_default=True, 
    help="Minimum intersection size for gene sets")
@click.option("--max_intersection", type=int, default=500, show_default=True, 
    help="Minimum intersection size for gene sets")
@click.option("--cmap", type=str, default="RdBu_r",
              help="matplotlib colormap name for heatmap plots.")
@click.option("--vmin", type=float, default=-0.5,
              help="minimum NES for heatmap plots")
@click.option("--vmax", type=float, default=0.5,
              help="maximum NES for heatmap plots")

@click.option("--no_plot", is_flag=True,
              help="Skip plotting geneset significance heatmaps")
@click.option('--cpus', type=int, default=cpus_available, show_default=True,
    help="Number of CPUs for MP-enabled tasks")
def cmd_ssgsea(output_dir, pkl_file, h5ad_file, gene_sets, min_intersection, max_intersection, cmap, vmin, vmax, no_plot, cpus):
    """
    Compute and plot ssGSEA Normalized Enrichment Scores (NES) for mosaicMPI programs. If a network_integration.pkl file
    is provided, ssGSEA is performed on reprepresentative programs from an integration. If a .h5ad file is provided, ssGSEA
    is performed on all programs.
    """
    try:
        import gseapy
    except ImportError:
        
        logging.error("gseapy is not installed. Please install using:\n\n\t"
                      "# if you have conda (MacOS_x86-64 and Linux only)\n\t"
                      "conda install -c bioconda gseapy\n\n\t"
                      "# Windows and MacOS_ARM64(M1/2-Chip)\n\t"
                      "pip install gseapy\n"
                      )
        sys.exit(1)

    from .genesets import program_ssgsea, order_genesets

    # create directory structure, warn if not empty
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        logging.warning(f"{output_dir} is not empty. Files may be overwritten.")

    # write to log file
    utils.start_logging(os.path.join(output_dir, "logfile.txt"))

    logging.info(f"Running ssGSEA...")
    if h5ad_file and not pkl_file:
        # run ssGSEA on all programs from a .h5ad file
        output_filename = "program_nes.txt"
        dataset = Dataset.from_h5ad(h5ad_file)
        programs = dataset.get_programs()
        
    elif pkl_file and not h5ad_file:
        # run ssGSEA on representative programs for each dataset in a network_integration.pkl file
        output_filename = "representative_program_nes.txt"
        network = Network.from_pkl(pkl_file)
        programs = network.get_representative_programs()
    else:
        logging.error("mosaicmpi ssgsea requires either a factorized dataset (.h5ad) file or a network_integration.pkl.gz file to run ssGSEA on programs.")
        sys.exit(1) 
        
    # run ssGSEA
    result = program_ssgsea(program_df=programs, gene_sets=gene_sets, min_intersection=min_intersection, max_intersection=max_intersection, cpus=cpus)
    result.prog_nes.to_csv(os.path.join(output_dir, output_filename), sep="\t")

    if h5ad_file and not pkl_file:
        # create ssGSEA NES heatmaps separately for each k
        n_k = dataset.adata.uns["kvals"].index.size
        for k in tqdm(dataset.adata.uns["kvals"].index, total=n_k, unit="k", desc="Plotting heatmaps"):
            df = result.prog_nes[k].dropna(how="all")
            df = df.sort_index(axis=1)
            df = order_genesets(df)
            df.to_csv(os.path.join(output_dir, f"ordered_genesets_k{k}.txt"), sep="\t")
            if not no_plot:
                fig, figlegend = plot_geneset_heatmap(df=df, cmap=cmap, vmin=vmin, vmax=vmax)
                ax = fig.axes[0]
                ax.set_title("ssGSEA NES\n" + gene_sets)
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
        # create ssGSEA NES heatmaps for all representative programs
        df = result.prog_nes.droplevel(axis=1, level=[2, 3]).dropna(how="all").loc[:, network.ordered_community_names]
        df = order_genesets(df)
        df.to_csv(os.path.join(output_dir, f"ordered_genesets.txt"), sep="\t")
        if not no_plot:
            logging.info("Plotting NES for representative programs...")
            fig, figlegend = plot_geneset_heatmap(df=df, cmap=cmap, vmin=vmin, vmax=vmax)
            ax = fig.axes[0]
            ax.set_title("ssGSEA NES\n" + gene_sets)
            ax.set_xlabel("")
            ax.set_ylabel("")
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('#aaaaaa') 
            ax.set_xlabel(f"Community - Dataset\n(Representative Program)")
            figlegend.savefig(os.path.join(output_dir, f"ordered_genesets.legend.pdf"))
            fig.savefig(os.path.join(output_dir, f"ordered_genesets.pdf"))
            plt.close(fig)
            plt.close(figlegend)

            # for smaller numbers of gene sets, create a better-formatted plot that emphasizes similarities and differences between datasets.
            if result.prog_nes.index.size < 200:
                fig, figlegend = plot_representative_program_nes(network=network, rep_nes=result.prog_nes, cmap=cmap, vmin=vmin, vmax=vmax, limit_geneset_label_length=100)
                utils.save_fig(fig, os.path.join(output_dir, "representative_program_nes"))
                utils.save_fig(figlegend, os.path.join(output_dir, "representative_program_nes.legend"))
            
    
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

    try:
        from gprofiler import GProfiler
    except ImportError:
        logging.error("gprofiler-official is not installed. Please install using:\n\n\t"
                      "conda install -c bioconda gprofiler-official"
                      )
        sys.exit(1)

    from .genesets import program_gprofiler, order_genesets

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
        n_k = dataset.adata.uns["kvals"].index.size
        for k in tqdm(dataset.adata.uns["kvals"].index, total=n_k, unit="k", desc="Plotting heatmaps"):
            df = result.summary["-log10pval"][k].dropna(how="all").fillna(0)
            df = df.sort_index(axis=1)
            df = order_genesets(df)
            df.to_csv(os.path.join(output_dir, f"ordered_genesets_k{k}.txt"), sep="\t")
            if not no_plot:
                fig, figlegend = plot_geneset_heatmap(df=df, cmap=cmap, vmin=vmin, vmax=vmax)
                ax = fig.axes[0]
                ax.set_title("g:Profiler -log10(p)\n" + ",".join(gene_sets))
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
                fig, figlegend = plot_geneset_heatmap(df=df)
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

@click.command(name="transfer-labels")
@click.option("-o", '--output_dir', type=click.Path(file_okay=False, exists=False), default=os.getcwd(), show_default=True,
    help="Output directory for results.")
@click.option('-n', '--pkl_file', type=click.Path(exists=True, dir_okay=False), required=True,
    help="Path to network_integration.pkl.gz file from `mosaicmpi integrate` step")
@click.option('-s', '--source', type=str, multiple=True,
    help="Only calculate Name of source dataset for sample labels.")
@click.option('-d', '--dest', type=str, multiple=True,
    help="Name of destination dataset")
@click.option('-c', '--categories', type=str, multiple=True, default=[], show_default=True,
    help="Name of categorical metadata field to transfer")
@click.option('-a', '--annotate', type=str, required=False, multiple=True,
    help="Annotate transfer heatmap using categorical data layer(s) from destination dataset")
@click.option('-m', '--metadata_colors_toml', type=click.Path(dir_okay=False, exists=True), required=False,
    help="TOML file with metadata_colors specification. If not provided, visually distinct colors will be chosen automatically.")
def cmd_transfer_labels(output_dir, pkl_file, source, dest, categories, annotate, metadata_colors_toml):
    """Transfer labels from a source to a destination dataset.
    """ 
    network = Network.from_pkl(pkl_file)
    os.makedirs(output_dir, exist_ok=True)

    transfer_df = network.transfer_labels(source=(source if source else None),
                                          dest=(dest if dest else None),
                                          categories=(categories if categories else None))
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
    if not categories:
        categories = network.integration.get_metadata_df(subset_datasets=source, include_numerical=False).columns
    
    total = len(source) * len(dest) * len(categories)

    for s, d, c in tqdm(product(source, dest, categories), total=total, unit="plot", desc="Creating plots"):
        if len(network.integration.datasets[s].get_metadata_df()[c].unique()) > 1:
            cgrid = plot_metadata_transfer(network=network, source=s, dest=d, categories=c, annotate=annotate, colors=colors)
            cgrid.fig.suptitle(f"source: {s}, dest: {d}, categories: {c}")
            cgrid.savefig(os.path.join(output_dir, f"s.{s}_d.{d}_l.{c}.pdf"))

    if annotate:
        # create legends for annotation tracks
        width = 3 * len(annotate)
        height = max([1 + len(colors.get_metadata_colors(a)) / 4 for a in annotate])
        fig, axes = plt.subplots(1, len(annotate), figsize=[width, height], layout="constrained", squeeze=False)
        for categories, ax in zip(annotate, axes[0]):
            colors.plot_metadata_colors_legend(categories=categories, ax=ax)
        utils.save_fig(fig, os.path.join(output_dir, "legend"), formats=("pdf", "png"), target_dpi=400, facecolor='white')


cli.add_command(cmd_txt_to_h5ad)
cli.add_command(cmd_update_h5ad_metadata)
cli.add_command(cmd_impute_knn)
cli.add_command(cmd_impute_zeros)
cli.add_command(cmd_check_h5ad)
cli.add_command(select_hvf)
cli.add_command(cmd_initialize_cnmf)
cli.add_command(cmd_factorize)
cli.add_command(cmd_postprocess)
cli.add_command(cmd_usage_heatmap)
cli.add_command(cmd_map_gene_ids)
cli.add_command(cmd_create_config)
cli.add_command(cmd_integrate)
cli.add_command(cmd_ssgsea)
cli.add_command(cmd_gprofiler)
cli.add_command(cmd_compare_integrations)
cli.add_command(cmd_transfer_labels)

# select_hvf subcommands
select_hvf.add_command(cmd_select_hvf_default)
select_hvf.add_command(cmd_select_hvf_stdeconvolve)
select_hvf.add_command(cmd_select_hvf_cnmf)
select_hvf.add_command(cmd_select_hvf_custom)