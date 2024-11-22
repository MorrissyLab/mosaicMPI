
from .dataset import Dataset
from .integration import Integration
from .colors import Colors
from .network import Network, compare_community_jaccard_similarity
from . import utils


from collections.abc import Iterable, Collection, Mapping
from typing import Union, Optional, Literal, Dict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch, Arc
import matplotlib.pyplot as plt
import upsetplot
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx


#########################
# Dataset-related plots #
#########################

def plot_feature_missingness(dataset: Dataset, ax: Optional[Axes] = None, proportion=False):
        
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=[10, 4], layout="constrained")
    else:
        ax_plot = ax

    if proportion:
        stat = "missingness"
    else:
        stat = "missing_values"

    if proportion:
        missingness = dataset.adata.var["missingness"]
        missingness.plot.hist(bins=100, color="k", ax=ax_plot)
    else:
        missing_values = dataset.adata.var["missing_values"]
        missing_values.value_counts().sort_index().plot.bar(width=0.9, color="k", ax=ax_plot)
    ax_plot.set_ylabel("Features")
    ax_plot.set_xlabel(stat)

    if ax is None:
        return fig


def plot_feature_dispersion(dataset: Dataset,
                            show_selected: bool = False,
                            show_model_curve: bool = True,
                            y_unit: Literal["log_variance", "odscore", "log_odscore", "vscore", "log_vscore"] = "log_variance",
                            modelled_features: bool = True,
                            unmodelled_features: bool = False,
                            ax: Optional[Axes] = None):

    
    df = dataset.adata.var
    df = df.sort_values("mean")
    model_curve_data = df.copy()

    if not modelled_features and not unmodelled_features:
        raise ValueError("at least one of modelled_features and unmodelled_features must be True.")
    elif modelled_features and not unmodelled_features:
        df = df[~df["odscore_excluded"]]
    elif not modelled_features and unmodelled_features:
        df = df[df["odscore_excluded"]]
    elif modelled_features and unmodelled_features:
        pass

    if "log_odscore" not in df.columns:  # required for compatibility with older h5ad files which do not have this column
        df["log_odscore"] = np.log10(df["odscore"])
        df["log_vscore"] = np.log10(df["vscore"])

    if ax is None:
        fig, ax_plot = plt.subplots(figsize=[4, 4], layout="tight")
    else:
        ax_plot = ax

    
    if show_selected:
        sns.histplot(df, x="log_mean", y=y_unit, hue="selected", bins=[100,100], ax=ax_plot, alpha=0.5, palette={True: "red", False: "blue"})
        ax_plot.legend(handles=[
            Patch(color="blue", alpha=0.5, label="False"),
            Patch(color="red", alpha=0.5, label="True"),
            Line2D([0], [0], color='green', label="model")
        ], title="selected")
    else:
        sns.histplot(df, x="log_mean", y=y_unit, bins=[100,100], ax=ax_plot, color="blue")
    
    # Show the model curve from the GAM
    if show_model_curve:
        if y_unit == "log_variance":
            ax_plot.plot(model_curve_data["log_mean"], model_curve_data["gam_fittedvalues"], color="green")
        elif y_unit == "odscore":
            ax_plot.axhline(1)
        elif y_unit == "log_odscore":
            ax_plot.axhline(0)
        else:
            raise ValueError('show_model_curve must be False if the y-unit is "vscore" or "log_vscore".')
    
    ax_plot.set_xlabel("log10(mean)")
    if y_unit == "log_variance":
        ax_plot.set_ylabel("log10(variance)")
    elif y_unit == "odscore":
        ax_plot.set_ylabel("odscore")
    elif y_unit == "log_odscore":
        ax_plot.set_ylabel("log10(odscore)")
    elif y_unit == "vscore":
        ax_plot.set_ylabel("vscore")
    elif y_unit == "log_vscore":
        ax_plot.set_ylabel("log10(vscore)")


    if ax is None:
        return fig


def plot_feature_overdispersion_histogram(dataset: Dataset,
                                          show_selected: bool = False,
                                          y_unit: Literal["odscore", "log_odscore"] = "log_variance",
                                          ax: Optional[Axes] = None):
    df = dataset.adata.var.sort_values("mean")
    if "log_odscore" not in df.columns:  # required for compatibility with older h5ad files which do not have this column
        df["log_odscore"] = np.log10(df["odscore"])
    
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=[4, 4], layout="tight")
    else:
        ax_plot = ax


    ax_plot.set_title("od-score Distribution")
    if df["odscore"].notnull().any():
        if show_selected:
            sns.histplot(df, x="odscore", hue="selected", bins=100, linewidth=0, ax=ax_plot, palette={True: "red", False: "blue"})
            ax_plot.legend(handles=[
                Patch(color="blue", alpha=0.5, label="False"),
                Patch(color="red", alpha=0.5, label="True")
            ], title="selected")
        else:
            sns.histplot(df, x=y_unit, bins=100, linewidth=0, ax=ax_plot, color="blue")
    
    if ax is None:
        return fig


def plot_stability_error(dataset: Dataset, figsize=(6, 4)):
    '''
    Borrowed from Alexandrov Et Al. 2013 Deciphering Mutational Signatures
    publication in Cell Reports
    '''
    
    stats = dataset.adata.uns["kvals"]

    fig, ax1 = plt.subplots(figsize=figsize, layout="tight")
    ax2 = ax1.twinx()


    ax1.plot(stats.index, stats.stability, 'o-', color='b')
    ax1.set_ylabel('Stability', color='b', fontsize=15)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2.plot(stats.index, stats.prediction_error, 'o-', color='r')
    ax2.set_ylabel('Error', color='r', fontsize=15)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax1.set_xlabel('Number of Components (k)', fontsize=15)
    ax1.grid('on')
    return fig



def annotated_heatmap(
        data, title, metadata=None, metadata_colors=None, missing_data_color="#BBBBBB", heatmap_cmap='YlOrRd', 
        row_cluster=True, col_cluster=True, plot_col_dendrogram=True, show_sample_labels=True, ylabel=None, cbar_label=None):
    n_columns = data.shape[1]
    if metadata is None:
        n_metadata_columns = 0
    else:
        n_metadata_columns = metadata.shape[1]

    fig = plt.figure(figsize=[20, 2 + n_columns/3 + n_metadata_columns/4])
    fig.suptitle(title, fontsize=14)
    gs0 = mpl.gridspec.GridSpec(2,2, figure=fig,
                                    height_ratios=[n_columns/3, 1 + n_metadata_columns/4], hspace=0.05,
                                    width_ratios=[5,1], wspace=0.05)
    
    # subdivide heatmap and dendrogram
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs0[0],
                                                    height_ratios=[1,n_columns],
                                                    hspace=0)

    # Heatmap
    ax_heatmap = fig.add_subplot(gs1[1])
    ax_col_dendrogram = fig.add_subplot(gs1[0], sharex=ax_heatmap)
    ax_col_dendrogram.set_axis_off()
    # ax_col_dendrogram.set_xlim(0, 10 * data.shape[1])

    # HAC clustering (compute linkage matrices)
    if col_cluster:
        col_links = linkage(data.dropna(axis=1), method='average', metric='euclidean')
        col_dendrogram = dendrogram(col_links, color_threshold=0, ax=ax_col_dendrogram, no_plot=not plot_col_dendrogram)
        xind = np.array(col_dendrogram['leaves'])
    else:
        xind = np.arange(0, data.shape[0])
    
    ax_col_dendrogram.set_xticks(np.arange(5, xind.shape[0] * 10 + 5, 10))
    ax_col_dendrogram.set_xticklabels(xind)

    
    if row_cluster:
        row_links = linkage(data.T, method='average', metric='euclidean')
        row_dendrogram = dendrogram(row_links, no_plot=True)
        yind = row_dendrogram['leaves']
    else:
        yind = np.arange(0, data.shape[1])

    xmin,xmax = ax_col_dendrogram.get_xlim()
    im_heatmap = ax_heatmap.imshow(data.iloc[xind,yind].T, aspect='auto', extent=[xmin,xmax,0,1], cmap=heatmap_cmap, vmin=0, vmax=1, interpolation='none')
    ax_heatmap.set_yticks((data.columns.astype("int").to_series() - 0.5).div(data.shape[1]))
    ax_heatmap.set_yticklabels(data.columns[yind][::-1])
    ax_heatmap.set_ylabel(ylabel, rotation=0, ha='right', va='center')
    ax_heatmap.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    

    # Annotations
    if metadata is not None:
        # data and metadata must have the same index
        metadata = metadata.loc[data.index]
        gs2 = mpl.gridspec.GridSpecFromSubplotSpec(metadata.shape[1], 1, subplot_spec=gs0[2])
        for i, (track, annot) in enumerate(metadata.items()):
            ax = fig.add_subplot(gs2[i], sharex=ax_heatmap)
            ax.set_facecolor(missing_data_color)
            if pd.api.types.is_categorical_dtype(annot):
                # convert categorical to object dtype to make mapping to colors easier
                annot = annot.iloc[xind].astype("object")

            if pd.api.types.is_object_dtype(annot):
                ordered_rgb = annot.iloc[xind]
                ordered_rgb[~ordered_rgb.isin(metadata_colors[track])] = np.nan  # omit samples for which no color exists
                ordered_rgb = ordered_rgb.replace(metadata_colors[track])
                if ordered_rgb.isnull().any():
                    ordered_rgb = ordered_rgb.fillna(missing_data_color)
                ordered_rgb = ordered_rgb.map(mpl.colors.to_rgb)
                ordered_rgb = np.array([list(rgb) for rgb in ordered_rgb])
                ax.imshow(np.stack([ordered_rgb, ordered_rgb]), aspect='auto', extent=[xmin,xmax,0,1], interpolation='none')
            else:
                ax.imshow(np.stack([annot.iloc[xind],annot.iloc[xind]]), aspect='auto', extent=[xmin,xmax,0,1], cmap='Blues', interpolation='none')
            ax.set_yticks([])
            ax.set_ylabel(track, rotation=0, ha='right', va='center')
            if ax.get_subplotspec().is_last_row():
                if show_sample_labels:
                    # ax.set_xticks(np.linspace(0, 1, data.shape[0], endpoint=False) + 1/(2 * data.shape[0]))
                    ax.set_xticklabels(data.index[xind], rotation=90)
                else:
                    ax.set_xticks([])
            else:
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # Colormap for heatmap
    ax = fig.add_subplot(gs0[1])
    ax.set_axis_off()
    plt.colorbar(im_heatmap, ax=ax, location="top", label=cbar_label)
    return fig

def plot_usage_heatmap(dataset: Dataset, k: Optional[int], colors: Colors, normalize_usage: bool = True, subset_metadata = None, subset_samples = None, title: str = None, cluster_programs = False, cluster_samples = True, show_sample_labels = True):
    assert dataset.has_cnmf_results
    df = dataset.get_usages(k=k, normalize=normalize_usage)
    if subset_samples is not None:
        df = df.loc[subset_samples]
    samples = df.index.to_series()
    metadata = dataset.get_metadata_df().loc[samples]
    if subset_metadata is not None:
        metadata = metadata.loc[:, subset_metadata]
    metadata_colors = {col: colors.get_metadata_colors(col) for col in metadata.columns}
    fig = annotated_heatmap(data=df, metadata=metadata,
                            metadata_colors=metadata_colors, 
                            missing_data_color=colors.missing_data_color, 
                            title=title,
                            row_cluster=cluster_programs,
                            col_cluster=cluster_samples,
                            show_sample_labels=show_sample_labels,
                            plot_col_dendrogram=True,
                            ylabel="Program")
    return fig

def plot_sample_numbers(dataset: Dataset, layer: str, figsize = None, ax = None):
    """Bar plot of sample numbers in each category based on a single categorical layer.

    :param dataset: dataset
    :type dataset: :class:`~mosaicmpi.dataset.Dataset`
    :param layer: categorical metadata layer
    :type layer: str
    :param figsize: figure size, defaults to None
    :type figsize: Collection[float, float], optional
    :param ax: axes object for adding to an existing plot, defaults to None
    :type ax: Axes, optional
    :return: figure or None
    :rtype: Union[Figure, None]
    """


    data = dataset.get_metadata_df(include_numerical=False)[layer].value_counts()
    
    if ax is None:
        if figsize is None:
            figsize = [1 + len(data) / 4, 4]
        fig, ax_plot = plt.subplots(figsize=figsize, layout="constrained")
    else:
        ax_plot = ax

    
    data.plot.bar(ax = ax_plot, width=0.8, color = "#666666")
    ax_plot.set_ylabel("# samples")

    return fig


#############################
# Integration-related plots #
#############################

def plot_program_correlation_matrix(integration: Integration, colors, figsize=(20,20), cmap="RdBu_r", hide_program_labels=False):
    ds_color_track = integration.corr_matrix.index.get_level_values(0).map(colors.dataset_colors)
    ds_color_track = [mpl.colors.to_rgb(c) for c in ds_color_track]
    cg = sns.clustermap(integration.corr_matrix, figsize=figsize,
                        cmap=cmap, center=0, vmin=-1, vmax=1,
                        row_colors=ds_color_track, col_colors=ds_color_track)
    cg.ax_heatmap.set_ylabel("")
    cg.ax_heatmap.set_xlabel("")
    if hide_program_labels:
        cg.ax_heatmap.set_xticks([])
        cg.ax_heatmap.set_yticks([])
    cg.ax_col_dendrogram.set_title("Program correlations")
    return cg.figure


def plot_rank_reduction(integration: Integration, figsize=None):
    n_ranks = integration.k_table.shape[0]
    n_datasets = len(integration.datasets)
    if figsize is None:
        figsize = [n_ranks * n_datasets / 2, 3]
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, squeeze=False)
    for dataset_name, ax in zip(integration.datasets, axes[0]):
        df = integration.k_table.loc[:, dataset_name]
        df = df.copy()    
        df["max_k"] = df.index
        ax.set_ylim([-1, 1])
        ax.set_xlim([df["max_k"].min() - 1, df["max_k"].max() + 1])
        ax.axhline(integration.max_median_corr, color="red")
        ax.set_title(dataset_name)
        ax.set_xticks(df["max_k"])
        sns.lineplot(data=df, x="max_k", y="max_k_median_corr", hue="max_k_filter_pass", ax=ax, marker="o")
        ax.get_legend().remove()
        plt.tight_layout()
    return fig

def plot_pairwise_corr(integration: Integration, subplot_size: Collection = [3, 3.5], overlaid=False, bins=50, inside_padding = 0.02, selected_k_filter = True, max_k_filter = True, sharey: bool = False) -> Figure:
    """Plot histograms of the pairwise correlation distribution for each dataset pair.

    :param integration: Integration object
    :type integration: :class:`~mosaicmpi.integration.Integration`
    :param subplot_size: width and height of each subplot, defaults to [3, 3.5]
    :type subplot_size: Collection, optional
    :param overlaid: Overlay negative and positive parts to compare the distribution, defaults to False
    :type overlaid: bool, optional
    :param bins: Number of bins for histogram, defaults to 50
    :type bins: int, optional
    :param inside_padding: Amount to shrink data within the axes, defaults to 0.02
    :type inside_padding: float, optional
    :param sharey: set y-axis limits to a common range, defaults to False
    :type sharey: bool, optional
    :return: figure
    :rtype: Figure
    """
    tril = integration.get_corr_matrix_lowertriangle(selected_k_filter=selected_k_filter, max_k_filter=max_k_filter)
    sps_width, sps_height = subplot_size
    n_datasets = integration.n_datasets
    fig, axes = plt.subplots(n_datasets, n_datasets,
                             figsize=[sps_width * n_datasets, sps_height * n_datasets], sharey=sharey, squeeze=False, layout="tight")
    fig.supxlabel("Correlation")
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            ax = axes[row,col]
            if row == 0 and col == integration.n_datasets - 1:
                handles = [
                    Rectangle((0, 0), width = 0.8, height = 0.8, facecolor='red', label="Included"),
                    Rectangle((0, 0), width = 0.8, height = 0.8, facecolor='grey', label="Excluded")
                ]
                ax.legend(handles=handles, loc='center', frameon=False)
                ax.set_axis_off()
            elif row < col:
                ax.set_axis_off()
            else:
                corr = pd.Series(tril.loc[dataset_row, dataset_col].values.flatten()).dropna()
                max_corr = corr.max()
                if integration.pairwise_thresholds is not None:
                    min_corr = integration.pairwise_thresholds.loc[(dataset_row, dataset_col)]
                    hist_kwargs = {
                        "hue":[("Included" if c else "Excluded") for c in (corr > min_corr)],
                        "palette": {"Included": "red", "Excluded": "gray"},
                        "hue_order": ["Excluded", "Included"]
                    }
                    included_fraction = (corr > min_corr).sum() / corr.shape[0]
                    ax.text(x=0.01, y=1.01, s=f"AUC = {included_fraction:.3f}\nrange = {min_corr:.3f}-{max_corr:.3f}", size=8, ha='left', va='bottom', transform=ax.transAxes, color="black")
                else:
                    hist_kwargs = {"color": "gray"}
                sns.histplot(x=corr, ax=ax,legend=False, bins=bins, linewidth=0, multiple="stack", **hist_kwargs)

                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(-1 - inside_padding, 1 + inside_padding)

    for row in axes:
        for ax in row:
            if sharey:
                ymax = max([ax.get_ylim()[1] for row in axes for ax in row])
                ax.set_ylim((
                    - ymax * inside_padding,
                    ymax + ymax * inside_padding
                ))
            else:
                ymax = ax.get_ylim()[1]
                ax.set_ylim((
                    - ymax * inside_padding,
                    ymax + ymax * inside_padding
                ))

    return fig

def plot_pairwise_corr_overlaid(integration: Integration, subplot_size = [3, 3.5], bins=50, inside_padding = 0.02, selected_k_filter = True, max_k_filter = True, sharey: bool = False):
    tril = integration.get_corr_matrix_lowertriangle(selected_k_filter = selected_k_filter, max_k_filter = max_k_filter)
    n_datasets = integration.n_datasets
    sps_width, sps_height = subplot_size
    fig, axes = plt.subplots(n_datasets, n_datasets,
                             figsize=[sps_width * n_datasets, sps_height * n_datasets], 
                             sharey=sharey, squeeze=False, layout="tight")
    fig.supxlabel("Correlation")
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            ax = axes[row,col]
            if row == 0 and col == integration.n_datasets - 1:
                handles = [
                    Rectangle((0, 0), width = 0.8, height = 0.8, facecolor='red', alpha=0.4, label="Positive"),
                    Rectangle((0, 0), width = 0.8, height = 0.8, facecolor='blue', alpha=0.4, label="Negative")
                ]
                ax.legend(handles=handles, loc='center', frameon=False)
                ax.set_axis_off()
            elif row < col:
                ax.set_axis_off()
            else:
                corr = pd.DataFrame({"corr": tril.loc[dataset_row, dataset_col].values.flatten()}).dropna()
                max_corr = corr["corr"].max()
                corr["sign"] = (corr["corr"] >= 0).map({True: "Positive", False: "Negative"})
                corr["abscorr"] = corr["corr"].abs()
                sns.histplot(data=corr, x="abscorr", hue="sign", palette= {"Positive": "red", "Negative": "blue"}, bins=bins, alpha=0.4, linewidth=0,
                             hue_order= ["Negative", "Positive"], ax=ax,legend=False)


                # show min_corr as text in top left of plot and vertical line
                min_corr = integration.pairwise_thresholds.loc[(dataset_row, dataset_col)]
                included_fraction = (corr["corr"] > min_corr).sum() / corr.shape[0]  # could also show the quantile of the min_corr threshold
                ax.text(x=0.01, y=1.01, s=f"AUC = {included_fraction:.3f}\nrange = {min_corr:.3f}-{max_corr:.3f}", size=8, ha='left', va='bottom', transform=ax.transAxes, color="black")
                ax.axvline(min_corr, color="black")
                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(0 - inside_padding, 1 + inside_padding)

    for row in axes:
        for ax in row:
            if sharey:
                ymax = max([ax.get_ylim()[1] for row in axes for ax in row])
                ax.set_ylim((
                    - ymax * inside_padding,
                    ymax + ymax * inside_padding
                ))
            else:
                ymax = ax.get_ylim()[1]
                ax.set_ylim((
                    - ymax * inside_padding,
                    ymax + ymax * inside_padding
                ))

    return fig

def plot_overdispersed_features_upset(integration: Integration, figsize: Collection = (6, 4), show_counts: bool = False) -> Figure:
    """Plot overlaps of overdispersed features between datasets

    :param integration: integration object
    :type integration: :class:`~mosaicmpi.integration.Integration`
    :param figsize: width and height of figure, defaults to [6, 4]
    :type figsize: Collection, optional
    :return: figure
    :rtype: Figure
    """
    overdispersed_feature_lists = {dataset_name: dataset.overdispersed_genes for dataset_name, dataset in integration.datasets.items()}
    fig = Figure(figsize=figsize)
    upsetplot.UpSet(upsetplot.from_contents(overdispersed_feature_lists), show_counts=show_counts).plot(fig=fig)
    fig.suptitle("Overdispersed features")
    return fig

def plot_features_upset(integration: Integration, figsize=[6, 4], show_counts: bool = False):
    """Plot overlaps of features between datasets

    :param integration: integration object
    :type integration: :class:`~mosaicmpi.integration.Integration`
    :param figsize: width and height of figure, defaults to [6, 4]
    :type figsize: Collection, optional
    :return: figure
    :rtype: Figure
    """
    feature_lists = {dataset_name: list(dataset.adata.var.index) for dataset_name, dataset in integration.datasets.items()}
    fig = Figure(figsize=figsize)
    upsetplot.UpSet(upsetplot.from_contents(feature_lists), show_counts=show_counts).plot(fig=fig)
    fig.suptitle("Features")
    return fig

#####################
# SNS-related plots #
#####################

def plot_community_usage_heatmap(network: Network,
                                 colors: Colors,
                                 subset_metadata = None,
                                 subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                 subset_samples = None,
                                 title = None,
                                 cluster_programs = False,
                                 cluster_samples = True,
                                 show_sample_labels = True,
                                 prepend_dataset_colors = True):
    

    
    if subset_datasets is None:
        subset_datasets = network.integration.datasets.keys()
    elif isinstance(subset_datasets, str):
        subset_datasets = [subset_datasets]
    elif isinstance(subset_datasets, Iterable):
        pass
    else:
        raise ValueError
    
    df = network.get_community_usage()
    
    if subset_samples is not None:
        df = df.loc[subset_samples]
    
    df = df.loc[subset_datasets]
            
    metadata = network.integration.get_metadata_df()
    metadata = metadata[metadata.index.get_level_values(0).isin(subset_datasets)]
    if subset_metadata is not None:
        metadata = metadata.loc[:, subset_metadata]
    metadata = metadata.dropna(axis=1, how="all")

    metadata_colors = {col: colors.get_metadata_colors(col) for col in metadata.columns}
    if prepend_dataset_colors:
        metadata.insert(0, "dataset", metadata.index.get_level_values(0))
        metadata_colors["dataset"] = colors.dataset_colors
    
    
    fig = annotated_heatmap(data=df, metadata=metadata,
                            metadata_colors=metadata_colors, 
                            missing_data_color=colors.missing_data_color, 
                            title=title,
                            row_cluster=cluster_programs,
                            col_cluster=cluster_samples,
                            show_sample_labels=show_sample_labels,
                            plot_col_dendrogram=True,
                            ylabel="Community",
                            cbar_label="Normalized Program Usage")
    return fig


def plot_community_usage_per_sample(network: Network,
                                    colors: Colors,
                                    dataset_name: str,
                                    layer: str
                                    ) -> Figure:
    df = network.get_community_usage().loc[dataset_name].copy()
    df.index = df.index.map(network.integration.datasets[dataset_name].get_metadata_df()[layer])
    df = df.dropna(axis=1, how="all")

    df = df.sort_index()

    n_row = -(df.index.nunique() // -4)  # aka ceiling division
    n_col = 4

    fig, axes = plt.subplots(n_row, n_col, figsize=[3 * n_col, 3 * n_row], sharex=True, sharey=True, layout="tight")

    for i, (celltype, subdf) in enumerate(df.groupby(level=0)):
        ax = axes[i // n_col, i % n_col]
        sns.violinplot(data=subdf, linewidth=0.1, ax=ax, cut=0, color=colors.community_colors)
        ax.set_title(celltype)
        is_last_row = i // n_col == n_row - 1
        if not is_last_row:
            ax.set_xlabel("")
    
    return fig

def plot_community_contribution(network: Network, colors: Colors, figsize: Collection = None, highlight_central_program: bool = True, orientation = "horizontal"):
    """_summary_

    :param network: _description_
    :type network: :class:`~mosaicmpi.network.Network`
    :param colors: _description_
    :type colors: :class:`~mosaicmpi.colors.Colors`
    :param figsize: figure size, defaults to None
    :type figsize: Collection, optional
    :param highlight_central_program: _description_, defaults to True
    :type highlight_central_program: bool, optional
    :param orientation: _description_, defaults to "horizontal"
    :type orientation: str, optional
    :return: _description_
    :rtype: _type_
    """

    marker_style = {
        1: ("s", 30),  # 1 factor: square markers, size 30
        2: (2, 30),     # 2 factors: marker #2 (up tick), size 30
        }

    if highlight_central_program:
        central_ranks = pd.DataFrame(network.get_representative_program_ids()).reset_index().set_index(["Community", "dataset"])["k"]

    n_datasets = network.integration.n_datasets

    if figsize is None:
        if orientation == "horizontal":
            figsize = [2 + n_datasets * 6, 1 + len(network.communities)/4]
        elif orientation == "vertical":
            figsize= [4, (2 + len(network.communities)/4) * n_datasets]
        else:
            raise ValueError(f"{orientation} is not a valid orientation. Please choose from `horizontal` or `vertical`")

    if orientation == "horizontal":
        fig, axes = plt.subplots(1, n_datasets+1, figsize=figsize, sharex=False, sharey=True, layout="tight")
    elif orientation == "vertical":
        fig, axes = plt.subplots(n_datasets+1, 1, figsize=figsize, sharex=False, sharey=True, layout="tight")
    
    all_k_values = set()
    for selected_k_values in network.integration.selected_k.values():
        all_k_values |= set(selected_k_values)
    all_k_values = sorted(list(all_k_values))

    for dataset, ax in zip(network.integration.datasets, axes):
        selected_k_values = network.integration.selected_k[dataset]
        x_positions = [all_k_values.index(k) for k in selected_k_values]
        for y, community in enumerate(network.ordered_community_names):
            members = network.communities[community]
            counts = pd.Series([m.rpartition("|")[0] for m in members]).value_counts()
            
            # plot line if any factors are present
            line_x = []
            line_y = []
            for x, rank in zip(x_positions, selected_k_values):
                line_x.append(x)
                if f"{dataset}|{rank}" in counts.index:
                    line_y.append(y)
                else:
                    line_y.append(np.nan)
            ax.plot(line_x, line_y, color=colors.dataset_colors[dataset], linewidth=2)
            
            for count, style in marker_style.items():
                # plot different markers depending on how many factors are present:
                scatter_x = []
                scatter_y = []
                for x, rank in zip(x_positions, selected_k_values):
                    factor_prefix = f"{dataset}|{rank}"
                    if factor_prefix in counts.index and counts[factor_prefix] == count:
                        scatter_x.append(x)
                        scatter_y.append(y)
                ax.scatter(scatter_x, scatter_y, color=colors.dataset_colors[dataset], marker=style[0], s=style[1])

            if highlight_central_program and (community, dataset) in central_ranks:
                x = all_k_values.index(central_ranks[(community, dataset)])
                ax.add_patch(Rectangle((x - 0.4, y-0.4), width = 0.8, height = 0.8, edgecolor="k", facecolor='none'))
        ax.set_yticks(list(range(len(network.ordered_community_names))), labels=network.ordered_community_names)
        ax.set_xticks(x_positions, labels=selected_k_values)
        ax.set_xlim(-0.6, len(all_k_values) - 0.4)

        if orientation == "vertical":
            ax.set_ylabel("Community")
            if ax is axes[-2]:
                ax.set_xlabel("Rank (k)")
                ax.xaxis.set_tick_params(labelbottom=True)
        elif orientation == "horizontal":
            ax.set_xlabel("Rank (k)")
            if ax is axes[0]:
                ax.set_ylabel("Community")

        ax.set_title(dataset)

    # Add legend
    cbdrlegend = []
    cbdrlegend.append(Line2D([0],[0], marker='s', color='black', label="1 Program", markerfacecolor="black", markersize=8))
    cbdrlegend.append(Line2D([0],[0], marker=2, color='black', label="2 Programs", markerfacecolor="black", markersize=8))
    cbdrlegend.append(Line2D([0],[0], marker=None, color='black', label="3+ Programs", markerfacecolor="black", markersize=8))
    if highlight_central_program:
        cbdrlegend.append(Rectangle((0, 0), width = 0.8, height = 0.8, edgecolor="k", facecolor='none', label="Central Program"))
    axes[-1].legend(handles=cbdrlegend, loc='center', frameon=False)
    axes[-1].set_axis_off()
    return fig


###########################################
#   Helper functions for circle bar plots #
###########################################

def draw_circle_bar_plot(position, enrichments, colors, size, ax, scale_factor: float=1, draw_labels: bool=False, label_font_size: float=1):

    x, y = position
    previous = np.pi
    for color, (label, enrichment) in zip(colors, enrichments.items()):
        this = previous - 2 * np.pi / len(enrichments)
        if enrichment > 0:
            # calculate the points of the pie pieces
            radius = size * np.sqrt(enrichment * scale_factor)
            n_edges_on_arc = max(2, 200 // len(enrichments))
            x_shape  = np.array([0] + np.cos(np.linspace(previous, this, n_edges_on_arc)).tolist()) * radius + x
            y_shape  = np.array([0] + np.sin(np.linspace(previous, this, n_edges_on_arc)).tolist()) * radius + y
            xy_shape = np.column_stack([x_shape, y_shape])
            ax.add_patch(Polygon(xy_shape, fill=True, closed=True, color=color, linewidth=0))

            # text
            if draw_labels:
                a = (previous - np.pi / len(enrichments))
                x_offset = (size * 1.1) * np.cos(a)
                y_offset = (size * 1.1) * np.sin(a)
                ax.text(x+x_offset, y+y_offset, label, rotation=np.rad2deg(a), ha="left", va="center", rotation_mode='anchor', fontsize=label_font_size)
        previous = this

def draw_circle_bar_scale(position, size, ax, scale_factor, label_font_size = 8, linewidth=0.8, rings=[0.2, 0.5, 1]):
    x, y = position
    max_ring_radius = np.sqrt(np.max(rings)) * size
    ax.add_line(Line2D([x, x], [y, y+max_ring_radius], color="k", linewidth=linewidth))
    ax.add_line(Line2D([x, x - 0.5 * max_ring_radius],
                       [y, y + max_ring_radius * np.cos(np.deg2rad(30))], color="k", linewidth=linewidth))

    for ring in rings:
        arc_diameter = 2 * (np.sqrt(ring) * size)
        ax.add_patch(Arc(position, width=arc_diameter, height=arc_diameter, angle=0, theta1 = 90, theta2 = 120, linewidth=linewidth))
    # ax.scatter(position[0], position[1], s=linewidth*2, color="black", marker=".")  # dot at centre of arc
    for ring in rings:
        value = ring/scale_factor
        ax.text(
            x + size * 0.05,
            y + np.sqrt(ring) * size,
            f"{value:.2f}",
            fontsize=label_font_size,
            verticalalignment="center")


###########################
#  Program Network Plots  #
###########################


def plot_overrepresentation_program_network(network: Network,
                                        colors: Colors,
                                        layer: str,
                                        subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                        ax: Optional[Axes] = None,
                                        pie_size: float = 0.3,
                                        figsize: Collection = (9, 6),
                                        edge_weights: Optional[str] = None,
                                        edge_color: str = "#AAAAAA88",
                                        legend_pie_size: float = 0.1,
                                        show_legend: bool = True) -> Optional[Figure]:
    
    if show_legend:
        assert ax is None
    
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Programs")
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_aspect("equal")
    ax_legend.set_axis_off()
    
    if edge_weights is None or not network.program_graph.edges:
        width = 0.2
    else:
        width = np.array(list(nx.get_edge_attributes(network.program_graph, edge_weights).values()))
        if width.sum() > 0:
            width = 20 * width / np.max(width)

    nx.draw_networkx_edges(network.program_graph, pos=network.layout, edge_color=edge_color, ax=ax_plot, width=width)
    overrepresentation = network.integration.get_category_overrepresentation(subset_datasets=subset_datasets, layer=layer)
    overrepresentation = overrepresentation.fillna(0)
    max_or = np.max(overrepresentation.values.flatten())
    if max_or == 0:
        scale_factor = 1
    else:
        scale_factor = 1 / max_or
    
    color_list = overrepresentation.index.map(colors.get_metadata_colors(layer))
    for program, program_or in overrepresentation.items():
        node = "|".join((str(p) for p in program))

        if node in network.program_graph and program_or.any():
            draw_circle_bar_plot(position=network.layout[node],
                                 enrichments=program_or,
                                 scale_factor=scale_factor,
                                 colors=color_list,
                                 size=pie_size, ax=ax_plot)

    if show_legend:
        # Add legends
        draw_circle_bar_plot(
            position=(0, 0.5),
            enrichments=pd.Series(max_or, index=overrepresentation.index),
            colors=overrepresentation.index.map(colors.get_metadata_colors(layer)),
            scale_factor=scale_factor,
            size=legend_pie_size,
            draw_labels=True,
            label_font_size=6, ax=ax_legend)
        draw_circle_bar_scale(
            position=(0, -0.75),
            scale_factor=scale_factor,
            size=pie_size,
            label_font_size=8, ax=ax_legend)
        ax_legend.set_title(f"{layer}")
        ax_legend.text(0, -1, "overrepresentation", ha="center", va="top", fontsize=10, fontweight="regular")
    
    # assert ax_plot.get_xlim() == ax_plot.get_ylim()
    # assert ax_plot.get_ylim() == ax_legend.get_ylim()
    
    if ax is None:
        return fig


def plot_metadata_correlation_program_network(network: Network,
                                          colors: Colors,
                                          layer: str,
                                          subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                          ax: Optional[Axes] = None,
                                          figsize=(7, 6),
                                          node_size=100,
                                          edge_weights: Optional[str] = None,
                                          edge_color="#AAAAAA88",
                                          method: str = "pearson",
                                          cmap: str = "RdBu_r",
                                          show_legend=True) -> Optional[Figure]:
    
    if show_legend:
        assert ax is None
    
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [6, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title(f"Programs correlated with\n{layer}")
    ax_legend.set_aspect("equal")
    ax_legend.set_axis_off()
    
    if edge_weights is None or not network.program_graph.edges:
        width = 0.2
    else:
        width = np.array(list(nx.get_edge_attributes(network.program_graph, edge_weights).values()))
        if width.sum() > 0:
            width = 20 * width / np.max(width)

    nx.draw_networkx_edges(network.program_graph, pos=network.layout, edge_color=edge_color, ax=ax_plot, width=width)
    md_corr = network.integration.get_metadata_correlation(subset_datasets=subset_datasets, layer=layer, method=method)

    color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1, 1), cmap=cmap)
    node_colors = []
    for node in network.program_graph:
        program = utils.node_to_program(node)
        if program in md_corr.index:
            node_colors.append(color_map.to_rgba(md_corr[program]))
        else:
            node_colors.append(mpl.colors.to_rgba(colors.missing_data_color))
    nx.draw(network.program_graph, pos=network.layout, node_color=node_colors, node_size=node_size, linewidths=0, width=width, edge_color=edge_color, with_labels=False, ax=ax_plot, font_size=20)
    
    if show_legend:
        plt.colorbar(color_map, ax=ax_legend, location="left", anchor=(0, 1), panchor=(1, 1), shrink=0.5, fraction=0.15, label=f"{method.capitalize()} correlation")
    
    # assert ax_plot.get_xlim() == ax_plot.get_ylim()
    # assert ax_plot.get_ylim() == ax_legend.get_ylim()
    
    if ax is None:
        return fig


def plot_program_network_datasets(network: Network, colors: Colors, figsize = (9,6), edge_color = "#AAAAAA88", node_size = 30, node_size_kval = False, labels = False, ax = None):
     
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Programs")
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_dataset_colors_legend(ax=ax_legend)
    
    node_colors = []
    for node in network.program_graph:
        node_colors.append(colors.dataset_colors[node.split("|")[0]])

    # Labels without dataset names
    node_labels = {}
    for node in network.program_graph:
        node_labels[node] = node.partition("|")[2]

    # Node sizes inversely proportional to k
    if node_size_kval:
        sizes = {}
        selected_k_anyds = network.integration.k_table.loc[:, (slice(None), "selected_k")].any(axis=1)
        median_rank = min(selected_k_anyds[selected_k_anyds].index)

        for node in network.program_graph:
            sizes[node] = node_size / (int(node.split("|")[1]) + 0.5 - median_rank)
        node_sizes = [(sizes[n] if n in sizes else 0) for n in network.program_graph]
    else:
        node_sizes = node_size
    # Plot nodes colored by dataset
    nx.draw(network.program_graph,
            pos=network.layout,
            with_labels=labels,
            node_color=node_colors,
            labels=node_labels,
            node_size=node_sizes,
            linewidths=0,
            width=0.2,
            edge_color=edge_color,
            font_size=4, ax=ax_plot)
    return fig


def plot_program_network_communities(network: Network,
                                 colors: Colors,
                                 figsize = (9,6),
                                 edge_color = "#AAAAAA88",
                                 node_size = 30,
                                 node_size_kval = False,
                                 ax = None):
     
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Programs")
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_community_colors_legend(ax=ax_legend)
    
    node_colors = []
    for node in network.program_graph:
        if node in network.node_to_community:
            node_colors.append(colors.community_colors[network.node_to_community[node]])
        else:
            # if node has been pruned
            node_colors.append("#00000000")

    # Labels without dataset names
    labels = {}
    for node in network.program_graph:
        labels[node] = node.partition("|")[2]

    # Node sizes inversely proportional to k
    if node_size_kval:
        sizes = {}
        selected_k_anyds = network.integration.k_table.loc[:, (slice(None), "selected_k")].any(axis=1)
        median_rank = min(selected_k_anyds[selected_k_anyds].index)

        for node in network.program_graph:
            sizes[node] = node_size / (int(node.split("|")[1]) + 0.5 - median_rank)
        node_sizes = [(sizes[n] if n in sizes else 0) for n in network.program_graph]
    else:
        node_sizes = node_size
    # Plot nodes colored by dataset
    nx.draw(network.program_graph,
            pos=network.layout,
            with_labels=False,
            node_color=node_colors,
            labels=labels,
            node_size=node_sizes,
            linewidths=0,
            width=0.2,
            edge_color=edge_color,
            font_size=4, ax=ax_plot)
    if ax is None:
        return fig


def plot_program_network_nsamples(network: Network,
                             colors: Colors,
                             figsize: Collection = (9, 6),
                             discretize = False,
                             edge_color = "#AAAAAA88",
                             node_size = 30,
                             font_size=6,
                             ax = None):
     
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Programs")
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_dataset_colors_legend(ax=ax_legend)

    usages = network.integration.get_usages(discretize = discretize,
                                           normalize = True)


    if discretize:
        labels = usages[network.programs_in_graph].sum().apply(lambda x: str(int(x))).to_dict()
    else:
        labels = usages[network.programs_in_graph].sum().apply(lambda x: f"{x:.1f}").to_dict()
    labels = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in labels.items()}
    
    scale_factor = node_size / usages[network.programs_in_graph].sum().max()
    sizes = (usages[network.programs_in_graph].sum() * scale_factor).to_dict()
    sizes = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in sizes.items()}
    
    colors = [colors.dataset_colors[node.partition("|")[0]] for node in network.program_graph]

    node_sizes = [(sizes[n] if n in sizes else 0) for n in network.program_graph]
    nx.draw(network.program_graph, network.layout,
    with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, linewidths=0, width=0.2, edge_color=edge_color, font_size=font_size, ax=ax_plot)
    if ax is None:
        return fig


def plot_program_network_npatients(network: Network,
                               colors: Colors,
                               figsize: Collection = (9, 6),
                               edge_color = "#AAAAAA88",
                               node_size = 30,
                               font_size=6,
                               ax = None):
    
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
        ax_legend.set_axis_off()
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Programs")
    ax_legend.set_xlim([-0.5, 0.5])
    colors.plot_dataset_colors_legend(ax=ax_legend)

    usages = network.integration.get_usages(discretize = True).fillna(0).astype(bool)
    s2p = network.integration.sample_to_patient
    if s2p is None:
        raise ValueError("No samples have valid patient IDs. Make sure to set the patient_id_col property for each Dataset")
    usages = usages[usages.index.isin(s2p.index)]
    usages.index = usages.index.map(s2p)
    usages = usages.groupby(level=[0,1]).any()
        
    labels = usages[network.programs_in_graph].sum().apply(lambda x: str(int(x))).to_dict()
    labels = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in labels.items()}
    
    scale_factor = node_size / usages[network.programs_in_graph].sum().max()
    sizes = (usages[network.programs_in_graph].sum() * scale_factor).to_dict()
    sizes = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in sizes.items()}
    
    colors = [colors.dataset_colors[node.partition("|")[0]] for node in network.program_graph]

    node_sizes = [(sizes[n] if n in sizes else 0) for n in network.program_graph]
    nx.draw(network.program_graph, network.layout,
    with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, linewidths=0, width=0.2, edge_color=edge_color, font_size=font_size, ax=ax_plot)
    if ax is None:
        return fig


#################################
#   Community Network Plots #
#################################


def plot_community_network_summary(network: Network,
                                   colors: Colors,
                                   figsize = (4, 4),
                                   edge_color = "#AAAAAA88",
                                   label_edges = False,
                                   node_size = 500,
                                   ax: Axes = None):
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=figsize, layout="tight")
    else:
        ax_plot = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Communities")
    
    G = network.comm_graph
    if G.edges:
        width = np.array(list(nx.get_edge_attributes(G, "n_edges").values()))
        if width.sum() > 0:
            width = 20 * width / np.max(width)
    else:
        width = None
    sizes = np.array([len(network.communities[node]) for node in G.nodes])
    sizes = node_size * sizes / np.max(sizes)
    node_colors = [colors.community_colors[node] for node in G]
    nx.draw(G, pos=network.comm_layout, node_color=node_colors, node_size=sizes, linewidths=0, width=width, edge_color=edge_color, with_labels=True, ax=ax_plot, font_size=20)
    
    if label_edges:
        edge_labels = {(n1, n2): str(n_edges) for n1, n2, n_edges in G.edges(data="n_edges")}
        nx.draw_networkx_edge_labels(G, pos=network.comm_layout, edge_labels=edge_labels)
    
    if ax is None:
        return fig


def plot_metadata_correlation_community_network(network: Network,
                                      colors: Colors,
                                      layer: str,
                                      method: str = "pearson",
                                      subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                      figsize: Collection = (5, 4),
                                      edge_color = "#AAAAAA88",
                                      node_size = 500,
                                      cmap = "RdBu_r",
                                      ax: Optional[Axes] = None,
                                      show_legend: bool =True
                                      ) -> Optional[Figure]:
    
    if show_legend:
        assert ax is None
    
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [4, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title(f"Communities\n{layer}")
    ax_legend.set_aspect("equal")
    ax_legend.set_axis_off()
    
    if network.comm_graph.edges:
        width = np.array(list(nx.get_edge_attributes(network.comm_graph, "n_edges").values()))
        if width.sum() > 0:
            width = 20 * width / np.max(width)
    else:
        width = None
    sizes = np.array([len(network.communities[node]) for node in network.comm_graph.nodes])
    sizes = node_size * sizes / np.max(sizes)
    
    md_corr = network.get_community_metadata_correlation(layer=layer, subset_datasets=subset_datasets, method=method)
    
    color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1, 1), cmap=cmap)
    node_colors = []
    for community in network.comm_graph:
        if community in md_corr.index:
            node_colors.append(color_map.to_rgba(md_corr[community]))
        else:
            node_colors.append(mpl.colors.to_rgba(colors.missing_data_color))
    nx.draw(network.comm_graph, pos=network.comm_layout, node_color=node_colors, node_size=sizes, linewidths=0, width=width, edge_color=edge_color, with_labels=True, ax=ax_plot, font_size=20)
    
    if show_legend:
        plt.colorbar(color_map, ax=ax_legend, location="left", anchor=(0, 1), panchor=(1, 1), shrink=0.5, fraction=0.15, label=f"{method.capitalize()} correlation")
        
    if ax is None:
        return fig


def plot_overrepresentation_community_network(network: Network,
                                        colors: Colors,
                                        layer: str,
                                        subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                        ax: Optional[Axes] = None,
                                        pie_size: float = 0.3,
                                        figsize: Collection = (6, 4),
                                        edge_color: str = "#AAAAAA88",
                                        metric: str = "pearson_residual",
                                        legend_pie_size: float = 0.1,
                                        show_legend: bool = True) -> Optional[Figure]:
    
    if show_legend:
        assert ax is None
    
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
        if show_legend:
            ax_legend.set_xlim([-0.5, 0.5])
            ax_legend.set_aspect("equal")
            ax_legend.set_axis_off()
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("Communities")
    
    if network.comm_graph.edges:
        width = np.array(list(nx.get_edge_attributes(network.comm_graph, "n_edges").values()))
        if width.sum() > 0:
            width = 20 * width / np.max(width)
    else:
        width = None

    nx.draw_networkx_edges(network.comm_graph, pos=network.comm_layout, edge_color=edge_color, ax=ax_plot, width=width)
    overrepresentation = network.get_community_category_overrepresentation(subset_datasets=subset_datasets, layer=layer)
    overrepresentation = overrepresentation.fillna(0)

    max_or = np.max(overrepresentation.values.flatten())
    if max_or == 0:
        scale_factor = 1
    else:
        scale_factor = 1 / max_or
    for community, comm_or in overrepresentation.items():
        if community in network.comm_graph and comm_or.any():
            color_list = comm_or.index.map(colors.get_metadata_colors(layer))
            draw_circle_bar_plot(position=network.comm_layout[community],
                                 enrichments=comm_or,
                                 scale_factor=scale_factor,
                                 colors=color_list,
                                 size=pie_size, ax=ax_plot)

    if show_legend:
        # Add legends
        draw_circle_bar_plot(
            position=(0, 0.5),
            enrichments=pd.Series(max_or, index=overrepresentation.index),
            colors=overrepresentation.index.map(colors.get_metadata_colors(layer)),
            scale_factor=scale_factor,
            size=legend_pie_size,
            draw_labels=True,
            label_font_size=6, ax=ax_legend)
        draw_circle_bar_scale(
            position=(0, -0.75),
            scale_factor=scale_factor,
            size=pie_size,
            label_font_size=8, ax=ax_legend)
        ax_legend.set_title(f"{layer}")
        ax_legend.text(0, -1, "overrepresentation", ha="center", va="top", fontsize=10, fontweight="regular")

    
    # assert ax_plot.get_xlim() == ax_plot.get_ylim()
    # assert ax_plot.get_ylim() == ax_legend.get_ylim()
    
    if ax is None:
        return fig


#################
# Program-level bar plots #
#################


# overrepresentation bar plots
def plot_overrepresentation_program_bar(network: Network,
                                     colors: Colors,
                                     dataset_name: str,
                                     layers: Optional[Union[str, Collection[str]]] = None,
                                     figsize: Optional[Collection] = None):
    dataset = network.integration.datasets[dataset_name]

    # layers to plot
    if layers is None:
        metadata_layers = dataset.get_metadata_df(include_numerical=False).dropna(how="all", axis=1).columns.to_list()
    elif isinstance(layers, str):
        metadata_layers = [layers]
    else:
        metadata_layers = layers

    # number of bars in each community for this dataset
    community_program_counts = [len([node for node in network.communities[c] if node.split("|")[0] == dataset_name]) for c in network.ordered_community_names]
    
    if figsize is None:
        figsize = (network.n_communities + 0.05 * sum(community_program_counts),
                   (len(metadata_layers) + 1)* 2)
    
    fig, axes = plt.subplots(
        len(metadata_layers) + 1, network.n_communities,
        figsize = figsize,
        sharey='row', squeeze=False,
        gridspec_kw={"width_ratios": community_program_counts},
        layout="tight")
    fig.supxlabel("Program")
    fig.supylabel("Overrepresentation")
    fig.suptitle("Community")
    
    # first subplot row has k-values
    axes[0, 0].set_ylabel("rank (k)")
    for col, community in enumerate(network.ordered_community_names):
        ax = axes[0, col]
        ax.set_title(community)
        programs = []
        for node in network.communities[community]:
            dataset_str, k_str, program_str = node.split("|")
            if dataset_str == dataset_name:
                programs.append((int(k_str), int(program_str)))
        if programs:
            programs = sorted(programs)
            kvals = pd.Series([k for k, program in programs], index=programs)
            kvals.plot.bar(width=0.9, ax=ax, legend=None, color="#666666")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", integer=True))  # y-ticks are integers!
    
    # subsequent subplot rows have metadata overrepresentation
    for row, layer in enumerate(metadata_layers, 1):
        overrepresentation = dataset.get_category_overrepresentation(layer=layer)
        for col, community in enumerate(network.ordered_community_names):
            ax = axes[row, col]
            programs = []
            for node in network.communities[community]:
                dataset_str, k_str, program_str = node.split("|")
                if dataset_str == dataset_name:
                    programs.append((int(k_str), int(program_str)))
            programs = sorted(programs)
            if programs:
                overrepresentation[programs].T.plot.bar(stacked=True, width=0.9, ax=ax, legend=None, color=colors.get_metadata_colors(layer))
            ax.set_xlabel("")
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(layer)
            if row == 0:
                ax.set_title(community, size=14)

    return fig


def plot_metadata_correlation_program_bar(network: Network,
                                      colors: Colors,
                                      dataset_name: str,
                                      layers: Optional[Union[str, Collection[str]]] = None,
                                      method: str = "pearson",
                                      figsize: Optional[Collection] = None
                                      ) -> Optional[Figure]:
    
    dataset = network.integration.datasets[dataset_name]

    # layers to plot
    if layers is None:
        metadata_layers = dataset.get_metadata_df(include_categorical=False).dropna(how="all", axis=1).columns.to_list()
    elif isinstance(layers, str):
        metadata_layers = [layers]
    else:
        metadata_layers = layers

    # number of bars in each community for this dataset
    community_program_counts = [len([node for node in network.communities[c] if node.split("|")[0] == dataset_name]) for c in network.ordered_community_names]
    
    if figsize is None:
        figsize = (network.n_communities + 0.05 * sum(community_program_counts),
                   (len(metadata_layers) + 1)* 2)
    
    fig, axes = plt.subplots(
        len(metadata_layers) + 1, network.n_communities,
        figsize = figsize,
        sharey='row', squeeze=False,
        gridspec_kw={"width_ratios": community_program_counts},
        layout="tight")
    fig.supxlabel("Program")
    fig.supylabel(method.capitalize() + " Correlation")
    fig.suptitle("Community")
    
    # first subplot row has k-values
    axes[0, 0].set_ylabel("rank (k)")
    for col, community in enumerate(network.ordered_community_names):
        ax = axes[0, col]
        programs = []
        for node in network.communities[community]:
            dataset_str, k_str, program_str = node.split("|")
            if dataset_str == dataset_name:
                programs.append((int(k_str), int(program_str)))
        if programs:
            programs = sorted(programs)
            kvals = pd.Series([k for k, program in programs], index=programs)
            kvals.plot.bar(width=0.9, ax=ax, legend=None, color="#666666")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", integer=True))  # y-ticks are integers!
    
    # subsequent subplot rows have metadata overrepresentation
    for row, layer in enumerate(metadata_layers, 1):
        md_corr = dataset.get_metadata_correlation(layer=layer, method=method)
        for col, community in enumerate(network.ordered_community_names):
            ax = axes[row, col]
            programs = []
            for node in network.communities[community]:
                dataset_str, k_str, program_str = node.split("|")
                if dataset_str == dataset_name:
                    programs.append((int(k_str), int(program_str)))
            programs = sorted(programs)
            if programs:
                md_corr[programs].T.plot.bar(width=0.9, ax=ax, legend=None, color="#444444")
            ax.set_xlabel("")
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(layer)
            if row == 0:
                ax.set_title(community, size=14)

    return fig
    

###########################
# Community bar plots #
###########################


def plot_overrepresentation_community_bar(network: Network,
                                      colors: Colors,
                                      layer: str,
                                      subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                      figsize: Optional[Collection] = None,
                                      truncate_negative = True,
                                      ax = None
                                      ) -> Figure:
    
    df = network.get_community_category_overrepresentation(subset_datasets=subset_datasets, layer=layer, truncate_negative=truncate_negative).fillna(0)
    # if existing axes object is provided, plots with legend on that axes. Otherwise, creates a new figure with a separate plot and legend Axes.
    if ax is None:
        if figsize is None:
            figsize = [0.2 * df.shape[1] + 4, 4]
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [df.shape[1]/5, 1]}, layout="tight")
        ax_legend.set_axis_off()
    else:
        ax_plot = ax
        ax_legend = ax
        
    # Overrepresentation plot
    df.T.plot.bar(stacked=True, ax=ax_plot, width = 0.9, color=colors.get_metadata_colors(layer), legend=False)
    ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=90)
    ax_plot.set_ylabel("Median overrepresentation")
    ax_plot.set_xlabel("Community")
    
    # Legend
    colors.plot_metadata_colors_legend(layer=layer, ax=ax_legend)

    if ax is None:
        return fig


def plot_metadata_correlation_community_bar(network: Network,
                                      colors: Colors,
                                      layer: str,
                                      subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                      figsize: Optional[Collection] = None,
                                      ax = None,
                                      method: str = "pearson"
                                      ) -> Figure:
    md_corr = network.get_community_metadata_correlation(subset_datasets=subset_datasets, layer=layer).fillna(0)
    
    # if existing axes object is provided, plots with legend on that axes. Otherwise, creates a new figure with a separate plot and legend Axes.
    if ax is None:
        if figsize is None:
            figsize = [0.3 * md_corr.shape[0], 3]
        fig, ax_plot = plt.subplots(1, 1, figsize=figsize, layout="tight")
    else:
        ax_plot = ax
        
    # Overrepresentation plot
    md_corr.plot.bar(ax=ax_plot, width = 0.9, color="#888888")
    ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=90)
    ax_plot.set_ylabel(f"Median {method.capitalize()} Correlation")
    ax_plot.set_xlabel("Community")
    
    if ax is None:
        return fig



def plot_overrepresentation_community_heatmap(network: Network,
                                      layer: str,
                                      subset_datasets: Optional[Union[str, Collection[str]]] = None,
                                      figsize: Optional[Collection] = None,
                                      truncate_negative = False,
                                      ax = None
                                      ) -> Figure:
    
    df = network.get_community_category_overrepresentation(subset_datasets=subset_datasets, layer=layer, truncate_negative=truncate_negative).fillna(0)
    # if existing axes object is provided, plots with legend on that axes. Otherwise, creates a new figure with a separate plot and legend Axes.
    if ax is None:
        if figsize is None:
            figsize = [0.2 * df.shape[1] + 4, 0.2 * df.shape[0] + 2]
        fig, ax_plot = plt.subplots(figsize=figsize, layout="constrained")
    else:
        ax_plot = ax
        
    # Overrepresentation plot
    sns.heatmap(data=df, ax=ax_plot, xticklabels = True, yticklabels=True, 
                center = 0, cmap="RdBu_r")
    
    ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=0)
    ax_plot.set_ylabel("Median overrepresentation")
    ax_plot.set_xlabel("Community")
    
    if ax is None:
        return fig


#################################
# Community-usage Entropy plots #
#################################


def plot_sample_entropy(network: Network,
                     colors: Colors,
                     layer: str = "Dataset",
                     subset_datasets: Optional[Union[str, Collection[str]]] = None,
                     figsize = None
                     ):    
    metadata = network.integration.get_metadata_df(prepend_dataset_column=True, subset_datasets=subset_datasets)[layer]
    diversity = network.get_sample_entropy()
    df = pd.concat([diversity.rename("entropy"), metadata], axis=1)
    
    if figsize == None:
        figsize= [df[metadata.name].nunique() * 0.5 + 1,3]
    fig, ax = plt.subplots(figsize=figsize)
    if metadata.name == "Dataset":
        palette = colors.dataset_colors
    else:
        palette = colors.get_metadata_colors(metadata.name)
    sns.stripplot(data=df, hue=metadata.name, x=metadata.name, y="entropy", palette=palette, size=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.boxplot(
        meanprops={'color': 'k', 'ls': '-', 'lw': 2},
        whiskerprops={'visible': False},
        zorder=10,
        x=metadata.name,
        y="entropy",
        data=df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax)
    ax.set_ybound(lower=0)
    ax.set_title("Shannon Entropy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig



########################
# Label-transfer plots #
########################


def plot_metadata_transfer(network: Network, source: str, dest: str, layer: str, figsize: Collection[float, float] = None, annotate: Optional[Union[str, Collection[str]]] = None, colors: Colors = None, ax: Axes = None) -> Figure:
    """Plot heatmap of metadata transfer scores

    :param network: network through which to propagate labels
    :type network: :class:`mosaicmpi.network.Network`
    :param colors: colors object with metadata colors if 'annotation' parameter is specified
    :type colors: :class:`mosaicmpi.colors.Colors`
    :param source: Source dataset for label transfer
    :type source: str
    :param dest: Target dataset for label transfer
    :type dest: str
    :param layer: name of categorical data layer from source dataset
    :type layer: str
    :param annotate: categorical metadata layer, defaults to None
    :type annotate: Union[str, Collection[str]], optional
    :return: figure object
    :rtype: Figure
    """
    if annotate is None:
        annotations = []
    elif isinstance(annotate, str):
        annotations = [annotate]
    elif isinstance(annotate, Collection):
        annotations = annotate

    transfer_df = network.transfer_labels(source=source, dest=dest, layer=layer)
    if annotate is not None:
        metadata = network.integration.datasets[dest].get_metadata_df().loc[:, annotations].copy()
        for annotation_layer in annotations:
            mapper = colors.get_metadata_colors(annotation_layer)
            color_vec = metadata[annotation_layer].map(mapper)
            if color_vec.dtype == "category" and colors.missing_data_color not in color_vec.cat.categories:  # TODO: checking for color_vec dtype is a bandaid solution - we need consistent handling of NaNs in categorical metadata
                color_vec = color_vec.cat.add_categories(colors.missing_data_color)
            color_vec = color_vec.fillna(colors.missing_data_color)
            metadata[annotation_layer] = color_vec
        col_colors = metadata
    else:
        col_colors = None
    if figsize is None:
        figsize = [8, 1 + transfer_df.shape[0] * 0.2 + len(annotations) * 0.3]

    fig = sns.clustermap(data = transfer_df, col_colors=col_colors, figsize=figsize, row_cluster=False, xticklabels=False, yticklabels=True, cmap="Blues", colors_ratio=0.08)
    
    return fig


    
################
# ssGSEA plots #
################


def plot_representative_program_nes(network: Network, rep_nes: pd.DataFrame):

    rep_ids = network.get_representative_program_ids()
    height_ratios = rep_ids.value_counts()[network.ordered_community_names].values
    ds_names = sorted(list(network.integration.datasets.keys()))

    df = rep_nes.T.copy()
    df = df.droplevel(axis=0, level=[2,3])
    figsize = [df.shape[1] * 0.2, df.shape[0] * 0.2 + 4]

    fig, axes = plt.subplots(nrows = len(height_ratios), figsize=figsize, sharex=True, gridspec_kw={"height_ratios": height_ratios}, layout="constrained")
    figlegend, axlegend = plt.subplots(figsize=[1,2], layout="tight")
    for community, ax in zip(network.ordered_community_names, axes):
        df_comm = df.loc[community].astype("float").sort_index(key = lambda x: x.map(ds_names.index))
        sns.heatmap(df_comm, square=True, yticklabels=True, cmap="RdBu_r", vmin = -0.5, vmax=0.5, center = 0, xticklabels=True, cbar_ax=axlegend, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, )
        ax.set_xlabel(None)
        ax.set_ylabel(community, rotation=0, fontweight = "bold", fontsize="large", va="center", ha="right")
        is_bottom_subplot = community == network.ordered_community_names[-1]
        ax.tick_params(top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelright=True,
                labelbottom=is_bottom_subplot)
    fig.supylabel(f"Community")
    return fig, figlegend

####################
# Compare Networks #
####################

def plot_compare_integrations(name1: str, network1: Network, name2: str, network2: str, colors: Colors, figsize: Optional[Collection[float]] = None) -> Figure:
    """Calculate jaccard similarity of communities from two Network objects.
       Jaccard similarity is calculated over programs from datasets in common between the two Network objects,
       with network 1 on the y-axis, and network 2 on the x-axis.

    :param name1: name of first network
    :type name1: str
    :param network1: first network
    :type network1: :class:`mosaicmpi.Network`
    :param name2: name of second network
    :type name2: str
    :param network2: second network
    :type network2: :class:`mosaicmpi.Network`
    :param colors: colors object with a color for each dataset from both networks.
    :type colors: :class:`mosaicmpi.Colors`
    :param figsize: (width, height), defaults to None
    :type figsize: Optional[Collection[float]], optional
    :return: figure
    :rtype: Figure
    """
    
    shared_datasets = set(network1.integration.datasets) & set(network2.integration.datasets)
    net1_datasets = set(network1.integration.datasets) - shared_datasets
    net2_datasets = set(network2.integration.datasets) - shared_datasets

    ds_colors = {"shared": colors.missing_data_color} | {ds: colors for ds, colors in colors.dataset_colors.items() if ds in (net1_datasets | net2_datasets)}

    jaccard = compare_community_jaccard_similarity(name1=name1, network1=network1, name2=name2, network2=network2)

    net1_bars = pd.DataFrame(np.nan, index=jaccard.index[::-1], columns=["shared"] + list(net1_datasets))
    for net1_comm in network1.ordered_community_names:
        net1_bars.loc[net1_comm, "shared"] = len(set(n for n in network1.communities[net1_comm] if utils.node_to_program(n)[0] in shared_datasets))
        for d in net1_datasets:
            net1_bars.loc[net1_comm, d] = len(set(n for n in network1.communities[net1_comm] if utils.node_to_program(n)[0] == d))
    net1_bars

    net2_bars = pd.DataFrame(np.nan, index=jaccard.columns, columns=["shared"] + list(net2_datasets))
    for net2_comm in network2.ordered_community_names:
        net2_bars.loc[net2_comm, "shared"] = len(set(n for n in network2.communities[net2_comm] if utils.node_to_program(n)[0] in shared_datasets))
        for d in net2_datasets:
            net2_bars.loc[net2_comm, d] = len(set(n for n in network2.communities[net2_comm] if utils.node_to_program(n)[0] == d))
    net2_bars

    max_bar_height = max(net1_bars.T.sum().max(), net2_bars.T.sum().max())

    if figsize is None:
        figsize = [0.5 * jaccard.shape[1] + 2, 0.5 * jaccard.shape[0] + 3]

    fig, axes = plt.subplots(3,3, figsize=figsize, gridspec_kw={"width_ratios": [10, 1, 0.2], "height_ratios": [1, 1, 10]},
                            sharex="col", sharey="row", layout="constrained")

    sns.heatmap(jaccard, xticklabels=True, cmap="Blues", ax=axes[2,0], cbar_ax=axes[0,2])
    axes[2,0].set_yticklabels(axes[2,0].get_yticklabels(), rotation=0)
    axes[2,0].set_ylabel(f"{name1}\ncommunity")
    axes[2,0].set_xlabel(f"{name2}\ncommunity")

    axes[0,2].set_title("Jaccard\nsimilarity")


    # horizontal bars
    y_tick_pos = [i + 0.5 for i in range(len(network1.ordered_community_names))]
    left = pd.Series(0, index=net1_bars.index[::-1])
    for ds, ds_width in net1_bars[::-1].items():
        axes[2,1].barh(y=y_tick_pos, width=ds_width, left=left, height=0.8, color=ds_colors[ds])
        left += ds_width
    axes[2,1].set_xlim([0,max_bar_height])
    axes[2,1].set_xlabel("Nodes")

    # vertical bars
    x_tick_pos = [i + 0.5 for i in range(len(network2.ordered_community_names))]
    bottom = pd.Series(0, index=net2_bars.index)
    for ds, ds_height in net2_bars.items():
        axes[1,0].bar(x=x_tick_pos, height=ds_height, bottom=bottom, width=0.8, color=ds_colors[ds])
        bottom += ds_height
    axes[1,0].set_ylim([0,max_bar_height])
    axes[1,0].set_ylabel("Nodes")


    axes[0,0].set_axis_off()
    axes[0,1].set_axis_off()
    axes[1,1].set_axis_off()
    axes[1,2].set_axis_off()
    axes[2,2].set_axis_off()

    legend_handles = [Patch(color=color, label=ds) for ds, color in ds_colors.items()]
    axes[0,0].legend(handles=legend_handles, loc = "center", ncols = 3)
    
    return fig
     
def plot_geneset_pval_heatmap(df: pd.DataFrame,
                              ax: Optional[Axes] = None,
                              axlegend: Optional[Axes] = None,
                              cmap: str = "Blues",
                              vmin: float = 0.,
                              vmax: float = 10.,
                              show_geneset_labels: bool = False,
                              limit_geneset_label_length: int = 200) -> Optional[Figure]:

    if ax is None:

        if show_geneset_labels:
            figsize = [10 + df.shape[1]/4,  0.15 * df.shape[0]]
        else:
            figsize = [0.5 + df.shape[1]/4, 8]

        fig, ax_plot = plt.subplots(figsize=figsize, layout="constrained")
    else:
        ax_plot = ax
        
    if axlegend is None:
        figlegend, axlegend_plot = plt.subplots(figsize=[1, 3], layout="constrained")
    else:
        axlegend_plot = axlegend
    
    if df.shape[0] > 0:
        if show_geneset_labels and limit_geneset_label_length > 0:
            yticklabels = df.index.str[0:limit_geneset_label_length]
        else:
            yticklabels=False
        sns.heatmap(df, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax_plot, cbar_ax = axlegend_plot, yticklabels=yticklabels)
        ax_plot.tick_params(top=False,
                            bottom=True,
                            left=False,
                            right=False,
                            labelleft=False,
                            labelright=show_geneset_labels,
                            labeltop= False,
                            labelbottom=True)
        if show_geneset_labels:
            ax_plot.tick_params(axis="y", labelsize=8, labelrotation=0)

        ax_plot.set_xlabel("")
        ax_plot.set_ylabel("")
        for _, spine in ax_plot.spines.items():
            spine.set_visible(True)
            spine.set_color('#aaaaaa')
    if ax is None and axlegend is None:
        return fig, figlegend
