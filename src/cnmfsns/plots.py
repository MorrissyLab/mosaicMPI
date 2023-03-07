
from . import Dataset, Integration, Colors, SNS, utils

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch
import matplotlib.pyplot as plt
import upsetplot
from typing import Optional, Union, Collection
from scipy.cluster.hierarchy import linkage, dendrogram
from anndata import read_h5ad
import networkx as nx

def plot_feature_dispersion(dataset: Dataset, show_selected:bool = False, all_plots: bool = False):
    """
    Create diagnostic plots for data from model_overdispersion()
    """
    df = dataset.adata.var.sort_values("mean")
    
    figs = {}

    # Figure: Default model mean and variance
    fig, axes = plt.subplots(1, 3, figsize=[12, 4], layout="tight")
    
    ax = axes[0]  # untransformed
    if show_selected:
        sns.histplot(df, x="log_mean", y="log_variance", hue="selected", bins=[100,100], ax=ax, alpha=0.5, palette={True: "red", False: "blue"})
        ax.legend(handles=[
            Patch(color="blue", alpha=0.5, label="False"),
            Patch(color="red", alpha=0.5, label="True"),
            Line2D([0], [0], color='green', label="model")
        ], title="selected")
    else:
        sns.histplot(df, x="log_mean", y="log_variance", bins=[100,100], ax=ax, color="blue")
    ax.plot(df["log_mean"], df["gam_fittedvalues"], color="green")
    ax.set_title("Mean-Variance")
    ax.set_xlabel("log10(mean)")
    ax.set_ylabel("log10(variance)")

    ax = axes[1]  # transformed
    if show_selected:
        sns.histplot(df, x="log_mean", y="odscore", hue="selected", bins=[100,100], ax=ax, palette={True: "red", False: "blue"})
        ax.legend(handles=[
            Patch(color="blue", alpha=0.5, label="False"),
            Patch(color="red", alpha=0.5, label="True"),
            Line2D([0], [0], color='green', label="model")
        ], title="selected")
    else:
        sns.histplot(df, x="log_mean", y="odscore", bins=[100,100], ax=ax, color="blue")
    ax.hlines(1, xmin=df["log_mean"].min(), xmax=df["log_mean"].max(), color="green")
    ax.set_title("Mean-Overdispersion (Default)")
    ax.set_xlabel("log10(mean)")
    ax.set_ylabel("od-score")
    
    ax=axes[2]  # thresholds
    ax.set_title("od-score Distribution")
    if df["odscore"].notnull().any():
        if show_selected:
            sns.histplot(df, x="odscore", hue="selected", bins=100, linewidth=0, ax=ax, palette={True: "red", False: "blue"})
            ax.legend(handles=[
                Patch(color="blue", alpha=0.5, label="False"),
                Patch(color="red", alpha=0.5, label="True")
            ], title="selected")
        else:
            sns.histplot(df, x="odscore", bins=100, linewidth=0, ax=ax, color="blue")
    ax.set_xlabel("od-score")
    figs["default"] = fig

    if all_plots:
        # Figure: cnmf model mean and variance
        fig, axes = plt.subplots(1, 3, figsize=[12, 4], layout="tight")
        ax = axes[0]
        if show_selected:
            sns.histplot(df, x="log_mean", y="log_variance", hue="selected", bins=[100,100], ax=ax, alpha=0.5, palette={True: "red", False: "blue"})
        else:
            sns.histplot(df, x="log_mean", y="log_variance", bins=[100,100], ax=ax, color="blue")
        ax.set_title("Mean-Variance")
        ax.set_xlabel("log10(mean)")
        ax.set_ylabel("log10(variance)")
        ax = axes[1]
        if show_selected:
            sns.histplot(df, x="log_mean", y="vscore", hue="selected", bins=[100,100], ax=ax, palette={True: "red", False: "blue"})
        else:
            sns.histplot(df, x="log_mean", y="vscore", bins=[100,100], ax=ax, color="blue")
        ax.set_title("Mean-Overdispersion (cnmf)")
        ax.set_xlabel("log10(mean)")
        ax.set_ylabel("v-score")
        ax=axes[2]  # thresholds
        ax.set_title("v-score Distribution")
        if df["vscore"].notnull().any():
            if show_selected:
                sns.histplot(df, x="vscore", hue="selected", bins=100, linewidth=0, ax=ax, palette={True: "red", False: "blue"})
                ax.legend(handles=[
                    Patch(color="blue", alpha=0.5, label="False"),
                    Patch(color="red", alpha=0.5, label="True")
                ], title="selected")
            else:
                sns.histplot(df, x="vscore", bins=100, linewidth=0, ax=ax, color="blue")
        ax.set_xlabel("v-score")
        figs["cnmf"] = fig

        fig, ax = plt.subplots(figsize=[12,12], layout="tight")
        sns.histplot(data=df.fillna(0), x="odscore", y="vscore", bins=[100, 100], ax=ax)
        ax.set_xlabel("od-score")
        ax.set_ylabel("v-score")
        figs["score_comparison"] = fig
        
    return figs


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
        row_cluster=True, col_cluster=True, plot_col_dendrogram=True, show_sample_labels=True, ylabel=""):
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
            if pd.api.types.is_categorical_dtype(annot) or pd.api.types.is_object_dtype(annot):
                ordered_rgb = annot.iloc[xind].replace(metadata_colors[track])
                if ordered_rgb.isnull().any():
                    ordered_rgb = ordered_rgb.cat.add_categories(missing_data_color)
                    ordered_rgb = ordered_rgb.fillna(missing_data_color)
                ordered_rgb = ordered_rgb.astype("object").map(mpl.colors.to_rgb)
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
    plt.colorbar(im_heatmap, ax=ax, location="top")
    return fig

def plot_usage_heatmap(dataset: Dataset, k: Optional[int], colors, subset_metadata = None, subset_samples = None, title = None, cluster_geps = False, cluster_samples = True, show_sample_labels = True):
    assert dataset.has_cnmf_results
    df = dataset.get_usages(k=k)
    if subset_samples is not None:
        df = df.loc[subset_samples]
    samples = df.index.to_series()
    metadata = dataset.adata.obs.loc[samples]
    if subset_metadata is not None:
        metadata = metadata.loc[:, subset_metadata]
    metadata_colors = {col: colors.get_metadata_colors(col) for col in metadata.columns}
    df = df.div(df.sum(axis=1), axis=0)
    fig = annotated_heatmap(data=df, metadata=metadata,
                            metadata_colors=metadata_colors, 
                            missing_data_color=colors.missing_data_color, 
                            title=title,
                            row_cluster=cluster_geps,
                            col_cluster=cluster_samples,
                            show_sample_labels=show_sample_labels,
                            plot_col_dendrogram=True,
                            ylabel="GEP")
    return fig

def plot_community_usage_heatmap(snsmap: SNS,
                                 colors: Colors,
                                 subset_metadata = None,
                                 subset_datasets = None,
                                 subset_samples = None,
                                 title = None,
                                 cluster_geps = False,
                                 cluster_samples = True,
                                 show_sample_labels = True):
    df = snsmap.get_community_usage()
    if subset_samples is not None:
        df = df.loc[subset_samples]
    if subset_datasets is not None:
        df = df.loc[subset_datasets]
        
    
    metadata = snsmap.integration.get_metadata_df()
    if subset_metadata is not None:
        metadata = metadata.loc[:, subset_metadata]
    
    
    metadata_colors = {col: colors.get_metadata_colors(col) for col in metadata.columns}
    df = df.div(df.sum(axis=1), axis=0)
    fig = annotated_heatmap(data=df, metadata=metadata,
                            metadata_colors=metadata_colors, 
                            missing_data_color=colors.missing_data_color, 
                            title=title,
                            row_cluster=cluster_geps,
                            col_cluster=cluster_samples,
                            show_sample_labels=show_sample_labels,
                            plot_col_dendrogram=True,
                            ylabel="Community")
    return fig


def plot_gep_correlation_matrix(integration: Integration, colors, figsize=(20,20), cmap="RdBu_r", hide_gep_labels=False):
    ds_color_track = integration.corr_matrix.index.get_level_values(0).map(colors.dataset_colors)
    ds_color_track = [mpl.colors.to_rgb(c) for c in ds_color_track]
    cg = sns.clustermap(integration.corr_matrix, figsize=figsize,
                        cmap=cmap, center=0, vmin=-1, vmax=1,
                        row_colors=ds_color_track, col_colors=ds_color_track)
    cg.ax_heatmap.set_ylabel("")
    cg.ax_heatmap.set_xlabel("")
    if hide_gep_labels:
        cg.ax_heatmap.set_xticks([])
        cg.ax_heatmap.set_yticks([])
    cg.ax_col_dendrogram.set_title("GEP correlations")
    return cg.figure


def plot_rank_reduction(integration: Integration, figsize=None):
    n_ranks = integration.k_table.shape[0]
    n_datasets = len(integration.datasets)
    if figsize is None:
        figsize = [n_ranks * n_datasets / 2, 3]
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize)
    for dataset_name, ax in zip(integration.datasets, axes):
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

def plot_pairwise_corr(integration: Integration, subplot_size = [3, 3.5], overlaid=False):
    tril = integration.get_corr_matrix_lowertriangle()
    n_datasets = len(tril.index.levels[0])
    sps_width, sps_height = subplot_size
    fig, axes = plt.subplots(n_datasets, n_datasets, figsize=[sps_width * n_datasets, sps_height * n_datasets], sharex=True, sharey=True, squeeze=False)
    fig.supxlabel("Correlation coefficient")
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            ax = axes[row,col]
            if row < col:
                ax.set_axis_off()
            else:
                corr = pd.Series(tril.loc[dataset_row, dataset_col].values.flatten()).dropna()
                if integration.pairwise_thresholds is not None:
                    min_corr = integration.pairwise_thresholds.loc[(dataset_row, dataset_col)]
                    hist_kwargs = {
                        "hue":[("Included" if c else "Excluded") for c in (corr > min_corr)],
                        "palette": {"Included": "red", "Excluded": "gray"},
                        "hue_order": ["Excluded", "Included"]
                    }
                    included_fraction = (corr > min_corr).sum() / corr.shape[0]
                    ax.text(x=0.01, y=1.01, s=f"quantile={included_fraction:.3f}\nmin_corr={min_corr:.3f}", size=8, ha='left', va='bottom', transform=ax.transAxes, color="black")
                else:
                    hist_kwargs = {"color": "gray"}
                sns.histplot(x=corr, ax=ax,legend=(row == 0)&(col == 0), bins=50, linewidth=0, **hist_kwargs)

                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(-1,1)
    fig.suptitle(f"Correlation distribution between datasets")
    plt.tight_layout()
    return fig

def plot_pairwise_corr_overlaid(integration: Integration, subplot_size = [3, 3.5]):
    tril = tril = integration.get_corr_matrix_lowertriangle(max_k_filter=True)
    sps_width, sps_height = subplot_size
    fig, axes = plt.subplots(integration.n_datasets, integration.n_datasets,
                             figsize=[sps_width * integration.n_datasets, sps_height * integration.n_datasets],
                             sharex=True, sharey=True, squeeze=False)
    fig.supxlabel("Correlation coefficient")
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            ax = axes[row,col]
            if row < col:
                ax.set_axis_off()
            else:
                corr = pd.DataFrame({"corr": tril.loc[dataset_row, dataset_col].values.flatten()}).dropna()
                corr["sign"] = (corr["corr"] >= 0).map({True: "Positive", False: "Negative"})
                corr["abscorr"] = corr["corr"].abs()
                sns.histplot(data=corr, x="abscorr", hue="sign", palette= {"Positive": "red", "Negative": "blue"}, bins=50, alpha=0.5, linewidth=0,
                             hue_order= ["Negative", "Positive"], ax=ax,legend=(row == 0)&(col == 0))

                # show min_corr as text in top left of plot and vertical line
                min_corr = integration.pairwise_thresholds.loc[(dataset_row, dataset_col)]
                included_fraction = (corr["corr"] > min_corr).sum() / corr.shape[0]  # could also show the quantile of the min_corr threshold
                ax.text(x=0.01, y=1.01, s=f"quantile={included_fraction:.3f}\nmin_corr={min_corr:.3f}", size=8, ha='left', va='bottom', transform=ax.transAxes, color="black")
                ax.axvline(min_corr, color="black")
                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(0,1)
    fig.suptitle(f"Correlation distribution between datasets")
    plt.tight_layout()
    return fig

def plot_overdispersed_features_upset(integration: Integration, figsize=[6, 4]):
    overdispersed_feature_lists = {dataset_name: dataset.overdispersed_genes for dataset_name, dataset in integration.datasets.items()}
    fig = plt.Figure(figsize=figsize)
    upsetplot.UpSet(upsetplot.from_contents(overdispersed_feature_lists)).plot(fig=fig)
    fig.suptitle("Overdispersed features")
    return fig

def plot_features_upset(integration: Integration, figsize=[6, 4]):
    feature_lists = {dataset_name: list(dataset.adata.var.index) for dataset_name, dataset in integration.datasets.items()}
    fig = plt.Figure(figsize=figsize)
    upsetplot.UpSet(upsetplot.from_contents(feature_lists)).plot(fig=fig)
    fig.suptitle("Features")
    return fig

def plot_community_by_dataset_rank(snsmap: SNS, colors: Colors, figsize: Collection = None):
    """
    Plot communities by dataset and rank representation
    """

    marker_style = {
        1: ("s", 30),  # 1 factor: square markers, size 30
        2: (2, 30)     # 2 factors: marker #2 (up tick), size 30
        }
    n_datasets = snsmap.integration.n_datasets
    if figsize is None:
        figsize = [1 + n_datasets * 5, 1 + len(snsmap.communities)/4]
    fig, axes = plt.subplots(1, n_datasets+1, figsize=figsize, sharex=True, sharey=True)
    for dataset, ax in zip(snsmap.integration.datasets, axes):
        for y, community in enumerate(snsmap.ordered_community_names):
            members = snsmap.communities[community]
            counts = pd.Series([m.rpartition("|")[0] for m in members]).value_counts()
            
            # plot line if any factors are present
            line_x = []
            line_y = []
            for x, rank in enumerate(snsmap.integration.selected_k[dataset]):
                line_x.append(x)
                if f"{dataset}|{rank}" in counts.index:
                    line_y.append(y)
                else:
                    line_y.append(np.NaN)
            ax.plot(line_x, line_y, color=colors.dataset_colors[dataset], linewidth=2)
            
            for count, style in marker_style.items():
                # plot different markers depending on how many factors are present:
                scatter_x = []
                scatter_y = []
                for x, rank in enumerate(snsmap.integration.selected_k[dataset]):
                    factor_prefix = f"{dataset}|{rank}"
                    if factor_prefix in counts.index and counts[factor_prefix] == count:
                        scatter_x.append(x)
                        scatter_y.append(y)
                ax.scatter(scatter_x, scatter_y, color=colors.dataset_colors[dataset], marker=style[0], s=style[1])
        ax.set_yticks(list(range(len(snsmap.ordered_community_names))))
        ax.set_yticklabels(snsmap.ordered_community_names)
        ax.set_xticks(list(range(len(snsmap.integration.selected_k[dataset]))))
        ax.set_xticklabels(snsmap.integration.selected_k[dataset])
        ax.set_title(dataset)

    fig.supxlabel("Rank (k)")
    fig.supylabel("Community")


    # Add legend
    cbdrlegend = []
    cbdrlegend.append(Line2D([0],[0], marker='s', color='black', label="1 GEP", markerfacecolor="black", markersize=8))
    cbdrlegend.append(Line2D([0],[0], marker=2, color='black', label="2 GEPs", markerfacecolor="black", markersize=8))
    cbdrlegend.append(Line2D([0],[0], marker=None, color='black', label="3+ GEPs", markerfacecolor="black", markersize=8))
    axes[-1].legend(handles=cbdrlegend, loc='center', frameon=False)
    axes[-1].set_axis_off()
    plt.tight_layout()
    return fig

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

def draw_circle_bar_scale(position, size, ax, scale_factor, label_font_size, linewidth=0.5):

    x, y = position
    for ring in [0.25,0.5,0.75,1]:
        ax.add_patch(plt.Circle(position, np.sqrt(ring) * size, color="black", fill=False, linewidth=linewidth))
    ax.add_patch(Rectangle(position, size * 1.01, size * 1.01, color="#FFFFFF"))
    for ring in [0.25,0.5,0.75,1]:
        value = ring/scale_factor
        ax.text(
            x + size * 0.05,
            y + np.sqrt(ring) * size,
            f"{value:.3f}",
            fontsize=label_font_size,
            verticalalignment="center")


def plot_gep_overrepresentation_network(snsmap,
                                        colors,
                                        layer,
                                        subset_datasets = None,
                                        ax: Optional[Axes] = None,
                                        pie_size=0.1,
                                        figsize=(9, 6),
                                        edge_weights=None,
                                        metric="pearson_residual",
                                        show_legends=True) -> Optional[mpl.figure.Figure]:
    
    if show_legends:
        assert ax is None
    
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_plot.set_title("GEP Network")
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_aspect("equal")
    ax_legend.set_axis_off()
    
    if edge_weights is None or not snsmap.gep_graph.edges:
        width = 0.2
    else:
        width = np.array(list(nx.get_edge_attributes(snsmap.gep_graph, edge_weights).values()))
        width = width / np.max(width)

    nx.draw_networkx_edges(snsmap.gep_graph, pos=snsmap.layout, edge_color="#888888", ax=ax_plot, width=width)
    overrepresentation = snsmap.integration.get_category_overrepresentation(subset_datasets=subset_datasets, layer=layer)
    overrepresentation = overrepresentation.fillna(0)

    max_or = np.max(overrepresentation.values.flatten())
    scale_factor = 1 / max_or
    for gep, gep_or in overrepresentation.items():
        node = "|".join((str(p) for p in gep))

        if node in snsmap.gep_graph and gep_or.any():
            color_list = gep_or.index.map(colors.get_metadata_colors(layer))
            draw_circle_bar_plot(position=snsmap.layout[node],
                                 enrichments=gep_or,
                                 scale_factor=scale_factor,
                                 colors=color_list,
                                 size=pie_size, ax=ax_plot)

    if show_legends:
        # Add legends
        draw_circle_bar_plot(
            position=(0, 0.5),
            enrichments=pd.Series(max_or, index=gep_or.index.sort_values().unique()),
            colors=overrepresentation.index.map(colors.get_metadata_colors(layer)),
            scale_factor=scale_factor,
            size=pie_size,
            draw_labels=True,
            label_font_size=6, ax=ax_legend)
        draw_circle_bar_scale(
            position=(0, -0.5),
            scale_factor=scale_factor,
            size=pie_size,
            label_font_size=4, ax=ax_legend)
        ax_legend.set_title(f"{layer}")
        ax_legend.text(0, -0.25, "pearson_residual", ha="center", va="center", )
    
    # assert ax_plot.get_xlim() == ax_plot.get_ylim()
    # assert ax_plot.get_ylim() == ax_legend.get_ylim()
    
    if ax is None:
        return fig


def plot_gep_network_datasets(snsmap: SNS, colors: Colors, figsize = (9,6), edge_color = "#888888", node_size = 30, node_size_kval = False, labels = False, ax = None):
     
    if ax is None:
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2, 1]}, layout="tight")
    else:
        ax_plot = ax
        ax_legend = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_dataset_colors_legend(ax=ax_legend)
    
    node_colors = []
    for node in snsmap.gep_graph:
        node_colors.append(colors.dataset_colors[node.split("|")[0]])

    # Labels without dataset names
    node_labels = {}
    for node in snsmap.gep_graph:
        node_labels[node] = node.partition("|")[2]

    # Node sizes inversely proportional to k
    if node_size_kval:
        sizes = {}
        selected_k_anyds = snsmap.integration.k_table.loc[:, (slice(None), "selected_k")].any(axis=1)
        median_rank = min(selected_k_anyds[selected_k_anyds].index)

        for node in snsmap.gep_graph:
            sizes[node] = node_size / (int(node.split("|")[1]) + 0.5 - median_rank)
        node_sizes = [(sizes[n] if n in sizes else 0) for n in snsmap.gep_graph]
    else:
        node_sizes = node_size
    # Plot nodes colored by dataset
    nx.draw(snsmap.gep_graph,
            pos=snsmap.layout,
            with_labels=labels,
            node_color=node_colors,
            labels=node_labels,
            node_size=node_sizes,
            linewidths=0,
            width=0.2,
            edge_color=edge_color,
            font_size=4, ax=ax_plot)
    ax_plot.set_title("GEP Network")
    return fig


def plot_gep_network_communities(snsmap: SNS,
                                 colors: Colors,
                                 figsize = (9,6),
                                 edge_color = "#888888",
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
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_community_colors_legend(ax=ax_legend)
    
    node_colors = []
    for node in snsmap.gep_graph:
        node_colors.append(colors.community_colors[snsmap.gep_communities[node]])

    # Labels without dataset names
    labels = {}
    for node in snsmap.gep_graph:
        labels[node] = node.partition("|")[2]

    # Node sizes inversely proportional to k
    if node_size_kval:
        sizes = {}
        selected_k_anyds = snsmap.integration.k_table.loc[:, (slice(None), "selected_k")].any(axis=1)
        median_rank = min(selected_k_anyds[selected_k_anyds].index)

        for node in snsmap.gep_graph:
            sizes[node] = node_size / (int(node.split("|")[1]) + 0.5 - median_rank)
        node_sizes = [(sizes[n] if n in sizes else 0) for n in snsmap.gep_graph]
    else:
        node_sizes = node_size
    # Plot nodes colored by dataset
    nx.draw(snsmap.gep_graph,
            pos=snsmap.layout,
            with_labels=False,
            node_color=node_colors,
            labels=labels,
            node_size=node_sizes,
            linewidths=0,
            width=0.2,
            edge_color=edge_color,
            font_size=4, ax=ax_plot)
    ax_plot.set_title("GEP Network")
    return fig

def plot_gep_network_samples(snsmap: SNS,
                             colors: Colors,
                             figsize: Collection = (9, 6),
                             discretize = False,
                             edge_color = "#888888",
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
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_dataset_colors_legend(ax=ax_legend)

    usages = snsmap.integration.get_usages(discretize = discretize,
                                           normalize = True)


    if discretize:
        labels = usages[snsmap.geps_in_graph].sum().apply(lambda x: str(int(x))).to_dict()
    else:
        labels = usages[snsmap.geps_in_graph].sum().apply(lambda x: f"{x:.1f}").to_dict()
    labels = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in labels.items()}
    
    scale_factor = node_size / usages[snsmap.geps_in_graph].sum().max()
    sizes = (usages[snsmap.geps_in_graph].sum() * scale_factor).to_dict()
    sizes = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in sizes.items()}
    
    colors = [colors.dataset_colors[node.partition("|")[0]] for node in snsmap.gep_graph]

    node_sizes = [(sizes[n] if n in sizes else 0) for n in snsmap.gep_graph]
    nx.draw(snsmap.gep_graph, snsmap.layout,
    with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, linewidths=0, width=0.2, edge_color=edge_color, font_size=font_size, ax=ax_plot)
    if ax is None:
        return fig

def plot_gep_network_patients(snsmap: SNS,
                             colors: Colors,
                             figsize: Collection = (9, 6),
                             edge_color = "#888888",
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
    ax_legend.set_xlim([-0.5, 0.5])
    ax_legend.set_axis_off()
    colors.plot_dataset_colors_legend(ax=ax_legend)

    usages = snsmap.integration.get_usages(discretize = True).fillna(0).astype(bool)
    usages.index = usages.index.map(snsmap.integration.sample_to_patient)
    usages = usages.groupby(axis=0, level=[0,1]).any()
        
    labels = usages[snsmap.geps_in_graph].sum().apply(lambda x: str(int(x))).to_dict()
    labels = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in labels.items()}
    
    scale_factor = node_size / usages[snsmap.geps_in_graph].sum().max()
    sizes = (usages[snsmap.geps_in_graph].sum() * scale_factor).to_dict()
    sizes = {f"{k[0]}|{k[1]}|{k[2]}": v for k,v in sizes.items()}
    
    colors = [colors.dataset_colors[node.partition("|")[0]] for node in snsmap.gep_graph]

    node_sizes = [(sizes[n] if n in sizes else 0) for n in snsmap.gep_graph]
    nx.draw(snsmap.gep_graph, snsmap.layout,
    with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, linewidths=0, width=0.2, edge_color=edge_color, font_size=font_size, ax=ax_plot)
    if ax is None:
        return fig

def plot_community_network_summary(snsmap: SNS,
                                   colors: Colors,
                                   figsize = (4, 4),
                                   edge_color = "#888888",
                                   node_size = 500,
                                   ax: Axes = None):
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=figsize, layout="tight")
    else:
        ax_plot = ax
    ax_plot.set_aspect("equal")
    ax_plot.set_axis_off()
    if snsmap.comm_graph.edges:
        width = np.array(list(nx.get_edge_attributes(snsmap.comm_graph, "n_edges").values()))
        width = 20 * width / np.max(width)
    else:
        width = None
    sizes = np.array([len(snsmap.communities[node]) for node in snsmap.comm_graph.nodes])
    sizes = node_size * sizes / np.max(sizes)
    node_colors = [colors.community_colors[node] for node in snsmap.comm_graph]
    nx.draw(snsmap.comm_graph, pos=snsmap.comm_layout, node_color=node_colors, node_size=sizes, linewidths=0, width=width, edge_color=edge_color, with_labels=True, ax=ax_plot, font_size=20)
    return fig

# overrepresentation bar plots
def plot_overrepresentation_geps_bar(snsmap: SNS, colors, dataset_name, figsize = None):
    dataset = snsmap.integration.datasets[dataset_name]
    metadata = dataset.get_metadata_df(include_numerical=False).dropna(how="all", axis=1)
    # number of bars in each community for this dataset
    communities = snsmap.communities
    community_gep_counts = [len([node for node in communities[c] if node.split("|")[0] == dataset_name]) for c in sorted(list(communities))]
    
    if figsize is None:
        figsize = (len(communities) + 0.05 * sum(community_gep_counts),
                   metadata.shape[1] * 2)
    
    fig, axes = plt.subplots(
        metadata.shape[1], len(communities),
        figsize = figsize,
        sharey='row', squeeze=False,
        gridspec_kw={"width_ratios": community_gep_counts})
    for row, layer in enumerate(metadata.columns):
        overrepresentation = dataset.get_category_overrepresentation(layer=layer)
        for col, community in enumerate(sorted(list(communities))):
            ax = axes[row, col]
            geps = []
            for node in communities[community]:
                dataset_str, k_str, gep_str = node.split("|")
                if dataset_str == dataset_name:
                    geps.append((int(k_str), int(gep_str)))
            geps = sorted(geps)
            if geps:
                overrepresentation[geps].T.plot.bar(stacked=True, width=0.9, ax=ax, legend=None, color=colors.get_metadata_colors(layer))
            ax.set_xlabel("")
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(layer)
            if row == 0:
                ax.set_title(community, size=14)

    fig.supxlabel("GEP")
    fig.supylabel("Overrepresentation")
    fig.suptitle("Community")
    fig.tight_layout()
    return fig


#####

def plot_annotated_geps_by_community(usage, config, communities):
    figs = {}
    for dataset_name, dataset in config.datasets.items():
        metadata = read_h5ad(dataset["filename"], backed="r").obs.select_dtypes(include="category")  # only use categorical data
        metadata = metadata.dropna(axis=1, how="all")
        # number of bars in each community for this dataset
        community_gep_counts = [len([node for node in communities[c] if node.split("|")[0] == dataset_name]) for c in sorted(list(communities))]
        width_ratios = [1 + gep_counts for gep_counts in community_gep_counts]
        fig, axes = plt.subplots(metadata.shape[1], len(communities), figsize=[2 + 0.1 * sum(community_gep_counts), metadata.shape[1] * 3], sharey='row', gridspec_kw={"width_ratios": community_gep_counts})
        for row, (annotation_layer, sample_to_class) in enumerate(metadata.items()):
            # usage subset to dataset
            ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
            ds_usage.index = ds_usage.index.map(sample_to_class)
            ds_usage = ds_usage[ds_usage.index.notnull()]
            observed = ds_usage.groupby(axis=0, level=0).sum()
            expected = []
            for k, obs_k in observed.groupby(axis=1, level=1):
                exp_k = pd.DataFrame(obs_k.sum(axis=1)) @ pd.DataFrame(obs_k.sum(axis=0)).T / obs_k.sum().sum()
                expected.append(exp_k)
            expected = pd.concat(expected, axis=1)
            chisq_resid = (observed - expected) / np.sqrt(expected)  # pearson residual of chi-squared test of contingency table
            overrepresentation = chisq_resid.clip(lower=0)
            for col, community in enumerate(sorted(list(communities))):
                ax = axes[row, col]
                geps = []
                for node in communities[community]:
                    dataset_str, k_str, gep_str = node.split("|")
                    if dataset_str == dataset_name:
                        geps.append((dataset_str, int(k_str), int(gep_str)))
                geps = sorted(geps)
                if geps:
                    overrepresentation[geps].T.plot.bar(stacked=True, width=0.9, ax=ax, legend=None, color=config.get_metadata_colors(annotation_layer))
                ax.set_xlabel("")
                ax.set_xticks([])
                if col == 0:
                    ax.set_ylabel(annotation_layer)
                if row == 0:
                    ax.set_title(community, size=14)

        fig.supxlabel("GEP")
        fig.supylabel("Overrepresentation")
        fig.suptitle("Community")
        figs[dataset_name] = fig
    return figs



def plot_metadata_correlation_geps_bar(usage, metadata, communities, dataset_name, config):
    metadata = metadata.dropna(axis=1, how="all")
    # usage subset to dataset
    ds_usage = usage.loc[:, (dataset_name, slice(None), slice(None))].dropna(how="all").droplevel(axis=0, level=0)
    # number of bars in each community for this dataset
    community_gep_counts = [len([node for node in communities[c] if node.split("|")[0] == dataset_name]) for c in sorted(list(communities))]
    fig, axes = plt.subplots(
        metadata.shape[1], len(communities),
        figsize=[len(communities) + 0.05 * sum(community_gep_counts), metadata.shape[1] * 2],
        sharey='row',
        gridspec_kw={"width_ratios": [gep_counts for gep_counts in community_gep_counts]}, squeeze=False)
    for row, (annotation_layer, sample_to_numeric) in enumerate(metadata.items()):
        association = ds_usage.corrwith(sample_to_numeric, method="spearman")
        for col, community in enumerate(sorted(list(communities))):
            ax = axes[row, col]
            geps = []
            for node in communities[community]:
                dataset_str, k_str, gep_str = node.split("|")
                if dataset_str == dataset_name:
                    geps.append((dataset_str, int(k_str), int(gep_str)))
            geps = sorted(geps)
            if geps:
                association[geps].T.plot.bar(width=0.9, ax=ax, legend=None)
            ax.set_xlabel("")
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(annotation_layer)
            if row == 0:
                ax.set_title(community, size=14)

    fig.supxlabel("GEP")
    fig.supylabel("Pearson corr")
    fig.suptitle("Community")
    fig.tight_layout()
    return fig

def plot_metadata_correlation_network(graph, layout, title, correlation, plot_size, node_size, config, edge_weights=None):
     
    if edge_weights is None or not graph.edges:
        width = 0.2
    else:
        width = np.array(list(nx.get_edge_attributes(graph, edge_weights).values()))
        width = width / np.max(width)   

    color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1, 1), cmap=config.colormaps["diverging"])
    nodes_in_dataset = []
    colors = []
    for node in graph:
        if node in correlation:
            nodes_in_dataset.append(node)
            colors.append(color_map.to_rgba(correlation[node]))

    fig, ax = plt.subplots(figsize=(plot_size[0] * 1.15, plot_size[1]))
    nx.draw(graph, nodelist = nodes_in_dataset, pos=layout,
            with_labels=False, node_color=colors, node_size=node_size, linewidths=0, width=width, edge_color="#888888", font_size=4)
    ax.set_title(title)
    fig.colorbar(color_map, location="right", anchor=(0, 1), panchor=(1, 0), shrink=0.2, fraction=0.15)
    plt.tight_layout()
    return fig


def plot_icu_diversity(metadata, diversity, config, title):
    metadata = metadata.cat.add_categories("").fillna("").cat.remove_unused_categories()
    df = pd.concat([diversity.rename("diversity"), metadata], axis=1)
    fig, ax = plt.subplots(figsize=[df[metadata.name].nunique() + 1,10])
    if metadata.name == "Dataset":
        palette = {ds: ds_attr["color"] for ds, ds_attr in config.datasets.items()}
    else:
        palette = config.get_metadata_colors(metadata.name)
    palette[""] = config.metadata_colors["missing_data"]
    sns.stripplot(data=df, hue=metadata.name, x=metadata.name, y="diversity", palette=palette)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.boxplot(
        meanprops={'color': 'k', 'ls': '-', 'lw': 2},
        whiskerprops={'visible': False},
        zorder=10,
        x=metadata.name,
        y="diversity",
        data=df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax)
    ax.set_ybound(lower=0)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig