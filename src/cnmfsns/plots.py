import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import upsetplot
from scipy.cluster.hierarchy import linkage, dendrogram
from anndata import read_h5ad

def annotated_heatmap(
        data, title, metadata=None, metadata_colors=None,
        row_cluster=True, col_cluster=True, plot_col_dendrogram=True, show_sample_labels=True):

    n_columns = data.columns.shape[0]
    if metadata is None:
        n_metadata_columns = 0
    else:
        n_metadata_columns = metadata.shape[1]

    fig = plt.figure(figsize=[20, 2 + n_columns/3 + n_metadata_columns/4])
    fig.suptitle(title, fontsize=14)
    gs0 = matplotlib.gridspec.GridSpec(2,2, figure=fig,
                                    height_ratios=[n_columns/3, 1 + n_metadata_columns/4], hspace=0.05,
                                    width_ratios=[5,1], wspace=0.05)
    
    # subdivide heatmap and dendrogram
    gs1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs0[0],
                                                    height_ratios=[1,n_columns],
                                                    hspace=0)

    # Heatmap
    ax_heatmap = fig.add_subplot(gs1[1])
    ax_col_dendrogram = fig.add_subplot(gs1[0], sharex=ax_heatmap)
    ax_col_dendrogram.set_axis_off()

    # HAC clustering (compute linkage matrices)
    if col_cluster:
        col_links = linkage(data, method='average', metric='euclidean')
        if plot_col_dendrogram:
            col_dendrogram = dendrogram(col_links, color_threshold=0, ax=ax_col_dendrogram)
        else:
            col_dendrogram = dendrogram(col_links, no_plot=True)
        xind = col_dendrogram['leaves']
    else:
        xind = np.arange(0, data.shape[0])
    if row_cluster:
        row_links = linkage(data.T, method='average', metric='euclidean')
        row_dendrogram = dendrogram(row_links, no_plot=True)
        yind = row_dendrogram['leaves']
    else:
        yind = np.arange(0, data.shape[1])

    xmin,xmax = ax_col_dendrogram.get_xlim()
    im_heatmap = ax_heatmap.imshow(data.iloc[xind,yind].T, aspect='auto', extent=[xmin,xmax,0,1], cmap='YlOrRd', vmin=0, vmax=1, interpolation='none')
    ax_heatmap.set_yticks((data.columns.astype("int").to_series() - 0.5).div(data.shape[1]))
    ax_heatmap.set_yticklabels(data.columns[yind][::-1])
    ax_heatmap.set_ylabel("GEP", rotation=0, ha='right', va='center')
    ax_heatmap.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    

    # Annotations
    if metadata is not None:
        # data and metadata must have the same index
        metadata = metadata.loc[data.index]
        missing_data_color = metadata_colors["missing_data"]
    gs2 = matplotlib.gridspec.GridSpecFromSubplotSpec(metadata.shape[1], 1, subplot_spec=gs0[2])
    for i, (track, annot) in enumerate(metadata.iteritems()):
        ax = fig.add_subplot(gs2[i], sharex=ax_heatmap)
        ax.set_facecolor(missing_data_color)
        if annot.dtype == "category" or annot.dtype == "object":
            ordered_rgb = annot.iloc[xind].replace(metadata_colors[track])
            if ordered_rgb.isnull().any():
                ordered_rgb = ordered_rgb.cat.add_categories(missing_data_color)
                ordered_rgb = ordered_rgb.fillna(missing_data_color)
            ordered_rgb = ordered_rgb.astype("object").map(matplotlib.colors.to_rgb)
            ordered_rgb = np.array([list(rgb) for rgb in ordered_rgb])
            ax.imshow(np.stack([ordered_rgb, ordered_rgb]), aspect='auto', extent=[xmin,xmax,0,1], interpolation='none')
        else:
            ax.imshow(np.stack([annot.iloc[xind],annot.iloc[xind]]), aspect='auto', extent=[xmin,xmax,0,1], cmap='Blues', interpolation='none')
        ax.set_yticks([])
        ax.set_ylabel(track, rotation=0, ha='right', va='center')
        if ax.get_subplotspec().is_last_row():
            if show_sample_labels:
                ax.set_xticks(np.linspace(0, 1, data.shape[0], endpoint=False) + 1/(2 * data.shape[0]))
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
    
    # Category Legend
    from matplotlib.patches import Patch
    ax = fig.add_subplot(gs0[3])
    # Add legend
    legend_elements = []
    for track, color_def in metadata_colors.items():
        if track in metadata.columns:
            legend_elements.append(Patch(label=track, facecolor='white', edgecolor=None, ))
            for cat, color in color_def.items():
                if cat in metadata[track].astype("category").cat.categories:
                    legend_elements.append(Patch(label=cat, facecolor=color, edgecolor=None))
            # if metadata[track].isnull().any():
            #     legend_elements.append(Patch(label="Other", facecolor=missing_data_color, edgecolor=None))

    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_axis_off()
    return fig

def plot_annotated_usages(df, metadata, metadata_colors, title, filename, cluster_geps, cluster_samples, show_sample_labels):
    samples = df.index.to_series()
    df = df.div(df.sum(axis=1), axis=0)
    annotations = metadata.loc[samples]
    fig = annotated_heatmap(data=df, metadata=annotations, metadata_colors=metadata_colors, title=title, row_cluster=cluster_geps, col_cluster=cluster_samples, show_sample_labels=show_sample_labels)
    fig.savefig(filename, transparent=False, bbox_inches = "tight")
    plt.close(fig)  
    return fig

def plot_rank_reduction(df, max_median_corr_threshold):
    df = df.copy()    
    df["max_k"] = df.index
    fig, ax = plt.subplots(figsize=[df["max_k"].shape[0]/8, 6])
    ax.set_ylim([-1, 1])
    ax.set_xlim([df["max_k"].min() - 1, df["max_k"].max() + 1])
    ax.axhline(max_median_corr_threshold, color="red")
    ax.set_title("Median of the Correlation Distribution vs max_k")
    sns.lineplot(data=df, x="max_k", y="max_k_median_corr", hue="max_k_filter_pass", ax=ax, marker="o")
    plt.tight_layout()
    return fig

def plot_pairwise_corr(tril, thresholds=None):
    n_datasets = len(tril.index.levels[0])
    fig, axes = plt.subplots(n_datasets, n_datasets, figsize=[3 * n_datasets, 3 * n_datasets], sharex=True, sharey=True)
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            ax = axes[row,col]
            if row < col:
                ax.set_axis_off()
            else:
                corr = pd.Series(tril.loc[dataset_row, dataset_col].values.flatten()).dropna()
                if thresholds is not None:
                    min_corr = thresholds.loc[(dataset_row, dataset_col)].values[0]
                    hist_kwargs = {
                        "hue":[("Included" if c else "Excluded") for c in (corr > min_corr)],
                        "palette": {"Included": "red", "Excluded": "gray"},
                        "hue_order": ["Excluded", "Included"],
                        "linewidth": 0
                    }
                    included_fraction = (corr > min_corr).sum() / corr.shape[0]
                    ax.text(x=0.01, y=0.99, s=f"{included_fraction:.3f}", ha='left', va='top', transform=ax.transAxes, color="red", alpha=0.5)
                else:
                    hist_kwargs = {
                        "color": "gray",
                        "linewidth": 0
                    }
                sns.histplot(x=corr, ax=ax,legend=(row == 0)&(col == 0), **hist_kwargs)
                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(-1,1)
    fig.suptitle(f"Correlation distribution between datasets")
    plt.tight_layout()
    return fig

def plot_pairwise_corr_overlaid(tril, thresholds):
    n_datasets = len(tril.index.levels[0])
    fig, axes = plt.subplots(n_datasets, n_datasets, figsize=[3 * n_datasets, 3 * n_datasets], sharex=True, sharey=True)
    for row, dataset_row in enumerate(tril.index.levels[0]):
        for col, dataset_col in enumerate(tril.columns.levels[0]):
            ax = axes[row,col]
            if row < col:
                ax.set_axis_off()
            else:
                corr = pd.DataFrame({"abscorr": tril.loc[dataset_row, dataset_col].values.flatten()}).dropna()
                corr["sign"] = (corr["abscorr"] >= 0).map({True: "Positive", False: "Negative"})
                corr["abscorr"] = corr["abscorr"].abs()
                sns.histplot(data=corr, x="abscorr", hue="sign", palette= {"Positive": "red", "Negative": "lightblue"}, alpha=0.5, linewidth=0,
                             hue_order= ["Negative", "Positive"], ax=ax,legend=(row == 0)&(col == 0))

                # show min_corr as text in top left of plot and vertical line
                min_corr = thresholds.loc[(dataset_row, dataset_col)].values[0]
                # included_fraction = (corr["abscorr"] > min_corr).sum() / corr.shape[0]  # could also show the quantile of the min_corr threshold
                ax.text(x=0.01, y=0.99, s=f"{min_corr:.3f}", ha='left', va='top', transform=ax.transAxes, color="black")
                ax.axvline(min_corr, color="black")
                
                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(0,1)
    fig.suptitle(f"Correlation distribution between datasets")
    plt.tight_layout()
    return fig

def plot_genelist_upsets(config):
    overdispersed_genelists = {}
    full_genelists = {}
    for dataset_name, d in config.datasets.items():
        adata = read_h5ad(d["filename"])
        overdispersed_genelists[dataset_name] = list(adata.uns["gene_list"])
        full_genelists[dataset_name] = list(adata.var.index)
    figs = {}
    fig = plt.Figure()
    upsetplot.UpSet(upsetplot.from_contents(overdispersed_genelists)).plot(fig=fig)
    figs["overdispersed_genes.upset"] = fig
    fig = plt.Figure()
    upsetplot.UpSet(upsetplot.from_contents(full_genelists)).plot(fig=fig)
    figs["all_genes.upset"] = fig
    return figs
