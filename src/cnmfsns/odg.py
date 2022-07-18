import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cnmf

def model_overdispersion(adata, odg_default_spline_degree=3, odg_default_dof=8, odg_cnmf_mean_threshold=0.5, selected_genes = []):

    # create dataframe of per-gene statistics
    df = pd.DataFrame(index=adata.var.index)
    df["mean"] = adata.X.mean(0)
    df["rank_mean"] = df["mean"].rank()
    df["variance"] = adata.X.var(0)
    df["sd"] = np.sqrt(df["variance"])
    df["missingness"] = np.isnan(adata.X).sum(0)/adata.shape[0]
    df[["log_mean", "log_variance"]] = np.log10(df[["mean", "variance"]])
    df["odscore_excluded"] = (df["missingness"] > 0) | df["log_mean"].isnull() | (df["mean"] == 0) | df["log_variance"].isnull()
    df = df.sort_values("mean")

    # model mean-variance relationship using generalized additive model with smooth components
    df_model = df[~df["odscore_excluded"]]
    bs = BSplines(df_model["mean"], df=odg_default_dof, degree=odg_default_spline_degree)
    gam = GLMGam.from_formula("log_variance ~ log_mean", data=df_model, smoother=bs).fit()
    df["resid_log_variance"] = gam.resid_response
    df["odscore"] = np.sqrt(10 ** df["resid_log_variance"])
    df["gam_fittedvalues"] = gam.fittedvalues


    # model mean-variance relationship using cNMF's method based on v-score and minimum expression threshold
    vscore_stats = pd.DataFrame(cnmf.cnmf.get_highvar_genes(input_counts=adata.X)[0])
    vscore_stats.index = adata.var.index
    df["vscore"] = vscore_stats["fano_ratio"]
    df["vscore_excluded"] = df["mean"] < odg_cnmf_mean_threshold
    df.loc[df["vscore_excluded"], "vscore"] = np.NaN
    # include selected genes if provided
    df["selected"] = df.index.isin(selected_genes)
    return df

def create_diagnostic_plots(df, show_selected):
    """
    Create diagnostic plots for data from model_overdispersion()
    """
    figs = {}

    # Figure: Default model mean and variance
    fig, axes = plt.subplots(1, 3, figsize=[12, 4])
    
    ax = axes[0]  # untransformed
    if show_selected:
        sns.histplot(df, x="log_mean", y="log_variance", hue="selected", bins=[100,100], ax=ax, alpha=0.5, palette={True: "red", False: "blue"})
        ax.legend(handles=[
            Patch(color="blue", alpha=0.5, label="False"),
            Patch(color="red", alpha=0.5, label="True"),
            Line2D([0], [0], color='green', label="model")
        ], title="selected")
    else:
        sns.histplot(df, x="log_mean", y="log_variance", bins=[100,100], ax=ax, cmap="Blues")
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
        sns.histplot(df, x="log_mean", y="odscore", bins=[100,100], ax=ax, cmap="Blues")
    ax.hlines(1, xmin=df["log_mean"].min(), xmax=df["log_mean"].max(), color="green")
    ax.set_title("Mean-Overdispersion (Default)")
    ax.set_xlabel("log10(mean)")
    ax.set_ylabel("od-score")
    
    ax=axes[2]  # thresholds
    ax2=ax.twinx()
    
    ax.set_title("od-score Distribution")
    if df["odscore"].notnull().any():
        if show_selected:
            sns.histplot(df, x="odscore", hue="selected", bins=100, linewidth=0, ax=ax, palette={True: "red", False: "blue"})
            ax.legend(handles=[
                Patch(color="blue", alpha=0.5, label="False"),
                Patch(color="red", alpha=0.5, label="True")
            ], title="selected")
        else:
            sns.histplot(df, x="odscore", bins=100, linewidth=0, ax=ax, cmap="Blues")
        sns.ecdfplot(df, x="odscore", ax=ax2, color="black", stat="count", complementary=True)
    ax.set_xlabel("od-score")
    ax2.set_ylabel("Total Gene Count")
    plt.tight_layout()
    plot_id = ("default",)
    figs[plot_id] = fig

    # Figure: cnmf model mean and variance
    fig, axes = plt.subplots(1, 3, figsize=[12, 4])
    ax = axes[0]
    if show_selected:
        sns.histplot(df, x="log_mean", y="log_variance", hue="selected", bins=[100,100], ax=ax, alpha=0.5, palette={True: "red", False: "blue"})
    else:
        sns.histplot(df, x="log_mean", y="log_variance", bins=[100,100], ax=ax, cmap="Blues")
    ax.set_title("Mean-Variance")
    ax.set_xlabel("log10(mean)")
    ax.set_ylabel("log10(variance)")
    ax = axes[1]
    if show_selected:
        sns.histplot(df, x="log_mean", y="vscore", hue="selected", bins=[100,100], ax=ax, palette={True: "red", False: "blue"})
    else:
        sns.histplot(df, x="log_mean", y="vscore", bins=[100,100], ax=ax, cmap="Blues")
    ax.set_title("Mean-Overdispersion (cnmf)")
    ax.set_xlabel("log10(mean)")
    ax.set_ylabel("v-score")
    ax=axes[2]  # thresholds
    ax2=ax.twinx()
    ax.set_title("v-score Distribution")
    if df["vscore"].notnull().any():
        if show_selected:
            sns.histplot(df, x="vscore", hue="selected", bins=100, linewidth=0, ax=ax, palette={True: "red", False: "blue"})
            ax.legend(handles=[
                Patch(color="blue", alpha=0.5, label="False"),
                Patch(color="red", alpha=0.5, label="True")
            ], title="selected")
        else:
            sns.histplot(df, x="vscore", bins=100, linewidth=0, ax=ax, cmap="Blues")
        sns.ecdfplot(df, x="vscore", ax=ax2, color="black", stat="count", complementary=True)
    ax.set_xlabel("v-score")
    ax2.set_ylabel("Total Gene Count")
    plt.tight_layout()
    plot_id = ("cnmf",)
    figs[plot_id] = fig

    fig, ax = plt.subplots(figsize=[12,12])
    sns.histplot(data=df.fillna(0), x="odscore", y="vscore", bins=[100, 100], ax=ax)
    ax.set_xlabel("od-score")
    ax.set_ylabel("v-score")
    plt.tight_layout()
    plot_id = ("score_comparison",)
    figs[plot_id] = fig
    return figs

def fetch_hgnc_protein_coding_genes():
    import httplib2 as http
    import json
    from urllib.parse import urlparse

    headers = {
    'Accept': 'application/json',
    }

    uri = 'http://rest.genenames.org'
    path = '/search/locus_type/%22gene with protein product22'
    target = urlparse(uri+path)
    method = 'GET'
    body = ''

    h = http.Http()

    response, content = h.request(
    target.geturl(),
    method,
    body,
    headers)

    if response['status'] == '200':
        # assume that content is a json reply
        # parse content with the json module 
        data = json.loads(content)
        # print('Symbol:' + data['response']['docs'][0]['symbol'])

    else:
        print('Error detected: ' + response['status'])

    protein_coding_genes = {entry["symbol"] for entry in data["response"]["docs"]}
    return protein_coding_genes