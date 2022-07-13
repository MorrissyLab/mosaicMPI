import mudata
import logging
import sys
from anndata import AnnData, read_h5ad
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from cnmfsns.config import Config

def add_cnmf_results_to_h5ad(cnmf_output_dir, cnmf_name, h5ad_path, local_density_threshold: float = None, force=False):
    adata = read_h5ad(h5ad_path)
    adata.uns["cnmf_name"] = cnmf_name
    cnmf_data_loaded =  "cnmf_usage" in adata.obsm or\
                        "cnmf_gep_score" in adata.varm or\
                        "cnmf_gep_tpm" in adata.varm or\
                        "cnmf_gep_raw" in adata.varm
    if cnmf_data_loaded and not force:
        logging.error(f"Error: {h5ad_path} already contains cNMF results. Use --force_h5ad_update to overwrite.")
        sys.exit(1)

    # infer from filenames which local density threshold was used
    sensed_ldts = set()
    for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.*spectra*.k_*")):
        ldt_str = os.path.basename(fn).split(".")[3]
        ldt = float(ldt_str.replace("dt_", "").replace("_", "."))
        sensed_ldts.add((ldt_str, ldt))
    if local_density_threshold is None and len(sensed_ldts) == 1:
        ldt_str, ldt = sensed_ldts.pop()
    elif local_density_threshold in (ldt[1] for ldt in sensed_ldts):
        ldt_str, ldt = [(ldt_str, ldt) for ldt_str, ldt in sensed_ldts if ldt == local_density_threshold].pop()
    else:
        logging.error(f"local_density_threshold of {local_density_threshold} does not match what is in the cNMF result directory: {sensed_ldts}")
        sys.exit(1)
    adata.uns["ldt"] = ldt
        
    # Import GEPs
    result_types = {
        "gene_spectra_score": "cnmf_gep_score",
        "gene_spectra_tpm": "cnmf_gep_tpm",
        "spectra": "cnmf_gep_raw"
        }
    for matchstr, result_type in result_types.items():
        meta_w = []
        for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.{matchstr}.k_*.{ldt_str}.*txt")):
            k = int(os.path.basename(fn).split(".")[2].replace("k_", ""))
            w = pd.read_table(fn, index_col=0)
            w.index = str(k) + "." + w.index.astype(str)
            meta_w.append(w)
        meta_w = pd.concat(meta_w, axis=0).T.reindex(adata.var.index).rename_axis(["k.gep"], axis=1)
        adata.varm[result_type] = meta_w

    # Import Usages
    usage = []
    for fn in glob(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}*.usages.k_*.{ldt_str}.*txt")):
        k = int(os.path.basename(fn).split(".")[2].replace("k_", ""))
        h = pd.read_table(fn, index_col=0)
        h.columns = str(k) + "." + h.columns.astype(str)
        usage.append(h)
    adata.obsm["cnmf_usage"] = pd.concat(usage, axis=1).sort_index(axis=1).rename_axis(["k.gep"], axis=1)
    
    # Import gene list used for factorization
    with open(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}.overdispersed_genes.txt")) as f:
        adata.uns["gene_list"] = [line.strip() for line in f.readlines()]

    # Import K-selection stats
    kvals = pd.DataFrame(**np.load(os.path.join(cnmf_output_dir, cnmf_name, f"{cnmf_name}.k_selection_stats.df.npz"), allow_pickle=True)).set_index("k")[["stability", "prediction_error"]]
    kvals.index = kvals.index.astype(int)
    adata.uns["kvals"] = kvals

    adata.write_h5ad(h5ad_path)



class CnmfResult(object):

    def __init__(self, run_name, ldt, gene_list, geps, usage, kvals, metadata):
        self.run_name = run_name
        self.ldt = ldt
        self.gene_list = gene_list
        self.geps = geps
        self.usage = usage
        self.kvals = kvals
        self.metadata = metadata
    
    def __repr__(self):
        repstr = [
            "CnmfResult",
            f"Run Name: {self.run_name}",
            f"Local Density Threshold: {self.ldt}",
            f"Usage: {self.usage.shape}",
            f"Gene Expression Programs (GEPs):"
        ]
        for result_type, df in self.geps.items():
            shape_no_nan = df.dropna(how="all").dropna(how="all", axis=1).shape
            repstr.append(f"  - {result_type}: {shape_no_nan}")
        return "\n".join(repstr)
    @property
    def gep_tpm(self):
        return self.geps["gene_spectra_tpm"]

    @property
    def gep_score(self):
        return self.geps["gene_spectra_score"]
    
    @property
    def gep_raw(self):
        return self.geps["spectra"]

    @classmethod
    def from_h5mu(cls, h5mu_file):
        logging.info(f"Reading {h5mu_file}")
        muobj = mudata.read_h5mu(h5mu_file)
        geps = {}
        for gep_type, ad in muobj.mod.items():
            meta_w = pd.concat(ad.varm, axis=1)
            meta_w.columns = pd.MultiIndex.from_tuples([(int(k), int(gep.split(".")[1])) for k, gep in meta_w.columns])
            geps[gep_type] = meta_w.sort_index(axis=1).T

        usage = pd.concat(muobj.mod["gene_spectra_score"].obsm, axis=1)
        usage.columns = pd.MultiIndex.from_tuples([(int(k), int(gep.split(".")[1])) for k, gep in usage.columns])
        usage = usage.sort_index(axis=1)
        return cls(**muobj.uns, geps=geps, usage=usage, metadata=muobj.obs)

    @classmethod
    def from_dir(cls, cnmf_result_dir, local_density_threshold: float = None):
        """
        if local_density_threshold is None, then infer from filenames (assuming only 1 threshold was used)
        """
        run_name = os.path.basename(os.path.normpath(cnmf_result_dir))
        
        # infer from filenames which local density threshold was used
        ldt_options = set()
        for fn in glob(os.path.join(cnmf_result_dir, f"*.*spectra*.k_*")):
            ldt_str = os.path.basename(fn).split(".")[3]
            ldt = float(ldt_str.replace("dt_", "").replace("_", "."))
            ldt_options.add((ldt_str, ldt))
        if local_density_threshold is None and len(ldt_options) == 1:
            ldt_str, ldt = ldt_options.pop()
        elif local_density_threshold in (ldt[1] for ldt in ldt_options):
            ldt_str, ldt = [(ldt_str, ldt) for ldt_str, ldt in ldt_options if ldt == local_density_threshold].pop()
        else:
            raise RuntimeError(f"local_density_threshold of {local_density_threshold} "
                               f"does not match what is in the cNMF result directory: {ldt_options}")
        
        # Import GEP files
        result_types = ["gene_spectra_score", "gene_spectra_tpm", "spectra"]
        geps = {}
        for result_type in result_types:
            meta_w = []
            for fn in glob(os.path.join(cnmf_result_dir, f"*.{result_type}.k_*.{ldt_str}.*txt")):
                k = int(os.path.basename(fn).split(".")[2].replace("k_", ""))
                w = pd.read_table(fn, index_col=0)
                w.index = pd.MultiIndex.from_arrays(([k] * w.shape[0], w.index))
                meta_w.append(w)
            meta_w = pd.concat(meta_w, axis=0).sort_index(axis=0).rename_axis(["k", "gep"], axis=0)
            geps[result_type] = meta_w

        # Import Usages matrix
        usage = []
        for fn in glob(os.path.join(cnmf_result_dir, f"*.usages.k_*.{ldt_str}.*txt")):
            k = int(os.path.basename(fn).split(".")[2].replace("k_", ""))
            h = pd.read_table(fn, index_col=0)
            h.columns = pd.MultiIndex.from_arrays(([k] * h.shape[1], h.columns.astype(int)))
            usage.append(h)
        usage = pd.concat(usage, axis=1).sort_index(axis=1).rename_axis(["k", "gep"], axis=1)
        
        # Import genes used for factorization
        with open(os.path.join(cnmf_result_dir, f"{run_name}.overdispersed_genes.txt")) as f:
            gene_list = [line.strip() for line in f.readlines()]

        # Import K-selection stats
        kvals = pd.DataFrame(**np.load(os.path.join(cnmf_result_dir, f"{run_name}.k_selection_stats.df.npz"), allow_pickle=True)).set_index("k")[["stability", "prediction_error"]]
        kvals.index = kvals.index.astype(int)

        # Import annotations from anndata
        metadata = read_h5ad(os.path.join(cnmf_result_dir, run_name + ".h5ad")).obs
        return cls(run_name, ldt, gene_list, geps, usage, kvals, metadata)

    def to_anndata(self, input_h5ad):
        input_adata = read_h5ad(input_h5ad)
        obsm = {}
        obsm["cnmf_usage"] = self.usage
        varm = {}
        varm["cnmf_gep_score"] = self.gep_score.T
        varm["cnmf_gep_tpm"] = self.gep_tpm.T
        varm["cnmf_gep_raw"] = self.gep_raw.T
        uns = {"run_name": self.run_name, "ldt": self.ldt, "gene_list": self.gene_list, "kvals": self.kvals}
        if input_adata:
            return AnnData(x=input_adata.X, raw=input_adata.raw, )
        else:
            return AnnData(X=pd.DataFrame(np.NaN, index=self.usage.index, columns=varm["cnmf_gep_score"].index), varm=varm, obsm=obsm, obs=self.metadata, uns=uns)

    def to_mudata(self):
        mu_dict = {gep_type: self.to_anndata(gep_type) for gep_type in self.geps.keys()}
        uns = {"run_name": self.run_name, "ldt": self.ldt, "gene_list": self.gene_list, "kvals": self.kvals}
        return mudata.MuData(mu_dict, uns=uns)


class Integration(object):
    def __init__(self, config: Config, output_dir: str, corr_method="pearson", min_corr=-1.0):
        self.cnmfresults = {name: CnmfResult.from_h5mu(os.path.join(output_dir, "input", "datasets", name + ".h5mu")) for name in config.datasets}
        self.corr_method = corr_method
        self.min_corr = min_corr
        self.output_dir = output_dir

        # merge GEPs into big (GEP x gene) dataframe. GEPs (rows) are indexed using (dataset, k, gep)
        geps = {dataset: cnmf_result.gep_score for dataset, cnmf_result in self.cnmfresults.items()}
        self.geps = pd.concat(geps)
        # merge usages into big (samples x GEP) dataframe. GEPs (columns) are indexed using (dataset, k, gep) and samples are indexed using (dataset, sample)
        usage = {
            dataset: pd.concat({dataset: cnmf_result.usage})
            for dataset, cnmf_result in self.cnmfresults.items()}
        self.usage = pd.concat(usage, axis=1)
        logging.info(f"Calculating {self.corr_method} correlation matrix...")
        if corr_method == "spearman":
            self.corr = self.geps.T.rank().corr(method="pearson")
        elif corr_method == "pearson":
            self.corr = self.geps.T.corr(method="pearson")
        else:
            raise ValueError(f"{self.corr_method} is not a valid correlation method. Please select either `pearson` or `spearman`.")

    def plot_pairwise_corr(self, show_threshold=True):
        n_datasets = len(self.cnmfresults)
        fig, axes = plt.subplots(n_datasets, n_datasets, figsize=[3 * n_datasets, 3 * n_datasets], sharex=True, sharey=True)
        for row, dataset_row in enumerate(self.corr.index.levels[0]):
            for col, dataset_col in enumerate(self.corr.columns.levels[0]):
                ax = axes[row,col]
                corr = self.corr.loc[dataset_row, dataset_col].values.flatten()
                if show_threshold:
                    hist_kwargs = {
                        "hue":[("Included" if c else "Excluded") for c in (corr > self.min_corr)],
                        "palette": {"Included": "red", "Excluded": "gray"},
                        "hue_order": ["Excluded", "Included"]
                    }
                    included_fraction = (corr > self.min_corr).sum() / corr.shape[0]
                    ax.text(x=0.01, y=0.99, s=f"{included_fraction:.3f}", ha='left', va='top', transform=ax.transAxes, color="red", alpha=0.5)
                else:
                    hist_kwargs = {"color": "gray"}
                sns.histplot(x=corr, ax=ax, linewidth=0,legend=(row == 0)&(col == 0), **hist_kwargs)
                secondary_ax = ax.twinx()
                sns.ecdfplot(x=corr, ax=secondary_ax, color="black", complementary=True)
                ax.set_ylabel(dataset_row)
                ax.set_xlabel(dataset_col)
                ax.set_xlim(-1,1)
                if col < n_datasets - 1:
                    secondary_ax.set_yticklabels([])
                    secondary_ax.set_ylabel("")
        fig.suptitle(f"{self.corr_method.capitalize()} correlation distribution between datasets")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "output", "correlation_distributions", self.corr_method + ".pdf"))
        fig.savefig(os.path.join(self.output_dir, "output", "correlation_distributions", self.corr_method + ".png"), dpi=600)
        return fig

    def plot_genelist_upset(self):
        import upsetplot
        genelists = {name: d.gene_list for name, d in self.cnmfresults.items()}
        fig = plt.Figure()
        upsetplot.UpSet(upsetplot.from_contents(genelists)).plot(fig=fig)
        fig.savefig(os.path.join(self.output_dir, "output", "overdispersed_genes", "upset_plot.pdf"))
        fig.savefig(os.path.join(self.output_dir, "output", "overdispersed_genes", "upset_plot.png"), dpi=600)
        return fig
        
