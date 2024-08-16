## Code adapted and improved from https://github.com/dylkot/cNMF/blob/master/src/cnmf/cnmf.py

from . import utils

import os
import errno
import datetime
import logging
import uuid
import itertools
import subprocess
from functools import partial
from multiprocessing.pool import Pool
from collections.abc import Collection, Iterable
from typing import Union, Optional

import yaml
import scipy.sparse as sp
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import squareform
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list
import matplotlib.pyplot as plt



def _worker_filter(iterable, worker_index, total_workers):
    return (p for i,p in enumerate(iterable) if (i-worker_index)%total_workers==0)

def fast_ols_all_cols(X, Y):
    pinv = np.linalg.pinv(X)
    beta = np.dot(pinv, Y)
    return(beta)


def get_highvar_genes_sparse(expression, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_mean = np.array(expression.mean(axis=0)).astype(float).reshape(-1)
    E2 = expression.copy(); E2.data **= 2; gene2_mean = np.array(E2.mean(axis=0)).reshape(-1)
    gene_var = pd.Series(gene2_mean - (gene_mean**2))
    del(E2)
    gene_mean = pd.Series(gene_mean)
    gene_fano = gene_var / gene_mean

    # Find parameters for expected fano line
    top_genes = gene_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_var)/gene_mean)[top_genes].min()
    
    w_mean_low, w_mean_high = gene_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_fano > w_fano_low) &
                    (gene_fano < w_fano_high) &
                    (gene_mean > w_mean_low) &
                    (gene_mean < w_mean_high))
    fano_median = gene_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_mean + (B**2)
    fano_ratio = (gene_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None


    else:
        if not expected_fano_threshold:
            T = (1. + gene_fano[winsor_box].std())
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_mean,
        'var': gene_var,
        'fano': gene_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)



def get_highvar_genes(input_counts, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_counts_mean = pd.Series(input_counts.mean(axis=0).astype(float))
    gene_counts_var = pd.Series(input_counts.var(ddof=0, axis=0).astype(float))
    gene_counts_fano = pd.Series(gene_counts_var/gene_counts_mean)

    # Find parameters for expected fano line
    top_genes = gene_counts_mean.sort_values(ascending=False).iloc[:20].index
    A = (np.sqrt(gene_counts_var)/gene_counts_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_counts_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_counts_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_counts_fano > w_fano_low) &
                    (gene_counts_fano < w_fano_high) &
                    (gene_counts_mean > w_mean_low) &
                    (gene_counts_mean < w_mean_high))
    fano_median = gene_counts_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_counts_mean + (B**2)

    fano_ratio = (gene_counts_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None


    else:
        if not expected_fano_threshold:
            T = (1. + gene_counts_fano[winsor_box].std())
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_counts_mean,
        'var': gene_counts_var,
        'fano': gene_counts_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)


class cNMF():
    """Legacy cNMF object based off of the cNMF package
    """

    def __init__(self, output_dir=".", name=None):
        """

        :param output_dir: Place to put cNMF resutls, defaults to "."
        :type output_dir: str, optional
        :param name: A name for this analysis. Will be prefixed to all output files, defaults to automatically generated timestamp (and random string).
        :type name: str, optional
        """


        self.output_dir = output_dir
        if name is None:
            now = datetime.datetime.now()
            rand_hash =  uuid.uuid4().hex[:6]
            name = '%s_%s' % (now.strftime("%Y_%m_%d"), rand_hash)
        self.name = name
        self.paths = None
        self._initialize_dirs()


    def _initialize_dirs(self):
        if self.paths is None:
            # Check that output directory exists, create it if needed.
            os.makedirs(os.path.join(self.output_dir, self.name, 'cnmf_tmp'), exist_ok=True)

            self.paths = {
                'normalized_counts' : os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.norm_counts.h5ad'),
                'nmf_replicate_parameters' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.nmf_params.df.npz'),
                'nmf_run_parameters' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.nmf_idvrun_params.yaml'),
                'nmf_genes_list' :  os.path.join(self.output_dir, self.name, self.name+'.overdispersed_genes.txt'),

                'tpm' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.tpm.h5ad'),
                'tpm_stats' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.tpm_stats.df.npz'),

                'iter_spectra' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.iter_%d.df.npz'),
                'iter_usages' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.usages.k_%d.iter_%d.df.npz'),
                'merged_spectra': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.merged.df.npz'),

                'local_density_cache': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.local_density_cache.k_%d.merged.df.npz'),
                'consensus_spectra': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.dt_%s.consensus.df.npz'),
                'consensus_spectra__txt': os.path.join(self.output_dir, self.name, self.name+'.spectra.k_%d.dt_%s.consensus.txt'),
                'consensus_usages': os.path.join(self.output_dir, self.name, 'cnmf_tmp',self.name+'.usages.k_%d.dt_%s.consensus.df.npz'),
                'consensus_usages__txt': os.path.join(self.output_dir, self.name, self.name+'.usages.k_%d.dt_%s.consensus.txt'),

                'consensus_stats': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.stats.k_%d.dt_%s.df.npz'),

                'clustering_plot': os.path.join(self.output_dir, self.name, self.name+'.clustering.k_%d.dt_%s.png'),
                'gene_spectra_score': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.gene_spectra_score.k_%d.dt_%s.df.npz'),
                'gene_spectra_score__txt': os.path.join(self.output_dir, self.name, self.name+'.gene_spectra_score.k_%d.dt_%s.txt'),
                'gene_spectra_tpm': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.gene_spectra_tpm.k_%d.dt_%s.df.npz'),
                'gene_spectra_tpm__txt': os.path.join(self.output_dir, self.name, self.name+'.gene_spectra_tpm.k_%d.dt_%s.txt'),

                'k_selection_plot' :  os.path.join(self.output_dir, self.name, self.name+'.k_selection.png'),
                'k_selection_stats' :  os.path.join(self.output_dir, self.name, self.name+'.k_selection_stats.df.npz'),
            }
        
        
    def get_nmf_iter_params(self, ks, n_iter = 100,
                               random_state_seed = None,
                               beta_loss = 'kullback-leibler',
                               alpha_usage=0.0, alpha_spectra=0.0,
                               init='random'):
        """_summary_

        :param ks: Number of topics (components) for factorization.
            Several values can be specified at the same time, which will be run independently.
        :type ks: integer or list-like
        :param n_iter: Number of iterations for factorization. If several ``k`` are specified, this many
            iterations will be run for each value of ``k``. defaults to 100
        :type n_iter: int, optional
        :param random_state_seed: Seed for sklearn random state. defaults to None
        :type random_state_seed: int, optional
        :param beta_loss: defaults to 'kullback-leibler'
        :type beta_loss: str, optional
        :param alpha_usage: Regularization parameter for NMF corresponding to alpha_W in scikit-learn, defaults to 0.0
        :type alpha_usage: float, optional
        :param alpha_spectra: Regularization parameter for NMF corresponding to alpha_H in scikit-learn, defaults to 0.0
        :type alpha_spectra: float, optional
        :param init: defaults to 'random'
        :type init: str, optional
        """

        if type(ks) is int:
            ks = [ks]

        # Remove any repeated k values, and order.
        k_list = sorted(set(list(ks)))

        n_runs = len(ks)* n_iter

        np.random.seed(seed=random_state_seed)
        nmf_seeds = np.random.randint(low=1, high=(2**16)-1, size=n_runs)

        replicate_params = []
        for i, (k, r) in enumerate(itertools.product(k_list, range(n_iter))):
            replicate_params.append([k, r, nmf_seeds[i]])
        replicate_params = pd.DataFrame(replicate_params, columns = ['n_components', 'iter', 'nmf_seed'])

        _nmf_kwargs = dict(
                        alpha_W=alpha_usage,
                        alpha_H=alpha_spectra,
                        l1_ratio=0.0,
                        beta_loss=beta_loss,
                        solver='mu',
                        tol=1e-4,
                        max_iter=1000,
                        init=init
                        )
        
        ## Coordinate descent is faster than multiplicative update but only works for frobenius
        if beta_loss == 'frobenius':
            _nmf_kwargs['solver'] = 'cd'

        return(replicate_params, _nmf_kwargs)


    def save_nmf_iter_params(self, replicate_params, run_params):
        self._initialize_dirs()
        utils.save_df_to_npz(replicate_params, self.paths['nmf_replicate_parameters'])
        with open(self.paths['nmf_run_parameters'], 'w') as F:
            yaml.dump(run_params, F)


    def _nmf(self, X, nmf_kwargs):
        """

        :param X: Normalized counts dataFrame to be factorized.
        :type X: pd.DataFrame
        :param nmf_kwargs: Arguments to be passed to ``non_negative_factorization``
        :type nmf_kwargs: dict
        """
        (usages, spectra, niter) = non_negative_factorization(X, **nmf_kwargs)

        return(spectra, usages)


    def factorize(self,
                worker_i=0, total_workers=1, verbose=True
                ):
        """
        Iteratively run NMF with prespecified parameters.
        Use the `worker_i` and `total_workers` parameters for parallelization.
        Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'], defaults below::

            non_negative_factorization default arguments:
                alpha=0.0
                l1_ratio=0.0
                beta_loss='kullback-leibler'
                solver='mu'
                tol=1e-4,
                max_iter=200
                regularization=None
                init='random'
                random_state, n_components are both set by the prespecified self.paths['nmf_replicate_parameters'].

        :param worker_i: worker index, defaults to 0
        :type worker_i: int, optional
        :param total_workers: total number of workers, defaults to 1
        :type total_workers: int, optional
        :param verbose: verbose, defaults to True
        :type verbose: bool, optional
        """
        run_params = utils.load_df_from_npz(self.paths['nmf_replicate_parameters'])
        norm_counts = ad.read_h5ad(self.paths['normalized_counts'])
        _nmf_kwargs = yaml.load(open(self.paths['nmf_run_parameters']), Loader=yaml.FullLoader)

        jobs_for_this_worker = _worker_filter(range(len(run_params)), worker_i, total_workers)
        for idx in jobs_for_this_worker:
            p = run_params.iloc[idx, :]
            if verbose:
                logging.info('[Worker %d] Starting task %d.' % (worker_i, idx))
            _nmf_kwargs['random_state'] = p['nmf_seed']
            _nmf_kwargs['n_components'] = p['n_components']

            (spectra, usages) = self._nmf(norm_counts.X, _nmf_kwargs)
            spectra = pd.DataFrame(spectra,
                                   index=np.arange(1, _nmf_kwargs['n_components']+1),
                                   columns=norm_counts.var.index)
            utils.save_df_to_npz(spectra, self.paths['iter_spectra'] % (p['n_components'], p['iter']))


    def combine_nmf(self, k, skip_missing_files=False, remove_individual_iterations=False):
        run_params = utils.load_df_from_npz(self.paths['nmf_replicate_parameters'])
        logging.info('Combining factorizations for k=%d.'%k)

        run_params_subset = run_params[run_params.n_components==k].sort_values('iter')
        combined_spectra = []

        for i,p in run_params_subset.iterrows():
            current_file = self.paths['iter_spectra'] % (p['n_components'], p['iter'])
            if not os.path.exists(current_file):
                if not skip_missing_files:
                    print('Missing file: %s, run with skip_missing=True to override' % current_file)
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), current_file)
                else:
                    print('Missing file: %s. Skipping.' % current_file)
            else:
                spectra = utils.load_df_from_npz(current_file)
                spectra.index = ['iter%d_topic%d' % (p['iter'], t+1) for t in range(k)]
                combined_spectra.append(spectra)
                
        if len(combined_spectra)>0:        
            combined_spectra = pd.concat(combined_spectra, axis=0)
            utils.save_df_to_npz(combined_spectra, self.paths['merged_spectra']%k)
        else:
            print('No spectra found for k=%d' % k)
        return combined_spectra
    
    
    def refit_usage(self, X, spectra) -> pd.DataFrame:
        """Takes an input data matrix and a fixed spectra and uses NNLS to find the optimal
        usage matrix. Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'].
        If input data are pandas.DataFrame, returns a DataFrame with row index matching X and
        columns index matching index of spectra

        :param X: Non-negative expression data to fit spectra to
        :type X: pd.DataFrame or np.ndarray, cells x genes
        :param spectra: Non-negative spectra of expression programs
        :type spectra: pandas.DataFrame or numpy.ndarray, programs X genes
        :return: refit usages
        :rtype: pd.DataFrame
        """
        refit_nmf_kwargs = yaml.load(open(self.paths['nmf_run_parameters']), Loader=yaml.FullLoader)
        if type(spectra) is pd.DataFrame:
            refit_nmf_kwargs.update(dict(n_components = spectra.shape[0], H = spectra.values, update_H = False))
        else:
            refit_nmf_kwargs.update(dict(n_components = spectra.shape[0], H = spectra, update_H = False))
            
        _, rf_usages = self._nmf(X, nmf_kwargs=refit_nmf_kwargs)
        if (type(X) is pd.DataFrame) and (type(spectra) is pd.DataFrame):
            rf_usages = pd.DataFrame(rf_usages, index=X.index, columns=spectra.index)
          
        return(rf_usages)
    
    
    def refit_spectra(self, X, usage) -> pd.DataFrame:
        """Takes an input data matrix and a fixed usage matrix and uses NNLS to find the optimal
        spectra matrix. Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'].
        If input data are pandas.DataFrame, returns a DataFrame with row index matching X and
        columns index matching index of spectra

        :param X: Non-negative expression data to fit spectra to
        :type X: pd.DataFrame or np.ndarray, cells x genes
        :param usage: Non-negative spectra of expression programs
        :type usage: pandas.DataFrame or numpy.ndarray, cells X genes
        :return: refit spectra
        :rtype: pd.DataFrame
        """

        return(self.refit_usage(X.T, usage.T).T)


    def consensus(self, k, density_threshold=0.5, local_neighborhood_size = 0.30,show_clustering = True,
                  skip_density_and_return_after_stats = False, close_clustergram_fig=False,
                  refit_usage=True):
        merged_spectra = utils.load_df_from_npz(self.paths['merged_spectra']%k)
        norm_counts = ad.read_h5ad(self.paths['normalized_counts'])

        density_threshold_str = str(density_threshold)
        if skip_density_and_return_after_stats:
            density_threshold_str = '2'
        density_threshold_repl = density_threshold_str.replace('.', '_')
        n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)

        # Rescale topics such to length of 1.
        l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T

        if not skip_density_and_return_after_stats:
            # Compute the local density matrix (if not previously cached)
            topics_dist = None
            if os.path.isfile(self.paths['local_density_cache'] % k):
                local_density = utils.load_df_from_npz(self.paths['local_density_cache'] % k)
            else:
                #   first find the full distance matrix
                topics_dist = euclidean_distances(l2_spectra.values)
                #   partition based on the first n neighbors
                partitioning_order  = np.argpartition(topics_dist, n_neighbors+1)[:, :n_neighbors+1]
                #   find the mean over those n_neighbors (excluding self, which has a distance of 0)
                distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
                local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1)/(n_neighbors),
                                             columns=['local_density'],
                                             index=l2_spectra.index)
                utils.save_df_to_npz(local_density, self.paths['local_density_cache'] % k)
                del(partitioning_order)
                del(distance_to_nearest_neighbors)

            density_filter = local_density.iloc[:, 0] < density_threshold
            l2_spectra = l2_spectra.loc[density_filter, :]

        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        kmeans_model.fit(l2_spectra)
        kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

        # Find median usage for each gene across cluster
        median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

        # Normalize median spectra to probability distributions.
        median_spectra = (median_spectra.T/median_spectra.sum(1)).T

        # Compute the silhouette score
        stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')

        # Obtain reconstructed count matrix by re-fitting usage and computing dot product: usage.dot(spectra)
        rf_usages = self.refit_usage(norm_counts.X, median_spectra)
        rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index, columns=median_spectra.index)        
        rf_pred_norm_counts = rf_usages.dot(median_spectra)
        
        # Re-order usage by total contribution
        norm_usages = rf_usages.div(rf_usages.sum(axis=1), axis=0)      
        reorder = norm_usages.sum(axis=0).sort_values(ascending=False)
        rf_usages = rf_usages.loc[:, reorder.index]
        norm_usages = norm_usages.loc[:, reorder.index]
        median_spectra = median_spectra.loc[reorder.index, :]
        rf_usages.columns = np.arange(1, rf_usages.shape[1]+1)
        norm_usages.columns = rf_usages.columns
        median_spectra.index = rf_usages.columns

        # Compute prediction error as a frobenius norm
        if sp.issparse(norm_counts.X):
            prediction_error = ((norm_counts.X.todense() - rf_pred_norm_counts)**2).sum().sum()
        else:
            prediction_error = ((norm_counts.X - rf_pred_norm_counts)**2).sum().sum()
        
        consensus_stats = pd.DataFrame([k, density_threshold, stability, prediction_error],
                    index = ['k', 'local_density_threshold', 'stability', 'prediction_error'],
                    columns = ['stats'])

        if skip_density_and_return_after_stats:
            return consensus_stats
        
        # Convert spectra to TPM units, and obtain results for all genes by running last step of NMF
        # with usages fixed and TPM as the input matrix
        tpm = ad.read_h5ad(self.paths['tpm'])
        tpm_stats = utils.load_df_from_npz(self.paths['tpm_stats'])
        norm_usages = norm_usages.fillna(0).astype(np.float32)  # replaces NaN usages for samples that have 0 HVG counts
        spectra_tpm = self.refit_spectra(tpm.X, norm_usages)
        spectra_tpm = pd.DataFrame(spectra_tpm, index=rf_usages.columns, columns=tpm.var.index)
        spectra_tpm = spectra_tpm.div(spectra_tpm.sum(axis=1), axis=0) * 1e6
        
        # Convert spectra to Z-score units, and obtain results for all genes by running last step of NMF
        # with usages fixed and Z-scored TPM as the input matrix
        if sp.issparse(tpm.X):
            norm_tpm = (np.array(tpm.X.todense()) - tpm_stats['__mean'].values) / tpm_stats['__std'].values
        else:
            norm_tpm = (tpm.X - tpm_stats['__mean'].values) / tpm_stats['__std'].values
        
        usage_coef = fast_ols_all_cols(rf_usages.values, norm_tpm)
        usage_coef = pd.DataFrame(usage_coef, index=rf_usages.columns, columns=tpm.var.index)
        
        if refit_usage:
            ## Re-fitting usage a final time on std-scaled HVG TPM seems to
            ## increase accuracy on simulated data
            hvgs = open(self.paths['nmf_genes_list']).read().split('\n')
            norm_tpm = tpm[:, hvgs]
            if sp.issparse(norm_tpm.X):
                raise NotImplementedError("Sparse functions not implemented in mosaicMPI yet")
            else:
                norm_tpm.X /= norm_tpm.X.std(axis=0, ddof=1)
                
            spectra_tpm_rf = spectra_tpm.loc[:,hvgs]
            tpm_stats.index = tpm.var.index

            spectra_tpm_rf = spectra_tpm_rf.div(tpm_stats.loc[hvgs, '__std'], axis=1)
            rf_usages = self.refit_usage(norm_tpm.X, spectra_tpm_rf)
            rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index, columns=spectra_tpm_rf.index)                                                                  
               
        utils.save_df_to_npz(median_spectra, self.paths['consensus_spectra']%(k, density_threshold_repl))
        utils.save_df_to_npz(rf_usages, self.paths['consensus_usages']%(k, density_threshold_repl))
        utils.save_df_to_npz(consensus_stats, self.paths['consensus_stats']%(k, density_threshold_repl))
        utils.save_df_to_text(median_spectra, self.paths['consensus_spectra__txt']%(k, density_threshold_repl))
        utils.save_df_to_text(rf_usages, self.paths['consensus_usages__txt']%(k, density_threshold_repl))
        utils.save_df_to_npz(spectra_tpm, self.paths['gene_spectra_tpm']%(k, density_threshold_repl))
        utils.save_df_to_text(spectra_tpm, self.paths['gene_spectra_tpm__txt']%(k, density_threshold_repl))
        utils.save_df_to_npz(usage_coef, self.paths['gene_spectra_score']%(k, density_threshold_repl))
        utils.save_df_to_text(usage_coef, self.paths['gene_spectra_score__txt']%(k, density_threshold_repl))
        if show_clustering:
            if topics_dist is None:
                topics_dist = euclidean_distances(l2_spectra.values)
                # (l2_spectra was already filtered using the density filter)
            else:
                # (but the previously computed topics_dist was not!)
                topics_dist = topics_dist[density_filter.values, :][:, density_filter.values]


            spectra_order = []
            for cl in sorted(set(kmeans_cluster_labels)):

                cl_filter = kmeans_cluster_labels==cl

                if cl_filter.sum() > 1:
                    cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter], checks=False)
                    cl_dist[cl_dist < 0] = 0 #Rarely get floating point arithmetic issues
                    cl_link = linkage(cl_dist, 'average')
                    cl_leaves_order = leaves_list(cl_link)

                    spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
                else:
                    ## Corner case where a component only has one element
                    spectra_order += list(np.where(cl_filter)[0])


            from matplotlib import gridspec
            import matplotlib.pyplot as plt

            width_ratios = [0.5, 9, 0.5, 4, 1]
            height_ratios = [0.5, 9]
            fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
                                    0.01, 0.01, 0.98, 0.98,
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios,
                                   wspace=0, hspace=0)

            dist_ax = fig.add_subplot(gs[1,1], xscale='linear', yscale='linear',
                                      xticks=[], yticks=[],xlabel='', ylabel='',
                                      frameon=True)

            D = topics_dist[spectra_order, :][:, spectra_order]
            dist_im = dist_ax.imshow(D, interpolation='none', cmap='viridis',
                                     aspect='auto', rasterized=True)

            left_ax = fig.add_subplot(gs[1,0], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
            left_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(-1, 1),
                            interpolation='none', cmap='Spectral', aspect='auto',
                            rasterized=True)


            top_ax = fig.add_subplot(gs[0,1], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
            top_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(1, -1),
                              interpolation='none', cmap='Spectral', aspect='auto',
                                rasterized=True)


            hist_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 3],
                                   wspace=0, hspace=0)

            hist_ax = fig.add_subplot(hist_gs[0,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Local density histogram')
            hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            hist_ax.yaxis.tick_right()

            xlim = hist_ax.get_xlim()
            ylim = hist_ax.get_ylim()
            if density_threshold < xlim[1]:
                hist_ax.axvline(density_threshold, linestyle='--', color='k')
                hist_ax.text(density_threshold  + 0.02, ylim[1] * 0.95, 'filtering\nthreshold\n\n', va='top')
            hist_ax.set_xlim(xlim)
            hist_ax.set_xlabel('Mean distance to k nearest neighbors\n\n%d/%d (%.0f%%) spectra above threshold\nwere removed prior to clustering'%(sum(~density_filter), len(density_filter), 100*(~density_filter).mean()))
            
            ## Add colorbar
            cbar_gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=hist_gs[1, 0],
                                   wspace=0, hspace=0)
            cbar_ax = fig.add_subplot(cbar_gs[4,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Euclidean Distance')
            vmin = D.min().min()
            vmax = D.max().max()
            fig.colorbar(dist_im, cax=cbar_ax,
            ticks=np.linspace(vmin, vmax, 3),
            orientation='horizontal')
            
            
            #hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            #hist_ax.yaxis.tick_right()            

            fig.savefig(self.paths['clustering_plot']%(k, density_threshold_repl), dpi=250)
            if close_clustergram_fig:
                plt.close(fig)

    def get_and_check_consensus(self, k, local_density_threshold, local_neighborhood_size):
        logging.info(f"Creating consensus programs and usages for k={k}")
        self.consensus(k, density_threshold=local_density_threshold,
            local_neighborhood_size=local_neighborhood_size,
            show_clustering=True,
            close_clustergram_fig=True)
        density_threshold_repl = str(local_density_threshold).replace(".", "_")
        filenames = [
            self.paths['consensus_spectra']%(k, density_threshold_repl),
            self.paths['consensus_spectra']%(k, density_threshold_repl),
            self.paths['consensus_usages']%(k, density_threshold_repl),
            self.paths['consensus_stats']%(k, density_threshold_repl),
            self.paths['consensus_spectra__txt']%(k, density_threshold_repl),
            self.paths['consensus_usages__txt']%(k, density_threshold_repl),
            self.paths['gene_spectra_tpm']%(k, density_threshold_repl),
            self.paths['gene_spectra_tpm__txt']%(k, density_threshold_repl),
            self.paths['gene_spectra_score']%(k, density_threshold_repl),
            self.paths['gene_spectra_score__txt']%(k, density_threshold_repl)
            ]
        for filename in filenames:
            if not os.path.exists(filename):
                raise ValueError(f"cNMF postprocessing could not find output file {filename}. This can arise in low memory conditions.")

    def k_selection_plot(self, close_fig=False):
        """
        :param close_fig: close figure after saving, defaults to False
        :type close_fig: bool, optional
        """
        run_params = utils.load_df_from_npz(self.paths['nmf_replicate_parameters'])
        stats = []
        for k in sorted(set(run_params.n_components)):

            stats.append(self.consensus(k, skip_density_and_return_after_stats=True,
                                        show_clustering=False, close_clustergram_fig=True).stats)

        stats = pd.DataFrame(stats)
        stats.reset_index(drop = True, inplace = True)

        utils.save_df_to_npz(stats, self.paths['k_selection_stats'])

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()


        ax1.plot(stats.k, stats.stability, 'o-', color='b')
        ax1.set_ylabel('Stability', color='b', fontsize=15)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        #ax1.set_xlabel('K', fontsize=15)

        ax2.plot(stats.k, stats.prediction_error, 'o-', color='r')
        ax2.set_ylabel('Error', color='r', fontsize=15)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        ax1.set_xlabel('Number of Components', fontsize=15)
        ax1.grid('on')
        plt.tight_layout()
        fig.savefig(self.paths['k_selection_plot'], dpi=250)
        if close_fig:
            plt.close(fig)
            
    def postprocess(self,
                    cpus: int = 1,
                    local_density_threshold: float = 2.0,
                    local_neighborhood_size: float = 0.3,
                    skip_missing_iterations: bool = False):
        
        run_params = utils.load_df_from_npz(self.paths['nmf_replicate_parameters'])
        # first check for combined outputs:
        missing_combined = []
        for k in sorted(set(run_params.n_components)):
            merged_result = self.paths['merged_spectra'] % k
            if not os.path.exists(merged_result) or os.path.getsize(merged_result) == 0:
                missing_combined.append(merged_result)
        if missing_combined:
            failed = []
            # Check if all output files and iterations exist
            for _, row in run_params.iterrows():
                iter_result = self.paths['iter_spectra'] % (row['n_components'], row['iter'])
                if not os.path.exists(iter_result) or os.path.getsize(iter_result) == 0:
                    failed.append(iter_result)
            if failed and not skip_missing_iterations:
                raise ValueError(
                    f"Postprocessing could not proceed. To skip missing iterations, use --skip_missing_iterations." + 
                    f"\n {(len(failed))} files from the factorization step are missing or empty:\n  - " + 
                    "\n  - ".join(failed)
                )
            elif failed and skip_missing_iterations:
                logging.warning("Missing files will be skipped")
            else:
                logging.info(f"Factorization outputs (individual iterations) were found for all values of k. No missing files were detected.")

            # combine individual iterations
            for k in sorted(set(run_params.n_components)):
                logging.info(f"Merging iterations for k={k}")
                self.combine_nmf(k, skip_missing_files=skip_missing_iterations)
        else:
            logging.info(f"Factorization outputs (merged iterations) were found for all values of k.")
        # calculate consensus programs and usages
        logging.info(f"Creating consensus programs and usages using {cpus} CPUs")
        call_consensus = partial(
            self.get_and_check_consensus,
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
        self.k_selection_plot(close_fig=True)
    
