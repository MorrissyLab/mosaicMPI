import os
import sys
import logging
import numpy as np
import pandas as pd
import networkx as nx
from anndata import read_h5ad

def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def load_df_from_npz(filename, multiindex=False):
    with np.load(filename, allow_pickle=True) as f:
        if any([isinstance(c, tuple) for c in (f["index"])]):
            index = pd.MultiIndex.from_tuples(f["index"])
        else:
            index = f["index"]
        if any([isinstance(c, tuple) for c in (f["columns"])]):
            columns = pd.MultiIndex.from_tuples(f["columns"])
        else:
            columns = f["columns"]
        obj = pd.DataFrame(f["data"], index=index, columns=columns)
    return obj

def get_corr_matrix(output_dir, config):
    corr_path = os.path.join(output_dir, "integrate", config.integration["corr_method"] + ".df.npz")
    if not os.path.exists(corr_path):
        logging.error(f"No correlation matrix found at {corr_path}. Make sure you have run `cnmfsns integrate` before running `cnmfsns create-sns`.")
    corr = load_df_from_npz(corr_path)
    logging.info(f"Loaded correlation matrix from {corr_path}")
    # Check that rows and columns of correlation matrix are identical
    assert (corr.index == corr.columns).all()
    return corr

def create_graph(output_dir, config):
    corr = get_corr_matrix(output_dir, config)
    # Lower triangular matrix contains each edge only once and removes diagonal (self-correlation)
    tril = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))

    # create quantile version of tril where correlations are replaced by quantile of intra-and inter-dataset correlations
    tril_quantile = tril.copy(deep=True)
    for ds1 in config.datasets:
        for ds2 in config.datasets:
            chunk = tril_quantile.loc[tril_quantile.index.get_level_values(0) == ds1, tril_quantile.index.get_level_values(0) == ds2]
            flattened_ranks = pd.Series(chunk.values.flatten()).rank() - 1
            flattened_quantiles = (flattened_ranks / flattened_ranks.max()).values
            quantile_chunk = pd.DataFrame(data=np.reshape(flattened_quantiles, newshape=chunk.values.shape), index=chunk.index, columns=chunk.columns)
            tril_quantile.loc[tril_quantile.index.get_level_values(0) == ds1, tril_quantile.index.get_level_values(0) == ds2] = quantile_chunk      

    # filter to selected k in each dataset
    selected_k_index = pd.MultiIndex.from_tuples([gep for gep in tril.index if gep[1] in config.datasets[gep[0]]["selected_k"]])
    subset = tril.loc[selected_k_index, selected_k_index]
    subset_quantile = tril_quantile.loc[selected_k_index, selected_k_index]

    # filter edges by inter and intra-dataset thresholds
    min_corr_thresholds = pd.read_table(os.path.join(output_dir, "integrate", "max_k_filtered.pairwise_corr_thresholds.txt"))
    for _, row in min_corr_thresholds.iterrows():
        dataset_row, dataset_col, threshold = row
        filtered_chunk = subset.loc[subset.index.get_level_values(0) == dataset_row, subset.columns.get_level_values(0) == dataset_col] <= threshold
        subset.loc[subset.index.get_level_values(0) == dataset_row, subset.columns.get_level_values(0) == dataset_col] = subset.loc[subset.index.get_level_values(0) == dataset_row, subset.columns.get_level_values(0) == dataset_col].mask(filtered_chunk)
        subset_quantile.loc[subset_quantile.index.get_level_values(0) == dataset_row, subset_quantile.columns.get_level_values(0) == dataset_col] = subset_quantile.loc[subset_quantile.index.get_level_values(0) == dataset_row, subset_quantile.columns.get_level_values(0) == dataset_col].mask(filtered_chunk)

    subset.index = pd.Index(["|".join((gep[0], str(gep[1]), str(gep[2]))) for gep in subset.index])
    subset.columns = pd.Index(["|".join((gep[0], str(gep[1]), str(gep[2]))) for gep in subset.columns])
    subset_quantile.index = pd.Index(["|".join((gep[0], str(gep[1]), str(gep[2]))) for gep in subset_quantile.index])
    subset_quantile.columns = pd.Index(["|".join((gep[0], str(gep[1]), str(gep[2]))) for gep in subset_quantile.columns])

    # Build graph
    links = subset.stack().reset_index()
    links.columns = ['node1', 'node2', 'corr']
    links_quantile = subset_quantile.stack().reset_index()
    links_quantile.columns = ['node1', 'node2', 'prefilter_quantile']
    links = links.merge(links_quantile)

    # add post-filter quantile column
    for ds1 in config.datasets:
        for ds2 in config.datasets:
            ds_pair_indices = (links['node1'].str.split("|").str[0] == ds1) & (links['node2'].str.split("|").str[0] == ds2)
            links_ds_pair = links.loc[ds_pair_indices]
            links.loc[ds_pair_indices, "postfilter_quantile"] = links_ds_pair["corr"].rank() / links_ds_pair["corr"].count()

    G = nx.from_pandas_edgelist(links, 'node1', 'node2', ["corr", "prefilter_quantile", "postfilter_quantile"])
    return G

def add_community_weights_to_graph(G, gep_communities, config):
    edge_attr = {}
    for edge in G.edges:
        weight = 1
        if gep_communities[edge[0]] == gep_communities[edge[1]]:
            weight *= config.sns["layouts"]["community_weighted_spring"]["within_community"]
        if edge[0].split("|")[0] == edge[1].split("|")[0]:
            weight *= config.sns["layouts"]["community_weighted_spring"]["within_dataset"]
        edge_attr[edge] = weight
    nx.set_edge_attributes(G, edge_attr, name="community_weight")


def community_search(G, config):
    weight_method = config.sns["edge_weight"]
    if weight_method == "none":
        weight_method = None
    elif weight_method not in ("corr", "prefilter_quantile", "postfilter_quantile"):
        logging.error(f"{weight_method} is not a valid weight_method. Please choose one of `corr`, `prefilter_quantile`, `postfilter_quantile`")

    # Community search
    community_algorithm = config.sns["community_algorithm"]
    if community_algorithm == "greedy_modularity":
        from networkx.algorithms.community.modularity_max import greedy_modularity_communities
        best_n_param = config.sns["communities"]["greedy_modularity"]["best_n"]
        if best_n_param == 'none':
            best_n_param = None
        communities = {
            name: nodes for name, nodes in
            enumerate(greedy_modularity_communities(G, resolution=config.sns["communities"]["greedy_modularity"]["resolution"],
                weight=weight_method,
                best_n=best_n_param), start=1)
            }
    elif community_algorithm == "leiden":
        import igraph
        G_igraph = igraph.Graph.from_networkx(G)
        communities = {}
        leiden_comm = G_igraph.community_leiden(resolution_parameter=config.sns["communities"]["leiden"]["resolution"], weights=weight_method)
        for community, member_nodes in enumerate(leiden_comm, start=1):
            communities[community] = G_igraph.vs[member_nodes]['_nx_name']
    else:
        logging.error(f"{community_algorithm} is not a valid community algorithm ")
        sys.exit(1)
    logging.info(f"Identified {len(communities)} communities")
    return communities

def get_graph_layout(G, config):
    logging.info(f"Computing network layout for {len(G)} nodes")
    layout_algorithm = config.sns["layout_algorithm"]
    if layout_algorithm == "neato":
        layout = nx.nx_agraph.graphviz_layout(G, prog="neato", args='-Goverlap=true')
    elif layout_algorithm == "spring":
        layout = nx.spring_layout(G)
        layout = {node: list(coords) for node, coords in layout.items()}
    elif layout_algorithm == "community_weighted_spring":
        layout = nx.spring_layout(G, weight="community_weight")
        layout = {node: list(coords) for node, coords in layout.items()}
    elif layout_algorithm == "umap":
        import umap
        from sklearn.preprocessing import StandardScaler, RobustScaler

        geps = {}
        for dataset_name, dataset in config.datasets.items():
            adata = read_h5ad(dataset["filename"], backed="r")
            df = adata.varm["cnmf_gep_score"]
            df.columns = pd.MultiIndex.from_tuples([(int(gep[0]), int(gep[1])) for gep in df.columns.str.split(".")])
            geps[dataset_name] = df.loc[:, dataset["selected_k"]]

        geps = pd.concat(geps, axis=1).sort_index(axis=1)
        # Standardize features for dimensionality reduction
        table = geps.dropna().T
        x = table.values
        x = RobustScaler().fit_transform(x)

        embedding = umap.UMAP(n_neighbors=25, min_dist=0.01).fit_transform(x)
        layout = {"|".join((gep[0], str(gep[1]), str(gep[2]))): list(emb.astype(float)) for gep, emb in zip(table.index, embedding)}  
    return layout


def get_max_corr_communities(communities, output_dir, config):
    corr = get_corr_matrix(output_dir, config)

    index = pd.MultiIndex.from_product([communities, config.datasets], names=["Community", "Dataset"])
    max_corr_communities = pd.DataFrame(index=index, columns=index)

    for (community_1, dataset_1) in index:
        nodes_1 = [(l[0], int(l[1]), int(l[2])) for l in pd.Index(list(communities[community_1])).str.split("|") if l[0] == dataset_1]
        if nodes_1:
            nodes_1 = pd.MultiIndex.from_tuples(nodes_1)
            for (community_2, dataset_2) in index:
                nodes_2 = [(l[0], int(l[1]), int(l[2])) for l in pd.Index(list(communities[community_2])).str.split("|") if l[0] == dataset_2]
                if nodes_2:
                    nodes_2 = pd.MultiIndex.from_tuples(nodes_2)
                    max_corr_communities.loc[(community_1, dataset_1), (community_2, dataset_2)] = corr.loc[nodes_1, nodes_2].max().max()
    return max_corr_communities

def get_category_overrepresentation(usage, sample_to_class):
    usage.index = usage.index.map(sample_to_class)
    observed = usage.groupby(axis=0, level=0).sum()
    expected = []
    for k, obs_k in observed.groupby(axis=1, level=1):
        exp_k = pd.DataFrame(obs_k.sum(axis=1)) @ pd.DataFrame(obs_k.sum(axis=0)).T / obs_k.sum().sum()
        expected.append(exp_k)
    expected = pd.concat(expected, axis=1)
    chisq_resid = (observed - expected) / np.sqrt(expected)  # pearson residual of chi-squared test of contingency table
    overrepresentation = chisq_resid.clip(lower=0)
    return overrepresentation