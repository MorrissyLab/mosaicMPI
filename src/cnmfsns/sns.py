
from .integration import Integration


from collections.abc import Collection, Iterable
from typing import Union, Optional
import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import igraph
import distinctipy
import tomli_w
import networkx as nx
from scipy.stats import entropy

class SNS():
    def __init__(self,
                 integration: Integration,
                 subset_nodes: Optional[Iterable] = None,
                 communities: Optional[dict] = None,
                 ):
        self.integration = integration
        self.subset_nodes = subset_nodes
        self.create_gep_network()
        self.communities = communities
        if subset_nodes is not None:
            self.gep_graph = nx.subgraph(self.gep_graph, subset_nodes)
    @property
    def n_communities(self):
        if self.communities is None:
            raise ValueError("Communities have not yet been defined.")
        else:
            return len(self.communities)

    @property
    def geps_in_graph(self):
        """Get the nodes in the GEP graph as (dataset, k, gep) tuples.
        This is helpful for indexing usage matrices etc., whereas the
        nodes from Dataset.gep_graph.nodes will be given as pipe-delimited
        strings.

        Returns:
            list of tuples: nodes in the GEP graph
        """
        nodes = []
        for node in self.gep_graph.nodes:
            dataset_name, k_str, gep_str = node.split("|")
            nodes.append((dataset_name, int(k_str), int(gep_str)))
        return nodes
    
    def get_community_usage(self, subset_datasets = None, normalized=True):
        usage = self.integration.get_usages()
        ic_usage = []
        
        if subset_datasets is None:
            subset_datasets = self.integration.datasets
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        
        for dataset_name in subset_datasets:
            data = []
            for community, nodes in self.communities.items():
                geps = []
                for node in nodes:
                    gep = node.split("|")
                    if gep[0] == dataset_name:
                        geps.append((gep[0], int(gep[1]), int(gep[2])))
                gep_comm = usage[geps]
                gep_comm = gep_comm / gep_comm.median()
                data.append(gep_comm.median(axis=1).rename((community, dataset_name)))
            data = pd.concat(data, axis=1).droplevel(axis=1, level=1).dropna(how="all")
            data.columns.rename("Community", inplace=True)
            ic_usage.append(data.sort_index(axis=0))
        ic_usage = pd.concat(ic_usage)
        if normalized:
            ic_usage = ic_usage.div(ic_usage.sum(axis=1), axis=0)
        return ic_usage
    
    def get_sample_entropy(self, subset_datasets = None):
        ic_usage = self.get_community_usage(subset_datasets = subset_datasets)
        diversity = ic_usage.apply(lambda x: entropy(x.dropna()), axis=1)
        return diversity
    
    def get_community_category_overrepresentation(self,
                                                  layer: str,
                                                  subset_datasets: Optional[Collection] = None
                                                  ) -> pd.DataFrame:
        df = self.integration.get_category_overrepresentation(layer=layer, subset_datasets=subset_datasets)
        mapper = {tuple([gep.split("|")[0], int(gep.split("|")[1]), int(gep.split("|")[2])]): comm for gep, comm in self.gep_communities.items()}
        df.columns = df.columns.map(mapper)
        df = df.groupby(axis=1, level=0).mean()
        df = df.reindex(self.ordered_community_names, axis=1)
        return df
    
    def get_community_metadata_correlation(self,
                                           layer: str,
                                           subset_datasets: Optional[Collection] = None,
                                           method: str = "pearson"
                                           ) -> pd.Series:
        ser = self.integration.get_metadata_correlation(layer=layer, subset_datasets=subset_datasets, method=method)
        mapper = {tuple([gep.split("|")[0], int(gep.split("|")[1]), int(gep.split("|")[2])]): comm for gep, comm in self.gep_communities.items()}
        ser.index = ser.index.map(mapper)
        ser = ser.groupby(axis=0, level=0).mean()
        ser = ser.reindex(self.ordered_community_names)
        return ser

    
    def create_gep_network(self):

        # get matrix of edges after filtering
        subset = self.integration.get_corr_matrix_lowertriangle(selected_k_filter=True)
        subset_quantile = self.integration.get_corr_matrix_lowertriangle(selected_k_filter=True, quantile_transformation=True)
        
        # filter edges by inter and intra-dataset min_corr thresholds
        for (dataset_row, dataset_col), threshold in self.integration.pairwise_thresholds.items():
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
        for ds1 in self.integration.datasets:
            for ds2 in self.integration.datasets:
                ds_pair_indices = (links['node1'].str.split("|").str[0] == ds1) & (links['node2'].str.split("|").str[0] == ds2)
                links_ds_pair = links.loc[ds_pair_indices]
                links.loc[ds_pair_indices, "postfilter_quantile"] = links_ds_pair["corr"].rank() / links_ds_pair["corr"].count()
        G = nx.from_pandas_edgelist(links, 'node1', 'node2', ["corr", "prefilter_quantile", "postfilter_quantile"])

        G.add_nodes_from(subset.index.to_list()) # ensures that nodes are not excluded because they are not connected
        self.gep_graph = G

    def community_search(self,
                           algorithm="greedy_modularity",
                           resolution: Optional[float] = 2.0,
                           k: Optional[int] = None,
                           edge_weight: Optional[str] = None
                           ):
        
        assert edge_weight in (None, "corr", "prefilter_quantile", "postfilter_quantile")
        
        G = self.gep_graph
            
        # Community search
        logging.info(f"Community search: algorithm = {algorithm}, resolution = {resolution}")
        if algorithm == "greedy_modularity":
            algo_output = nx.algorithms.community.modularity_max.greedy_modularity_communities(
                G, resolution=resolution, weight=edge_weight)
            communities = {name: nodes for name, nodes in enumerate(algo_output, start=1)}
        elif algorithm == "leiden":
            G_igraph = igraph.Graph.from_networkx(G)
            algo_output = G_igraph.community_leiden(resolution_parameter=resolution, weights=edge_weight)
            communities = {}
            for community, member_nodes in enumerate(algo_output, start=1):
                communities[community] = G_igraph.vs[member_nodes]['_nx_name']
        elif algorithm == "asyn_lpa":
            assert resolution is None
            algo_output = nx.algorithms.community.label_propagation.asyn_lpa_communities(G, weight=edge_weight)
            communities = {name: nodes for name, nodes in enumerate(algo_output, start=1)}
        elif algorithm =="girvan_newman":
            assert resolution is None
            assert isinstance(k, int)
            algo_output = nx.algorithms.community.centrality.girvan_newman(G)
            for partition in algo_output:
                if len(partition) >= k:
                    break
            
            communities = {}
            for name, nodes in enumerate(partition, start=1):
                communities[name] = nodes
        elif algorithm == "asyn_fluidc":
            assert resolution is None
            assert isinstance(k, int)
            algo_output = nx.algorithms.community.asyn_fluidc(G, k, max_iter=100)
            communities = {name: nodes for name, nodes in enumerate(algo_output, start=1)}
        else:
            raise ValueError(f"{algorithm} is not a valid community detection algorithm")

        # community names must be strings to avoid problems with TOML persistence
        communities = {str(k): v for k, v in communities.items()}

        self.communities = communities
        self.gep_communities = {gep: community for community, geps in communities.items() for gep in geps}
        self.create_community_network()

    def prune_communities(self,
                          min_nodes: int = 1,
                          min_datasets: int = 1,
                          min_nodes_per_dataset: int = 0,
                          renumber = False, recolor = False):
        pruned = {}
        for community, nodes in self.communities.items():
            if len(nodes) < min_nodes:
                continue
            
            dataset_counts = {dsname: 0 for dsname in self.integration.datasets}
            for node in nodes:
                dsname = node.split("|")[0]
                dataset_counts[dsname] += 1
                
            n_datasets = np.sum([x > 0 for x in dataset_counts.values()])
            if n_datasets < min_datasets:
                continue
            
            if any([x < min_nodes_per_dataset for x in dataset_counts.values()]):
                continue
            
            pruned[community] = nodes
            
        if renumber:
            renumbered = list(pruned.values())
            renumbered.sort(key=lambda x: len(x), reverse=True)
            renumbered = {str(num): nodes for num, nodes in enumerate(renumbered, 1)}
            pruned = renumbered
        
        self.communities = pruned
        self.gep_communities = {gep: community for community, geps in self.communities.items() for gep in geps}
        self.create_community_network()


    @property
    def ordered_community_names(self):
        community_names = sorted(self.communities.keys(), key = lambda cstr: [int(lvl) for lvl in cstr.split(".")])
        return community_names
    
    def add_community_weights_to_graph(self, shared_community_weight = 0.3, shared_dataset_weight = 0.1):
        edge_attr = {}
        for edge in self.gep_graph.edges:
            weight = 1
            if edge[0] in self.gep_communities and edge[1] in self.gep_communities:  # nodes might not have communities due to pruning, and so will be treated as if they are in different communities for layout purposes
                if self.gep_communities[edge[0]] == self.gep_communities[edge[1]]:
                    weight *= shared_community_weight
            if edge[0].split("|")[0] == edge[1].split("|")[0]:
                weight *= shared_dataset_weight
            edge_attr[edge] = weight
        nx.set_edge_attributes(self.gep_graph, edge_attr, name="community_weight")

    def write_communities_toml(self, filename):
        toml_conformed = {str(community): sorted(geps, key=lambda x: int(x.split("|")[1])) for community, geps in self.communities.items()}
        with open(filename, "wb") as fh:
            tomli_w.dump(toml_conformed, fh)

    def write_gep_network_graphml(self, filename):
        nx.write_graphml(self.gep_graph, filename)
    
    def write_community_network_graphml(self, filename):
        nx.write_graphml(self.comm_graph, filename)

    def compute_layout(self, algorithm="community_weighted_spring", shared_community_weight = 200, shared_dataset_weight = 1.05, community_layout_algorithm="spring"):
        if algorithm == "neato":
            layout = nx.nx_agraph.graphviz_layout(self.gep_graph, prog="neato", args='-Goverlap=true')
        elif algorithm == "spring":
            layout = nx.spring_layout(self.gep_graph)
            layout = {node: list(coords) for node, coords in layout.items()}
        elif algorithm == "community_weighted_spring":
            self.add_community_weights_to_graph(shared_community_weight=shared_community_weight, shared_dataset_weight=shared_dataset_weight)
            layout = nx.spring_layout(self.gep_graph, weight="community_weight")

            layout = {node: list(coords) for node, coords in layout.items()}
        elif algorithm == "umap":
            import umap
            from sklearn.preprocessing import StandardScaler, RobustScaler

            geps = {}
            for dataset_name, dataset in self.integration.datasets.items():
                selected_k = self.integration.k_table[(dataset_name, "selected_k")]
                selected_k = selected_k[selected_k].index.to_list()
                df = dataset.get_geps(k=selected_k)
                geps[dataset_name] = df.loc[:, ]

            geps = pd.concat(geps, axis=1).sort_index(axis=1)
            # Standardize features for dimensionality reduction
            table = geps.dropna().T
            x = table.values
            x = RobustScaler().fit_transform(x)

            embedding = umap.UMAP(n_neighbors=25, min_dist=0.01).fit_transform(x)
            layout = {"|".join((gep[0], str(gep[1]), str(gep[2]))): list(emb.astype(float)) for gep, emb in zip(table.index, embedding)}
        else:
            raise ValueError(f"{algorithm} is not a valid layout algorithm ")
            
        # rescale layout
        xmax = max(x for x, y in layout.values())
        xmin = min(x for x, y in layout.values())
        ymax = max(y for x, y in layout.values())
        ymin = min(y for x, y in layout.values())
        def rescale_point(xy, xmin, xmax, ymin, ymax, new_range):
            x, y = xy
            x = (new_range[1] - new_range[0]) * (x - xmin) / (xmax - xmin) + new_range[0]
            y = (new_range[1] - new_range[0]) * (y - ymin) / (ymax - ymin) + new_range[0]
            return (x, y)
        layout = {
            name: rescale_point(xy, xmin, xmax, ymin, ymax, (-1, 1))
            for name, xy in layout.items()}
        self.layout = layout
        
        self.compute_community_network_layout(method=community_layout_algorithm)

    def compute_community_network_layout(self, method: str = "neato"):
        
        logging.info(f"Computing community layout using {method} method.")
        if method == "centroid":
            # Centroid method for community layout
            self.comm_layout = {}
            for community_name, nodes in self.communities.items():
                points = np.array([self.layout[node] for node in nodes])
                centroid = (np.median(points[:, 0]), np.median(points[:, 1]))
                self.comm_layout[community_name] = centroid
                
        elif method == "neato":
            layout = nx.nx_agraph.graphviz_layout(self.comm_graph, prog="neato", args='-Goverlap=true')
            # rescale layout
            xmax = max(x for x, y in layout.values())
            xmin = min(x for x, y in layout.values())
            ymax = max(y for x, y in layout.values())
            ymin = min(y for x, y in layout.values())
            def rescale_point(xy, xmin, xmax, ymin, ymax, new_range):
                x, y = xy
                x = (new_range[1] - new_range[0]) * (x - xmin) / (xmax - xmin) + new_range[0]
                y = (new_range[1] - new_range[0]) * (y - ymin) / (ymax - ymin) + new_range[0]
                return (x, y)
            layout = {
                name: rescale_point(xy, xmin, xmax, ymin, ymax, (-1, 1))
                for name, xy in layout.items()}
            self.comm_layout = layout
        
        elif method == "spring":
            layout = nx.spring_layout(self.comm_graph, weight="weight")
            # rescale layout
            xmax = max(x for x, y in layout.values())
            xmin = min(x for x, y in layout.values())
            ymax = max(y for x, y in layout.values())
            ymin = min(y for x, y in layout.values())
            def rescale_point(xy, xmin, xmax, ymin, ymax, new_range):
                x, y = xy
                x = (new_range[1] - new_range[0]) * (x - xmin) / (xmax - xmin) + new_range[0]
                y = (new_range[1] - new_range[0]) * (y - ymin) / (ymax - ymin) + new_range[0]
                return (x, y)
            layout = {
                name: rescale_point(xy, xmin, xmax, ymin, ymax, (-1, 1))
                for name, xy in layout.items()}
            self.comm_layout = layout
        else:
            raise NotImplementedError
    
    def get_max_corr_communities(self) -> pd.DataFrame:
        corr = self.integration.corr_matrix

        index = pd.MultiIndex.from_product([self.communities, self.integration.datasets.keys()], names=["Community", "Dataset"])
        max_corr_communities = pd.DataFrame(index=index, columns=index)

        for (community_1, dataset_1) in index:
            nodes_1 = [(l[0], int(l[1]), int(l[2])) for l in pd.Index(list(self.communities[community_1])).str.split("|") if l[0] == dataset_1]
            if nodes_1:
                nodes_1 = pd.MultiIndex.from_tuples(nodes_1)
                for (community_2, dataset_2) in index:
                    nodes_2 = [(l[0], int(l[1]), int(l[2])) for l in pd.Index(list(self.communities[community_2])).str.split("|") if l[0] == dataset_2]
                    if nodes_2:
                        nodes_2 = pd.MultiIndex.from_tuples(nodes_2)
                        max_corr_communities.loc[(community_1, dataset_1), (community_2, dataset_2)] = corr.loc[nodes_1, nodes_2].max().max()
        return max_corr_communities

    def create_community_network(self) -> None:
        logging.info("Creating community network")
        edge_list = []
        for c1, n1 in self.communities.items():
            for c2, n2 in self.communities.items():
                if c1 != c2:  # no self-loops
                    n_edges = len(list(nx.edge_boundary(self.gep_graph, n1, n2)))
                    weight = np.sqrt(n_edges)
                    edge_list.append((c1, c2, n_edges, weight))

        edge_list = pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges", "weight"))
        self.comm_graph = nx.from_pandas_edgelist(df = pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges", "weight")),
                                                  source = "comm1",
                                                  target = "comm2",
                                                  edge_attr=["n_edges", "weight"])
        self.comm_graph.add_nodes_from(self.communities.keys())
        
    def get_representative_gep_table(self,
                                method = "min_k",
                                min_k: int = 2
                                ) -> pd.DataFrame:
        if method == "min_k":
            logging.info(f"Selecting representative ranks for each community with min_k = {min_k}")
            # get minimum k GEPs for each community/dataset combination.
            table = []
            for community, nodes in self.communities.items():
                nodes = pd.DataFrame([n.split("|") for n in nodes], columns=["dataset", "k", "GEP"])
                nodes = nodes[nodes["k"].astype(int) >= min_k]
                for dataset_name in self.integration.datasets:
                    dataset_nodes = nodes[nodes["dataset"] == dataset_name].copy()
                    if dataset_nodes.shape[0]:
                        block = dataset_nodes[dataset_nodes["k"].astype(int) == dataset_nodes["k"].astype(int).min()].copy()
                        block["Community"] = community
                        table.append(block)
            table = pd.concat(table).set_index("Community")
        else:
            raise ValueError
        
        return table
        

    def get_representative_geps(self,
                                method = "min_k",
                                min_k: int = 2
                                ) -> pd.DataFrame:
        geps = self.integration.get_geps()
        
        table = self.get_representative_gep_table(method = method, min_k = min_k)
        selected_geps = []
        for community, gep in table.iterrows():
            gep = geps[gep['dataset'], int(gep['k']), int(gep['GEP'])]
            gep = gep.rename((community,) + gep.name)
            selected_geps.append(gep)
        selected_geps = pd.concat(selected_geps, axis=1)
        selected_geps.columns.rename(("community", "dataset", "k", "GEP"), inplace=True)
        return selected_geps
        