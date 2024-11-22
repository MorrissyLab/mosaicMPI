
from .integration import Integration
from .utils import node_to_program


from collections.abc import Collection, Iterable
from typing import Union, Optional, Dict, List, Literal
import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import igraph
import pickle
import gzip
import distinctipy
import tomli
import tomli_w
import networkx as nx
from scipy.stats import entropy

class Network():
    def __init__(self,
                 integration: Integration,
                 subset_nodes: Optional[Iterable[str]] = None,
                 communities: Optional[Dict[str, Collection[str]]] = None,
                 ):
        """Create a program network from an :class:`~mosaicmpi.integration.Integration` object.

        :param integration: Integration of multiple datasets.
        :type integration: :class:`~mosaicmpi.integration.Integration`
        :param subset_nodes: Create an SNS from a subset of the larger program graph, defaults to None
        :type subset_nodes: Iterable[str], optional
        :param communities: Use pre-defined communities, defaults to None
        :type communities: Dict[str, Collection[str]], optional
        """
        self.integration = integration
        self.subset_nodes = subset_nodes
        self.create_program_network()
        self.communities = communities
        if subset_nodes is not None:
            self.program_graph = nx.subgraph(self.program_graph, subset_nodes)
    
    # Reading and writing from .pkl and pkl.gz files

    @classmethod
    def from_pkl(cls, filename) -> "Network":
        """Read an Network object from a file.

        :param filename: path to .pkl or .pkl.gz file
        :type filename: str
        :return: Network
        :rtype: :class:`mosaicmpi.network.Network`
        """
        if filename.endswith(".pkl.gz"):
            with gzip.open(filename, "rb") as handle:
                network = pickle.load(handle)
        elif filename.endswith(".pkl"):
            with open(filename, "rb") as handle:
                network = pickle.load(handle)
        else:
            ext = "." + filename.rpartition(".")[2]
            raise ValueError(f"Filename endswith an invalid extension: {ext}")
        return network
            
    def to_pkl(self,
               filename: str):
        """
        Persists the SNS object using python's pickle format with optional gzip compression
        :param filename: path to .pkl or .pkl.gz file
        :type filename: str
        """
        if filename.endswith(".pkl.gz"):
            with gzip.open(filename, "wb") as handle:
                pickle.dump(self, handle)
        elif filename.endswith(".pkl"):
            with open(filename, "wb") as handle:
                pickle.dump(self, handle)
        else:
            ext = "." + filename.rpartition(".")[2]
            raise ValueError(f"Filename endswith an invalid extension: {ext}")


    @property
    def n_communities(self) -> int:
        """Get the number of communities in the Network.

        :return: Number of communities
        :rtype: int
        """
        if self.communities is None:
            raise ValueError("Communities have not yet been defined.")
        else:
            return len(self.communities)

    @property
    def programs_in_graph(self):
        """
        Get the nodes in the program graph as (dataset, k, program) tuples.
        This is helpful for indexing usage matrices etc., whereas the
        nodes from Dataset.program_graph.nodes will be given as pipe-delimited
        strings.

        :return: list of programs
        :rtype: list of tuples
        """
        nodes = []
        for node in self.program_graph.nodes:
            dataset_name, k_str, program_str = node.split("|")
            nodes.append((dataset_name, int(k_str), int(program_str)))
        return nodes
    
    def get_node_table(self) -> pd.DataFrame:
        """Get node counts before and after various node and edge filters.

        :return: Summary table of node counts
        :rtype: pd.DataFrame
        """
        node_table = self.integration.get_node_table()
        node_table[("network", "")] = pd.Series([node_to_program(node)[0] for node in self.program_graph.nodes]).value_counts()
        return node_table

    def get_community_usage(self,
                            subset_datasets: Optional[Union[str, Iterable[str]]] = None,
                            normalize: bool = True,
                            discretize: bool = False):
        """
        Get median usage of each community of programs for each samples.  # TODO: migrate to representative programs

        :param subset_datasets: dataset name or iterable of dataset names to subset the results, defaults to None
        :type subset_datasets: str or Iterable[str], optional
        :param normalize: Normalize the community usage matrix such that for each value of k, usage of all communities of programs sums to 1. Defaults to False
        :type normalize: bool, optional
        :param discretize: Discretizes the community usage matrix such that for each value of k, each sample has usage of only 1 community (the one with the maximum usage). Defaults to False
        :type discretize: bool, optional
        :return: observations × communities matrix
        :rtype: pd.DataFrame
        """
        usage = self.integration.get_usages(normalize=normalize)
        ic_usage = []
        
        if subset_datasets is None:
            subset_datasets = self.integration.datasets
        elif isinstance(subset_datasets, str):
            subset_datasets = [subset_datasets]
        
        for dataset_name in subset_datasets:
            data = []
            for community, nodes in self.communities.items():
                programs = []
                for node in nodes:
                    program = node.split("|")
                    if program[0] == dataset_name:
                        programs.append((program[0], int(program[1]), int(program[2])))
                program_comm = usage[programs]
                program_comm = program_comm / program_comm.median()
                data.append(program_comm.median(axis=1).rename((community, dataset_name)))
            data = pd.concat(data, axis=1).droplevel(axis=1, level=1).dropna(how="all")
            data.columns.rename("Community", inplace=True)
            ic_usage.append(data.sort_index(axis=0))
        ic_usage = pd.concat(ic_usage)
        if normalize:
            ic_usage = ic_usage.div(ic_usage.sum(axis=1), axis=0)
        if discretize:
            ic_usage = ic_usage.eq(ic_usage.max(axis=1), axis=0).astype(int)
        return ic_usage
    
    def get_sample_entropy(self, subset_datasets: Optional[Union[str, Iterable]] = None):
        """Get shannon diversity of Community Usage for each sample.

        :param subset_datasets: dataset name or iterable of dataset names to subset the results, defaults to None
        :type subset_datasets: str or Iterable[str], optional
        :return: Shannon Entropy for each dataset and sample
        :rtype: pd.Series
        """
        ic_usage = self.get_community_usage(subset_datasets = subset_datasets)
        diversity = ic_usage.apply(lambda x: entropy(x.dropna()), axis=1)
        return diversity
    
    def get_community_category_overrepresentation(self,
                                                  layer: str,
                                                  subset_datasets: Optional[Union[str, Iterable]] = None,
                                                  truncate_negative: bool = True,
                                                  subset_categories: Collection[str] = None
                                                  ) -> pd.DataFrame:
        """_summary_

        :param layer: name of categorical data layer
        :type layer: str
        :param subset_datasets: dataset name or iterable of dataset names to subset the results, defaults to None
        :type subset_datasets: str or Iterable[str], optional
        :param truncate_negative: Truncate negative residuals to 0, defaults to True
        :type truncate_negative: bool, optional
        :param subset_categories: Provide a subset of categories for calculating overrepresentation
        :type subset_categories: Collection[str]
        :return: category × program matrix of overrepresentation values
        :rtype: pd.DataFrame
        """
        df = self.integration.get_category_overrepresentation(layer=layer,
                                                              subset_datasets=subset_datasets,
                                                              truncate_negative=False,
                                                              subset_categories=subset_categories)
        mapper = {tuple([program.split("|")[0], int(program.split("|")[1]), int(program.split("|")[2])]): comm for program, comm in self.node_to_community.items()}
        df.columns = df.columns.map(mapper)
        df = df.T.groupby(level=0).mean().T
        df = df.reindex(self.ordered_community_names, axis=1)
        if truncate_negative:
            df = df.clip(lower=0)
        return df
    
    def get_community_metadata_correlation(self,
                                           layer: str,
                                           subset_datasets: Optional[Union[str, Iterable]] = None,
                                           method: str = "pearson"
                                           ) -> pd.Series:
        """Calculate Pearson correlation of program usage to numerical metadata across samples/observations.

        :param layer: name of numerical data layer
        :type layer: str
        :param subset_datasets: _description_, defaults to None
        :type subset_datasets: Optional[Union[str, Iterable]], optional
        :param method: Correlation method: "pearson", "spearman", or "kendall". Defaults to "pearson"
        :type method: str, optional
        :return: median correlation of program communities to metadata
        :rtype: pd.Series
        """

        ser = self.integration.get_metadata_correlation(layer=layer, subset_datasets=subset_datasets, method=method)
        mapper = {tuple([program.split("|")[0], int(program.split("|")[1]), int(program.split("|")[2])]): comm for program, comm in self.node_to_community.items()}
        ser.index = ser.index.map(mapper)
        ser = ser.groupby(level=0).mean()
        ser = ser.reindex(self.ordered_community_names)
        return ser

    
    def create_program_network(self):
        """
        Creates a program graph based on pairwise correlation thresholds and selected ranks.
        """
        # get matrix of edges after filtering
        subset = self.integration.get_corr_matrix_lowertriangle(selected_k_filter=True)
        subset_quantile = self.integration.get_corr_matrix_lowertriangle(selected_k_filter=True, quantile_transformation=True)
        
        # filter edges by inter and intra-dataset min_corr thresholds
        for (dataset_row, dataset_col), threshold in self.integration.pairwise_thresholds.items():
            filtered_chunk = subset.loc[subset.index.get_level_values(0) == dataset_row, subset.columns.get_level_values(0) == dataset_col] <= threshold
            subset.loc[subset.index.get_level_values(0) == dataset_row, subset.columns.get_level_values(0) == dataset_col] = subset.loc[subset.index.get_level_values(0) == dataset_row, subset.columns.get_level_values(0) == dataset_col].mask(filtered_chunk)
            subset_quantile.loc[subset_quantile.index.get_level_values(0) == dataset_row, subset_quantile.columns.get_level_values(0) == dataset_col] = subset_quantile.loc[subset_quantile.index.get_level_values(0) == dataset_row, subset_quantile.columns.get_level_values(0) == dataset_col].mask(filtered_chunk)

        subset.index = pd.Index(["|".join((program[0], str(program[1]), str(program[2]))) for program in subset.index])
        subset.columns = pd.Index(["|".join((program[0], str(program[1]), str(program[2]))) for program in subset.columns])
        subset_quantile.index = pd.Index(["|".join((program[0], str(program[1]), str(program[2]))) for program in subset_quantile.index])
        subset_quantile.columns = pd.Index(["|".join((program[0], str(program[1]), str(program[2]))) for program in subset_quantile.columns])
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
        self.program_graph = G

    def community_search(self,
                           algorithm="greedy_modularity",
                           resolution: Optional[float] = 2.0,
                           k: Optional[int] = None,
                           edge_weight: Optional[str] = None
                           ):
        """Identifies communities from the program graph.

        :param algorithm: Valid algorithms include: "greedy_modularity", "leiden", "asyn_lpa", "girvan_newman". Defaults to "greedy_modularity"
        :type algorithm: str, optional
        :param resolution: Resolution parameter is related to the relative size and number of communities. (resolution must not be set for asyn_lpa and girvan_newman algorithms), defaults to 2.0
        :type resolution: float, optional
        :param k: Number of clusters (applies to the girvan_newman algorithm only), defaults to None
        :type k: int, optional
        :param edge_weight: Add edge weights, using either the correlation (corr), and quantile-transformed correlation,
            which can be calculated prior to or after filtering (prefilter_quantile and postfilter_quantile, respectively). Defaults to None
        :type edge_weight: str, optional
        """
        
        G = self.program_graph
            
        # Check parameters
        logging.info(f"Community search: algorithm = {algorithm}, resolution = {resolution}")
        if k is not None and (algorithm in ("greedy_modularity","leiden","asyn_lpa")):
            raise ValueError(f"{algorithm} community search algorithm does not support parameter k.")
        if resolution is not None and algorithm in ("asyn_lpa", "girvan_newman", "asyn_fluidc"):
            raise ValueError(f"{algorithm} community search algorithm does not support the resolution parameter.")
        if edge_weight not in (None, "corr", "prefilter_quantile", "postfilter_quantile"):
            raise ValueError(f'{edge_weight} is not a valid edge weighting method. Choose from: "corr", "prefilter_quantile", "postfilter_quantile".')

        # Community search algorithms
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
            algo_output = nx.algorithms.community.label_propagation.asyn_lpa_communities(G, weight=edge_weight)
            communities = {name: nodes for name, nodes in enumerate(algo_output, start=1)}
        elif algorithm =="girvan_newman":
            algo_output = nx.algorithms.community.centrality.girvan_newman(G)
            for partition in algo_output:
                if len(partition) >= k:
                    break
            communities = {}
            for name, nodes in enumerate(partition, start=1):
                communities[name] = nodes
        elif algorithm == "asyn_fluidc":
            algo_output = nx.algorithms.community.asyn_fluidc(G, k, max_iter=100)
            communities = {name: nodes for name, nodes in enumerate(algo_output, start=1)}
        else:
            raise ValueError(f"{algorithm} is not a valid community detection algorithm")

        # community names must be strings to avoid problems with TOML persistence and flexibility for custom community naming (eg., for subclustering)
        communities = {str(k): v for k, v in communities.items()}

        self.communities = communities

        # also create a community network with default parameters, so that self.comm_graph is available if needed. However, self.create_community_network can be called again
        self.create_community_network()

    def read_communities_from_toml(self, toml_file):

        with open(toml_file, 'rb') as f:
            communities = tomli.load(f)
            
        # community names must be strings to avoid problems with TOML persistence and flexibility for custom community naming (eg., for subclustering)
        communities = {str(k): v for k, v in communities.items()}
        
        self.communities = communities

        # also create a community network with default parameters, so that self.comm_graph is available if needed. However, self.create_community_network can be called again
        self.create_community_network()

    def prune_communities(self,
                          min_nodes: int = 1,
                          min_datasets: int = 1,
                          min_nodes_per_dataset: int = 0,
                          renumber = False):
        """Prune communities based on one or more filters.

        :param min_nodes: Minimum number of nodes per community, defaults to 1
        :type min_nodes: int, optional
        :param min_datasets: Minimum number of datasets with nodes in a community, defaults to 1
        :type min_datasets: int, optional
        :param min_nodes_per_dataset: Minimum number of nodes per dataset in a community, defaults to 0
        :type min_nodes_per_dataset: int, optional
        :param renumber: Reset the names of the communities after pruning, defaults to False
        :type renumber: bool, optional
        """
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
        self.create_community_network()
        self.program_graph = self.program_graph.subgraph([node for nodes in self.communities.values() for node in nodes])  # removes nodes not in a pruned community


    @property
    def ordered_community_names(self) -> List[str]:
        """Get community names, ordered numerically after separating clusters and subclusters. For example, this algorithm can properly sort communities labelled 1.1, 1.2, 1.3, 2.1, 2.2, 2.10, 2.15.

        :return: list of communities
        :rtype: list[str]
        """
        community_names = sorted(self.communities.keys(), key = lambda cstr: [int(lvl) for lvl in cstr.split(".")])
        return community_names
    
    @property
    def node_to_community(self) -> Dict[str, str]:
        dict_of_programs = {program: community for community, programs in self.communities.items() for program in programs}
        return dict_of_programs


    def get_vectorized_community_sort_key(self, community_names: pd.Index) -> pd.Index:
        """Return a vector of sort_indicesGet community names, ordered numerically after separating clusters and subclusters. For example, this algorithm can properly sort communities labelled 1.1, 1.2, 1.3, 2.1, 2.2, 2.10, 2.15.

        :return: sort indices
        :rtype: pd.Index
        """
        
        return pd.Index([self.ordered_community_names.index(c) for c in community_names])
    
    def add_community_weights_to_graph(self, base_weight = 1.0, shared_community_weight = 500, shared_dataset_weight = 1.05):
        """Add attributes to the program graph for generating the community-weighted network. If an edge connects two programs 

        :param base_weight: Starting weight for all edges, defaults to 1.0
        :type base_weight: float, optional
        :param shared_community_weight: Multiplier if edges connect programs in the same community, defaults to 500
        :type shared_community_weight: float, optional
        :param shared_dataset_weight: Multiplier if edges connect programs in the same dataset, defaults to 1.05
        :type shared_dataset_weight: float, optional
        """
        edge_attr = {}
        for edge in self.program_graph.edges:
            weight = base_weight
            if edge[0] in self.node_to_community and edge[1] in self.node_to_community:  # nodes might not have communities due to pruning, and so will be treated as if they are in different communities for layout purposes
                if self.node_to_community[edge[0]] == self.node_to_community[edge[1]]:
                    weight *= shared_community_weight
            if edge[0].split("|")[0] == edge[1].split("|")[0]:
                weight *= shared_dataset_weight
            edge_attr[edge] = weight
        nx.set_edge_attributes(self.program_graph, edge_attr, name="community_weight")

    def write_communities_toml(self, filename: str):
        """Write communities to TOML file.

        :param filename: path to TOML file
        :type filename: str
        """
        toml_conformed = {str(community): sorted(programs, key=lambda x: int(x.split("|")[1])) for community, programs in self.communities.items()}
        with open(filename, "wb") as fh:
            tomli_w.dump(toml_conformed, fh)

    def write_program_network_graphml(self, filename: str):
        """Output the program network in graphml format.

        :param filename: path to .graphml file
        :type filename: str
        """
        nx.write_graphml(self.program_graph, filename)
    
    def write_community_network_graphml(self, filename):
        """Output the community network in graphml format.

        :param filename: path to .graphml file
        :type filename: str
        """
        nx.write_graphml(self.comm_graph, filename)

    def compute_layout(self,
                       algorithm: str = "community_weighted_spring",
                       base_weight: float = 1.0,
                       shared_community_weight: float = 500,
                       shared_dataset_weight: float = 1.05,
                       community_layout_algorithm: str = "spring",
                       **kwargs):
        """Compute the network layout using a specified algorithm.

        :param algorithm: Algorithm for network layout. Choose from "neato" (from pyGraphViz, minimizes edge and node overlap), "spring", "community_weighted_spring" (weights
            the network for optimal separation of communities and/or datasets), "umap", defaults to "community_weighted_spring"
        :type algorithm: str, optional
        :param base_weight: Starting weight for all edges (applies to community_weighted_spring algorithm only), defaults to 1.0
        :type base_weight: float, optional
        :param shared_community_weight: Multiplier if edges connect programs in the same community (applies to community_weighted_spring algorithm only), defaults to 500
        :type shared_community_weight: float, optional
        :param shared_dataset_weight: Multiplier if edges connect programs in the same dataset (applies to community_weighted_spring algorithm only), defaults to 1.05
        :type shared_dataset_weight: float, optional
        :param community_layout_algorithm: Algorithm for layout of community network. Choose from "centroid" (centroid of all community programs based on the program graph),
            "spring", and "neato" (from pyGraphViz, minimizes edge and node overlap). Defaults to "spring"
        :type community_layout_algorithm: str, optional
        """
        if algorithm == "neato":
            layout = nx.nx_agraph.graphviz_layout(self.program_graph, prog="neato", args='-Goverlap=true', **kwargs)
        elif algorithm == "spring":
            layout = nx.spring_layout(self.program_graph, **kwargs)
            layout = {node: list(coords) for node, coords in layout.items()}
        elif algorithm == "community_weighted_spring":
            self.add_community_weights_to_graph(base_weight = base_weight,
                                                shared_community_weight=shared_community_weight,
                                                shared_dataset_weight=shared_dataset_weight)
            layout = nx.spring_layout(self.program_graph, weight="community_weight", **kwargs)
            layout = {node: list(coords) for node, coords in layout.items()}
        elif algorithm == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError("umap-learn is not installed. Please install using:\n\n\t"
                            "conda install -c conda-forge umap-learn"
                            )
            from sklearn.preprocessing import StandardScaler, RobustScaler

            programs = {}
            for dataset_name, dataset in self.integration.datasets.items():
                selected_k = self.integration.k_table[(dataset_name, "selected_k")]
                selected_k = selected_k[selected_k].index.to_list()
                df = dataset.get_programs(k=selected_k)
                programs[dataset_name] = df.loc[:, ]

            programs = pd.concat(programs, axis=1).sort_index(axis=1)
            # Standardize features for dimensionality reduction
            table = programs.dropna().T
            x = RobustScaler().fit_transform(table.values)
            embedding = umap.UMAP(n_neighbors=25, min_dist=0.01, **kwargs).fit_transform(x)
            layout = {"|".join((program[0], str(program[1]), str(program[2]))): list(emb.astype(float)) for program, emb in zip(table.index, embedding)}
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
        
        self.compute_community_network_layout(algorithm=community_layout_algorithm)

    def compute_community_network_layout(self, algorithm: str = "spring", weight="weight", **kwargs):
        """_summary_

        :param algorithm: Algorithm for layout of community network. Choose from "centroid" (centroid of all community programs based on the program graph),
            "spring", and "neato" (from pyGraphViz, minimizes edge and node overlap). Defaults to "spring", defaults to "spring"
        :type algorithm: str, optional
        :param weight: Edge weights. Options include: "weight" (sqrt(n_edges)), "n_edges" (number of edges), or a custom string which represents edge attributes in self.comm_graph. Defaults to "weight"
        :type weight: str, optional
        :raises NotImplementedError: Error if invalid algorithm is chosen
        """
        
        logging.info(f"Computing community layout using {algorithm} method.")
        if algorithm == "centroid":
            assert not kwargs
            # Centroid method for community layout
            self.comm_layout = {}
            for community_name, nodes in self.communities.items():
                points = np.array([self.layout[node] for node in nodes])
                centroid = (np.median(points[:, 0]), np.median(points[:, 1]))
                self.comm_layout[community_name] = centroid
                
        elif algorithm == "neato":
            layout = nx.nx_agraph.graphviz_layout(self.comm_graph, prog="neato", args='-Goverlap=true', **kwargs)
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
        
        elif algorithm == "spring":
            layout = nx.spring_layout(self.comm_graph, weight=weight, **kwargs)
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
        """Create a matrix with community and dataset on each axis. Returns the highest correlation coefficient between nodes in each subset (based on community and dataset).

        :return: maximum correlation coefficient matrix for communities/datasets
        :rtype: pd.DataFrame
        """
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
        """Creates community network after community search.
        """
        logging.info("Creating community network")
        edge_list = []
        for c1, n1 in self.communities.items():
            for c2, n2 in self.communities.items():
                if c1 != c2:  # no self-loops
                    n_edges = len(list(nx.edge_boundary(self.program_graph, n1, n2)))
                    weight = np.sqrt(n_edges)
                    edge_list.append((c1, c2, n_edges, weight))

        edge_list = pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges", "weight"))
        self.comm_graph = nx.from_pandas_edgelist(df = pd.DataFrame(edge_list, columns = ("comm1", "comm2", "n_edges", "weight")),
                                                  source = "comm1",
                                                  target = "comm2",
                                                  edge_attr=["n_edges", "weight"])
        self.comm_graph.add_nodes_from(self.communities.keys())
        
    def get_representative_program_ids(self,
                         correlation_axis: Literal["programs", "usage"] = "programs"
                         ) -> pd.Series:
        """Select programs based on correlation with the median of all programs in each community
        
        :param correlation_axis: axis on which to compute correlations between programs whether correlating the programs or the usages, defaults to "programs"
        :type correlation_axis: int or dict
        :return: Communities, indexed by the most central program for each dataset
        :rtype: pd.Series
        """
        selected_programs = {}

        if correlation_axis == "programs":
            cg = self.consensus()
            programs_df = self.integration.get_programs()
            for community, nodes in self.communities.items():
                for dataset_name in self.integration.datasets:
                    programs = [node_to_program(node) for node in nodes]
                    programs = [program for program in programs if program[0] == dataset_name]
                    if programs:
                        top_program = programs_df[programs].corrwith(cg[(community, dataset_name)]).idxmax()
                        selected_programs[top_program] = community
            

        elif correlation_axis == "usage":
            cu = self.get_community_usage()
            usages = self.integration.get_usages(normalize=True)
            for community, nodes in self.communities.items():
                for dataset_name in self.integration.datasets:
                    programs = [node_to_program(node) for node in nodes]
                    programs = [program for program in programs if program[0] == dataset_name]
                    if programs:
                        top_program = usages.loc[dataset_name, programs].corrwith(cu.loc[dataset_name, community]).idxmax()
                        selected_programs[top_program] = community

        selected_programs = pd.Series(selected_programs, name="Community").rename_axis(index=("dataset", "k", "program"))
        selected_programs = selected_programs.sort_index().sort_values(key=lambda x: x.map(self.ordered_community_names.index))

        return selected_programs

    def get_selected_rank_program_ids(self,
                            k: Union[int, Dict[str, int]]) -> pd.Series:
        """Select programs based on rank. k may be either a single value for all datasets
        as an integer, or as a dict with separate values for each dataset.

        :param k: a rank or dict with dataset keys and rank values, defaults to None
        :type k: int or dict
        :return: Communities, indexed by the lowest rank program(s) for each dataset
        :rtype: pd.Series
        """
        programs = pd.Series(self.node_to_community, name="Community")
        programs.index = pd.MultiIndex.from_tuples((node_to_program(node) for node in programs.index), names=["dataset", "k", "program"])
        programs = programs.sort_index()
        if isinstance(k, int):
            selected_programs = programs.xs(k, level=1, drop_level=False)
        elif isinstance(k, dict):
            selected_programs = []
            for dataset_name, ds_k in k.items():
                selected_programs.append(programs.xs(dataset_name, drop_level=False).xs(ds_k, level=1, drop_level=False))
            selected_programs = pd.concat(selected_programs)
        selected_programs = selected_programs.sort_index().sort_values(key=lambda x: x.map(self.ordered_community_names.index))
        return selected_programs

    def get_representative_programs(self,
                         correlation_axis: Literal["programs", "usage"] = "programs"
                         ) -> pd.DataFrame:
        """Select programs based on correlation with the median of all programs in each community
        
        :param correlation_axis: axis on which to compute correlations between programs whether correlating the programs or the usages, defaults to "programs"
        :type correlation_axis: int or dict
        :return: Features × programs matrix
        :rtype: pd.DataFrame
        """
        rep_programs_ids = self.get_representative_program_ids(correlation_axis=correlation_axis)
        rep_programs = self.integration.get_programs()[rep_programs_ids.index]
        rep_programs.columns = pd.MultiIndex.from_tuples([[community] + list(program_id) for community, program_id in zip(rep_programs_ids, rep_programs_ids.index)], names=["Community", "dataset", "k", "Program"])
        return rep_programs  # TODO: Check return types

    def get_lowest_rank_programs(self,
                                min_k: Optional[Union[int, Dict[str, int]]] = None,
                            ) -> pd.Series:
        """Identify the programS that are the lowest rank for each dataset. A minimum rank to be considered may be supplied,
        either for all datasets as an integer, or as a dict with separate thresholds for each dataset.

        :param min_k: a minimum rank to consider for the minimal-rank programs, defaults to None
        :type min_k: int or dict, optional
        :return: Communities, indexed by the lowest rank program(s) for each dataset
        :rtype: pd.Series
        """
        programs = pd.Series(self.node_to_community, name="Community")
        programs.index = pd.MultiIndex.from_tuples((node_to_program(node) for node in programs.index), names=["dataset", "k", "program"])
        programs = programs.sort_index()
        if isinstance(min_k, int):
            programs = programs[programs.index.get_level_values(1) >= min_k]
        elif isinstance(min_k, dict):
            
            keep = []
            for program in programs.index:
                if not program[0] in min_k:
                    keep.append(True)
                elif program[1] >= min_k[program[0]]:
                    keep.append(True)
                else:
                    keep.append(False)
            programs = programs[keep]

        selected_programs = []
        for community, cprograms in programs.groupby(programs):
            for dataset_name, cdprograms in cprograms.groupby(level=0):
                min_rank = cdprograms.index.get_level_values(1).min()
                min_rank_programs = cdprograms[cdprograms.index.get_level_values(1) == min_rank]
                selected_programs.append(min_rank_programs)
        selected_programs = pd.concat(selected_programs)
        selected_programs = selected_programs.sort_index().sort_values(key=lambda x: x.map(self.ordered_community_names.index))
        return selected_programs  # TODO: Check return types

    def count_intracommunity_edges(self):
        """Counts edges within each community that are within and between datasets.

        :return: Table with number of edges for each dataset and dataset pair
        :rtype: pd.DataFrame
        """
        edgedata = {}
        for cname, cmembers in self.communities.items():
            community_graph = self.program_graph.subgraph(cmembers)
            stats = {}
            for edge in community_graph.edges:
                datasets = tuple(sorted((edge[0].split("|")[0], edge[1].split("|")[0])))
                if datasets not in stats:
                    stats[datasets] = 0
                else:
                    stats[datasets] += 1
            edgedata[cname] = stats
        edgedata = pd.DataFrame(edgedata).T.fillna(0)
        edgedata.columns = ["-".join(sorted(items)) for items in edgedata.columns]
        return edgedata

    def most_correlated_edge_between_datasets(self,
                                              ds1: str,
                                              ds2: str,
                                              ds1_rank: Optional[int] = None,
                                              ds2_rank: Optional[int] = None):

        """Identifies the most correlated edge between two datasets within each community.
        Optionally, you can fix the rank of 1 or both datasets to restrict the search space.

        :param ds1: name of dataset 1
        :type ds1: str
        :param ds2: name of dataset 2
        :type ds2: str
        :param ds1_rank: rank of dataset 1
        :type ds1_rank: int, optional
        :param ds2_rank: rank of dataset 2
        :type ds2_rank: int, optional
        :return: High-correlation edges, one per community
        :rtype: pd.DataFrame
        """
        selected_edges = []
        for cname, cmembers in self.communities.items():
            # get correlation values
            community_idx = pd.MultiIndex.from_tuples([node_to_program(node) for node in cmembers])
            ds1_idx = community_idx[community_idx.get_level_values(0) == ds1]
            if ds1_rank is not None:
                ds1_idx = ds1_idx[ds1_idx.get_level_values(1) == ds1_rank]
            ds2_idx = community_idx[community_idx.get_level_values(0) == ds2]
            if ds2_rank is not None:
                ds2_idx = ds2_idx[ds2_idx.get_level_values(1) == ds2_rank]
            community_corr = self.integration.corr_matrix.loc[ds1_idx, ds2_idx]
            maxcorr = community_corr[community_corr == community_corr.max().max()]
            maxcorr = maxcorr.dropna(how="all", axis=0).dropna(how="all", axis=1)
            maxcorr = maxcorr.melt(ignore_index=False).dropna()
            maxcorr["Community"] = cname 
            maxcorr = maxcorr.reset_index().set_index("Community").drop(columns="value")
            maxcorr.columns = pd.MultiIndex.from_product([(ds1, ds2), ("dataset", "k", "program")])
            selected_edges.append(maxcorr)
        selected_edges = pd.concat(selected_edges)
        return selected_edges

    
    def consensus(self,
                  method: str = "median",
                  min_k: int = 2
                  ) -> pd.DataFrame:
        """Generate a 'consensus' program for each community and dataset by taking the median of all constituent programs, separately for each dataset.

        :param method: Choose from: "mean", "median". Defaults to "median"
        :type method: str, optional
        :param min_k: Minimum k value to filter programs prior to consensus, defaults to 2
        :type min_k: int, optional
        :return: Communities-Datasets × variables consensus matrix
        :rtype: pd.DataFrame
        """
        program_to_community = {node_to_program(node): community for node, community in self.node_to_community.items()}
        programs = self.integration.get_programs()
        programs = programs.loc[:, programs.columns.get_level_values(1).astype(int) >= min_k]
        programs.columns = pd.MultiIndex.from_arrays([programs.columns.map(program_to_community),
                                programs.columns.get_level_values(0)
                                ], names = ("Community", "Dataset"))
        programs = programs.loc[:, ~programs.columns.to_frame().isnull().any(axis=1)]  # Remove programs not in any community
        # aggregate programs to community
        if method == "median":
            community_scores = programs.T.groupby(level=[0,1]).median().T
        elif method == "mean":
            community_scores = programs.T.groupby(level=[0,1]).mean().T
        else:
            raise NotImplementedError
        community_scores = community_scores.loc[:, self.ordered_community_names]  # sort community names
        return community_scores


    def transfer_labels(self,
                          source: Optional[Union[str, Collection[str]]] = None,
                          dest: Optional[Union[str, Collection[str]]] = None,
                          layer: Optional[Union[str, Collection[str]]] = None,
                          subset_categories: Collection[str] = None,
                          simplify: bool = True
                          ) -> pd.DataFrame:
        """Transfer sample categories between datasets using usage of representative programs as a proxy.

        :param source: Source dataset(s) for label transfer, defaults to None
        :type source: Union[str, Collection[str]], optional
        :param dest: Target dataset(s) for label transfer, defaults to None
        :type dest: Union[str, Collection[str]], optional
        :param layer: name of categorical data layer(s) from source dataset, defaults to None
        :type layer: Union[str, Collection[str]], optional
        :param subset_categories: a subset of categories for calculating overrepresentation, defaults to None
        :type subset_categories: Collection[str], optional
        :param simplify: Simplify multi-index results when only one source, dest, or layer are specified, defaults to True
        :type simplify: bool, optional
        :raises ValueError: if source or dest is not a correct type
        :return: transfer score
        :rtype: pd.DataFrame
        """

        # parameter flexibility, by default everything
        if source is None:
            sources = self.integration.datasets.keys()
        elif isinstance(source, str):
            sources = [source]
        elif isinstance(source, Collection):
            sources = source
        else:
            raise ValueError
        
        if layer is None:
            # get all layers for all selected source datasets
            layers = self.integration.get_metadata_df(include_numerical=False, subset_datasets=sources).columns
        elif isinstance(layer, str):
            layers = [layer]
        elif isinstance(layer, Collection):
            layers = layer
        else:
            raise ValueError
        
        if dest is None:
            dests = self.integration.datasets.keys()
        elif isinstance(dest, str):
            dests = [dest]
        elif isinstance(dest, Collection):
            dests = dest
        else:
            raise ValueError
        

        agg = []
        for source_name in sources:
            for layer_name in layers:
                rowblock = []
                for dest_name in dests:
                    rprogs = self.get_representative_program_ids()  # representative programs
                    source_progs = rprogs.xs(source_name)
                    source_or = self.integration.datasets[source_name].get_category_overrepresentation(layer=layer_name, subset_categories=subset_categories)[source_progs.index]
                    source_or.columns = source_or.columns.map(source_progs)  # community-level overrepresentation based on representative programs
                    dest_progs = rprogs.xs(dest_name)
                    dest_usage = self.integration.datasets[dest_name].get_usages(normalize=False)
                    dest_cu = dest_usage[dest_progs.index]
                    dest_cu.columns = dest_cu.columns.map(dest_progs) # community-level usage based on representative programs
                    intgc = dest_cu.columns.intersection(source_or.columns)  # integrative communities (i.e., shared communities)
                    
                    source_or = source_or[intgc].div(source_or[intgc].sum(axis=1), axis=0)
                    dest_cu = dest_cu[intgc].div(dest_cu[intgc].sum(axis=1), axis=0)

                    transfer_df = source_or @ dest_cu[intgc].T  # multiply usage by overrepresentation for integrative communities
                    transfer_df.index = pd.MultiIndex.from_arrays([[source_name]*transfer_df.shape[0],
                                                                   [layer_name]*transfer_df.shape[0],
                                                                   transfer_df.index],
                                                                   names=["source_dataset", "layer", "category"])

                    transfer_df.columns = pd.MultiIndex.from_arrays([[dest_name]*transfer_df.shape[1], transfer_df.columns], names=["dest_dataset", "sample"])
                    rowblock.append(transfer_df)
                agg.append(pd.concat(rowblock, axis=1))
        agg = pd.concat(agg)

        if simplify:
            if isinstance(layer, str):
                agg = agg.droplevel(axis=0, level="layer")
            if isinstance(source, str):
                agg = agg.droplevel(axis=0, level="source_dataset")
            if isinstance(dest, str):
                agg = agg.droplevel(axis=1, level="dest_dataset")
        return agg


def compare_community_jaccard_similarity(name1: str, network1: Network, name2: str, network2: Network, subset_to_shared_datasets: bool = True):
    """Calculates the jaccard similarity of communities between two networks with overlapping nodes. One use case is to compare integrations
    after adding new datasets; the other is to compare integrations at different community resolutions.

    :param name1: name of first network
    :type name1: str
    :param network1: first network
    :type network1: :class:`mosaicmpi.Network`
    :param name2: name of second network
    :type name2: str
    :param network2: second network
    :type network2: :class:`mosaicmpi.Network`
    :param subset_to_shared_datasets: calculate jaccard similarity over shared datasets only, defaults to True
    :type subset_to_shared_datasets: bool, optional
    """

    all_datasets = set(network1.integration.datasets) | set(network2.integration.datasets)
    shared_datasets = set(network1.integration.datasets) & set(network2.integration.datasets)
    net1_datasets = set(network1.integration.datasets) - shared_datasets
    net2_datasets = set(network2.integration.datasets) - shared_datasets

    jaccard = pd.DataFrame(np.nan,
                           index=pd.Index(network1.ordered_community_names, name=name1),
                           columns=pd.Index(network2.ordered_community_names, name=name2))
    for net1_comm, net1_nodes in network1.communities.items():
        for net2_comm, net2_nodes in network2.communities.items():
            
            intersection = len(net2_nodes & net1_nodes)
            if subset_to_shared_datasets:
                # jaccard is calculated based on shared datasets only
                union = len(set(n for n in (net1_nodes | net2_nodes) if node_to_program(n)[0] in shared_datasets))
            else:
                union = len(net1_nodes | net2_nodes)
            jaccard.loc[net1_comm, net2_comm] = intersection / union

    jaccard = jaccard[jaccard.idxmax(0).astype(int).sort_values().index]

    return jaccard