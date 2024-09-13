from types import SimpleNamespace
from typing import Union, Optional, Literal
from collections.abc import Collection, Iterable

import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm

def program_gprofiler(program_df: pd.DataFrame,
                      species: Literal["hsapiens", "mmusculus"],
                      n_hsg: int = 1000,
                      gene_sets: Collection[str] = [],
                      no_iea: bool = True,
                      min_intersection: int = 5,
                      max_intersection: int = 500,
                      batch_size: int = 20,
                      show_progress_bar: bool = True
                      ) -> SimpleNamespace:
    
    from gprofiler import GProfiler

    result = SimpleNamespace()
    result.background = program_df.dropna(how="all").index.to_list()  # all genes in program_df
    result.hsg = program_df.rank(ascending=False) <= n_hsg  # bool dataframe of high scoring genes across programs
    prog_names_str = program_df.columns.map(str)  # gProfiler multi-query only supports string names for queries
    # but we can decode them
    if program_df.columns.nlevels == 1:
        prog_names_str_decoder = {progstr: [prog] for progstr, prog in zip(prog_names_str, program_df.columns)}
    else:
        prog_names_str_decoder = {progstr: list(prog) for progstr, prog in zip(prog_names_str, program_df.columns)}
    prog_level_names = program_df.columns.names
    result.query = {prog_str: genes[genes].index.to_list() for (prog, genes), prog_str in zip(result.hsg.items(), prog_names_str)}
    result.gene_sets = gene_sets
    result.no_iea = no_iea

    gp = GProfiler(return_dataframe=True)

    result.gprofiler_output = []
    batch_query = []
    for i, query in enumerate(tqdm(result.query.keys(), total=len(result.query), unit="program", desc="Querying g:Profiler", disable=not show_progress_bar), start=1):
        batch_query.append(query)
        if i % batch_size == 0 or i == len(result.query):
            batch_result = gp.profile(organism=species, query={q: result.query[q] for q in batch_query},
                                    sources=gene_sets, no_iea=no_iea, background=result.background)
            result.gprofiler_output.append(batch_result)
            batch_query = []
        
    result.gprofiler_output = pd.concat(result.gprofiler_output)
    result.gprofiler_output["-log10pval"] = -np.log10(result.gprofiler_output["p_value"])
    subset = ((result.gprofiler_output["intersection_size"] <= max_intersection) &
              (result.gprofiler_output["intersection_size"] >= min_intersection))
    result.summary = result.gprofiler_output[subset].pivot(index=["source", "native", "name", "description", "term_size"], columns="query")
    stats = ["-log10pval", "query_size", "intersection_size"]
    result.summary = result.summary[stats]
    result.summary.columns = pd.MultiIndex.from_tuples([([c[0]] + prog_names_str_decoder[c[1]]) for c in result.summary.columns], names=["stat"] + prog_level_names)

    # conform column order to input dataframe
    if program_df.columns.nlevels == 1:
        sorted_cols = pd.MultiIndex.from_tuples([(stat, prog) for stat in stats for prog in program_df.columns])
    else:
        sorted_cols = pd.MultiIndex.from_tuples([tuple([stat] + list(prog)) for stat in stats for prog in program_df.columns])
    result.summary = result.summary.reindex(columns=sorted_cols)
    return result

def order_genesets(df: pd.DataFrame):
    """Order genesets by the column with highest significance, followed by the max significance value.

    :param df: A geneset × program/sample matrix with -log10(pvals) as values.
    :type df: pd.DataFrame
    """
    # sort gene sets by highest column and then highest value of that column
    if df.shape[0] > 0:
        stats = pd.DataFrame({"col": df.idxmax(axis=1), "max": df.max(axis=1)})
        ordered = []
        for col in df.columns:
            ordered.append(stats[stats["col"] == col].sort_values("max", ascending=False))
        ordered = pd.concat(ordered)
        ordered_df = df.loc[ordered.index]
    else:
        ordered_df = df
    return ordered_df

def program_ssgsea():
    pass