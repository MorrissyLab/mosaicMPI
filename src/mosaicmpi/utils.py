import logging
import os
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from collections.abc import Collection
from . import __version__, logging_started


def start_logging(output_path=None):
    """
    Starts logging to the console. If output_path is specified, logging messages are also appended to a log file.

    :param output_path: path to log file, defaults to None
    :type output_path: str, optional
    """

    if output_path is None:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
            handlers=[
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
            handlers=[
                logging.FileHandler(output_path, mode="a"),
                logging.StreamHandler()
            ]
        )
    global logging_started
    if not logging_started:
        logging.info(f"mosaicMPI version {__version__}")
    logging_started = True

def newline_wrap(string, length=40):
    """Adds \\n characters to a string at a specified length. Helps to control the width of :class:`Text` objects in plots.

    :param string: input string
    :type string: str
    :param length: number of characters per line, defaults to 40
    :type length: int, optional
    :return: wrapped string
    :rtype: str
    """
    return '\n'.join(string[i:i + length] for i in range(0, len(string), length))

def save_df_to_text(obj, filename):
    """Save DataFrame to tab-separated text file.

    :param obj: dataframe or array object
    :type obj: Union[pd.DataFrame, np.array]
    :param filename: path to .txt file
    :type filename: str
    """
    obj.to_csv(filename, sep='\t')

def save_df_to_npz(obj: Union[pd.DataFrame, np.array], filename: str):
    """Save DataFrame to compressed npz file. Compatible with MultiIndex.

    :param obj: dataframe or array object
    :type obj: Union[pd.DataFrame, np.array]
    :param filename: path to .npz file
    :type filename: str
    """
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def load_df_from_npz(filename: str) -> pd.DataFrame:
    """
    Load DataFrame from compressed npz file. Compatible with MultiIndex.

    :param filename: path to .npz file
    :type filename: str
    :return: Dataframe
    :rtype: pd.DataFrame
    """
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

def node_to_program(node_str) -> Tuple[str, int, int]:
    """Converts nodes like "CPTAC|3|5" into program IDs like ("CPTAC", 3, 5).

    :param node_str: Node ID from program-level graph.
    :type node_str: str
    :return: Program ID for indexing dataframes
    :rtype: Tuple[str, int, int]
    """
    
    node_str = node_str.split("|")
    program = (node_str[0], int(node_str[1]), int(node_str[2]))
    return program

def program_to_node(program: Tuple[str, int, int]) -> str:
    """Converts program IDs like ("CPTAC", 3, 5) into node IDs like "CPTAC|3|5".

    :param program: Program ID from dataframe indices
    :type program: Tuple[str, int, int]
    :return: Node ID for program-level graph.
    :rtype: str
    """
    node = "|".join((str(p) for p in program))
    return node
    

def get_adjusted_dpi(fig: Figure, target_dpi: float = 200) -> float:
    
    """Lower the DPI if figure size in pixels exceeds 2 ^ 16 in either direction.

    :param fig: figure
    :type fig: Figure
    :param target: maximum DPI, defaults to 200
    :type target: float, optional
    :return: DPI for error-free PNG outputs
    :rtype: float
    """

    max_dpi = (2**16 - 1)/(fig.get_size_inches().max())
    return min(max_dpi, target_dpi)

def save_fig(fig: Figure, filepath_no_ext: str, target_dpi: Optional[float] = 200, formats: Union[str, Collection[str]] = "pdf", close: bool = True, **kwargs):
    """Save figure to one or more files with the same basename. Directories are created as needed.

    :param fig: figure
    :type fig: Figure
    :param basename: path to filename excluding extension
    :type basename: str
    :param target_dpi: target DPI (for PNG output only), defaults to 200
    :type target_dpi: float, optional
    :param formats: figure extension or list of extensions. Values can be "pdf" or "png", or ("pdf", "png") for both, defaults to "pdf"
    :type formats: Union[str, Collection[str]], optional
    :param close: close figure after saving, defaults to True
    :type close: bool, optional
    """
    dpi = get_adjusted_dpi(fig, target_dpi)
    if isinstance(formats, str):
        formats = [formats]
    elif isinstance(formats, Collection):
        pass
    else:
        formats_type = type(formats)
        raise TypeError(f"{formats_type} is not a valid format or collection of formats. `formats` must be a string or Collection.")
    
    os.makedirs(os.path.dirname(filepath_no_ext), exist_ok=True)
    for format in formats:
        fig.savefig(fname = filepath_no_ext + "." + format, dpi=dpi, **kwargs)
    plt.close(fig)
    