

import logging
import numpy as np
import pandas as pd

def start_logging(output_path=None):
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
    return

def newline_wrap(string, length=40):
    return '\n'.join(string[i:i + length] for i in range(0, len(string), length))


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
        raise ValueError('Fetching HGNC Protein coding genes failed: ' + response['status'])

    protein_coding_genes = {entry["symbol"] for entry in data["response"]["docs"]}
    return protein_coding_genes

def save_df_to_text(obj, filename):
    obj.to_csv(filename, sep='\t')

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