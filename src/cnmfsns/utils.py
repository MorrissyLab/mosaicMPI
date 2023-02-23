

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