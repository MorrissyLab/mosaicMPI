import os
from glob import glob
from cnmfsns.containers import CnmfResult

for directory in glob("path/to/cnmf/runs/*/"):
    obj = CnmfResult.from_dir(directory)
    outfile = os.path.normpath(directory)
    obj.to_mudata().write(outfile + ".h5mu")

    # or just write the anndata object for the GEP scores:
    obj.to_anndata("gene_spectra_score").write(outfile + ".h5ad")