import os
from glob import glob
from nmfsns.containers import CnmfResult

for directory in glob("path/to/cnmf/runs/*/")
    obj = CnmfResult(directory)
    outfile = os.path.join(os.path.normpath(directory) + ".h5ad")
    obj.write_h5ad(outfile)