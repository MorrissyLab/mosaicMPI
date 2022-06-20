import os
from glob import glob
from cnmfsns.containers import CnmfResult

for directory in glob("path/to/cnmf/runs/*/"):
    obj = CnmfResult.from_dir(directory)
    outfile = os.path.join(os.path.normpath(directory) + ".h5mu")
    obj.to_mudata().write(outfile)