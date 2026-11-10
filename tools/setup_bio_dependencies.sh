#!/usr/bin/env bash
# Login-node-only additions for reading FANTOM bigWig and tabix files.
set -euo pipefail
module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
python -m pip install pyBigWig==0.3.24
python -m pip check
python - <<'PY'
import pyBigWig
import numpy
import scipy
import torch
print('bio_dependencies=PASS')
print('torch', torch.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)
PY
echo "Dependencies were installed into ddsm-repro. A child script cannot activate the parent shell."
echo "next=bash tools/run_bio_selftest.sh"
