#!/usr/bin/env bash
# Activate the same environment used by Slurm, then run all four synthetic checks.
set -euo pipefail
wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${wrapper_root}/.ddsm_charisma.env"
module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
export PYTHONPATH="${wrapper_root}:${PYTHONPATH:-}"
python "${wrapper_root}/tools/selftest_bio_core.py" \
  --ddsm-upstream "${DDSM_WORK_ROOT}/ddsm"
