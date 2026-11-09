#!/usr/bin/env bash
# Read-only cluster inspection and submission validation; does not submit a job.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
ws_find "${DDSM_WORKSPACE_ID:-ft-models}"
scontrol show partition test
sinfo -p test -N -o '%N %t %c %m %G %f'
sbatch --test-only --partition=test jobs/gpu_probe_test.sbatch
