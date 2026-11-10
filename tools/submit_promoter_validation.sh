#!/usr/bin/env bash
# Submit 4 methods x 3 seeds and a dependent Sei evaluation job.
set -euo pipefail

wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${wrapper_root}"
source .dna_benchmark.env

validation_job_id="$(sbatch --parsable jobs/promoter_four_multiseed_validation.sbatch)"
validation_job_id="${validation_job_id%%;*}"
run_root="${DNA_BENCHMARK_ROOT}/runs/validation-${validation_job_id}"
sei_job_id="$(sbatch --parsable \
  --dependency="afterok:${validation_job_id}" \
  --export="ALL,PROMOTER_RUN_ROOT=${run_root}" \
  jobs/promoter_sei_evaluate.sbatch)"
sei_job_id="${sei_job_id%%;*}"

echo "validation_job_id=${validation_job_id}"
echo "sei_job_id=${sei_job_id} dependency=afterok:${validation_job_id}"
echo "run_root=${run_root}"
echo "monitor=squeue -r -j ${validation_job_id},${sei_job_id}"
echo "summary=python tools/summarize_promoter_runs.py ${run_root}"
