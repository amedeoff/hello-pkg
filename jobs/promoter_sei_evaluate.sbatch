#!/usr/bin/env bash
#SBATCH --job-name=promoter-sei-eval
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=promoter-sei-eval-%j.log
#SBATCH --error=promoter-sei-eval-%j.err
set -euo pipefail

wrapper_root="${DDSM_WRAPPER_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "${wrapper_root}/.ddsm_charisma.env"
source "${wrapper_root}/.dna_benchmark.env"
: "${PROMOTER_RUN_ROOT:?Submit with --export=ALL,PROMOTER_RUN_ROOT=/absolute/run/path}"

data_dir="${DNA_BENCHMARK_ROOT}/data/promoter_design"
upstream_dir="${DDSM_WORK_ROOT}/ddsm"

module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${wrapper_root}:${PYTHONPATH:-}"

echo "job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION} host=$(hostname)"
echo "run_root=${PROMOTER_RUN_ROOT}"
test "${SLURM_JOB_PARTITION}" = test
test -d "${PROMOTER_RUN_ROOT}"
test -s "${data_dir}/best.sei.model.pth.tar"
test -s "${data_dir}/target.sei.names"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${wrapper_root}"
python -u tools/evaluate_sei_sp_mse.py \
  "${PROMOTER_RUN_ROOT}" \
  --data-dir "${data_dir}" \
  --ddsm-upstream "${upstream_dir}" \
  --weights "${data_dir}/best.sei.model.pth.tar" \
  --features "${data_dir}/target.sei.names" \
  --batch-size "${SEI_BATCH_SIZE:-8}"

echo "sei_evaluation=PASS"
