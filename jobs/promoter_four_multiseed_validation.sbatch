#!/usr/bin/env bash
#SBATCH --job-name=promoter-4-val
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-11%1
#SBATCH --output=promoter-4-val-%A_%a.log
#SBATCH --error=promoter-4-val-%A_%a.err
set -euo pipefail

wrapper_root="${DDSM_WRAPPER_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "${wrapper_root}/.ddsm_charisma.env"
source "${wrapper_root}/.dna_benchmark.env"
methods=(d3pm discrete_fm ddsm dirichlet_fm)
seeds=(42 43 44)
method_index=$((SLURM_ARRAY_TASK_ID % 4))
seed_index=$((SLURM_ARRAY_TASK_ID / 4))
method="${methods[method_index]}"
seed="${seeds[seed_index]}"
data_dir="${DNA_BENCHMARK_ROOT}/data/promoter_design"
upstream_dir="${DDSM_WORK_ROOT}/ddsm"
noise_dir="${DDSM_WORK_ROOT}/noise"
noise_path="${noise_dir}/steps32.cat4.speed_balance.time4.0.samples512.pth"
run_dir="${DNA_BENCHMARK_ROOT}/runs/validation-${SLURM_ARRAY_JOB_ID}/${method}-seed${seed}-task${SLURM_ARRAY_TASK_ID}"

module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${wrapper_root}:${PYTHONPATH:-}"
mkdir -p "${noise_dir}" "${run_dir}"

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} method=${method} seed=${seed}"
echo "partition=${SLURM_JOB_PARTITION} host=$(hostname) run_dir=${run_dir}"
test "${SLURM_JOB_PARTITION}" = test
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${method}" == ddsm && ! -f "${noise_path}" ]]; then
  cd "${upstream_dir}"
  python -u presample_noise.py \
    --num_samples 512 --num_cat 4 --num_time_steps 32 --max_time 4 \
    --speed_balance --order 64 --steps_per_tick 16 --out_path "${noise_dir}"
fi

cd "${wrapper_root}"
python -u tools/train_promoter_smoke.py \
  --method "${method}" \
  --data-dir "${data_dir}" \
  --output-dir "${run_dir}" \
  --ddsm-upstream "${upstream_dir}" \
  --ddsm-noise "${noise_path}" \
  --seed "${seed}" \
  --max-steps "${VALIDATION_MAX_STEPS:-20000}" \
  --batch-size "${VALIDATION_BATCH_SIZE:-32}" \
  --train-samples "${VALIDATION_TRAIN_SAMPLES:-32768}" \
  --valid-samples "${VALIDATION_VALID_SAMPLES:-2048}" \
  --validation-batches "${VALIDATION_BATCHES:-8}" \
  --sample-count "${VALIDATION_SAMPLE_COUNT:-256}" \
  --sample-steps "${VALIDATION_SAMPLE_STEPS:-100}" \
  --process-steps "${VALIDATION_PROCESS_STEPS:-100}" \
  --channels 64 \
  --layers 6 \
  --learning-rate 5e-4 \
  --checkpoint-every 1000 \
  --log-every 250 \
  --training-wall-seconds 1350 \
  --scope "three-seed common-backbone promoter validation; test chromosomes untouched" \
  --num-workers 2

echo "completed_method=${method} seed=${seed}"
find "${run_dir}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
