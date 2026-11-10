#!/usr/bin/env bash
#SBATCH --job-name=promoter-4-pilot
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-3%2
#SBATCH --output=promoter-4-pilot-%A_%a.log
#SBATCH --error=promoter-4-pilot-%A_%a.err
set -euo pipefail

wrapper_root="${DDSM_WRAPPER_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "${wrapper_root}/.ddsm_charisma.env"
source "${wrapper_root}/.dna_benchmark.env"
methods=(d3pm discrete_fm ddsm dirichlet_fm)
method="${methods[SLURM_ARRAY_TASK_ID]}"
data_dir="${DNA_BENCHMARK_ROOT}/data/promoter_design"
upstream_dir="${DDSM_WORK_ROOT}/ddsm"
noise_dir="${DDSM_WORK_ROOT}/noise"
noise_path="${noise_dir}/steps32.cat4.speed_balance.time4.0.samples512.pth"
run_dir="${DNA_BENCHMARK_ROOT}/runs/pilot-${SLURM_ARRAY_JOB_ID}/${method}-task${SLURM_ARRAY_TASK_ID}"

module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${wrapper_root}:${PYTHONPATH:-}"
mkdir -p "${noise_dir}" "${run_dir}"

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} method=${method}"
echo "partition=${SLURM_JOB_PARTITION} host=$(hostname) run_dir=${run_dir}"
test "${SLURM_JOB_PARTITION}" = test
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch
assert torch.cuda.is_available()
print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))
PY

if [[ "${method}" == ddsm && ! -f "${noise_path}" ]]; then
  cd "${upstream_dir}"
  python -u presample_noise.py \
    --num_samples 512 \
    --num_cat 4 \
    --num_time_steps 32 \
    --max_time 4 \
    --speed_balance \
    --order 64 \
    --steps_per_tick 16 \
    --out_path "${noise_dir}"
fi

cd "${wrapper_root}"
python -u tools/train_promoter_smoke.py \
  --method "${method}" \
  --data-dir "${data_dir}" \
  --output-dir "${run_dir}" \
  --ddsm-upstream "${upstream_dir}" \
  --ddsm-noise "${noise_path}" \
  --seed 42 \
  --max-steps "${PILOT_MAX_STEPS:-1000}" \
  --batch-size "${PILOT_BATCH_SIZE:-32}" \
  --train-samples "${PILOT_TRAIN_SAMPLES:-8192}" \
  --valid-samples "${PILOT_VALID_SAMPLES:-1024}" \
  --validation-batches "${PILOT_VALIDATION_BATCHES:-4}" \
  --sample-count "${PILOT_SAMPLE_COUNT:-128}" \
  --sample-steps "${PILOT_SAMPLE_STEPS:-64}" \
  --process-steps "${PILOT_PROCESS_STEPS:-64}" \
  --channels 64 \
  --layers 6 \
  --learning-rate 5e-4 \
  --checkpoint-every 100 \
  --log-every 25 \
  --training-wall-seconds 1050 \
  --scope "common-backbone promoter pilot; not paper-scale evaluation" \
  --num-workers 2

echo "completed_method=${method}"
find "${run_dir}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
