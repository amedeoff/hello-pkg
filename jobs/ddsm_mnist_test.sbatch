#!/usr/bin/env bash
#SBATCH --job-name=ddsm-mnist-test
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=ddsm-mnist-test-%j.log
#SBATCH --error=ddsm-mnist-test-%j.err

set -euo pipefail

wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${wrapper_root}/.ddsm_charisma.env"
if [[ ! -f "${config_path}" ]]; then
  echo "Missing ${config_path}; run bash setup_charisma.sh on the login node first." >&2
  exit 1
fi
source "${config_path}"
work_root="${DDSM_WORK_ROOT}"
upstream_dir="${work_root}/ddsm"
data_dir="${work_root}/data"
noise_dir="${work_root}/noise"
run_dir="${work_root}/runs/test-${SLURM_JOB_ID}"
noise_path="${noise_dir}/steps32.cat2.time1.0.samples256.pth"

module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro

mkdir -p "${noise_dir}" "${run_dir}"
cd "${upstream_dir}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "work_root=${work_root}"
echo "upstream_commit=$(git rev-parse HEAD)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -u - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm allocation")
print("gpu", torch.cuda.get_device_name(0))
PY

if [[ ! -f "${noise_path}" ]]; then
  python -u presample_noise.py \
    --num_samples 256 \
    --num_cat 2 \
    --num_time_steps 32 \
    --max_time 1 \
    --order 64 \
    --steps_per_tick 16 \
    --out_path "${noise_dir}"
fi

python -u experiments/bin_mnist/train_ddsm.py \
  --data-dir "${data_dir}" \
  --noise-path "${noise_path}" \
  --output-dir "${run_dir}" \
  --device cuda \
  --seed 42 \
  --epochs 1 \
  --max-steps 20 \
  --max-train-samples 1024 \
  --max-valid-samples 256 \
  --batch-size 64 \
  --num-workers 4 \
  --weight-estimation-batches 1 \
  --validate-every 1 \
  --sample-count 16 \
  --sample-steps 10 \
  --base-channels 32 \
  --channel-multipliers 1 2 \
  --attention-levels 1 \
  --res-blocks 1

echo "run_dir=${run_dir}"
find "${run_dir}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
