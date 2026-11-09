#!/usr/bin/env bash
#SBATCH --job-name=dna-gpu-probe
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=dna-gpu-probe-%j.log
#SBATCH --error=dna-gpu-probe-%j.err
set -euo pipefail
echo "job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION host=$(hostname)"
test "$SLURM_JOB_PARTITION" = test
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
python -u - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.randn(256, 256, device='cuda', requires_grad=True)
x.square().mean().backward()
torch.cuda.synchronize()
print('torch=', torch.__version__, 'gpu=', torch.cuda.get_device_name(0))
print('cuda_forward_backward=PASS')
PY
