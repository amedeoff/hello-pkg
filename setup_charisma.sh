#!/usr/bin/env bash
# Run on the cHARISMa login server. Compute nodes do not have Internet access.
set -euo pipefail

wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_id="${DDSM_WORKSPACE_ID:-ft-models}"
workspace_root="$(ws_find "${workspace_id}")"
if [[ -z "${workspace_root}" || ! -d "${workspace_root}" ]]; then
  echo "Unable to resolve cHARISMa workspace '${workspace_id}'." >&2
  echo "Available workspaces:" >&2
  ws_list >&2 || true
  exit 1
fi
work_root="${DDSM_WORK_ROOT:-${workspace_root}/ddsm-reproduction}"
upstream_dir="${work_root}/ddsm"
data_dir="${work_root}/data"
upstream_commit="ff20be49138589ae0f477b4039feafa7343de2a2"
env_name="ddsm-repro"

module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
  conda create -y -n "${env_name}" python=3.11
fi
conda activate "${env_name}"

python -m pip install --upgrade pip
# Pin a mutually compatible PyTorch pair. The Linux wheels include CUDA
# runtime libraries; the compute job verifies driver/runtime compatibility.
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install \
  numpy==1.26.4 \
  scipy==1.13.1 \
  tqdm==4.66.5 \
  joblib==1.4.2 \
  matplotlib==3.9.2

mkdir -p "${work_root}" "${data_dir}" "${work_root}/noise" "${work_root}/runs"

printf 'export DDSM_WORKSPACE_ID=%q\nexport DDSM_WORK_ROOT=%q\n' \
  "${workspace_id}" "${work_root}" > "${wrapper_root}/.ddsm_charisma.env"

if [[ ! -d "${upstream_dir}/.git" ]]; then
  git clone https://github.com/jzhoulab/ddsm.git "${upstream_dir}"
fi
git -C "${upstream_dir}" fetch --all --tags
git -C "${upstream_dir}" checkout --detach "${upstream_commit}"

mkdir -p "${upstream_dir}/experiments/bin_mnist"
cp "${wrapper_root}/tools/train_ddsm.py" \
  "${upstream_dir}/experiments/bin_mnist/train_ddsm.py"
cp "${wrapper_root}/tools/prepare_data.py" \
  "${upstream_dir}/experiments/bin_mnist/prepare_data.py"

python "${upstream_dir}/experiments/bin_mnist/prepare_data.py" \
  --data-dir "${data_dir}"

python - <<'PY'
import torch
import torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("built_cuda", torch.version.cuda)
PY

echo "setup_complete=true"
echo "environment=${env_name}"
echo "workspace_id=${workspace_id}"
echo "workspace_root=${workspace_root}"
echo "work_root=${work_root}"
echo "upstream=${upstream_dir}"
echo "upstream_commit=$(git -C "${upstream_dir}" rev-parse HEAD)"
echo "data=${data_dir}"
echo "next=sbatch ${wrapper_root}/jobs/ddsm_mnist_test.sbatch"
