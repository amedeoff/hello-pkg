#!/usr/bin/env bash
#SBATCH --job-name=promoter-memmap
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=promoter-memmap-%j.log
#SBATCH --error=promoter-memmap-%j.err
set -euo pipefail

wrapper_root="${DDSM_WRAPPER_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "${wrapper_root}/.dna_benchmark.env"
data_dir="${DNA_BENCHMARK_ROOT}/data/promoter_design"
fasta="${data_dir}/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

module purge
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddsm-repro
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION} host=$(hostname)"
test "${SLURM_JOB_PARTITION}" = test
python -u "${wrapper_root}/tools/build_genome_memmap.py" \
  --fasta "${fasta}" \
  --fai "${fasta}.fai" \
  --output "${fasta}.mmap"
ls -lh "${fasta}.mmap" "${fasta}.mmap.metadata.json"
