#!/usr/bin/env bash
# Extract the verified Zenodo archives and unpack nested promoter files.
set -euo pipefail

wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(ws_find "${DDSM_WORKSPACE_ID:-ft-models}")"
benchmark_root="${DNA_BENCHMARK_ROOT:-${workspace_root}/dna-benchmark}"
downloads="${benchmark_root}/downloads"
data_dir="${benchmark_root}/data/promoter_design"
stage="${benchmark_root}/.prepare-promoter-staging"

for archive in data.tar.gz best_models.tar.gz; do
  [[ -f "${downloads}/${archive}" ]] || {
    echo "Missing ${downloads}/${archive}; run tools/download_promoter_data.sh first." >&2
    exit 1
  }
done

if [[ ! -d "${data_dir}" ]]; then
  if [[ -e "${stage}" ]]; then
    echo "Incomplete staging directory exists: ${stage}" >&2
    echo "Inspect it before moving or removing it; no files were overwritten." >&2
    exit 1
  fi
  mkdir -p "${stage}"
  tar --extract --gzip --file "${downloads}/data.tar.gz" \
    --directory "${stage}" --exclude='*/._*' data/promoter_design
  tar --extract --gzip --file "${downloads}/best_models.tar.gz" \
    --directory "${stage}" best_model_weights/promoter_designer/promoter.cond.4l.pth
else
  echo "Using existing ${data_dir}; extraction skipped."
fi

if [[ -d "${stage}/data/promoter_design" ]]; then
  stage_data="${stage}/data/promoter_design"
  for compressed in \
    Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
    FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv.gz \
    agg.plus.bw.bedgraph.bw.gz \
    agg.minus.bw.bedgraph.bw.gz \
    best.sei.model.pth.tar.gz; do
    gzip --test "${stage_data}/${compressed}"
    gzip --decompress --keep "${stage_data}/${compressed}"
  done
  mkdir -p "${benchmark_root}"
  mv "${stage}/data" "${benchmark_root}/data"
  mv "${stage}/best_model_weights" "${benchmark_root}/best_model_weights"
  rmdir "${stage}"
fi

required=(
  Homo_sapiens.GRCh38.dna.primary_assembly.fa
  Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai
  FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv
  target.sei.names
  agg.plus.bw.bedgraph.bw
  agg.minus.bw.bedgraph.bw
  fantom.blacklist8.plus.bed.gz
  fantom.blacklist8.plus.bed.gz.tbi
  fantom.blacklist8.minus.bed.gz
  fantom.blacklist8.minus.bed.gz.tbi
  best.sei.model.pth.tar
  README.md
)
for path in "${required[@]}"; do
  [[ -s "${data_dir}/${path}" ]] || {
    echo "Required file is missing or empty: ${data_dir}/${path}" >&2
    exit 1
  }
done
[[ -s "${benchmark_root}/best_model_weights/promoter_designer/promoter.cond.4l.pth" ]]

printf 'export DNA_BENCHMARK_ROOT=%q\n' "${benchmark_root}" > "${wrapper_root}/.dna_benchmark.env"
(
  cd "${benchmark_root}"
  sha256sum \
    data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai \
    data/promoter_design/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv \
    data/promoter_design/target.sei.names \
    data/promoter_design/README.md \
    best_model_weights/promoter_designer/promoter.cond.4l.pth \
    > promoter_manifest.sha256
)

echo "promoter_data_ready=${data_dir}"
du -sh "${data_dir}" "${benchmark_root}/best_model_weights"
echo "next=sbatch --partition=test ${wrapper_root}/jobs/promoter_memmap_test.sbatch"
