#!/usr/bin/env bash
# Download only, on the login node. Extraction and genome preprocessing are separate.
set -euo pipefail
workspace_root="$(ws_find "${DDSM_WORKSPACE_ID:-ft-models}")"
test -d "$workspace_root"
dest="$workspace_root/dna-benchmark/downloads"
mkdir -p "$dest"
cd "$dest"
df -h .
fetch() {
  local filename="$1" expected="$2"
  if [[ ! -f "$filename" ]]; then
    curl --fail --location --retry 5 --continue-at - \
      "https://zenodo.org/records/7943307/files/$filename?download=1" \
      --output "$filename.part"
    printf '%s  %s\n' "$expected" "$filename.part" | md5sum --check -
    mv "$filename.part" "$filename"
  fi
  printf '%s  %s\n' "$expected" "$filename" | md5sum --check -
}
fetch data.tar.gz aa53f13ddcf1474962548bc02cbb9b93
fetch best_models.tar.gz 8a4940b23e275e52fe237babddc8e18f
echo "verified_downloads=$dest"
echo 'Archives verified. Read DNA_REVIEW.md before extraction and preprocessing.'
