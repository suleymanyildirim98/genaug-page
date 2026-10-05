#!/usr/bin/env bash
# Publish the reviewed GenAug project-page changes to GitHub Pages.
# Usage: bash publish.sh ["Optional commit message"]

set -euo pipefail

module load conda3/latest
source activate base
conda activate genaug-pt2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -d .git ]]; then
  echo "This script must be run from the initialized genaug-page repository." >&2
  exit 1
fi

git add -A

if git diff --cached --quiet; then
  echo "No uncommitted page changes; pushing any local commits."
  git push origin main
  conda deactivate
  exit 0
fi

if [[ $# -gt 0 ]]; then
  COMMIT_MESSAGE="$*"
else
  COMMIT_MESSAGE="Publish evaluation results $(date +%Y-%m-%d_%H-%M-%S)"
fi

git commit -m "${COMMIT_MESSAGE}"
git push origin main

echo "Published GenAug page updates."
conda deactivate
