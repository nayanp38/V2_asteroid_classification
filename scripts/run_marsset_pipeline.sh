#!/usr/bin/env bash
# Train the simplified pipeline on the Marsset-expanded manifest.
# SSL encoder pretraining is disabled (pretrain.enabled: false in config).
#
# Usage:
#   bash scripts/run_marsset_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p .mpl_cache runs/_marsset_log
export MPLCONFIGDIR="$(pwd)/.mpl_cache"

LOG=runs/_marsset_log/session.log

exec > >(tee -a "$LOG") 2>&1

echo "=== Marsset pipeline started at $(date) ==="
echo "SSL pretraining: disabled (pretrain.enabled: false)"

python scripts/build_manifest.py

for MODEL in spectranet_lite spectrum_cnn; do
  for SPLIT in run1 run2 run3; do
    echo ""
    echo ">>> Train ${MODEL} on ${SPLIT} (phase 2)"
    python3 -u -m asteroid_ml.train --model "${MODEL}" --split "${SPLIT}" --phase 2
  done

  echo ""
  echo ">>> Train ${MODEL} on cv5 (5 folds, phase 2)"
  python3 -u -m asteroid_ml.train --model "${MODEL}" --split cv5 --phase 2
done

echo ""
echo "=== Marsset pipeline completed at $(date) ==="
