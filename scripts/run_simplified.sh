#!/usr/bin/env bash
# Phase H orchestrator: train all splits x 2 models (hierarchical + augmentation).
# SSL encoder pretraining is disabled (pretrain.enabled: false in config).
#
# Usage:
#   bash scripts/run_simplified.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p .mpl_cache
export MPLCONFIGDIR="$(pwd)/.mpl_cache"

LOG_DIR=runs/_simplified_log
mkdir -p "$LOG_DIR"

echo "=== Simplified pipeline run started at $(date) ==="
echo "SSL pretraining: disabled (pretrain.enabled: false)"

for MODEL in spectranet_lite spectrum_cnn; do
  for SPLIT in run1 run2 run3; do
    echo ""
    echo ">>> Train ${MODEL} on ${SPLIT} (phase 2, random encoder init)"
    python3 -u -m asteroid_ml.train --model "${MODEL}" --split "${SPLIT}" --phase 2
  done

  echo ""
  echo ">>> Train ${MODEL} on cv5 (5 folds, phase 2)"
  python3 -u -m asteroid_ml.train --model "${MODEL}" --split cv5 --phase 2
done

echo ""
echo "=== Simplified pipeline run completed at $(date) ==="
