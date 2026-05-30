#!/usr/bin/env bash
# Phase H orchestrator: pretrain both encoders (or reuse), then train all
# 4 splits x 2 models under the simplified pipeline (hierarchical + SSL
# pretrain, no ensemble).
#
# Usage:
#   bash scripts/run_simplified.sh [pretrain_epochs]
# Default pretrain_epochs = 100.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p .mpl_cache
export MPLCONFIGDIR="$(pwd)/.mpl_cache"

PRETRAIN_EPOCHS=${1:-100}
LOG_DIR=runs/_simplified_log
mkdir -p "$LOG_DIR"

echo "=== Simplified pipeline run started at $(date) ==="
echo "Pretrain epochs: $PRETRAIN_EPOCHS"

for MODEL in spectranet_lite spectrum_cnn; do
  # Reuse the most recent pretrain if available, otherwise pretrain fresh.
  ENC_DIR=$(ls -dt runs/pretrain_*_${MODEL} 2>/dev/null | head -1 || true)
  if [[ -n "${ENC_DIR}" && -f "${ENC_DIR}/encoder.pt" ]]; then
    echo ""
    echo ">>> Reusing pretrained encoder for ${MODEL}: ${ENC_DIR}/encoder.pt"
  else
    echo ""
    echo ">>> Pretrain ${MODEL} (${PRETRAIN_EPOCHS} epochs)"
    python3 -u -m asteroid_ml.pretrain --model "${MODEL}" --epochs "${PRETRAIN_EPOCHS}"
    ENC_DIR=$(ls -dt runs/pretrain_*_${MODEL} | head -1)
  fi
  ENC="${ENC_DIR}/encoder.pt"
  echo "  encoder: ${ENC}"

  for SPLIT in run1 run2 run3; do
    echo ""
    echo ">>> Train ${MODEL} on ${SPLIT} (phase 2) with pretrained encoder"
    python3 -u -m asteroid_ml.train --model "${MODEL}" --split "${SPLIT}" --phase 2 --pretrained "${ENC}"
  done

  echo ""
  echo ">>> Train ${MODEL} on cv5 (5 folds, phase 2) with pretrained encoder"
  python3 -u -m asteroid_ml.train --model "${MODEL}" --split cv5 --phase 2 --pretrained "${ENC}"
done

echo ""
echo "=== Simplified pipeline run completed at $(date) ==="
