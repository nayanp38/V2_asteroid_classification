#!/usr/bin/env bash
# Pretraining ablation is disabled while pretrain.enabled is false in config.
# Re-enable pretrain.enabled and use train.py --pretrained for manual comparisons.
set -euo pipefail

echo "run_pretrain_ablation.sh: SSL pretraining is disabled repo-wide."
echo "  Set pretrain.enabled: true in configs/default.yaml to re-enable."
echo "  See README § Pretraining ablation for archived results."
exit 1
