#!/usr/bin/env bash
# Example Phase 2 workflow (from v2/ with PYTHONPATH=src)
set -euo pipefail
export PYTHONPATH=src

echo "=== 1. Optuna search (adjust --n-trials) ==="
python -m asteroid_ml.tune --model spectranet_lite --split run3 --n-trials 5

STUDY=$(ls -td runs/optuna_* 2>/dev/null | head -1)
PARAMS="${STUDY}/best_params.json"

echo "=== 2. Train phase 2 with tuned params ==="
python -m asteroid_ml.train --phase 2 --model spectranet_lite --split run3 --tuned-params "$PARAMS"

RUN=$(ls -td runs/*_p2 2>/dev/null | head -1)
echo "=== 3. Evaluate + optional extra Grad-CAM ==="
python -m asteroid_ml.evaluate --run "$RUN" --gradcam

echo "=== 4. Export release bundle ==="
python -m asteroid_ml.export_bundle --run "$RUN"

echo "Done. Run dir: $RUN"
