#!/usr/bin/env bash
# Run all Phase 1 + Phase 2 training/eval per AGENT_HANDOFF.md / README
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source .venv/bin/activate

LOG_DIR="runs/_batch_log"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== Batch started $(date) ==="

evaluate_run() {
  local run_dir="$1"
  echo "--- evaluate: $run_dir ---"
  python -m asteroid_ml.evaluate --run "$run_dir" || true
}

# --- Phase 1: run1, run2 (both models) ---
for split in run1 run2; do
  for model in spectrum_cnn spectranet_lite; do
    echo "=== Phase 1 train: $model / $split ==="
    python -m asteroid_ml.train --model "$model" --split "$split" --phase 1
    RUN=$(ls -td runs/*_"${model}"_"${split}"_p1 2>/dev/null | head -1)
    evaluate_run "$RUN"
  done
done

# --- Phase 1: run3 (evaluate existing; re-train optional — skip if metrics exist) ---
for run_dir in runs/20260529_130628_spectrum_cnn_run3 runs/20260529_131354_spectranet_lite_run3; do
  if [[ -d "$run_dir" && -f "$run_dir/best.pt" ]]; then
    echo "=== Phase 1 eval existing: $run_dir ==="
    python -m asteroid_ml.evaluate --run "$run_dir" --gradcam --gradcam-samples 8 || evaluate_run "$run_dir"
  fi
done

# --- Phase 1: cv5 (both models, all 5 folds each) ---
for model in spectrum_cnn spectranet_lite; do
  echo "=== Phase 1 train: $model / cv5 (5 folds) ==="
  python -m asteroid_ml.train --model "$model" --split cv5 --phase 1
  for run_dir in runs/*_"${model}"_cv5_fold*_p1; do
    [[ -d "$run_dir" ]] || continue
    evaluate_run "$run_dir"
  done
done

# --- Phase 2: Optuna (20 trials, run3, spectranet_lite) ---
echo "=== Phase 2 Optuna (20 trials) ==="
python -m asteroid_ml.tune --model spectranet_lite --split run3 --n-trials 20
STUDY=$(ls -td runs/optuna_* 2>/dev/null | head -1)
PARAMS="${STUDY}/best_params.json"
echo "Optuna study: $STUDY"

# --- Phase 2: train both models on run3 with tuned params + diagnostics ---
for model in spectrum_cnn spectranet_lite; do
  echo "=== Phase 2 train: $model / run3 (tuned) ==="
  python -m asteroid_ml.train --phase 2 --model "$model" --split run3 --tuned-params "$PARAMS"
  RUN=$(ls -td runs/*_"${model}"_run3_p2 2>/dev/null | head -1)
  python -m asteroid_ml.evaluate --run "$RUN" --gradcam --gradcam-samples 12
  python -m asteroid_ml.infer --run "$RUN" || true
done

# Export best phase-2 spectranet run
P2_SN=$(ls -td runs/*_spectranet_lite_run3_p2 2>/dev/null | head -1)
if [[ -n "${P2_SN:-}" ]]; then
  python -m asteroid_ml.export_bundle --run "$P2_SN"
fi

echo "=== Batch finished $(date) ==="
echo "Log: $MAIN_LOG"
