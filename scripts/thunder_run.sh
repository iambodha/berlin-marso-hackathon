#!/usr/bin/env bash
# Run the full train + eval pipeline on Thunder Compute.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Run scripts/thunder_setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source "$(dirname "$0")/thunder_env.sh"

# Override any of these on the command line, e.g.:
#   TOTAL_ITERS=50000 DEMO_DIR=medium bash scripts/thunder_run.sh
export DEMO_DIR="${DEMO_DIR:-easy}"
export TOTAL_ITERS="${TOTAL_ITERS:-30000}"
export EVAL_FREQ="${EVAL_FREQ:-5000}"
export EXP_NAME="${EXP_NAME:-warehouse_rgb_dp}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"

EXTRA=()
if [[ "$TRAIN_ONLY" == "1" ]]; then
  EXTRA+=(--train-only --skip-eval)
fi

exec python run_pipeline.py \
  --demo-dir "$DEMO_DIR" \
  --total-iters "$TOTAL_ITERS" \
  --eval-freq "$EVAL_FREQ" \
  --exp-name "$EXP_NAME" \
  "${EXTRA[@]}" \
  "$@"
