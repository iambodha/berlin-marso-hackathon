#!/usr/bin/env bash
# Sync this repo (+ demo data) from your Mac to a Thunder Compute instance.
#
# Usage:
#   bash scripts/thunder_sync.sh           # push to instance 0
#   bash scripts/thunder_sync.sh 1         # push to instance 1
#   bash scripts/thunder_sync.sh pull      # pull from instance 0
#   bash scripts/thunder_sync.sh pull 1    # pull from instance 1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INSTANCE="0"
MODE="push"

if [[ $# -eq 0 ]]; then
  :
elif [[ "$1" == "pull" ]]; then
  MODE="pull"
  INSTANCE="${2:-0}"
elif [[ "$1" == "push" ]]; then
  MODE="push"
  INSTANCE="${2:-0}"
elif [[ "$1" =~ ^[0-9]+$ ]]; then
  MODE="push"
  INSTANCE="$1"
else
  echo "Unknown argument: $1" >&2
  echo "Usage: bash scripts/thunder_sync.sh [pull] [instance_index]" >&2
  echo "  bash scripts/thunder_sync.sh           # push to instance 0" >&2
  echo "  bash scripts/thunder_sync.sh pull      # pull from instance 0" >&2
  exit 1
fi

REMOTE="tnr-${INSTANCE}"
REMOTE_DIR="~/berlin-marso-hackathon"

if ! ssh -G "$REMOTE" >/dev/null 2>&1; then
  echo "SSH host '$REMOTE' not configured." >&2
  echo "Connect once with:  tnr connect ${INSTANCE}" >&2
  exit 1
fi

RSYNC_EXCLUDES=(
  --exclude '.venv'
  --exclude '.git'
  --exclude '.pixi'
  --exclude '.matplotlib'
  --exclude '__pycache__'
  --exclude 'outputs'
  --exclude '*.mp4'
  --exclude 'il/baselines/diffusion_policy/runs'
)

if [[ "$MODE" == "push" ]]; then
  echo "==> Pushing code to ${REMOTE}:${REMOTE_DIR}"
  rsync -avz --progress "${RSYNC_EXCLUDES[@]}" ./ "${REMOTE}:${REMOTE_DIR}/"

  if [[ -d il/demos/easy ]]; then
    echo "==> Pushing demo datasets (il/demos/) — ~450 MB total"
    rsync -avz --progress il/demos/ "${REMOTE}:${REMOTE_DIR}/il/demos/"
  else
    echo "WARNING: il/demos/ not found locally. Run download_kaggle_data.py first." >&2
  fi

  echo
  echo "Done. On the instance:"
  echo "  tnr connect ${INSTANCE}"
  echo "  cd berlin-marso-hackathon && bash scripts/thunder_setup.sh   # first time only"
  echo "  bash scripts/thunder_run.sh"

elif [[ "$MODE" == "pull" ]]; then
  LOCAL_RUNS="${ROOT}/il/baselines/diffusion_policy/runs"
  mkdir -p "$LOCAL_RUNS"
  echo "==> Pulling checkpoints from ${REMOTE}:${REMOTE_DIR}/il/baselines/diffusion_policy/runs/"
  rsync -avz --progress "${REMOTE}:${REMOTE_DIR}/il/baselines/diffusion_policy/runs/" "$LOCAL_RUNS/"
  echo "Checkpoints saved under: $LOCAL_RUNS"
else
  echo "Unknown mode: $MODE (use push or pull)" >&2
  exit 1
fi
