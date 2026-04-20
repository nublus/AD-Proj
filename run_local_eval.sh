#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOUT_DIR="$PROJECT_ROOT/handout"

MODE="${1:-detection}"
DATAROOT="${DATAROOT:-$PROJECT_ROOT/material/v1.0-mini}"
VERSION="${VERSION:-v1.0-mini}"
OBSERVATION_RATIO="${OBSERVATION_RATIO:-0.75}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "/opt/miniconda3/envs/cs5493-a1-mps/bin/python" ]]; then
  PYTHON_CMD="/opt/miniconda3/envs/cs5493-a1-mps/bin/python"
else
  PYTHON_CMD="python3"
fi

CACHE_DIR="$PROJECT_ROOT/.cache"
export XDG_CACHE_HOME="$CACHE_DIR"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

if [[ ! -d "$HANDOUT_DIR" ]]; then
  echo "ERROR: handout directory not found at: $HANDOUT_DIR" >&2
  exit 1
fi

if [[ ! -d "$DATAROOT" ]]; then
  echo "ERROR: nuScenes dataroot not found at: $DATAROOT" >&2
  echo "Tip: set DATAROOT=/your/path/to/v1.0-mini and rerun." >&2
  exit 1
fi

run_detection() {
  echo "==> Running Phase 1 detection and scoring"
  "$PYTHON_CMD" run_detection.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --output submissions/detection_results.json \
    --observation_ratio "$OBSERVATION_RATIO"
}

run_tracking() {
  echo "==> Running Phase 2 tracking and scoring"
  "$PYTHON_CMD" run_tracking.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --detection submissions/detection_results.json \
    --output submissions/tracking_results.json \
    --observation_ratio "$OBSERVATION_RATIO"
}

run_full_pipeline() {
  echo "==> Running full pipeline evaluation"
  "$PYTHON_CMD" run_pipeline.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --detection submissions/detection_results.json \
    --tracking submissions/tracking_results.json \
    --observation_ratio "$OBSERVATION_RATIO"
}

cd "$HANDOUT_DIR"

case "$MODE" in
  detection)
    run_detection
    ;;
  tracking)
    run_tracking
    ;;
  full)
    if [[ ! -f "submissions/detection_results.json" ]]; then
      run_detection
    fi
    if [[ ! -f "submissions/tracking_results.json" ]]; then
      run_tracking
    fi
    run_full_pipeline
    ;;
  all)
    run_detection
    run_tracking
    run_full_pipeline
    ;;
  *)
    cat <<'EOF'
Usage:
  ./run_local_eval.sh [detection|tracking|full|all]

Modes:
  detection  Run Phase 1 detection and immediate grading.
  tracking   Run Phase 2 tracking grading using existing detection results.
  full       Reuse existing outputs when available, then run full evaluation.
  all        Force-run detection, tracking, then full evaluation in sequence.

Optional environment variables:
  DATAROOT=...           Path to nuScenes root. Default: ./material/v1.0-mini
  VERSION=...            nuScenes version. Default: v1.0-mini
  OBSERVATION_RATIO=...  Default: 0.75
  PYTHON_BIN=...         Python interpreter to use
EOF
    exit 1
    ;;
esac

echo
echo "Done."
