#!/usr/bin/env bash
#
# run_all_baselines.sh — Generate all baseline results end-to-end.
#
# This script runs:
#   1. CenterPoint detection  ->  tracking  ->  evaluation
#
# Usage:
#   cd project-done/provided_baselines
#   bash scripts/run_all_baselines.sh /path/to/nuscenes [DEVICE]
#
# Arguments:
#   $1  DATAROOT   (required)  Path to nuScenes data root
#   $2  DEVICE     (optional)  Inference device, default: cuda:0 if available
#
set -euo pipefail

DATAROOT="${1:?Usage: $0 <DATAROOT> [DEVICE]}"
DEVICE="${2:-cuda:0}"
VERSION="v1.0-mini"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BASELINE_DIR")"
OUTDIR="$BASELINE_DIR/baseline_submissions"

mkdir -p "$OUTDIR"

echo "========================================"
echo "  Baseline Pipeline"
echo "  DATAROOT : $DATAROOT"
echo "  DEVICE   : $DEVICE"
echo "  OUTPUT   : $OUTDIR"
echo "========================================"

# ── Helper ────────────────────────────────────────────────────────────
run_model() {
    local MODEL="$1"
    echo ""
    echo "================================================================"
    echo "  Model: $MODEL"
    echo "================================================================"

    DET_OUT="$OUTDIR/${MODEL}_detection_results.json"
    TRK_OUT="$OUTDIR/${MODEL}_tracking_results.json"
    EVAL_OUT="$OUTDIR/${MODEL}_evaluation_report.json"

    # ── Step 1: Detection ──
    echo ""
    echo ">>> [1/3] Detection ($MODEL) …"
    python "$SCRIPT_DIR/detect_baseline.py" \
        --dataroot "$DATAROOT" \
        --version  "$VERSION" \
        --model    "$MODEL" \
        --device   "$DEVICE" \
        --output   "$DET_OUT"

    # ── Step 2: Tracking  (Simple Kalman Filter) ──
    echo ""
    echo ">>> [2/3] Tracking (Simple KF on $MODEL detections) …"
    python "$SCRIPT_DIR/track_baseline.py" \
        --dataroot  "$DATAROOT" \
        --version   "$VERSION" \
        --detection "$DET_OUT" \
        --output    "$TRK_OUT"

    # ── Step 3: Evaluation (including CV Prediction) ──
    echo ""
    echo ">>> [3/3] Evaluation (CV Prediction + Full Pipeline on $MODEL) …"
    python "$SCRIPT_DIR/evaluate_baseline.py" \
        --dataroot  "$DATAROOT" \
        --version   "$VERSION" \
        --detection "$DET_OUT" \
        --tracking  "$TRK_OUT" \
        --output    "$EVAL_OUT"
}

# ── Run baseline ──────────────────────────────────────────────────────
run_model "centerpoint"

echo ""
echo "========================================"
echo "  All baselines generated successfully!"
echo "  Results in: $OUTDIR/"
echo "========================================"
ls -lh "$OUTDIR/"
