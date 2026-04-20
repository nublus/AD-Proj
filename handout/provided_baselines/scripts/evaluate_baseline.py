#!/usr/bin/env python3
"""
Evaluation Baseline — Constant-Velocity Prediction + Full Pipeline Eval.

Runs the project grader's full evaluation:
    1. Detection evaluation  (reads detection JSON)
    2. Tracking evaluation   (reads tracking JSON)
    3. Prediction evaluation (constant-velocity predictor, inline)

Usage:
    python scripts/evaluate_baseline.py \
        --dataroot   /path/to/nuscenes \
        --detection  baseline_submissions/centerpoint_detection_results.json \
        --tracking   baseline_submissions/centerpoint_tracking_results.json \
        --output     baseline_submissions/centerpoint_evaluation_report.json
"""

import argparse
import json
import os
import sys

import numpy as np

# ── Ensure project root is importable ─────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BASELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from nuscenes.nuscenes import NuScenes
from grading.evaluate import ProjectGrader


# ======================================================================
# Constant-Velocity Predictor  (self-contained baseline)
# ======================================================================

def predict_constant_velocity(
    track_history: list[dict],
    num_future_steps: int,
) -> list[tuple[float, float]]:
    """
    Constant-velocity prediction baseline.

    Uses the last observed velocity to linearly extrapolate future
    positions:

        x_{t+k} = x_t + vx · k · Δt
        y_{t+k} = y_t + vy · k · Δt

    where Δt = 0.5 s  (nuScenes keyframe rate: 2 Hz).

    Falls back to finite-difference velocity estimation when the stored
    velocity is near zero.

    Args:
        track_history:    List of dicts with keys:
                            x, y, vx, vy, timestamp
                          sorted by timestamp ascending.
        num_future_steps: Number of future positions to predict.

    Returns:
        List of (x, y) tuples, one per future step.
    """
    if len(track_history) == 0:
        return [(0.0, 0.0)] * num_future_steps

    dt = 0.5  # seconds between keyframes

    last = track_history[-1]
    x0, y0 = last["x"], last["y"]

    # ── estimate velocity ──
    vx = last.get("vx", 0.0)
    vy = last.get("vy", 0.0)

    # If stored velocity is negligible, estimate from last two observations
    if abs(vx) < 1e-6 and abs(vy) < 1e-6 and len(track_history) >= 2:
        prev = track_history[-2]
        t_diff = (last["timestamp"] - prev["timestamp"]) / 1e6  # → seconds
        if t_diff > 0:
            vx = (last["x"] - prev["x"]) / t_diff
            vy = (last["y"] - prev["y"]) / t_diff

    predictions = []
    for k in range(1, num_future_steps + 1):
        px = x0 + vx * k * dt
        py = y0 + vy * k * dt
        predictions.append((px, py))

    return predictions


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run full baseline evaluation (detection + tracking + CV prediction)"
    )
    parser.add_argument("--dataroot", type=str, required=True,
                        help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--detection", type=str, required=True,
                        help="Path to detection results JSON")
    parser.add_argument("--tracking", type=str, required=True,
                        help="Path to tracking results JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Evaluation report JSON path")
    parser.add_argument("--observation_ratio", type=float, default=0.75)
    args = parser.parse_args()

    # ── Pre-flight checks ──
    for label, path in [("Detection", args.detection), ("Tracking", args.tracking)]:
        if not os.path.isfile(path):
            print(f"WARNING: {label} results not found at {path}")
            print("         The corresponding stage will be skipped.\n")

    # ── Default output path ──
    if args.output is None:
        trk_basename = os.path.basename(args.tracking)
        report_name = trk_basename.replace("_tracking_results", "_evaluation_report")
        args.output = os.path.join(os.path.dirname(args.tracking), report_name)

    # ── Load nuScenes ──
    print(f"Loading nuScenes {args.version} from {args.dataroot} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)
    print(f"Observation tokens: {len(grader.get_observation_sample_tokens())}")
    print(f"Prediction tokens:  {len(grader.get_prediction_sample_tokens())}")

    # ── Full evaluation (detection + tracking + CV prediction) ──
    report = grader.run_full_evaluation(
        detection_path=args.detection,
        tracking_path=args.tracking,
        predict_fn=predict_constant_velocity,
    )

    # ── Save report ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    serialisable = {}
    for key, val in report.items():
        if val is not None:
            serialisable[key] = {k: v for k, v in val.items() if not callable(v)}
    with open(args.output, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    print(f"\nFull evaluation report saved → {args.output}")


if __name__ == "__main__":
    main()
