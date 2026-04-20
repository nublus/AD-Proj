#!/usr/bin/env python3
"""
Tracking Baseline — Simple Kalman Filter + Hungarian Algorithm.

Reads detection results JSON, runs a straightforward multi-object tracker
(constant-velocity Kalman filter with Hungarian matching), and saves
tracking results in the project's tracking JSON format.

This implementation is **self-contained** — it does not depend on anything
in src/tracking/ so it can serve as an independent baseline reference.

Usage:
    python scripts/track_baseline.py \
        --dataroot   /path/to/nuscenes \
        --detection  baseline_submissions/centerpoint_detection_results.json \
        --output     baseline_submissions/centerpoint_tracking_results.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ── Ensure project root is importable ─────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BASELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from nuscenes.nuscenes import NuScenes
from grading.evaluate import ProjectGrader


# ======================================================================
# Standalone Kalman Filter (constant-velocity, 3-D)
# ======================================================================

class SimpleKalmanFilter3D:
    """
    Minimal constant-velocity Kalman filter for 3-D tracking.

    State:  [px, py, pz, vx, vy, vz]     (6-D)
    Measurement:  [px, py, pz]            (3-D)
    """

    def __init__(self, position, dt=0.5, q=1.0, r=0.5):
        self.dt = dt
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = position

        # State covariance — high initial velocity uncertainty
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

        # Transition: constant velocity
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Observation matrix: observe position only
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise (discrete white-noise acceleration model)
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q = np.zeros((6, 6))
        for i in range(3):
            self.Q[i, i]         = q * dt4 / 4.0
            self.Q[i, i + 3]    = q * dt3 / 2.0
            self.Q[i + 3, i]    = q * dt3 / 2.0
            self.Q[i + 3, i + 3] = q * dt2

        # Measurement noise
        self.R = np.eye(3) * (r ** 2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:6]


# ======================================================================
# Track object
# ======================================================================

class Track:
    _next_id = 0

    def __init__(self, detection, dt=0.5):
        self.track_id = f"baseline_track_{Track._next_id:04d}"
        Track._next_id += 1

        self.det_name = detection.get("detection_name", "car")
        self.size = list(detection["size"])
        self.rotation = list(detection["rotation"])
        self.score = float(detection.get("detection_score", 0.5))

        pos = np.array(detection["translation"][:3], dtype=float)
        self.kf = SimpleKalmanFilter3D(pos, dt=dt)

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        pos = np.array(detection["translation"][:3], dtype=float)
        self.kf.update(pos)
        self.hits += 1
        self.time_since_update = 0
        self.size = list(detection["size"])
        self.rotation = list(detection["rotation"])
        self.det_name = detection.get("detection_name", self.det_name)
        self.score = float(detection.get("detection_score", self.score))


# ======================================================================
# Multi-Object Tracker
# ======================================================================

class SimpleMultiObjectTracker:
    """
    Baseline tracker: Kalman Filter + Hungarian (centre-distance).

    Parameters match the project's reference tracker with sensible
    defaults that work well on nuScenes-mini.
    """

    def __init__(self, max_age=3, min_hits=2, assoc_threshold=3.0, dt=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.assoc_threshold = assoc_threshold
        self.dt = dt
        self.tracks: list[Track] = []

    def reset(self):
        self.tracks = []
        Track._next_id = 0

    def update(self, detections, sample_token):
        # 1. Predict all existing tracks forward
        for track in self.tracks:
            track.predict()

        N = len(self.tracks)
        M = len(detections)

        if N == 0 and M == 0:
            return []

        matched_tracks = set()
        matched_dets = set()

        if N > 0 and M > 0:
            # 2. Cost matrix (centre distance, xy only)
            cost = np.full((N, M), 1e6)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    tp = track.kf.position[:2]
                    dp = np.array(det["translation"][:2], dtype=float)
                    cost[i, j] = np.linalg.norm(tp - dp)

            # 3. Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.assoc_threshold:
                    self.tracks[r].update(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # 4. Create new tracks for unmatched detections
        for j in range(M):
            if j not in matched_dets:
                self.tracks.append(Track(detections[j], dt=self.dt))

        # 5. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6. Return confirmed tracks (only those with enough hits & recently updated)
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                pos = track.kf.position
                vel = track.kf.velocity
                results.append({
                    "sample_token": sample_token,
                    "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "size": [float(s) for s in track.size],
                    "rotation": [float(r) for r in track.rotation],
                    "velocity": [float(vel[0]), float(vel[1])],
                    "tracking_id": track.track_id,
                    "tracking_name": track.det_name,
                    "tracking_score": float(track.score),
                })
        return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run simple Kalman-filter tracking baseline"
    )
    parser.add_argument("--dataroot", type=str, required=True,
                        help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--detection", type=str, required=True,
                        help="Path to detection results JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output tracking JSON path")
    parser.add_argument("--max_age", type=int, default=3,
                        help="Delete track after N frames without match")
    parser.add_argument("--min_hits", type=int, default=2,
                        help="Track must be matched N times before output")
    parser.add_argument("--assoc_threshold", type=float, default=3.0,
                        help="Max centre distance (m) for matching")
    parser.add_argument("--observation_ratio", type=float, default=0.75)
    args = parser.parse_args()

    if not os.path.isfile(args.detection):
        print(f"ERROR: detection file not found: {args.detection}")
        sys.exit(1)

    # ── Default output path ──
    if args.output is None:
        det_basename = os.path.basename(args.detection)
        trk_name = det_basename.replace("_detection_", "_tracking_")
        args.output = os.path.join(os.path.dirname(args.detection), trk_name)

    # ── Load detection results ──
    with open(args.detection) as f:
        det_data = json.load(f)
    det_results = det_data.get("results", {})
    print(f"Loaded detections: {sum(len(v) for v in det_results.values())} boxes "
          f"across {len(det_results)} frames")

    # ── Load nuScenes ──
    print(f"Loading nuScenes {args.version} from {args.dataroot} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)

    # ── Run tracking per scene ──
    tracker = SimpleMultiObjectTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        assoc_threshold=args.assoc_threshold,
    )
    all_tracking: dict[str, list] = {}

    for scene_name, split_data in grader.scene_splits.items():
        tracker.reset()
        obs_tokens = split_data["observation_tokens"]
        print(f"\nScene: {scene_name}  ({len(obs_tokens)} observation frames)")

        for token in tqdm(obs_tokens, desc=f"  Tracking {scene_name}"):
            dets = det_results.get(token, [])
            tracked = tracker.update(dets, token)
            all_tracking[token] = tracked

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": all_tracking,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    total_tracked = sum(len(v) for v in all_tracking.values())
    print(f"\nSaved tracking baseline → {args.output}")
    print(f"  Total tracked entries: {total_tracked}")

    # ── Quick evaluation ──
    print("\n--- Tracking Evaluation ---")
    grader.evaluate_tracking(args.output)


if __name__ == "__main__":
    main()
