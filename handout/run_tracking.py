import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nuscenes.nuscenes import NuScenes
from grading.evaluate import ProjectGrader
from src.tracking.tracker import MultiObjectTracker
from src.utils import save_tracking_results


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Object Tracking")
    parser.add_argument("--dataroot", type=str, required=True, help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument(
        "--detection", type=str, default="submissions/detection_results.json",
        help="Path to detection results",
    )
    parser.add_argument(
        "--output", type=str, default="submissions/tracking_results.json",
    )
    parser.add_argument("--observation_ratio", type=float, default=0.75)
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Load detection results
    if not os.path.exists(args.detection):
        print(f"ERROR: detection file not found: {args.detection}")
        print("Run run_detection.py first.")
        sys.exit(1)

    with open(args.detection) as f:
        det_data = json.load(f)
    det_results = det_data.get("results", {})

    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)

    tracker = MultiObjectTracker()
    all_tracking_results: dict[str, list] = {}

    # Process each scene independently (reset tracker between scenes)
    for scene_name, split_data in grader.scene_splits.items():
        print(f"\nScene: {scene_name}")
        tracker.reset()

        obs_tokens = split_data["observation_tokens"]
        for token in tqdm(obs_tokens, desc=f"  Tracking {scene_name}"):
            detections = det_results.get(token, [])
            tracked = tracker.update(detections, token)
            all_tracking_results[token] = tracked

    save_tracking_results(all_tracking_results, args.output)

    # Quick evaluation
    print("--- Tracking Evaluation ---")
    grader.evaluate_tracking(args.output)


if __name__ == "__main__":
    main()
