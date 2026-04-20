import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nuscenes.nuscenes import NuScenes
from grading.evaluate import ProjectGrader
from src.detection.detector import LidarDetector


def main():
    parser = argparse.ArgumentParser(description="Run LiDAR 3D Object Detection")
    parser.add_argument("--dataroot", type=str, required=True, help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output", type=str, default="submissions/detection_results.json")
    parser.add_argument("--observation_ratio", type=float, default=0.75)
    parser.add_argument(
        "--detector-preset",
        type=str,
        default="balanced",
        choices=sorted(LidarDetector.PRESETS.keys()),
        help="Detector hyper-parameter preset",
    )
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Get observation tokens
    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)
    obs_tokens = grader.get_observation_sample_tokens()
    print(f"Observation frames: {len(obs_tokens)}")
    print(f"Detector preset: {args.detector_preset}")

    # Run detector
    detector = LidarDetector(nusc, preset=args.detector_preset)
    detector.detect_all(obs_tokens, args.output)

    # Quick evaluation
    print("--- Detection Evaluation ---")
    grader.evaluate_detection(args.output)


if __name__ == "__main__":
    main()
