import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nuscenes.nuscenes import NuScenes
from grading.evaluate import ProjectGrader
from src.prediction.predictor import predict_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Run & evaluate the full perception pipeline"
    )
    parser.add_argument("--dataroot", type=str, required=True, help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument(
        "--detection", type=str, default="submissions/detection_results.json"
    )
    parser.add_argument(
        "--tracking", type=str, default="submissions/tracking_results.json"
    )
    parser.add_argument("--observation_ratio", type=float, default=0.75)
    args = parser.parse_args()

    # --- Pre-flight checks ---
    for label, path in [("Detection", args.detection), ("Tracking", args.tracking)]:
        if not os.path.exists(path):
            print(f"WARNING: {label} results not found at {path}")
            print("         Run run_detection.py / run_tracking.py first,")
            print("         or the corresponding stage will be skipped.\n")

    # --- Load nuScenes ---
    print(f"Loading nuScenes {args.version} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # --- Grader ---
    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)
    print(f"Observation tokens: {len(grader.get_observation_sample_tokens())}")
    print(f"Prediction tokens:  {len(grader.get_prediction_sample_tokens())}")

    # --- Full evaluation ---
    report = grader.run_full_evaluation(
        detection_path=args.detection,
        tracking_path=args.tracking,
        predict_fn=predict_trajectory,
    )

    # --- Save report ---
    import json

    os.makedirs("submissions", exist_ok=True)
    report_path = "submissions/evaluation_report.json"
    serialisable = {}
    for key, val in report.items():
        if val is not None:
            serialisable[key] = {k: v for k, v in val.items() if not callable(v)}
    with open(report_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"Full report saved to {report_path}")


if __name__ == "__main__":
    main()
