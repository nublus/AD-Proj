#!/usr/bin/env python3
"""
Detection Baseline — Run CenterPoint or PointPillar inference on nuScenes.

Loads a pre-trained mmdetection3d model, runs inference on all observation
frames, and saves results in the project's detection JSON format.

Usage:
    # CenterPoint
    python scripts/detect_baseline.py \
        --dataroot /path/to/nuscenes \
        --model    centerpoint \
        --config   scripts/configs/centerpoint_pillar02_nus.py \
        --checkpoint checkpoints/centerpoint_mini.pth

Outputs:
    baseline_submissions/centerpoint_detection_results.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

# ── Ensure project root is importable ─────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BASELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from nuscenes.nuscenes import NuScenes
from grading.evaluate import ProjectGrader

# ── Class names (must match training config order) ────────────────────
NUSCENES_DET_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Default config / checkpoint lookup per model name
_MODEL_DEFAULTS = {
    "centerpoint": {
        "config": os.path.join(SCRIPT_DIR, "configs", "centerpoint_pillar02_nus.py"),
        "checkpoint": os.path.join(BASELINE_DIR, "checkpoints", "centerpoint_mini.pth"),
    },
}


# ======================================================================
# Coordinate transform:  mmdet3d LiDAR frame  →  nuScenes global frame
# ======================================================================

def _get_lidar_to_global(nusc, sample_token):
    """Return (R, t, Q) for LiDAR → global transform and LiDAR file path."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])

    Q_le = Quaternion(cs["rotation"])       # LiDAR → ego
    t_le = np.array(cs["translation"])
    Q_eg = Quaternion(pose["rotation"])     # ego → global
    t_eg = np.array(pose["translation"])

    # Combined rotation & translation
    R_le = Q_le.rotation_matrix
    R_eg = Q_eg.rotation_matrix
    R = R_eg @ R_le
    t = R_eg @ t_le + t_eg
    Q = Q_eg * Q_le

    lidar_path = os.path.join(nusc.dataroot, sd["filename"])
    return R, t, Q, lidar_path


def convert_boxes_to_nuscenes(
    bboxes_3d,
    scores,
    labels,
    R,
    t,
    Q_lidar_to_global,
    sample_token,
    class_names,
    score_threshold=0.1,
):
    """
    Convert mmdet3d predicted boxes to the project's detection entry format.

    Args:
        bboxes_3d:  LiDARInstance3DBoxes  (N, 7+)
                    columns: [x, y, z, dx, dy, dz, yaw, ...]
        scores:     (N,) tensor
        labels:     (N,) tensor  (class index into class_names)
        R:          (3, 3) rotation  LiDAR → global
        t:          (3,)   translation  LiDAR → global
        Q_lidar_to_global:  Quaternion  LiDAR → global
        sample_token: str
        class_names:  list[str]
        score_threshold: float — discard boxes below this score

    Returns:
        list[dict]  detection entries
    """
    boxes = bboxes_3d.tensor.cpu().numpy()   # (N, 7) or (N, 9)
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    entries = []
    for i in range(len(boxes)):
        sc = float(scores_np[i])
        if sc < score_threshold:
            continue

        x, y, z = boxes[i, 0], boxes[i, 1], boxes[i, 2]
        dx, dy, dz = boxes[i, 3], boxes[i, 4], boxes[i, 5]
        yaw = float(boxes[i, 6])

        # --- centre: LiDAR → global ---
        center_lidar = np.array([x, y, z])
        center_global = R @ center_lidar + t

        # --- size: nuScenes [w, l, h] = [dy, dx, dz] ---
        size = [float(dy), float(dx), float(dz)]

        # --- rotation: yaw in LiDAR → quaternion in global ---
        Q_yaw = Quaternion(axis=[0, 0, 1], angle=yaw)
        Q_global = Q_lidar_to_global * Q_yaw
        rotation = [float(Q_global.w), float(Q_global.x),
                    float(Q_global.y), float(Q_global.z)]

        # --- class name ---
        label_idx = int(labels_np[i])
        det_name = class_names[label_idx] if label_idx < len(class_names) else "car"

        entries.append({
            "sample_token": sample_token,
            "translation": center_global.tolist(),
            "size": size,
            "rotation": rotation,
            "detection_name": det_name,
            "detection_score": sc,
        })

    return entries


# ======================================================================
# mmdet3d inference wrapper (supports v1.0 and v1.1+)
# ======================================================================

def run_inference(model, pcd_path):
    """
    Run inference on a single .bin point cloud file.

    Returns:
        (bboxes_3d, scores_3d, labels_3d)
    """
    from mmdet3d.apis import inference_detector

    result = inference_detector(model, pcd_path)

    # mmdet3d >= 1.1 (mmengine) — returns Det3DDataSample
    if hasattr(result, "pred_instances_3d"):
        pred = result.pred_instances_3d
        return pred.bboxes_3d, pred.scores_3d, pred.labels_3d

    # mmdet3d < 1.1 — returns list of dicts
    if isinstance(result, (list, tuple)):
        if isinstance(result[0], dict):
            pts = result[0].get("pts_bbox", result[0])
        else:
            # Some older APIs return (result, data)
            pts = result[0][0].get("pts_bbox", result[0][0])
        return pts["boxes_3d"], pts["scores_3d"], pts["labels_3d"]

    raise RuntimeError(f"Unexpected inference result type: {type(result)}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run detection baseline with a pre-trained mmdet3d model"
    )
    parser.add_argument("--dataroot", type=str, required=True,
                        help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--model", type=str, default="centerpoint",
                        choices=["centerpoint"],
                        help="Model name (selects default config/checkpoint)")
    parser.add_argument("--config", type=str, default=None,
                        help="mmdet3d config path (overrides --model default)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (overrides --model default)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--score_threshold", type=float, default=0.1,
                        help="Minimum detection score to keep")
    parser.add_argument("--observation_ratio", type=float, default=0.75)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: baseline_submissions/<model>_detection_results.json)")
    args = parser.parse_args()

    # ── Resolve config & checkpoint ──
    defaults = _MODEL_DEFAULTS[args.model]
    config_path = args.config or defaults["config"]
    ckpt_path = args.checkpoint or defaults["checkpoint"]

    if not os.path.isfile(config_path):
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_path = args.output or os.path.join(
        BASELINE_DIR, "baseline_submissions", f"{args.model}_detection_results.json"
    )

    # ── Load mmdet3d ──
    try:
        from mmdet3d.apis import init_model
    except ImportError:
        print("ERROR: mmdetection3d is not installed.")
        print("  Install it with:  pip install mmdet3d mmdet mmengine mmcv")
        sys.exit(1)

    # ── Load nuScenes ──
    print(f"Loading nuScenes {args.version} from {args.dataroot} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)
    obs_tokens = grader.get_observation_sample_tokens()
    print(f"Observation frames: {len(obs_tokens)}")

    # ── Init model ──
    print(f"Loading model: {args.model}")
    print(f"  Config:     {config_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device:     {args.device}")
    model = init_model(config_path, ckpt_path, device=args.device)

    # ── Try to extract class names from config ──
    cfg_class_names = None
    if hasattr(model, "cfg"):
        cfg_class_names = getattr(model.cfg, "class_names", None)
    if cfg_class_names is None:
        cfg_class_names = NUSCENES_DET_NAMES
        print(f"  Using default class names (len={len(cfg_class_names)})")
    else:
        print(f"  Class names from config (len={len(cfg_class_names)}): {cfg_class_names}")

    # ── Run inference ──
    all_results: dict[str, list] = {}
    total_boxes = 0

    for token in tqdm(obs_tokens, desc=f"Detecting ({args.model})"):
        try:
            R, t, Q_lg, lidar_path = _get_lidar_to_global(nusc, token)

            bboxes, scores, labels = run_inference(model, lidar_path)

            entries = convert_boxes_to_nuscenes(
                bboxes, scores, labels,
                R, t, Q_lg,
                token, cfg_class_names,
                score_threshold=args.score_threshold,
            )
            all_results[token] = entries
            total_boxes += len(entries)
        except Exception as e:
            warnings.warn(f"Detection failed for {token}: {e}")
            all_results[token] = []

    # ── Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {args.model} detection baseline → {output_path}")
    print(f"  Total boxes: {total_boxes}  across {len(all_results)} frames")

    # ── Quick evaluation ──
    print("\n--- Detection Evaluation ---")
    grader.evaluate_detection(output_path)


if __name__ == "__main__":
    main()
