import json
import os

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud


# ====================================================================
# Category mapping (same as grading script — kept here for convenience)
# ====================================================================

DETECTION_NAMES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]

CATEGORY_TO_DETECTION_NAME = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}


# ====================================================================
# Coordinate transformations
# ====================================================================

def get_lidar_points_in_global(nusc, sample_token: str) -> np.ndarray:
    """
    Load the LiDAR point cloud for a sample and transform it to the
    **global** coordinate frame.

    Transformation chain:
        sensor (LiDAR) → ego vehicle → global

    Args:
        nusc: loaded NuScenes instance
        sample_token: sample token string

    Returns:
        points  (N, 4) – columns [x, y, z, intensity] in the global frame
    """
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    # Load point cloud in sensor frame
    pcl_path = os.path.join(nusc.dataroot, lidar_data["filename"])
    pc = LidarPointCloud.from_file(pcl_path)  # (4, N)

    # --- sensor → ego ---
    cs = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))

    # --- ego → global ---
    pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
    pc.translate(np.array(pose["translation"]))

    return pc.points.T  # (N, 4)


def get_lidar_points_in_ego(nusc, sample_token: str) -> np.ndarray:
    """
    Load the LiDAR point cloud and transform it to the **ego-vehicle**
    coordinate frame.

    Args:
        nusc: loaded NuScenes instance
        sample_token: sample token string

    Returns:
        points  (N, 4) – columns [x, y, z, intensity] in ego frame
    """
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    pcl_path = os.path.join(nusc.dataroot, lidar_data["filename"])
    pc = LidarPointCloud.from_file(pcl_path)

    # sensor → ego
    cs = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))

    return pc.points.T  # (N, 4)


def get_sensor_to_global(nusc, sample_token: str, sensor_channel: str = "LIDAR_TOP"):
    """
    Return the 4 × 4 homogeneous transformation matrix:
        sensor → global

    Args:
        nusc: NuScenes instance
        sample_token: sample token
        sensor_channel: e.g. 'LIDAR_TOP', 'CAM_FRONT'

    Returns:
        T_global_sensor  (4, 4) np.ndarray
    """
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][sensor_channel]
    sd = nusc.get("sample_data", sd_token)

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])

    # sensor → ego
    R_es = Quaternion(cs["rotation"]).rotation_matrix
    t_es = np.array(cs["translation"])
    T_es = np.eye(4)
    T_es[:3, :3] = R_es
    T_es[:3, 3] = t_es

    # ego → global
    R_ge = Quaternion(pose["rotation"]).rotation_matrix
    t_ge = np.array(pose["translation"])
    T_ge = np.eye(4)
    T_ge[:3, :3] = R_ge
    T_ge[:3, 3] = t_ge

    return T_ge @ T_es


def transform_points_to_global(points_sensor: np.ndarray, T_global_sensor: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 homogeneous transform to Nx3 points.

    Args:
        points_sensor: (N, 3) array in source frame
        T_global_sensor: (4, 4) transformation matrix

    Returns:
        points_global: (N, 3) array in target frame
    """
    N = points_sensor.shape[0]
    ones = np.ones((N, 1))
    pts_h = np.hstack([points_sensor, ones])  # (N, 4)
    pts_t = (T_global_sensor @ pts_h.T).T     # (N, 4)
    return pts_t[:, :3]


def get_ego_pose(nusc, sample_token: str):
    """Return (translation, rotation_quaternion) of the ego vehicle in global frame."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(pose["translation"]), Quaternion(pose["rotation"])


# ====================================================================
# nuScenes result format helpers
# ====================================================================

def make_detection_entry(
    sample_token: str,
    translation: list,
    size: list,
    rotation: list,
    detection_name: str,
    detection_score: float,
) -> dict:
    """Create a single entry for detection_results.json."""
    return {
        "sample_token": sample_token,
        "translation": list(map(float, translation)),
        "size": list(map(float, size)),
        "rotation": list(map(float, rotation)),
        "detection_name": detection_name,
        "detection_score": float(detection_score),
    }


def make_tracking_entry(
    sample_token: str,
    translation: list,
    size: list,
    rotation: list,
    velocity: list,
    tracking_id: str,
    tracking_name: str,
    tracking_score: float,
) -> dict:
    """Create a single entry for tracking_results.json."""
    return {
        "sample_token": sample_token,
        "translation": list(map(float, translation)),
        "size": list(map(float, size)),
        "rotation": list(map(float, rotation)),
        "velocity": list(map(float, velocity)),
        "tracking_id": tracking_id,
        "tracking_name": tracking_name,
        "tracking_score": float(tracking_score),
    }


def save_detection_results(results_dict: dict, path: str):
    """
    Save detection results in nuScenes format.

    Args:
        results_dict: {sample_token: [detection_entry, ...]}
        path: output JSON path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    output = {
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": results_dict,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Detection results saved to {path}  ({len(results_dict)} samples)")


def save_tracking_results(results_dict: dict, path: str):
    """
    Save tracking results in nuScenes format.

    Args:
        results_dict: {sample_token: [tracking_entry, ...]}
        path: output JSON path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    output = {
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": results_dict,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Tracking results saved to {path}  ({len(results_dict)} samples)")
