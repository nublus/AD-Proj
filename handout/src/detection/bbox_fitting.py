import numpy as np
from pyquaternion import Quaternion


def fit_bounding_box(cluster_points: np.ndarray) -> dict:
    """
    Fit a 3D oriented bounding box to a cluster of points.

    The result should be expressed in the **global** frame and follow
    the nuScenes convention:

        - translation: centre of the box  [x, y, z]
        - size:        [width, length, height]   (w along x, l along y, h along z
                       in the box-local frame)
        - rotation:    unit quaternion  [qw, qx, qy, qz]

    Args:
        cluster_points: (N, 3)  points belonging to one cluster.

    Returns:
        dict with keys:
            'translation':  [cx, cy, cz]
            'size':         [w, l, h]
            'rotation':     [qw, qx, qy, qz]
    """
    cluster_points = np.asarray(cluster_points, dtype=float)
    if cluster_points.ndim != 2 or cluster_points.shape[1] < 3:
        raise ValueError("cluster_points must have shape (N, 3) or (N, C>=3)")
    if len(cluster_points) == 0:
        raise ValueError("cluster_points must be non-empty")

    xyz = cluster_points[:, :3]
    xy = xyz[:, :2]
    xy_mean = xy.mean(axis=0)
    centered_xy = xy - xy_mean

    if len(xy) >= 2 and np.linalg.norm(centered_xy) > 1e-8:
        covariance = np.cov(centered_xy, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        principal_yaw = float(np.arctan2(principal_axis[1], principal_axis[0]))
        yaw = principal_yaw - np.pi / 2.0
    else:
        yaw = 0.0

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rotation_to_local = np.array(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
        dtype=float,
    )
    local_xy = centered_xy @ rotation_to_local.T

    xy_min = local_xy.min(axis=0)
    xy_max = local_xy.max(axis=0)
    local_center_xy = 0.5 * (xy_min + xy_max)

    cos_inv = np.cos(yaw)
    sin_inv = np.sin(yaw)
    rotation_to_global = np.array(
        [[cos_inv, -sin_inv], [sin_inv, cos_inv]],
        dtype=float,
    )
    center_xy = xy_mean + local_center_xy @ rotation_to_global.T

    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())
    center_z = 0.5 * (z_min + z_max)

    width = float(max(xy_max[0] - xy_min[0], 1e-2))
    length = float(max(xy_max[1] - xy_min[1], 1e-2))
    height = float(max(z_max - z_min, 1e-2))

    if length < width:
        width, length = length, width
        yaw += np.pi / 2.0

    quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
    return {
        "translation": [float(center_xy[0]), float(center_xy[1]), center_z],
        "size": [width, length, height],
        "rotation": [float(quat.w), float(quat.x), float(quat.y), float(quat.z)],
    }


def classify_cluster(size: list[float], num_points: int = 0) -> tuple[str, float]:
    """
    Classify a cluster as a nuScenes detection class using geometric
    heuristics (box dimensions and optionally point count).

    Rough guidelines (metres):
        car:                 3.5 < length < 6.0,   1.5 < width < 2.5
        truck:               6.0 < length < 12.0
        bus:                 length > 10
        pedestrian:          length < 1.0,  height 1.2 – 2.0
        bicycle/motorcycle:  1.2 < length < 2.5,   height < 1.8
        traffic_cone:        very small,  height < 1.0
        barrier:             long & thin,  height < 1.2

    Args:
        size:        [width, length, height] of the fitted box.
        num_points:  Number of points in the cluster (optional heuristic).

    Returns:
        (detection_name, detection_score)
        detection_name:  one of the 10 nuScenes detection classes.
        detection_score: a confidence estimate in [0, 1].
    """
    width, length, height = map(float, size)
    num_points = int(num_points)

    # Keep scores conservative so dubious clusters are ranked later by AP.
    point_bonus = min(0.18, max(num_points - 10, 0) * 0.003)

    if 9.0 < length < 18.0 and 2.2 < width < 4.5 and 2.5 < height < 4.5 and num_points >= 35:
        return "bus", 0.60 + point_bonus
    if 5.5 < length < 12.0 and 1.8 < width < 3.5 and 1.8 < height < 4.0 and num_points >= 25:
        return "truck", 0.56 + point_bonus
    if 4.0 < length < 12.0 and width > 2.2 and height > 2.5 and num_points >= 30:
        return "construction_vehicle", 0.48 + point_bonus
    if 5.5 < length < 14.0 and 1.7 < width < 3.5 and 1.0 < height < 4.0 and num_points >= 20:
        return "trailer", 0.42 + point_bonus
    if 2.8 < length < 6.5 and 1.3 < width < 2.8 and 1.0 < height < 3.0 and num_points >= 18:
        return "car", 0.62 + point_bonus
    if length > 1.5 and width < 1.0 and height < 1.4 and num_points >= 8:
        return "barrier", 0.24 + min(point_bonus, 0.08)
    if length < 1.1 and width < 1.1 and 0.2 < height < 1.2 and num_points >= 5:
        return "traffic_cone", 0.16 + min(point_bonus, 0.05)
    if length < 1.1 and width < 1.1 and 1.0 < height < 2.4 and num_points >= 6:
        return "pedestrian", 0.20 + min(point_bonus, 0.07)
    if 1.2 < length < 2.8 and width < 1.1 and 0.6 < height < 1.8 and num_points >= 10:
        if num_points >= 18:
            return "motorcycle", 0.24 + min(point_bonus, 0.08)
        return "bicycle", 0.22 + min(point_bonus, 0.07)

    # Low-confidence fallback for ambiguous medium objects.
    if 2.5 < length < 7.0 and 1.0 < width < 3.5 and 0.8 < height < 3.5 and num_points >= 15:
        return "car", 0.16 + min(point_bonus, 0.06)
    return "traffic_cone", 0.08
