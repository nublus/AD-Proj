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
    if True:
        raise NotImplementedError("TODO: implement fit_bounding_box()")
    
    return {'translation': [None, None, None], 'size': [None, None, None], 'rotation': [None, None, None, None]}


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

    raise NotImplementedError("TODO: implement classify_cluster()")

    return "NONE", None
