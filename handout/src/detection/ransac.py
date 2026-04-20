import numpy as np


def ransac_ground_removal(
    points: np.ndarray,
    distance_threshold: float = 0.3,
    max_iterations: int = 1000,
    n_sample: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove ground points by fitting a plane with RANSAC.

    Algorithm outline:
        1. Repeat for ``max_iterations``:
            a. Randomly sample ``n_sample`` points.
            b. Fit a plane through those points  (solve for [a, b, c, d]).
            c. Compute distances of ALL points to the plane.
            d. Count inliers (distance < ``distance_threshold``).
            e. Keep the model with the most inliers.
        2. Separate inliers (ground) from outliers (non-ground).

    Plane equation:  ax + by + cz + d = 0,  with  ||[a,b,c]|| = 1.

    Args:
        points:              (N, 3+)  point cloud (at least x, y, z columns).
        distance_threshold:  Maximum point-to-plane distance for an inlier.
        max_iterations:      Number of RANSAC iterations.
        n_sample:            Points sampled per iteration (>=3 for a plane).

    Returns:
        non_ground_points:  (M, C)  points classified as non-ground.
        ground_points:      (K, C)  points classified as ground.
        plane_model:        (4,)    [a, b, c, d] of the best plane.
    """

    raise NotImplementedError("TODO: implement ransac_ground_removal()")
