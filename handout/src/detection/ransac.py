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
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, C) with C >= 3")
    if n_sample < 3:
        raise ValueError("n_sample must be at least 3")

    n_points = len(points)
    if n_points == 0:
        return (
            np.empty((0, points.shape[1]), dtype=float),
            np.empty((0, points.shape[1]), dtype=float),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
        )
    if n_points < n_sample:
        return (
            points.copy(),
            np.empty((0, points.shape[1]), dtype=float),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
        )

    xyz = points[:, :3]
    rng = np.random.default_rng()

    best_inlier_mask = None
    best_plane_model = None
    best_inlier_count = -1
    best_mean_distance = np.inf

    for _ in range(max_iterations):
        sample_idx = rng.choice(n_points, size=n_sample, replace=False)
        sample_points = xyz[sample_idx]
        centroid = sample_points.mean(axis=0)

        centered = sample_points - centroid
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            continue
        normal = normal / normal_norm

        # Ground in nuScenes should be mostly horizontal in the global frame.
        if abs(normal[2]) < 0.7:
            continue
        if normal[2] < 0:
            normal = -normal

        d = -float(np.dot(normal, centroid))
        distances = np.abs(xyz @ normal + d)
        inlier_mask = distances < distance_threshold
        inlier_count = int(inlier_mask.sum())
        if inlier_count == 0:
            continue

        mean_distance = float(distances[inlier_mask].mean())
        is_better = inlier_count > best_inlier_count
        if inlier_count == best_inlier_count and mean_distance < best_mean_distance:
            is_better = True

        if is_better:
            best_inlier_count = inlier_count
            best_mean_distance = mean_distance
            best_inlier_mask = inlier_mask
            best_plane_model = np.array([normal[0], normal[1], normal[2], d], dtype=float)

    if best_inlier_mask is None or best_plane_model is None:
        return (
            points.copy(),
            np.empty((0, points.shape[1]), dtype=float),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
        )

    ground_points = points[best_inlier_mask]
    non_ground_points = points[~best_inlier_mask]
    return non_ground_points, ground_points, best_plane_model
