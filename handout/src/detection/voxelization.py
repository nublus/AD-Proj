import numpy as np


def voxelize(
    points: np.ndarray,
    voxel_size: list[float],
    pc_range: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition a point cloud into a 3D voxel grid and return non-empty voxels.

    Algorithm outline:
        1. Clip / filter points to ``pc_range``.
        2. Compute the integer voxel index (ix, iy, iz) for each point.
        3. Group points by their voxel index.
        4. For each non-empty voxel, compute the centre coordinate and the
           mean feature vector (e.g. mean x, y, z, intensity).

    Args:
        points:     (N, C)  point cloud, with at least columns [x, y, z].
                    Typical C = 4 for [x, y, z, intensity].
        voxel_size: [vx, vy, vz]  size of each voxel in metres.
        pc_range:   [x_min, y_min, z_min, x_max, y_max, z_max]
                    The axis-aligned bounding box to consider.

    Returns:
        voxel_centers:   (M, 3)  centre coordinates of each non-empty voxel.
        voxel_features:  (M, C)  mean feature vector inside each voxel.
        voxel_counts:    (M,)    number of points in each voxel.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, C) with C >= 3")

    voxel_size_arr = np.asarray(voxel_size, dtype=float)
    pc_range_arr = np.asarray(pc_range, dtype=float)
    if voxel_size_arr.shape != (3,) or pc_range_arr.shape != (6,):
        raise ValueError("voxel_size must be (3,) and pc_range must be (6,)")

    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, points.shape[1]), dtype=float),
            np.empty((0,), dtype=int),
        )

    mins = pc_range_arr[:3]
    maxs = pc_range_arr[3:]
    mask = np.all((points[:, :3] >= mins) & (points[:, :3] < maxs), axis=1)
    filtered = points[mask]
    if len(filtered) == 0:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, points.shape[1]), dtype=float),
            np.empty((0,), dtype=int),
        )

    voxel_indices = np.floor((filtered[:, :3] - mins) / voxel_size_arr).astype(int)
    unique_indices, inverse, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )

    voxel_features = np.zeros((len(unique_indices), filtered.shape[1]), dtype=float)
    np.add.at(voxel_features, inverse, filtered)
    voxel_features /= counts[:, None]

    voxel_centers = mins + (unique_indices.astype(float) + 0.5) * voxel_size_arr
    voxel_counts = counts.astype(int)
    return voxel_centers, voxel_features, voxel_counts
