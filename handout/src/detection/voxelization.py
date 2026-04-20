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

    raise NotImplementedError("TODO: implement voxelize()")
