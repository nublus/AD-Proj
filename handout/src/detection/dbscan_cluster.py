import numpy as np


def dbscan_cluster(
    points: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
) -> tuple[np.ndarray, int]:
    """
    Cluster a point cloud using DBSCAN.

    Args:
        points:      (N, 3)  point cloud [x, y, z].
        eps:         Maximum distance between two samples in the same cluster.
        min_samples: Minimum number of points to form a dense region (core point).

    Returns:
        labels:     (N,)  cluster label for each point.
                    -1 means noise (not assigned to any cluster).
        n_clusters: Total number of clusters found (excluding noise).
    """

    raise NotImplementedError("TODO: implement dbscan_cluster()")
