import numpy as np
from sklearn.cluster import DBSCAN


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
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, 3) or (N, C>=3)")

    if len(points) == 0:
        return np.empty((0,), dtype=int), 0

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(points[:, :3]).astype(int)
    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    return labels, n_clusters
