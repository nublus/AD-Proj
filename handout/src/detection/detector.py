import numpy as np
from tqdm import tqdm

from src.utils import (
    get_lidar_points_in_global,
    make_detection_entry,
    save_detection_results,
)
from src.detection.voxelization import voxelize
from src.detection.ransac import ransac_ground_removal
from src.detection.dbscan_cluster import dbscan_cluster
from src.detection.bbox_fitting import fit_bounding_box, classify_cluster


class LidarDetector:
    """Rule-based 3D object detector for LiDAR point clouds."""

    # Default hyper-parameters — feel free to tune
    DEFAULT_CONFIG = {
        # Voxelization
        "voxel_size": [0.2, 0.2, 0.2],
        "pc_range": [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],

        # RANSAC ground segmentation
        "ransac_distance_threshold": 0.3,
        "ransac_max_iterations": 1000,

        # DBSCAN clustering
        "dbscan_eps": 0.6,
        "dbscan_min_samples": 10,

        # Bounding box / cluster filtering
        "min_cluster_points": 5,
        "max_cluster_points": 50000,
        "min_box_volume": 0.1,       # m³
        "max_box_volume": 1000.0,    # m³
    }

    def __init__(self, nusc, config: dict | None = None):
        """
        Args:
            nusc: loaded NuScenes instance.
            config: override defaults with a dict of hyper-parameters.
        """
        self.nusc = nusc
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}

    # ----------------------------------------------------------------
    # Core detection routine (per sample)
    # ----------------------------------------------------------------

    def detect_single_sample(self, sample_token: str) -> list[dict]:
        """
        Run detection on one sample and return a list of nuScenes
        detection entries (dicts).

        Pipeline:
            1. Load point cloud in global frame.
            2. (Optional) Voxelize.
            3. RANSAC ground removal.
            4. DBSCAN clustering on non-ground points.
            5. Fit bounding box to each cluster.
            6. Classify each cluster heuristically.
            7. Build detection entries.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            List of detection dicts ready for ``detection_results.json``.
        """
        # Load point cloud (provided)
        points = get_lidar_points_in_global(self.nusc, sample_token)  # (N, 4)
        if len(points) == 0:
            return []

        # The helper loads points in the global frame, so we build the voxel
        # range dynamically instead of using sensor-centric bounds.
        xyz = points[:, :3]
        xyz_min = xyz.min(axis=0) - 1e-3
        xyz_max = xyz.max(axis=0) + 1e-3
        dynamic_pc_range = [*xyz_min.tolist(), *xyz_max.tolist()]

        _, voxel_features, _ = voxelize(
            points,
            voxel_size=self.cfg["voxel_size"],
            pc_range=dynamic_pc_range,
        )
        non_ground_points, _, _ = ransac_ground_removal(
            xyz,
            distance_threshold=self.cfg["ransac_distance_threshold"],
            max_iterations=self.cfg["ransac_max_iterations"],
        )
        if len(non_ground_points) == 0:
            return []

        labels, n_clusters = dbscan_cluster(
            non_ground_points,
            eps=self.cfg["dbscan_eps"],
            min_samples=self.cfg["dbscan_min_samples"],
        )
        if n_clusters == 0:
            return []

        detections: list[dict] = []
        for cluster_id in range(n_clusters):
            cluster_points = non_ground_points[labels == cluster_id]
            n_cluster_points = len(cluster_points)
            if n_cluster_points < self.cfg["min_cluster_points"]:
                continue
            if n_cluster_points > self.cfg["max_cluster_points"]:
                continue

            box = fit_bounding_box(cluster_points)
            size = box["size"]
            volume = float(np.prod(size))
            if volume < self.cfg["min_box_volume"]:
                continue
            if volume > self.cfg["max_box_volume"]:
                continue

            detection_name, detection_score = classify_cluster(
                size, num_points=n_cluster_points
            )
            detections.append(
                make_detection_entry(
                    sample_token=sample_token,
                    translation=box["translation"],
                    size=size,
                    rotation=box["rotation"],
                    detection_name=detection_name,
                    detection_score=detection_score,
                )
            )

        return detections

    # ----------------------------------------------------------------
    # Batch detection
    # ----------------------------------------------------------------

    def detect_all(self, sample_tokens: list[str], output_path: str):
        """
        Run detection on a list of samples and save results to JSON.

        Args:
            sample_tokens: list of nuScenes sample tokens (observation only).
            output_path:   path to write detection_results.json.
        """
        results: dict[str, list] = {}
        for token in tqdm(sample_tokens, desc="Detecting"):
            try:
                dets = self.detect_single_sample(token)
                results[token] = dets
            except NotImplementedError:
                raise
            except Exception as e:
                print(f"  Warning: detection failed for {token}: {e}")
                results[token] = []

        save_detection_results(results, output_path)
