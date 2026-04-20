import numpy as np
from tqdm import tqdm

from src.utils import (
    get_ego_pose,
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
        "dbscan_eps": 0.8,
        "dbscan_min_samples": 12,

        # Bounding box / cluster filtering
        "min_cluster_points": 10,
        "max_cluster_points": 4000,
        "min_box_volume": 0.1,       # m³
        "max_box_volume": 300.0,     # m³
        "max_detection_range_m": 45.0,
        "min_relative_z_m": -2.5,
        "max_relative_z_m": 4.0,
        "min_detection_score": 0.12,
    }

    PRESETS = {
        "balanced": {},
        "tracking_recall": {
            "dbscan_eps": 0.85,
            "dbscan_min_samples": 10,
            "min_cluster_points": 8,
            "max_detection_range_m": 50.0,
            "min_relative_z_m": -3.0,
            "max_relative_z_m": 4.5,
            "min_detection_score": 0.08,
        },
        "precision": {
            "dbscan_eps": 0.75,
            "dbscan_min_samples": 14,
            "min_cluster_points": 12,
            "max_detection_range_m": 40.0,
            "min_detection_score": 0.16,
        },
    }

    def __init__(self, nusc, config: dict | None = None, preset: str = "balanced"):
        """
        Args:
            nusc: loaded NuScenes instance.
            config: override defaults with a dict of hyper-parameters.
        """
        self.nusc = nusc
        if preset not in self.PRESETS:
            raise ValueError(
                f"Unknown detector preset: {preset}. "
                f"Available presets: {', '.join(sorted(self.PRESETS))}"
            )
        self.preset = preset
        self.cfg = {
            **self.DEFAULT_CONFIG,
            **self.PRESETS[preset],
            **(config or {}),
        }

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

        ego_translation, _ = get_ego_pose(self.nusc, sample_token)
        xyz = points[:, :3]
        rel = xyz - ego_translation
        roi_mask = (
            (np.linalg.norm(rel[:, :2], axis=1) <= self.cfg["max_detection_range_m"])
            & (rel[:, 2] >= self.cfg["min_relative_z_m"])
            & (rel[:, 2] <= self.cfg["max_relative_z_m"])
        )
        points = points[roi_mask]
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
            if size[0] < 0.2 or size[1] < 0.2 or size[2] < 0.2:
                continue
            if size[0] > 5.0 or size[1] > 20.0 or size[2] > 5.0:
                continue

            center = np.asarray(box["translation"], dtype=float)
            rel_center = center - ego_translation
            center_distance = float(np.linalg.norm(rel_center[:2]))
            if center_distance > self.cfg["max_detection_range_m"]:
                continue
            if rel_center[2] < self.cfg["min_relative_z_m"] or rel_center[2] > self.cfg["max_relative_z_m"]:
                continue

            detection_name, detection_score = classify_cluster(
                size, num_points=n_cluster_points
            )
            # Down-rank far or weakly supported clusters to improve ranking.
            distance_scale = max(0.35, 1.0 - center_distance / 70.0)
            support_scale = min(1.0, 0.65 + 0.02 * np.log1p(n_cluster_points))
            detection_score = float(detection_score * distance_scale * support_scale)
            if detection_score < self.cfg["min_detection_score"]:
                continue
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
