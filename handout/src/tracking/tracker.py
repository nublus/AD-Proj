import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman_filter import KalmanFilter3D
from src.utils import make_tracking_entry


class Track:
    """Internal bookkeeping for a single tracked object."""

    _next_id = 0

    def __init__(self, detection: dict, dt: float = 0.5):
        """
        Args:
            detection: One detection dict from the detector
                       (with keys 'translation', 'size', 'rotation',
                        'detection_name', 'detection_score').
            dt: Time step (seconds) between keyframes.
        """
        self.track_id = f"track_{Track._next_id:04d}"
        Track._next_id += 1

        self.detection_name = detection.get("detection_name", "car")
        self.size = list(detection["size"])
        self.rotation = list(detection["rotation"])
        self.score = float(detection.get("detection_score", 0.5))

        pos = np.array(detection["translation"][:3], dtype=float)
        self.kf = KalmanFilter3D(initial_position=pos, dt=dt)

        self.hits = 1           # consecutive frames matched
        self.age = 0            # frames since creation
        self.time_since_update = 0
        self.total_visible_count = 1

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        pos = np.array(detection["translation"][:3], dtype=float)
        self.kf.update(pos)
        self.hits += 1
        self.time_since_update = 0
        self.total_visible_count += 1
        # optionally refresh size / rotation / name / score
        self.size = list(detection["size"])
        self.rotation = list(detection["rotation"])
        self.detection_name = detection.get("detection_name", self.detection_name)
        det_score = float(detection.get("detection_score", self.score))
        self.score = 0.7 * self.score + 0.3 * det_score

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity


class MultiObjectTracker:
    """
    Multi-Object Tracker using Kalman Filter + Hungarian algorithm.

    Usage:
        tracker = MultiObjectTracker()
        for frame_detections in all_frames:
            tracked_objects = tracker.update(frame_detections, sample_token)
    """

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        association_threshold: float = 3.0,
        dt: float = 0.5,
    ):
        """
        Args:
            max_age:               Delete a track after this many
                                   consecutive frames without a match.
            min_hits:              A track must be matched this many
                                   times before appearing in output.
            association_threshold: Max centre distance (m) for matching.
            dt:                    Time between keyframes (s).
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.assoc_threshold = association_threshold
        self.dt = dt
        self.tracks: list[Track] = []
        self.class_mismatch_penalty = 0.3
        self.max_output_age = 0
        self.new_track_score_threshold = 0.14
        self.allowed_tracking_classes = {"car", "traffic_cone"}

    def _should_start_track(self, detection: dict) -> bool:
        score = float(detection.get("detection_score", 0.0))
        name = detection.get("detection_name", "")
        if name not in self.allowed_tracking_classes:
            return False

        # Small-object detections are noisier in the current detector, so
        # require slightly stronger evidence before spawning a new track.
        threshold = self.new_track_score_threshold
        if name in {"traffic_cone", "bicycle", "pedestrian"}:
            threshold = max(threshold, 0.18)
        elif name in {"motorcycle", "barrier"}:
            threshold = max(threshold, 0.16)
        return score >= threshold

    def _build_cost_matrix(self, detections: list[dict]) -> np.ndarray:
        if not self.tracks or not detections:
            return np.empty((len(self.tracks), len(detections)), dtype=float)

        cost = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for i, track in enumerate(self.tracks):
            track_xy = track.position[:2]
            for j, det in enumerate(detections):
                det_xy = np.asarray(det["translation"][:2], dtype=float)
                distance = float(np.linalg.norm(track_xy - det_xy))

                # A mild class-consistency penalty helps reduce ID switches
                # without making noisy detection classes unusable.
                if det.get("detection_name") != track.detection_name:
                    distance += self.class_mismatch_penalty
                cost[i, j] = distance
        return cost

    def reset(self):
        """Clear all tracks (call between scenes)."""
        self.tracks = []
        Track._next_id = 0

    def update(self, detections: list[dict], sample_token: str) -> list[dict]:
        """
        Process one frame of detections and return tracking entries.

        Steps:
            1. Predict all existing tracks forward.
            2. Build cost matrix (centre distance).
            3. Solve assignment with Hungarian algorithm.
            4. Update matched tracks, create new tracks for unmatched
               detections, age unmatched tracks.
            5. Prune dead tracks.
            6. Return nuScenes tracking entries for confirmed tracks.

        Args:
            detections:   List of detection dicts for this frame.
            sample_token: The nuScenes sample token.

        Returns:
            List of tracking entry dicts (nuScenes format) for this frame.
        """
        detections = [
            det
            for det in detections
            if det.get("detection_name") in self.allowed_tracking_classes
        ]

        # 1. Predict all existing tracks.
        for track in self.tracks:
            track.predict()

        # 2. Associate tracks with new detections.
        matched_track_indices: set[int] = set()
        matched_detection_indices: set[int] = set()

        if self.tracks and detections:
            cost = self._build_cost_matrix(detections)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.assoc_threshold:
                    matched_track_indices.add(int(r))
                    matched_detection_indices.add(int(c))
                    self.tracks[r].update(detections[c])

        # 3. Create new tracks for unmatched detections.
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detection_indices:
                if not self._should_start_track(detection):
                    continue
                self.tracks.append(Track(detection=detection, dt=self.dt))

        # 4. Prune stale tracks.
        self.tracks = [
            track for track in self.tracks if track.time_since_update <= self.max_age
        ]

        # 5. Emit confirmed tracks, including short-term predictions for
        # recently unmatched tracks. This helps bridge brief detector dropouts.
        outputs: list[dict] = []
        for track in self.tracks:
            if track.hits < self.min_hits:
                continue
            if track.time_since_update > self.max_output_age:
                continue
            if track.time_since_update > 0 and track.total_visible_count < 4:
                continue

            position = track.position
            velocity = track.velocity
            freshness = max(0.35, 1.0 - 0.25 * track.time_since_update)
            tracking_score = float(
                min(
                    0.99,
                    max(
                        0.05,
                        track.score
                        * min(1.0, 0.75 + 0.05 * track.total_visible_count)
                        * freshness,
                    ),
                )
            )
            outputs.append(
                make_tracking_entry(
                    sample_token=sample_token,
                    translation=position.tolist(),
                    size=track.size,
                    rotation=track.rotation,
                    velocity=velocity[:2].tolist(),
                    tracking_id=track.track_id,
                    tracking_name=track.detection_name,
                    tracking_score=tracking_score,
                )
            )

        return outputs
