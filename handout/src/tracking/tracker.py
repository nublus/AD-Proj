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

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        pos = np.array(detection["translation"][:3], dtype=float)
        self.kf.update(pos)
        self.hits += 1
        self.time_since_update = 0
        # optionally refresh size / rotation / name / score
        self.size = list(detection["size"])
        self.rotation = list(detection["rotation"])
        self.detection_name = detection.get("detection_name", self.detection_name)
        self.score = float(detection.get("detection_score", self.score))


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
        max_age: int = 3,
        min_hits: int = 1,
        association_threshold: float = 4.0,
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

        raise NotImplementedError(
            "TODO: implement MultiObjectTracker.update()")
