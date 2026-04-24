# CS5493 Autonomous Driving Perception Pipeline Report

## 1. Introduction

This project implements a complete autonomous-driving perception pipeline on the nuScenes v1.0-mini split. The pipeline contains three sequential modules: LiDAR-based 3D object detection, multi-object tracking, and trajectory prediction. The official grading protocol divides each scene into an observation portion and a prediction portion. Detection and tracking are evaluated on the observation frames, while trajectory prediction is evaluated on the later prediction frames using the history produced by our own tracker.

The final implementation is a classical geometry-based system rather than a trained deep-learning model. This choice keeps the project reproducible on a local MacBook and avoids GPU training dependencies. The pipeline uses LiDAR point clouds, RANSAC ground removal, DBSCAN clustering, PCA-based 3D bounding box fitting, Kalman filtering, Hungarian assignment, and damped constant-velocity trajectory extrapolation.

## 2. Coordinate Frames

nuScenes uses three important coordinate systems. The sensor frame is local to a sensor such as the top LiDAR. The ego frame is attached to the vehicle at a given timestamp. The global frame is a fixed world coordinate system shared across a scene. Since the evaluation JSON expects object boxes in the global frame, all detected boxes and tracked states must be expressed in global coordinates before saving results.

Our loader follows the standard transformation chain:

```text
LiDAR sensor frame -> ego vehicle frame -> global frame
```

The helper `get_lidar_points_in_global()` loads raw LiDAR points and applies the calibrated sensor pose followed by the ego pose. Detection therefore operates directly on global-frame point clouds. Because points are already global, the detector builds a dynamic point-cloud range per frame instead of assuming a sensor-centered range such as `[-50, 50]`. Each final detection entry stores global `translation`, `size`, and quaternion `rotation` fields in nuScenes result format.

## 3. Detection Method

The detector is implemented in `handout/src/detection`. It is a rule-based LiDAR detector with the following stages:

1. Load LiDAR points in global coordinates.
2. Apply an ego-centered region of interest to remove distant and vertically irrelevant points.
3. Voxelize the filtered point cloud for structured point grouping.
4. Remove ground points with RANSAC plane fitting.
5. Cluster non-ground points with DBSCAN.
6. Fit each cluster with an oriented 3D bounding box using PCA in bird's-eye view.
7. Classify boxes with geometric heuristics and assign a confidence score.
8. Export `submissions/detection_results.json`.

The detector is mostly class-agnostic. It estimates labels using box dimensions and point counts. For example, medium boxes with vehicle-like width, length, and height are labeled as cars, while small and low boxes can be labeled as traffic cones. Orientation is estimated from the principal direction of the clustered points. Confidence is based on class heuristics, cluster support, and distance from the ego vehicle. This simple design is reproducible but less accurate than a learned 3D detector, especially for small or partially observed objects.

To improve reproducibility, RANSAC uses a fixed random seed. This avoids run-to-run changes caused by random plane samples.

## 4. Point Cloud Density

We measured the number of LiDAR points per sample in nuScenes v1.0-mini.

| Split | Frames | Mean points/frame | Std. dev. | Min | Max |
|---|---:|---:|---:|---:|---:|
| Full mini split | 404 | 34718.34 | 43.25 | 34368 | 34816 |
| Observation frames | 299 | 34716.47 | 48.02 | 34368 | 34816 |

The point count is very stable because nuScenes LiDAR sweeps have a consistent scan pattern. However, the useful density on each object still varies strongly with range, occlusion, surface reflectance, and object size. Voxel-based methods benefit from regular spatial discretization, but they can lose fine detail if the voxel size is too large. Point-based methods preserve local geometry, but clustering and neighbor search become sensitive to density variation and noise. In our detector, DBSCAN is affected by this directly: one fixed `eps` and `min_samples` value cannot be optimal for both near dense objects and far sparse objects.

## 5. Architecture: Voxel vs. Pillar Representations

Voxel-based representations divide the 3D space into cells along x, y, and z. They preserve vertical structure and are useful for estimating object height and 3D shape. The drawback is computational cost: dense 3D grids can be expensive and sparse, especially over a large detection range.

Pillar-based representations collapse the vertical dimension and divide space only in the bird's-eye-view x-y plane. This reduces memory and computation, making the method efficient for real-time perception. The trade-off is that vertical information is compressed into features rather than represented as explicit z cells. Pillars are therefore faster, while full voxels can preserve more geometric detail.

Our implementation uses explicit voxelization as a classical preprocessing module, but the final object proposals are generated by RANSAC, DBSCAN, and geometric box fitting rather than by a learned voxel or pillar neural network.

## 6. Tracking Method

Tracking is implemented in `handout/src/tracking`. Each track maintains a 3D Kalman filter with state:

```text
[px, py, pz, vx, vy, vz]
```

The `predict()` step propagates each track forward with a constant-velocity motion model. The `update()` step corrects the predicted position using a matched detection. This predict-update cycle smooths noisy detections and estimates velocity from sequential positions.

Data association is solved with the Hungarian algorithm. The cost matrix is based on center distance in the x-y plane, with a small class mismatch penalty. Hungarian assignment handles unequal numbers of tracks and detections by returning the minimum-cost one-to-one matches. Matches above a distance threshold are rejected. Unmatched detections can create new tracks, and stale unmatched tracks are removed.

During optimization, the main problem was false positives and ID switches from noisy detections. The final tracker therefore uses conservative track management:

| Parameter | Final value |
|---|---:|
| Maximum age | 1 frame |
| Minimum hits before output | 3 |
| Association threshold | 3.0 m |
| Output age | 0 |
| New-track score threshold | 0.14 |
| Tracking classes | `car`, `traffic_cone` |

Restricting tracking to the two most reliable detector classes greatly reduced false positives and ID switches. This improves MOTA, although it increases false negatives because many non-car objects are no longer output as tracks.

## 7. Prediction Method

The required prediction entry point is:

```python
predict_trajectory(track_history, num_future_steps)
```

We implemented both a constant-velocity baseline and an improved damped-velocity model. The baseline extrapolates the final observed position using the final reported velocity. The improved method estimates velocity robustly from recent tracker velocities and finite differences, then applies stronger damping at longer horizons. The damping reduces runaway extrapolation when the tracker history is short or noisy.

For prediction, the input history comes from our own tracking output rather than ground truth. This makes upstream tracking quality important. A cleaner but more conservative tracker gives fewer prediction targets, but the matched targets have more stable histories and lower prediction error.

## 8. Evaluation Results

All results below were produced with:

```bash
bash run_local_eval.sh all
```

### Detection

| Metric | Value |
|---|---:|
| AP, class-agnostic | 0.0413 |
| mATE | 0.8994 m |
| mASE | 0.6300 |
| mAOE | 1.6288 rad |
| NDS | 0.0847 |
| Label accuracy | 0.4278 |

Detection is the main bottleneck. The geometric detector can find some object-like clusters, but it has limited recall and coarse class labels. The relatively high orientation error comes from unstable PCA directions for small, sparse, or nearly square clusters.

### Tracking

| Metric | Value |
|---|---:|
| MOTA | 0.0078 |
| MOTP | 0.6159 m |
| ID switches | 3 |
| Fragmentation | 89 |
| Total FP | 165 |
| Total FN | 13889 |
| Total matches | 278 |

Tracking improved after filtering low-quality classes and requiring three hits before output. The final MOTA is slightly positive, with very low ID switches. The largest remaining issue is false negatives, caused mainly by limited detection recall and conservative track output.

### Prediction

| Method | L1 mean | L2 mean | L1 median | L2 median | Targets found | Targets predicted |
|---|---:|---:|---:|---:|---:|---:|
| CV baseline | 2.6662 m | 2.0346 m | 1.5539 m | 1.1678 m | 20 | 14 |
| Improved damped CV | 2.4692 m | 1.8640 m | 1.4480 m | 1.0912 m | 20 | 14 |

Per-step L2 error for the improved method generally grows with prediction horizon:

| Step | L2 mean |
|---:|---:|
| 1 | 1.1071 m |
| 2 | 1.2000 m |
| 3 | 1.3681 m |
| 4 | 1.5486 m |
| 5 | 1.7680 m |
| 6 | 1.9812 m |
| 7 | 2.2378 m |
| 8 | 2.6087 m |
| 9 | 2.5737 m |
| 10 | 2.8538 m |
| 11 | 1.0343 m |

The final step is lower because fewer ground-truth instances are available at that step, so the average is computed over a different subset.

## 9. Analysis Questions

### Q1. Coordinate frames

Sensor coordinates describe measurements relative to the LiDAR. Ego coordinates describe positions relative to the vehicle. Global coordinates describe positions in a scene-level world frame. The detector loads points through the sensor-to-ego and ego-to-global transforms, so all cluster centers and boxes are produced in global coordinates. This matches the expected JSON output format.

### Q2. Point cloud density

The mini split has approximately 34.7k points per LiDAR sample with a small frame-level standard deviation. Stable frame-level density helps keep voxel memory and DBSCAN runtime predictable. However, object-level density still decreases with range and occlusion. This makes fixed-resolution voxelization and fixed-parameter DBSCAN imperfect: near objects may be over-segmented while far objects may be missed.

### Q3. Voxel-based vs. pillar-based architecture

Voxel methods discretize 3D space in x, y, and z and preserve height structure directly. Pillar methods discretize only x and y and encode vertical information into features. Voxels can represent geometry more faithfully but cost more memory and computation. Pillars are more efficient and common in real-time systems, but they sacrifice explicit vertical resolution.

### Q4. Tracking

Kalman `predict()` estimates the next state before seeing a detection. Kalman `update()` corrects that estimate using the matched detection. Hungarian matching solves the assignment between existing tracks and current detections even when their counts differ. Unmatched detections can start new tracks, while unmatched tracks age and are removed after a short timeout.

### Q5. Prediction

The CV baseline works well for short horizons but tends to over-extrapolate when tracker velocity is noisy. The improved method reduces this by taking a robust median of recent velocity estimates and damping long-horizon displacement. Both methods show increasing error over time because small velocity errors accumulate with horizon. The improved method lowers mean L2 error from 2.0346 m to 1.8640 m.

## 10. Limitations and Future Work

The current system is reproducible and lightweight, but detection recall remains the main limitation. A learned detector such as PointPillars or CenterPoint would likely improve AP and reduce downstream false negatives. Within the classical pipeline, further gains could come from range-adaptive DBSCAN, class-specific box filters, multi-frame point accumulation, and a tracker that maintains class-specific thresholds without discarding too many valid objects.

## 11. Team Contributions

| Member | Contributions |
|---|---|
| Member 1 | Detection preprocessing, RANSAC, DBSCAN, bounding box fitting |
| Member 2 | Kalman filter, Hungarian tracking, track management |
| Member 3 | Trajectory prediction, evaluation, report writing |

