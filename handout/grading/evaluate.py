"""
Project Grading Script — CS5493 Autonomous Driving Perception Pipeline
======================================================================

Evaluates all three stages of the perception pipeline:
  1. 3D Object Detection   (nuScenes-style mAP, NDS, TP errors)
  2. Multi-Object Tracking  (MOTA, MOTP, ID Switches, Fragmentation)
  3. Trajectory Prediction   (L1 / L2 errors, per-step breakdown)

Usage (as library):
    from nuscenes.nuscenes import NuScenes
    from grading.evaluate import ProjectGrader

    nusc = NuScenes(version='v1.0-mini', dataroot='...', verbose=True)
    grader = ProjectGrader(nusc, observation_ratio=0.75)
    report = grader.run_full_evaluation(
        detection_path='submissions/detection_results.json',
        tracking_path='submissions/tracking_results.json',
        predict_fn=predict_trajectory,
    )

Usage (standalone):
    python -m grading.evaluate --dataroot /path/to/nuscenes \\
        --detection submissions/detection_results.json \\
        --tracking submissions/tracking_results.json \\
        --predict_module src.prediction.predictor

DO NOT MODIFY THIS FILE — it is the authoritative evaluator.
"""

import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment

# ====================================================================
# Constants
# ====================================================================

DETECTION_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

# Map nuScenes fine-grained categories → 10 detection classes
CATEGORY_TO_DETECTION_NAME = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}

# Center-distance thresholds (metres) used for AP computation
DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]

# Matching threshold for tracking evaluation (metres)
TRACKING_MATCH_THRESHOLD = 2.0

# Matching threshold for prediction target association (metres)
PREDICTION_MATCH_THRESHOLD = 4.0

# ====================================================================
# Utility helpers
# ====================================================================


def _quaternion_yaw(q: Quaternion) -> float:
    """Extract yaw angle (rotation around z-axis) from a Quaternion."""
    v = np.array([1.0, 0.0, 0.0])
    v_rot = q.rotate(v)
    return float(np.arctan2(v_rot[1], v_rot[0]))


def _angle_diff(a: float, b: float) -> float:
    """Smallest unsigned angle difference in [0, pi]."""
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def _volume(size):
    """Compute box volume from [w, l, h]."""
    return float(size[0]) * float(size[1]) * float(size[2])


# ====================================================================
# ProjectGrader
# ====================================================================


class ProjectGrader:
    """Evaluate the full autonomous-driving perception pipeline."""

    # ----------------------------------------------------------------
    # Initialisation & splits
    # ----------------------------------------------------------------

    def __init__(self, nusc, observation_ratio: float = 0.75):
        """
        Args:
            nusc: A loaded NuScenes instance.
            observation_ratio: Fraction of each scene's keyframes used for
                detection / tracking (the remainder is for prediction).
        """
        self.nusc = nusc
        self.observation_ratio = observation_ratio
        self.scene_splits: dict = {}
        self._compute_splits()
        self._save_scene_splits()

    # ---- split computation ----------------------------------------

    def _compute_splits(self):
        for scene in self.nusc.scene:
            tokens = []
            tok = scene["first_sample_token"]
            while tok:
                tokens.append(tok)
                s = self.nusc.get("sample", tok)
                tok = s["next"] if s["next"] != "" else None

            n_obs = max(1, int(len(tokens) * self.observation_ratio))
            self.scene_splits[scene["name"]] = {
                "scene_token": scene["token"],
                "all_tokens": tokens,
                "observation_tokens": tokens[:n_obs],
                "prediction_tokens": tokens[n_obs:],
            }

    def _save_scene_splits(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_splits.json")
        data = {}
        for name, sp in self.scene_splits.items():
            data[name] = {
                "observation_tokens": sp["observation_tokens"],
                "prediction_tokens": sp["prediction_tokens"],
                "n_observation": len(sp["observation_tokens"]),
                "n_prediction": len(sp["prediction_tokens"]),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ---- public token accessors -----------------------------------

    def get_observation_sample_tokens(self) -> list:
        """Return observation sample tokens across all scenes (ordered)."""
        out = []
        for sp in self.scene_splits.values():
            out.extend(sp["observation_tokens"])
        return out

    def get_prediction_sample_tokens(self) -> list:
        """Return prediction sample tokens across all scenes (ordered)."""
        out = []
        for sp in self.scene_splits.values():
            out.extend(sp["prediction_tokens"])
        return out

    # ----------------------------------------------------------------
    # Ground-truth helpers
    # ----------------------------------------------------------------

    def _get_gt_boxes(self, sample_token: str) -> list:
        """
        Return list of GT dicts for a sample, each containing:
            sample_token, translation, size, rotation, detection_name,
            instance_token, velocity, num_lidar_pts
        """
        sample = self.nusc.get("sample", sample_token)
        boxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            det_name = CATEGORY_TO_DETECTION_NAME.get(ann["category_name"])
            if det_name is None:
                continue

            vel = self.nusc.box_velocity(ann_token)  # (3,)
            if np.any(np.isnan(vel)):
                vel = np.array([0.0, 0.0, 0.0])

            boxes.append(
                {
                    "sample_token": sample_token,
                    "translation": list(ann["translation"]),
                    "size": list(ann["size"]),
                    "rotation": list(ann["rotation"]),
                    "detection_name": det_name,
                    "instance_token": ann["instance_token"],
                    "velocity": vel[:2].tolist(),
                    "num_lidar_pts": ann["num_lidar_pts"],
                }
            )
        return boxes

    # ================================================================
    # 1. DETECTION EVALUATION
    # ================================================================

    def evaluate_detection(self, detection_path: str) -> dict:
        """
        Evaluate detection results with **class-agnostic** metrics.

        Matching is based on centre distance only — the predicted class
        label does NOT need to match the ground-truth class.  Label
        accuracy is reported as an informational metric.

        Returns dict with keys:
            AP, mATE, mASE, mAOE, NDS, label_accuracy
        """
        _header("DETECTION EVALUATION (class-agnostic)")

        with open(detection_path) as f:
            det_data = json.load(f)
        det_results = det_data.get("results", {})

        obs_tokens = self.get_observation_sample_tokens()

        # Validate tokens
        invalid = [t for t in det_results if t not in obs_tokens]
        if invalid:
            warnings.warn(
                f"{len(invalid)} sample token(s) in detection results are NOT "
                f"in the observation set and will be ignored."
            )

        # Pre-fetch GT per frame
        frame_gt = {t: self._get_gt_boxes(t) for t in obs_tokens}
        frame_pred = {t: det_results.get(t, []) for t in obs_tokens}

        # Class-agnostic AP at each distance threshold
        aps = []
        tp_errors_at_2m = []
        label_correct = 0
        label_total = 0
        for d_th in DIST_THRESHOLDS:
            ap, tp_errs, n_correct, n_total = self._ap_class_agnostic(
                frame_gt, frame_pred, obs_tokens, d_th
            )
            aps.append(ap)
            if d_th == 2.0:
                tp_errors_at_2m = tp_errs
                label_correct = n_correct
                label_total = n_total

        AP = float(np.mean(aps))

        # TP errors from matched pairs
        if tp_errors_at_2m:
            mATE = float(np.mean([e["ate"] for e in tp_errors_at_2m]))
            mASE = float(np.mean([e["ase"] for e in tp_errors_at_2m]))
            mAOE = float(np.mean([e["aoe"] for e in tp_errors_at_2m]))
        else:
            mATE, mASE, mAOE = 1.0, 1.0, 1.0

        label_accuracy = label_correct / max(label_total, 1)

        # Simplified NDS (class-agnostic)
        tp_score = max(1.0 - mATE, 0.0) + max(1.0 - mASE, 0.0) + max(1.0 - mAOE, 0.0)
        NDS = (5.0 * AP + tp_score) / 8.0

        # ---- pretty print ----
        print(f"\n  {'Metric':<25} {'Value':>10}")
        print("  " + "-" * 37)
        print(f"  {'AP (class-agnostic)':<25} {AP:>10.4f}")
        print(f"  {'mATE (m)':<25} {mATE:>10.4f}")
        print(f"  {'mASE':<25} {mASE:>10.4f}")
        print(f"  {'mAOE (rad)':<25} {mAOE:>10.4f}")
        print(f"  {'NDS':<25} {NDS:>10.4f}")
        print(f"  {'Label Accuracy (info)':<25} {label_accuracy:>10.4f}  ({label_correct}/{label_total})")

        return {
            "AP": AP,
            "mATE": mATE,
            "mASE": mASE,
            "mAOE": mAOE,
            "NDS": NDS,
            "label_accuracy": label_accuracy,
        }

    # ---- AP computation (class-agnostic) ----------------------------

    def _ap_class_agnostic(self, frame_gt, frame_pred, tokens, dist_th):
        """
        Compute **class-agnostic** AP at one distance threshold.

        All GT boxes and all predicted boxes are pooled regardless of
        class label.  Matching is purely by centre distance.

        Returns:
            (ap, tp_errors, label_correct, label_total)
        """
        gt_list = []
        pred_list = []
        gt_matched = {}

        for token in tokens:
            gts = [b for b in frame_gt[token]]         # ALL GT, no class filter
            preds = [b for b in frame_pred[token]]     # ALL preds, no class filter
            for i, g in enumerate(gts):
                gid = f"{token}_{i}"
                gt_list.append({**g, "_gid": gid})
                gt_matched[gid] = False
            for p in preds:
                pred_list.append(
                    {**p, "_token": token, "_score": float(p.get("detection_score", 0.0))}
                )

        n_gt = len(gt_list)
        if n_gt == 0:
            return 0.0, [], 0, 0

        # sort predictions by score desc
        pred_list.sort(key=lambda x: x["_score"], reverse=True)

        tp = np.zeros(len(pred_list))
        fp = np.zeros(len(pred_list))
        tp_errors = []
        label_correct = 0
        label_total = 0

        # group GT by token for fast lookup
        gt_by_tok = defaultdict(list)
        for g in gt_list:
            gt_by_tok[g["sample_token"]].append(g)

        for idx, pred in enumerate(pred_list):
            tok_gts = gt_by_tok.get(pred["_token"], [])
            best_dist = float("inf")
            best_gt = None
            for g in tok_gts:
                if gt_matched[g["_gid"]]:
                    continue
                d = np.linalg.norm(
                    np.array(pred["translation"][:2]) - np.array(g["translation"][:2])
                )
                if d < best_dist:
                    best_dist = d
                    best_gt = g

            if best_gt is not None and best_dist < dist_th:
                tp[idx] = 1
                gt_matched[best_gt["_gid"]] = True

                # TP errors
                ate = float(
                    np.linalg.norm(
                        np.array(pred["translation"]) - np.array(best_gt["translation"])
                    )
                )
                pv = _volume(pred.get("size", [1, 1, 1]))
                gv = _volume(best_gt["size"])
                ase = 1.0 - min(pv, gv) / max(pv, gv) if pv > 0 and gv > 0 else 1.0

                pred_yaw = _quaternion_yaw(Quaternion(pred.get("rotation", [1, 0, 0, 0])))
                gt_yaw = _quaternion_yaw(Quaternion(best_gt["rotation"]))
                aoe = _angle_diff(pred_yaw, gt_yaw)

                tp_errors.append({"ate": ate, "ase": ase, "aoe": aoe})

                # Label accuracy (informational)
                label_total += 1
                pred_name = pred.get("detection_name", "")
                gt_name = best_gt.get("detection_name", "")
                if pred_name == gt_name:
                    label_correct += 1
            else:
                fp[idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        ap = _ap_interp(precision, recall)
        return float(ap), tp_errors, label_correct, label_total

    # ================================================================
    # 2. TRACKING EVALUATION
    # ================================================================

    def evaluate_tracking(self, tracking_path: str) -> dict:
        """
        Evaluate tracking results using MOT metrics.

        Returns dict with keys:
            MOTA, MOTP, ID_Switches, Fragmentation, …
        """
        _header("TRACKING EVALUATION")

        with open(tracking_path) as f:
            trk_data = json.load(f)
        trk_results = trk_data.get("results", {})

        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_frag = 0
        total_matches = 0
        total_dist = 0.0

        for scene_name, sp in self.scene_splits.items():
            obs_tokens = sp["observation_tokens"]
            prev_map: dict[str, str] = {}  # instance_token -> tracking_id
            prev_tracked: dict[str, bool] = {}  # instance_token -> was tracked

            for token in obs_tokens:
                gt_boxes = self._get_gt_boxes(token)
                pred_boxes = trk_results.get(token, [])
                n_gt = len(gt_boxes)
                n_pred = len(pred_boxes)
                total_gt += n_gt

                if n_gt == 0 and n_pred == 0:
                    continue
                if n_gt == 0:
                    total_fp += n_pred
                    continue
                if n_pred == 0:
                    total_fn += n_gt
                    for g in gt_boxes:
                        inst = g["instance_token"]
                        if prev_tracked.get(inst, False):
                            total_frag += 1
                        prev_tracked[inst] = False
                    continue

                # cost matrix (centre distance)
                cost = np.full((n_gt, n_pred), 1e6)
                for i, g in enumerate(gt_boxes):
                    for j, p in enumerate(pred_boxes):
                        cost[i, j] = np.linalg.norm(
                            np.array(g["translation"][:2]) - np.array(p["translation"][:2])
                        )

                ri, ci = linear_sum_assignment(cost)

                cur_map: dict[str, str] = {}
                matched_gt: set[int] = set()
                matched_pr: set[int] = set()

                for r, c in zip(ri, ci):
                    if cost[r, c] < TRACKING_MATCH_THRESHOLD:
                        inst = gt_boxes[r]["instance_token"]
                        tid = pred_boxes[c].get("tracking_id", f"unk_{c}")
                        matched_gt.add(r)
                        matched_pr.add(c)
                        cur_map[inst] = tid
                        total_matches += 1
                        total_dist += cost[r, c]
                        if inst in prev_map and prev_map[inst] != tid:
                            total_idsw += 1

                total_fn += n_gt - len(matched_gt)
                total_fp += n_pred - len(matched_pr)

                for i, g in enumerate(gt_boxes):
                    inst = g["instance_token"]
                    is_tracked = i in matched_gt
                    if prev_tracked.get(inst, False) and not is_tracked:
                        total_frag += 1
                    prev_tracked[inst] = is_tracked

                prev_map = cur_map

        mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)
        motp = total_dist / max(total_matches, 1)

        print(f"{'MOTA':<25} {mota:>10.4f}")
        print(f"{'MOTP (m)':<25} {motp:>10.4f}")
        print(f"{'ID Switches':<25} {total_idsw:>10d}")
        print(f"{'Fragmentation':<25} {total_frag:>10d}")
        print(f"{'Total GT':<25} {total_gt:>10d}")
        print(f"{'Total FP':<25} {total_fp:>10d}")
        print(f"{'Total FN':<25} {total_fn:>10d}")
        print(f"{'Total Matches':<25} {total_matches:>10d}")

        return {
            "MOTA": float(mota),
            "MOTP": float(motp),
            "ID_Switches": int(total_idsw),
            "Fragmentation": int(total_frag),
            "Total_GT": int(total_gt),
            "Total_FP": int(total_fp),
            "Total_FN": int(total_fn),
            "Total_Matches": int(total_matches),
        }

    # ================================================================
    # 3. PREDICTION EVALUATION
    # ================================================================

    def evaluate_prediction(self, tracking_path: str, predict_fn) -> dict:
        """
        Evaluate trajectory prediction.

        The grading script:
        1. Finds GT instances that appear in both observation and prediction
           portions.
        2. Matches each to the closest student track in the last observation
           frame.
        3. Extracts the student's track history and calls ``predict_fn``.
        4. Compares predictions against GT positions in prediction frames.

        Returns dict with L1/L2 mean/median errors + per-step breakdown.
        """
        _header("PREDICTION EVALUATION")

        with open(tracking_path) as f:
            trk_results = json.load(f).get("results", {})

        all_l1: list[float] = []
        all_l2: list[float] = []
        per_step_l1: dict[int, list] = defaultdict(list)
        per_step_l2: dict[int, list] = defaultdict(list)
        n_targets = 0
        n_predicted = 0

        for scene_name, sp in self.scene_splits.items():
            obs_tokens = sp["observation_tokens"]
            pred_tokens = sp["prediction_tokens"]
            if not pred_tokens:
                continue
            n_future = len(pred_tokens)

            # --- GT in prediction portion (instance → step → position) ---
            pred_gt: dict[str, dict[int, list]] = defaultdict(dict)
            for step, tok in enumerate(pred_tokens):
                for g in self._get_gt_boxes(tok):
                    pred_gt[g["instance_token"]][step] = g["translation"][:2]

            # --- GT instances also in observation portion ---
            obs_instances: set[str] = set()
            for tok in obs_tokens:
                for g in self._get_gt_boxes(tok):
                    obs_instances.add(g["instance_token"])
            target_instances = obs_instances & set(pred_gt.keys())
            if not target_instances:
                continue

            # --- match targets to student tracks (last obs frame) ---
            last_tok = obs_tokens[-1]
            last_gt = self._get_gt_boxes(last_tok)
            last_pred = trk_results.get(last_tok, [])
            inst_to_track: dict[str, str] = {}

            for g in last_gt:
                inst = g["instance_token"]
                if inst not in target_instances:
                    continue
                gp = np.array(g["translation"][:2])
                best_d, best_tid = float("inf"), None
                for p in last_pred:
                    d = np.linalg.norm(gp - np.array(p["translation"][:2]))
                    if d < best_d and d < PREDICTION_MATCH_THRESHOLD:
                        best_d, best_tid = d, p.get("tracking_id")
                if best_tid is not None:
                    inst_to_track[inst] = best_tid

            # --- predict & evaluate ---
            for inst, tid in inst_to_track.items():
                n_targets += 1

                # collect track history from student results
                history: list[dict] = []
                for tok in obs_tokens:
                    sample = self.nusc.get("sample", tok)
                    ts = sample["timestamp"]
                    for p in trk_results.get(tok, []):
                        if p.get("tracking_id") == tid:
                            history.append(
                                {
                                    "x": float(p["translation"][0]),
                                    "y": float(p["translation"][1]),
                                    "vx": float(p.get("velocity", [0, 0])[0]),
                                    "vy": float(p.get("velocity", [0, 0])[1]),
                                    "timestamp": int(ts),
                                }
                            )
                            break

                if len(history) < 2:
                    continue
                history.sort(key=lambda h: h["timestamp"])

                try:
                    preds = predict_fn(history, n_future)
                except Exception as exc:
                    warnings.warn(f"predict_fn failed for track {tid}: {exc}")
                    continue
                if preds is None or len(preds) == 0:
                    continue

                n_predicted += 1
                gt_pos = pred_gt[inst]
                for step in range(min(len(preds), n_future)):
                    if step not in gt_pos:
                        continue
                    pp = np.array(preds[step][:2], dtype=float)
                    gp = np.array(gt_pos[step], dtype=float)
                    l1 = float(np.sum(np.abs(pp - gp)))
                    l2 = float(np.linalg.norm(pp - gp))
                    all_l1.append(l1)
                    all_l2.append(l2)
                    per_step_l1[step].append(l1)
                    per_step_l2[step].append(l2)

        # ---- aggregate ----
        if not all_l1:
            print("No prediction targets matched or all predictions failed.")
            return {
                "L1_mean": float("inf"),
                "L2_mean": float("inf"),
                "L1_median": float("inf"),
                "L2_median": float("inf"),
                "n_targets": n_targets,
                "n_predicted": 0,
            }

        metrics = {
            "L1_mean": float(np.mean(all_l1)),
            "L2_mean": float(np.mean(all_l2)),
            "L1_median": float(np.median(all_l1)),
            "L2_median": float(np.median(all_l2)),
            "n_targets": n_targets,
            "n_predicted": n_predicted,
        }

        print(f"{'L1 Error (Mean)':<25} {metrics['L1_mean']:>10.4f} m")
        print(f"{'L2 Error (Mean)':<25} {metrics['L2_mean']:>10.4f} m")
        print(f"{'L1 Error (Median)':<25} {metrics['L1_median']:>10.4f} m")
        print(f"{'L2 Error (Median)':<25} {metrics['L2_median']:>10.4f} m")
        print(f"{'Targets Found':<25} {n_targets:>10d}")
        print(f"{'Targets Predicted':<25} {n_predicted:>10d}")

        if per_step_l2:
            print(f"Per-step L2 Error (mean):")
            for step in sorted(per_step_l2.keys()):
                print(f"Step {step + 1:>2d}:  {np.mean(per_step_l2[step]):>8.4f} m")

        metrics["per_step_l1"] = {int(k): float(np.mean(v)) for k, v in per_step_l1.items()}
        metrics["per_step_l2"] = {int(k): float(np.mean(v)) for k, v in per_step_l2.items()}
        return metrics

    # ================================================================
    # Full pipeline
    # ================================================================

    def run_full_evaluation(
        self,
        detection_path: str,
        tracking_path: str,
        predict_fn=None,
    ) -> dict:
        """Run the complete pipeline evaluation and return a report dict."""
        _header("FULL PIPELINE EVALUATION", char="#")
        report: dict = {}

        # 1 — Detection
        if detection_path and os.path.exists(detection_path):
            report["detection"] = self.evaluate_detection(detection_path)
        else:
            print(f"[SKIP] Detection results not found: {detection_path}")
            report["detection"] = None

        # 2 — Tracking
        if tracking_path and os.path.exists(tracking_path):
            report["tracking"] = self.evaluate_tracking(tracking_path)
        else:
            print(f"[SKIP] Tracking results not found: {tracking_path}")
            report["tracking"] = None

        # 3 — Prediction
        if tracking_path and os.path.exists(tracking_path) and predict_fn is not None:
            report["prediction"] = self.evaluate_prediction(tracking_path, predict_fn)
        else:
            print("[SKIP] Prediction (missing tracking results or predict_fn)")
            report["prediction"] = None

        # ---- Summary ----
        _header("EVALUATION SUMMARY", char="#")
        if report["detection"]:
            d = report["detection"]
            print(f"  Detection  AP  = {d['AP']:.4f}   NDS = {d['NDS']:.4f}   Label Acc = {d['label_accuracy']:.4f}")
        if report["tracking"]:
            t = report["tracking"]
            print(f"Tracking   MOTA = {t['MOTA']:.4f}   MOTP = {t['MOTP']:.4f} m")
        if report["prediction"]:
            p = report["prediction"]
            print(f"Prediction L2   = {p['L2_mean']:.4f} m (mean)  {p['L2_median']:.4f} m (median)")

        return report


# ====================================================================
# Internal helpers
# ====================================================================


def _header(title: str, char: str = "="):
    print(f"\n{title}\n")


def _ap_interp(precision: np.ndarray, recall: np.ndarray) -> float:
    """All-point AP interpolation (area under precision–recall envelope)."""
    if len(precision) == 0:
        return 0.0
    rec = np.concatenate([[0.0], recall, [1.0]])
    prec = np.concatenate([[1.0], precision, [0.0]])
    # monotonically decreasing precision
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum(np.diff(rec) * prec[1:]))


# ====================================================================
# Standalone CLI
# ====================================================================

if __name__ == "__main__":
    import argparse
    import importlib
    import importlib.util

    parser = argparse.ArgumentParser(
        description="CS5493 — Evaluate autonomous-driving perception pipeline"
    )
    parser.add_argument("--dataroot", type=str, required=True, help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes version")
    parser.add_argument(
        "--detection",
        type=str,
        default="submissions/detection_results.json",
        help="Path to detection results JSON",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        default="submissions/tracking_results.json",
        help="Path to tracking results JSON",
    )
    parser.add_argument(
        "--predict_module",
        type=str,
        default="src.prediction.predictor",
        help="Dotted module path containing predict_trajectory()",
    )
    parser.add_argument(
        "--observation_ratio",
        type=float,
        default=0.75,
        help="Fraction of frames used for observation",
    )
    args = parser.parse_args()

    # nuscenes import here so --help works without the package
    from nuscenes.nuscenes import NuScenes

    print(f"Loading nuScenes {args.version} from {args.dataroot} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # load prediction function
    predict_fn = None
    try:
        # allow both dotted module and file path
        if os.path.isfile(args.predict_module.replace(".", "/") + ".py"):
            # convert dotted path → importable
            mod = importlib.import_module(args.predict_module)
        else:
            mod = importlib.import_module(args.predict_module)
        predict_fn = getattr(mod, "predict_trajectory", None)
        if predict_fn:
            print(f"Loaded predict_trajectory from {args.predict_module}")
    except Exception as exc:
        print(f"Warning: could not load prediction module: {exc}")

    grader = ProjectGrader(nusc, observation_ratio=args.observation_ratio)
    report = grader.run_full_evaluation(
        detection_path=args.detection,
        tracking_path=args.tracking,
        predict_fn=predict_fn,
    )

    # persist report
    os.makedirs("submissions", exist_ok=True)
    report_path = "submissions/evaluation_report.json"
    serialisable = {}
    for key, val in report.items():
        if val is not None:
            serialisable[key] = {k: v for k, v in val.items() if not callable(v)}
    with open(report_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"Report saved to {report_path}")
