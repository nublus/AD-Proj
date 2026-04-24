import numpy as np

def predict_trajectory(
    track_history: list[dict],
    num_future_steps: int,
) -> list[tuple[float, float]]:
    """
    Predict future trajectory of an object.

    This is the **entry point** called by ``grading/evaluate.py``.
    Internally you may dispatch to any method you wish.

    Args:
        track_history:    List of dicts, sorted by timestamp, each with:
                            - 'x':  float  (global x position)
                            - 'y':  float  (global y position)
                            - 'vx': float  (x velocity)
                            - 'vy': float  (y velocity)
                            - 'timestamp': int  (microseconds)
        num_future_steps: Number of future positions to predict.

    Returns:
        List of (x, y) tuples — one per future step.
    """
    return predict_improved(track_history, num_future_steps)


# ====================================================================
# Method 1:  Constant Velocity (CV) Baseline
# ====================================================================

def predict_cv(
    track_history: list[dict],
    num_future_steps: int,
) -> list[tuple[float, float]]:
    """
    Constant-velocity prediction.

    Uses the last observed velocity to linearly extrapolate future
    positions:

        x_{t+k} = x_t + v_x · k · Δt
        y_{t+k} = y_t + v_y · k · Δt

    where Δt = 0.5 s (nuScenes 2 Hz keyframe rate).
    """
    if len(track_history) == 0:
        return [(0.0, 0.0)] * num_future_steps

    dt = 0.5  # seconds between keyframes

    last = track_history[-1]
    x0 = last["x"]
    y0 = last["y"]

    # --- estimate velocity -------------------------------------------------
    # Prefer the stored velocity; fall back to finite-difference if zero.
    vx = last.get("vx", 0.0)
    vy = last.get("vy", 0.0)

    if abs(vx) < 1e-6 and abs(vy) < 1e-6 and len(track_history) >= 2:
        prev = track_history[-2]
        t_diff = (last["timestamp"] - prev["timestamp"]) / 1e6  # -> seconds
        if t_diff > 0:
            vx = (last["x"] - prev["x"]) / t_diff
            vy = (last["y"] - prev["y"]) / t_diff

    predictions = []
    for k in range(1, num_future_steps + 1):
        px = x0 + vx * k * dt
        py = y0 + vy * k * dt
        predictions.append((px, py))

    return predictions


# ====================================================================
# Method 2:  Improved Prediction  (TODO — student implementation)
# ====================================================================

def predict_improved(
    track_history: list[dict],
    num_future_steps: int,
) -> list[tuple[float, float]]:
    """
    Improved trajectory prediction.

    Suggested approaches (pick ≥ 1):
        • Kalman-Filter based prediction (reuse your KF from tracking).
        • Polynomial / spline fitting on the position history.
        • Acceleration-aware model (estimate acceleration from Δv).
        • Category-aware prediction (different model per object type).

    Args / Returns: same as ``predict_trajectory``.
    """
    if len(track_history) == 0:
        return [(0.0, 0.0)] * num_future_steps
    hist = sorted(track_history, key=lambda h: h["timestamp"])
    last = hist[-1]
    if len(hist) == 1:
        return [(float(last["x"]), float(last["y"]))] * num_future_steps

    dt = 0.5
    vel_candidates: list[tuple[float, float]] = []

    # Blend tracker-reported velocity with short finite differences to
    # stay robust when the history is short or slightly noisy.
    for h in hist[-4:]:
        vx = float(h.get("vx", 0.0))
        vy = float(h.get("vy", 0.0))
        if np.isfinite(vx) and np.isfinite(vy):
            vel_candidates.append((vx, vy))

    for i in range(max(1, len(hist) - 4), len(hist)):
        t_prev = float(hist[i - 1]["timestamp"]) / 1e6
        t_cur = float(hist[i]["timestamp"]) / 1e6
        dt_i = t_cur - t_prev
        if dt_i <= 1e-6:
            continue
        vel_candidates.append(
            (
                (float(hist[i]["x"]) - float(hist[i - 1]["x"])) / dt_i,
                (float(hist[i]["y"]) - float(hist[i - 1]["y"])) / dt_i,
            )
        )

    if vel_candidates:
        vel_array = np.asarray(vel_candidates[-5:], dtype=float)
        vx = float(np.median(vel_array[:, 0]))
        vy = float(np.median(vel_array[:, 1]))
    else:
        vx = vy = 0.0

    x0 = float(last["x"])
    y0 = float(last["y"])
    predictions = []
    for step in range(1, num_future_steps + 1):
        # Damp long-horizon motion to avoid runaway extrapolation.
        scale = max(0.55, 1.0 - 0.05 * step)
        horizon = dt * step * scale
        predictions.append((x0 + vx * horizon, y0 + vy * horizon))

    return predictions
