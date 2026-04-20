import numpy as np


class KalmanFilter3D:
    """
    Linear Kalman Filter with constant-velocity motion model for 3-D
    object tracking.

    The filter maintains:
        x  (6,)    state estimate  [px, py, pz, vx, vy, vz]
        P  (6, 6)  state covariance

    Motion model (discrete, dt seconds):
        x_{k+1} = F · x_k + w       w ~ N(0, Q)

    Measurement model:
        z_k     = H · x_k + v       v ~ N(0, R)
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        dt: float = 0.5,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 0.5,
    ):
        """
        Initialise the filter.

        Args:
            initial_position: (3,) initial [px, py, pz].
            dt:               Time step between frames (seconds).
                              nuScenes keyframes are at 2 Hz → dt = 0.5 s.
            process_noise_std:     Standard deviation for process noise.
            measurement_noise_std: Standard deviation for measurement noise.
        """

        self.dt = dt
        initial_position = np.asarray(initial_position, dtype=float).reshape(3)

        # State: [px, py, pz, vx, vy, vz]
        self.x = np.zeros(6, dtype=float)
        self.x[:3] = initial_position

        # Constant-velocity dynamics.
        self.F = np.eye(6, dtype=float)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # We only observe position.
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        pos_var = float(measurement_noise_std) ** 2
        vel_var = max(float(process_noise_std) ** 2, 1e-3)
        self.P = np.diag([pos_var, pos_var, pos_var, 10.0, 10.0, 10.0]).astype(float)

        # Constant-velocity process noise with white acceleration.
        q = float(process_noise_std) ** 2
        q_block = np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        ) * q
        self.Q = np.zeros((6, 6), dtype=float)
        self.Q[np.ix_([0, 3], [0, 3])] = q_block
        self.Q[np.ix_([1, 4], [1, 4])] = q_block
        self.Q[np.ix_([2, 5], [2, 5])] = q_block

        self.R = np.eye(3, dtype=float) * pos_var

    def predict(self) -> np.ndarray:
        """
        Prediction step: propagate state and covariance forward by dt.

            x⁻ = F · x
            P⁻ = F · P · Fᵀ + Q

        Returns:
            predicted state  (6,)
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step: correct the prediction with a new measurement.

            y  = z - H · x⁻               (innovation)
            S  = H · P⁻ · Hᵀ + R          (innovation covariance)
            K  = P⁻ · Hᵀ · S⁻¹            (Kalman gain)
            x  = x⁻ + K · y
            P  = (I - K · H) · P⁻

        Args:
            measurement: (3,) observed position [px, py, pz].

        Returns:
            updated state  (6,)
        """
        measurement = np.asarray(measurement, dtype=float).reshape(3)

        innovation = measurement - self.H @ self.x
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)

        self.x = self.x + kalman_gain @ innovation
        identity = np.eye(6, dtype=float)
        self.P = (identity - kalman_gain @ self.H) @ self.P
        return self.x.copy()

    # ----------------------------------------------------------------
    # Convenience accessors
    # ----------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Current estimated position (3,)."""
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity (3,)."""
        return self.x[3:].copy()
