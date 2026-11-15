"""
Kalman Filter implementation for 2D kinematic tracking (Stage 3).
Supports both a bounded arena (elastic wall reflection, the project default) and a
toroidal arena (wrap-around), selected via ArenaConfig.toroidal_wrap.
"""

from typing import List, Optional, Tuple
import numpy as np
from src.config import ArenaConfig, KalmanConfig
from src.geometry import extrapolate, fold_axis


class KalmanFilter2D:
    """
    Constant-Velocity (CV) Linear Kalman Filter.
    State vector: [x, y, vx, vy]^T
    Handles 2D position radar measurements with measurement noise R,
    process noise Q, and toroidal arena boundary unwrapping.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float],
        initial_vel: Tuple[float, float] = (0.0, 0.0),
        config: Optional[KalmanConfig] = None,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
        wrap: Optional[bool] = None,
        margin: float = 0.0,
    ):
        self.config = config or KalmanConfig()
        self.arena_width = arena_width
        self.arena_height = arena_height
        # Boundary model: False = bounded arena with elastic walls (matches the game engine)
        self.wrap = ArenaConfig().toroidal_wrap if wrap is None else bool(wrap)
        # Object radius: asteroids bounce when their edge (not centre) touches a wall
        self.margin = float(margin)

        # State vector: [x, y, vx, vy]^T
        self.x = np.array([initial_pos[0], initial_pos[1], initial_vel[0], initial_vel[1]], dtype=np.float64)

        # State covariance P
        pos_var = self.config.measurement_noise_pos ** 2
        vel_var = 100.0  # Initial velocity uncertainty
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var])

        # Measurement matrix H: maps [x, y, vx, vy] -> [x, y]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (self.config.measurement_noise_pos ** 2)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:]))

    def _make_Q(self, dt: float) -> np.ndarray:
        """
        Discrete White Noise Acceleration (DWNA) process noise covariance matrix.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q_pos = self.config.process_noise_pos ** 2
        q_vel = self.config.process_noise_vel ** 2

        Q = np.array([
            [dt4 * q_pos, 0.0, dt3 * q_pos, 0.0],
            [0.0, dt4 * q_pos, 0.0, dt3 * q_pos],
            [dt3 * q_pos, 0.0, dt2 * q_vel, 0.0],
            [0.0, dt3 * q_pos, 0.0, dt2 * q_vel],
        ], dtype=np.float64)
        return Q

    def predict(self, dt: float) -> np.ndarray:
        """
        Kalman prediction step:
        x_hat = F * x
        P = F * P * F^T + Q
        """
        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Predict state & covariance
        self.x = F @ self.x
        Q = self._make_Q(dt)
        self.P = F @ self.P @ F.T + Q

        self._apply_boundary()
        return self.x.copy()

    def _apply_boundary(self) -> None:
        """
        Keep the state consistent with the arena boundary model.
        Toroidal: wrap position.  Bounded: reflect position and flip the velocity
        component (exactly what Asteroid.update does on wall contact).  The covariance
        is transformed with the same Jacobian J so that P stays consistent.
        """
        if self.wrap:
            self.x[0] = self.x[0] % self.arena_width
            self.x[1] = self.x[1] % self.arena_height
            return

        J = np.eye(4, dtype=np.float64)
        x, vx, fx = fold_axis(self.x[0], self.x[2], self.margin, self.arena_width - self.margin)
        y, vy, fy = fold_axis(self.x[1], self.x[3], self.margin, self.arena_height - self.margin)
        if fx:
            J[0, 0] = -1.0
            J[2, 2] = -1.0
        if fy:
            J[1, 1] = -1.0
            J[3, 3] = -1.0
        self.x[:] = [x, y, vx, vy]
        if fx or fy:
            self.P = J @ self.P @ J.T

    def update(self, measured_pos: Tuple[float, float]) -> None:
        """
        Kalman measurement update step with boundary unwrapping:
        y = z - H * x
        S = H * P * H^T + R
        K = P * H^T * inv(S)
        x = x + K * y
        P = (I - K * H) * P
        """
        z = np.array(measured_pos, dtype=np.float64)
        expected_z = self.H @ self.x

        # Calculate measurement residual (innovation)
        dy = z - expected_z

        # Unwrap innovation across toroidal borders to prevent velocity spikes
        # (only meaningful in a wrap-around arena)
        if self.wrap:
            if dy[0] > self.arena_width / 2.0:
                dy[0] -= self.arena_width
            elif dy[0] < -self.arena_width / 2.0:
                dy[0] += self.arena_width

            if dy[1] > self.arena_height / 2.0:
                dy[1] -= self.arena_height
            elif dy[1] < -self.arena_height / 2.0:
                dy[1] += self.arena_height

        # Innovation covariance S
        S = self.H @ self.P @ self.H.T + self.R

        # Optimal Kalman gain K
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ dy

        # Ensure updated position stays in bounds
        if self.wrap:
            self.x[0] = self.x[0] % self.arena_width
            self.x[1] = self.x[1] % self.arena_height
        else:
            self.x[0] = float(np.clip(self.x[0], 0.0, self.arena_width))
            self.x[1] = float(np.clip(self.x[1], 0.0, self.arena_height))

        # Joseph form covariance update for numerical stability: (I - KH) P (I - KH)^T + K R K^T
        I = np.eye(4, dtype=np.float64)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def predict_future(self, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict position and velocity at t + horizon seconds into the future,
        honouring the arena boundary model (wall reflection or wrap-around).
        Returns (predicted_position, predicted_velocity).
        """
        return extrapolate(
            self.x[:2], self.x[2:], horizon,
            self.arena_width, self.arena_height,
            wrap=self.wrap, margin=self.margin,
        )

    def predict_trajectory(self, horizon: float = 2.0, steps: int = 20) -> List[Tuple[float, float]]:
        """
        Generates a discrete sequence of future coordinates from t to t + horizon.
        Essential for A* dynamic obstacle planning and lead-angle aiming.
        """
        dt_step = horizon / max(1, steps)
        trajectory = []
        for i in range(1, steps + 1):
            tau = i * dt_step
            pos, _ = self.predict_future(tau)
            trajectory.append((float(pos[0]), float(pos[1])))
        return trajectory
