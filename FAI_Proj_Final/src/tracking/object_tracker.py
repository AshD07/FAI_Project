"""
Multi-Object Tracker (MOT) managing active Kalman filter tracks and future trajectory predictions (Stage 3).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import KalmanConfig
from src.config import ArenaConfig
from src.environment.game_engine import wrapped_distance
from src.geometry import extrapolate
from src.sensors.radar_sensor import RadarDetection, SensorObservation
from src.tracking.kalman_filter import KalmanFilter2D


@dataclass
class TrackedEntity:
    """
    Filtered, state-estimated entity provided to downstream AI systems (Rules, A*, CSP).
    """
    track_id: int
    entity_type: str                   # "asteroid" or "ship"
    position: np.ndarray               # Filtered [x, y]
    velocity: np.ndarray               # Filtered [vx, vy]
    speed: float
    estimated_radius: float
    size_category: int
    is_confirmed: bool
    true_id: Optional[int] = None      # Debug reference only
    future_trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def predict_at(self, seconds_ahead: float, arena_width: float = 1000.0, arena_height: float = 800.0) -> np.ndarray:
        """Extrapolate future position using filtered velocity, reflecting off the arena walls."""
        pos, _ = extrapolate(
            self.position, self.velocity, seconds_ahead, arena_width, arena_height,
            wrap=ArenaConfig().toroidal_wrap, margin=self.estimated_radius,
        )
        return pos


class Track:
    """Internal Kalman tracking state for a single persistent physical object."""
    _next_id = 1

    def __init__(
        self,
        detection: RadarDetection,
        config: Optional[KalmanConfig] = None,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
    ):
        self.track_id = Track._next_id
        Track._next_id += 1

        self.config = config or KalmanConfig()
        self.entity_type = detection.entity_type
        self.size_category = detection.size_category
        self.estimated_radius = detection.estimated_radius
        self.true_id = detection.true_id

        # Initialize Kalman Filter with first measured position and velocity
        self.kf = KalmanFilter2D(
            initial_pos=detection.measured_position,
            initial_vel=detection.measured_velocity,
            config=self.config,
            arena_width=arena_width,
            arena_height=arena_height,
            margin=detection.estimated_radius,
        )

        self.hits: int = 1
        self.misses: int = 0
        self.last_seen: float = detection.timestamp
        self.is_confirmed: bool = False

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return self.kf.speed

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)

    def update(self, detection: RadarDetection) -> None:
        self.kf.update(detection.measured_position)
        self.hits += 1
        self.misses = 0
        self.last_seen = detection.timestamp
        self.estimated_radius = detection.estimated_radius
        self.size_category = detection.size_category
        self.true_id = detection.true_id
        if self.hits >= 2:
            self.is_confirmed = True

    def mark_missed(self) -> None:
        self.misses += 1


class MultiObjectTracker:
    """
    Maintains all active tracks across the arena.
    Performs Hungarian data association between noisy radar detections and existing tracks.
    Eliminates false tracks and extrapolates 2.0s trajectory predictions.
    """

    def __init__(
        self,
        config: Optional[KalmanConfig] = None,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
    ):
        self.config = config or KalmanConfig()
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.last_timestamp: Optional[float] = None

        self.tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        """Forget all tracks (call at the start of every new match/episode)."""
        self.tracks.clear()
        self.last_timestamp = None

    def update(self, obs: SensorObservation) -> List[TrackedEntity]:
        """
        Process incoming radar sensor observation:
        1. Predict active tracks forward by dt
        2. Associate detections with tracks via Hungarian algorithm
        3. Update matched tracks, spawn new tracks, prune stale tracks
        4. Generate future trajectory predictions
        """
        current_time = obs.timestamp
        dt = (current_time - self.last_timestamp) if self.last_timestamp is not None else (1.0 / 30.0)
        if dt <= 0.0:
            # Clock went backwards => a new episode started without reset(); drop stale tracks
            self.reset()
            dt = 1.0 / 30.0
        self.last_timestamp = current_time

        # 1. Prediction step for all tracks
        for track in self.tracks.values():
            track.predict(dt)

        # Combine all incoming radar detections
        detections = obs.radar_asteroids + obs.radar_rivals

        # 2. Data Association
        active_track_ids = list(self.tracks.keys())
        num_tracks = len(active_track_ids)
        num_dets = len(detections)

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            # Build cost matrix (Euclidean distance with wrap)
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float64)
            for i, tid in enumerate(active_track_ids):
                track_pos = self.tracks[tid].position
                for j, det in enumerate(detections):
                    det_pos = np.array(det.measured_position, dtype=np.float64)
                    dist, _ = wrapped_distance(track_pos, det_pos, self.arena_width, self.arena_height)
                    cost_matrix[i, j] = dist

            # Hungarian algorithm optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                dist = cost_matrix[r, c]
                # Distance gating: reject matching if outside threshold
                if dist <= self.config.gate_threshold:
                    tid = active_track_ids[r]
                    self.tracks[tid].update(detections[c])
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # 3. Handle unmatched tracks (missed detections)
        stale_track_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.mark_missed()
                if track.misses > self.config.max_missed_frames:
                    stale_track_ids.append(tid)

        for tid in stale_track_ids:
            del self.tracks[tid]

        # 4. Handle unmatched detections (spawn new candidate tracks)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = Track(
                    detection=det,
                    config=self.config,
                    arena_width=self.arena_width,
                    arena_height=self.arena_height,
                )
                self.tracks[new_track.track_id] = new_track

        # 5. Build output TrackedEntity list with 2.0s future predictions
        tracked_entities: List[TrackedEntity] = []
        horizon = self.config.prediction_horizon

        for track in self.tracks.values():
            # Generate future trajectory (2.0s lookahead, 20 intervals)
            traj = track.kf.predict_trajectory(horizon=horizon, steps=20)

            tracked_entities.append(
                TrackedEntity(
                    track_id=track.track_id,
                    entity_type=track.entity_type,
                    position=track.position,
                    velocity=track.velocity,
                    speed=track.speed,
                    estimated_radius=track.estimated_radius,
                    size_category=track.size_category,
                    is_confirmed=track.is_confirmed,
                    true_id=track.true_id,
                    future_trajectory=traj,
                )
            )

        return tracked_entities

    def get_asteroids(self, entities: List[TrackedEntity]) -> List[TrackedEntity]:
        """Convenience filter for tracked asteroids."""
        return [e for e in entities if e.entity_type == "asteroid" and e.is_confirmed]

    def get_rivals(self, entities: List[TrackedEntity]) -> List[TrackedEntity]:
        """Convenience filter for tracked rival ships."""
        return [e for e in entities if e.entity_type == "ship"]
