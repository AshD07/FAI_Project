"""
Radar Sensor module: Telemetry acquisition and noisy radar measurement generator (Stage 2).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import SensorConfig
from src.environment.game_engine import GameEngine, wrapped_distance


@dataclass
class RadarDetection:
    """
    A single noisy sensor detection produced by the radar.
    Contains uncertainty (Gaussian measurement noise).
    """
    measured_position: Tuple[float, float]
    measured_velocity: Tuple[float, float]
    size_category: int                 # 3: Large, 2: Medium, 1: Small (or 0 for rival ship)
    estimated_radius: float
    distance_to_ship: float
    bearing_from_ship: float           # Bearing in degrees relative to ship
    timestamp: float
    entity_type: str = "asteroid"      # "asteroid" or "ship"
    true_id: Optional[int] = None      # Debug reference only (not used by AI)


@dataclass
class ShipTelemetry:
    """Accurate internal inertial telemetry of the observing ship."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    heading: float
    speed: float
    lives: int
    ammo: int
    mines_remaining: int
    bullet_cooldown: float
    mine_cooldown: float
    is_invulnerable: bool


@dataclass
class SensorObservation:
    """
    Complete sensory percept delivered to the AI pilot every game tick.
    """
    timestamp: float
    telemetry: ShipTelemetry
    radar_asteroids: List[RadarDetection] = field(default_factory=list)
    radar_rivals: List[RadarDetection] = field(default_factory=list)
    arena_size: Tuple[float, float] = (1000.0, 800.0)


class RadarSensor:
    """
    Simulates on-board radar scanning the environment.
    Applies realistic Gaussian noise to position and velocity observations.
    """

    def __init__(self, config: Optional[SensorConfig] = None, seed: Optional[int] = None):
        self.config = config or SensorConfig()
        self.rng = np.random.default_rng(seed)

    def observe(self, game: GameEngine, observing_ship_id: int = 1) -> SensorObservation:
        """
        Generate sensor percepts for the specified spacecraft.
        Telemetry is accurate (internal sensors); radar returns noisy estimates.
        """
        ship = game.ships.get(observing_ship_id)
        if not ship:
            raise ValueError(f"Ship ID {observing_ship_id} not found in game engine.")

        # 1. Ship Telemetry
        telemetry = ShipTelemetry(
            position=(float(ship.position[0]), float(ship.position[1])),
            velocity=(float(ship.velocity[0]), float(ship.velocity[1])),
            heading=float(ship.heading),
            speed=float(ship.speed),
            lives=ship.lives,
            ammo=ship.ammo,
            mines_remaining=ship.mines_remaining,
            bullet_cooldown=float(ship.bullet_cooldown_timer),
            mine_cooldown=float(ship.mine_cooldown_timer),
            is_invulnerable=ship.is_invulnerable,
        )

        radar_asteroids: List[RadarDetection] = []
        radar_rivals: List[RadarDetection] = []
        time = game.time
        width, height = game.arena.width, game.arena.height

        # 2. Asteroids Radar Scan
        for ast in game.asteroids:
            if not ast.is_alive:
                continue

            dist, rel_vec = wrapped_distance(ship.position, ast.position, width, height)
            if dist > self.config.detection_range:
                continue

            # Stochastic detection probability (occasional radar flicker / miss)
            if self.rng.uniform(0.0, 1.0) > self.config.detection_prob:
                continue

            # Inject Gaussian measurement noise into position
            noise_pos = self.rng.normal(0.0, self.config.radar_noise_std_pos, size=2)
            measured_pos = ast.position + noise_pos

            # Inject Gaussian measurement noise into velocity
            noise_vel = self.rng.normal(0.0, self.config.radar_noise_std_vel, size=2)
            measured_vel = ast.velocity + noise_vel

            # Calculate relative bearing from ship
            bearing_deg = float(np.degrees(np.arctan2(rel_vec[0], -rel_vec[1])) % 360.0)

            radar_asteroids.append(
                RadarDetection(
                    measured_position=(float(measured_pos[0]), float(measured_pos[1])),
                    measured_velocity=(float(measured_vel[0]), float(measured_vel[1])),
                    size_category=ast.size,
                    estimated_radius=ast.radius,
                    distance_to_ship=dist,
                    bearing_from_ship=bearing_deg,
                    timestamp=time,
                    entity_type="asteroid",
                    true_id=ast.id,
                )
            )

        # 3. Rival Spacecraft Radar Scan
        for other_ship in game.ships.values():
            if other_ship.id == observing_ship_id or not other_ship.is_alive:
                continue

            dist, rel_vec = wrapped_distance(ship.position, other_ship.position, width, height)
            if dist > self.config.detection_range:
                continue

            if self.rng.uniform(0.0, 1.0) > self.config.detection_prob:
                continue

            noise_pos = self.rng.normal(0.0, self.config.radar_noise_std_pos, size=2)
            measured_pos = other_ship.position + noise_pos

            noise_vel = self.rng.normal(0.0, self.config.radar_noise_std_vel, size=2)
            measured_vel = other_ship.velocity + noise_vel

            bearing_deg = float(np.degrees(np.arctan2(rel_vec[0], -rel_vec[1])) % 360.0)

            radar_rivals.append(
                RadarDetection(
                    measured_position=(float(measured_pos[0]), float(measured_pos[1])),
                    measured_velocity=(float(measured_vel[0]), float(measured_vel[1])),
                    size_category=0,
                    estimated_radius=other_ship.radius,
                    distance_to_ship=dist,
                    bearing_from_ship=bearing_deg,
                    timestamp=time,
                    entity_type="ship",
                    true_id=other_ship.id,
                )
            )

        return SensorObservation(
            timestamp=time,
            telemetry=telemetry,
            radar_asteroids=radar_asteroids,
            radar_rivals=radar_rivals,
            arena_size=(width, height),
        )
