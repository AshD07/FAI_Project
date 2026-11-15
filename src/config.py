"""
Global configuration and physics constants for the AI Piloted Asteroid Game.
Covers Stages 1-3 (Environment, Sensors, Kalman Filter Tracking).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ArenaConfig:
    width: float = 1000.0
    height: float = 800.0
    toroidal_wrap: bool = False        # Screen boundaries are hard walls; ship cannot exit screen
    wall_buffer: float = 65.0          # Boundary safety buffer for wall avoidance reflexes


@dataclass
class ShipConfig:
    radius: float = 16.0
    mass: float = 100.0
    max_speed: float = 300.0
    thrust_power: float = 400.0        # Acceleration in px/s^2 when thrusting
    linear_drag: float = 0.05          # Subtle space resistance (or 0.0 for pure Newtonian drift)
    turn_rate: float = 180.0           # Max turn rate in deg/s
    initial_lives: int = 1             # Single life only (permadeath survival mode)
    respawn_invulnerability: float = 2.5 # Seconds of shield/invulnerability after respawn
    
    # Weapons
    bullet_cooldown: float = 0.15      # Seconds between shots (~6.6 shots/sec)
    bullet_speed: float = 750.0        # Bullet speed in px/s relative to ship orientation
    bullet_lifetime: float = 1.2       # Seconds before bullet expires
    bullet_radius: float = 3.0
    bullet_mass: float = 1.0           # Projectile mass for XFC inelastic momentum transfer
    initial_ammo: int = 200            # Total ammo capacity
    
    # Mines (Disabled for solo survival mode)
    initial_mines: int = 0             # No mines
    mine_deploy_cooldown: float = 1.0  # Seconds between dropping mines
    mine_fuse_time: float = 3.0        # Countdown until automatic detonation
    mine_blast_radius: float = 120.0   # AOE blast radius in px
    mine_radius: float = 8.0


@dataclass
class AsteroidConfig:
    # XFC Guidelines: 4 Size categories (4: Large, 3: Medium-Large, 2: Medium, 1: Small)
    # Radii: size * 8.0 px (Size 4: 32px, Size 3: 24px, Size 2: 16px, Size 1: 8px)
    radii = {4: 32.0, 3: 24.0, 2: 16.0, 1: 8.0}
    # Masses: 0.25 * pi * radius^2
    masses = {
        4: 0.25 * 3.1415926535 * (32.0 ** 2),
        3: 0.25 * 3.1415926535 * (24.0 ** 2),
        2: 0.25 * 3.1415926535 * (16.0 ** 2),
        1: 0.25 * 3.1415926535 * (8.0 ** 2),
    }
    # Speeds: max_speed = 60.0 * (2.0 + (4.0 - size) / 4.0)
    speeds = {
        4: (30.0, 120.0),
        3: (40.0, 135.0),
        2: (50.0, 150.0),
        1: (60.0, 165.0),
    }
    split_count: int = 3               # Number of fragments created when split (XFC guideline: 3 children)
    score_values = {4: 100, 3: 50, 2: 25, 1: 10}


@dataclass
class SensorConfig:
    # Sensor noise characteristics for Stage 2
    radar_noise_std_pos: float = 12.0  # Standard deviation of radar position noise (pixels)
    radar_noise_std_vel: float = 3.5   # Standard deviation of velocity measurement noise
    detection_range: float = 1200.0    # Maximum radar range (detects across whole arena by default)
    detection_prob: float = 0.98       # Probability of detection per tick


@dataclass
class KalmanConfig:
    # Kalman Filter parameters for Stage 3
    process_noise_pos: float = 0.5     # Acceleration variance / process noise for position
    process_noise_vel: float = 2.0     # Process noise for velocity
    measurement_noise_pos: float = 12.0# Expected radar position noise sigma
    gate_threshold: float = 50.0       # Gating distance for track association
    max_missed_frames: int = 15        # Drop track after missed updates (~0.5s at 30 FPS)
    prediction_horizon: float = 2.0    # Lookahead time in seconds for future prediction


@dataclass
class GameConfig:
    fps: int = 30
    dt: float = 1.0 / 30.0
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    ship: ShipConfig = field(default_factory=ShipConfig)
    asteroid: AsteroidConfig = field(default_factory=AsteroidConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
