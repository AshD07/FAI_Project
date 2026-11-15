"""
Stage 10: Multi-Scenario Benchmark Suite and Baseline Agent Implementations.
Defines standardized testing scenarios (Accuracy, Swarm, Rival Adversary, High-Speed)
and baseline controllers for comparative evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.agent.autonomous_pilot import AutonomousPilot
from src.config import GameConfig
from src.environment.asteroid import Asteroid
from src.environment.game_engine import GameEngine, wrapped_distance
from src.environment.ship import heading_to_vector
from src.optimization.genetic_algorithm import AgentGenome
from src.sensors.radar_sensor import RadarSensor, SensorObservation


@dataclass
class BenchmarkScenario:
    """Definition of a reproducible game benchmark test scenario."""
    name: str
    description: str
    num_asteroids: int
    asteroid_sizes: List[int]
    asteroid_speed_multiplier: float
    has_rival: bool
    rival_behavior: str               # "none", "passive", "aggressive"
    time_limit: float = 30.0


# Standardized Scenarios for Evaluation
BENCHMARK_SCENARIOS = [
    BenchmarkScenario(
        name="1_Target_Accuracy",
        description="Low-density field testing ballistic lead-angle intercept precision and CSP constraint gating.",
        num_asteroids=4,
        asteroid_sizes=[3, 3, 2, 2],
        asteroid_speed_multiplier=1.0,
        has_rival=False,
        rival_behavior="none",
        time_limit=30.0,
    ),
    BenchmarkScenario(
        name="2_Hazard_Swarm",
        description="High-density asteroid field testing local A* path planning and emergency collision avoidance.",
        num_asteroids=12,
        asteroid_sizes=[3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1],
        asteroid_speed_multiplier=1.1,
        has_rival=False,
        rival_behavior="none",
        time_limit=30.0,
    ),
    BenchmarkScenario(
        name="3_Passive_Rival",
        description="Standard asteroid field with a drifting rival spacecraft testing target prioritization.",
        num_asteroids=6,
        asteroid_sizes=[3, 3, 2, 2, 1, 1],
        asteroid_speed_multiplier=1.0,
        has_rival=True,
        rival_behavior="passive",
        time_limit=30.0,
    ),
    BenchmarkScenario(
        name="4_Adversarial_Duel",
        description="Active combat against an aggressive rival bot testing 2-ply Minimax tactical decision making.",
        num_asteroids=6,
        asteroid_sizes=[3, 3, 2, 2, 2, 2],
        asteroid_speed_multiplier=1.0,
        has_rival=True,
        rival_behavior="aggressive",
        time_limit=30.0,
    ),
    BenchmarkScenario(
        name="5_High_Speed_Debris",
        description="Fast-moving debris field (1.8x speed) testing Kalman prediction and reflex timing.",
        num_asteroids=7,
        asteroid_sizes=[3, 2, 2, 2, 1, 1, 1],
        asteroid_speed_multiplier=1.8,
        has_rival=False,
        rival_behavior="none",
        time_limit=30.0,
    ),
]


class BaseController:
    """Base interface for evaluation controllers."""
    def decide(self, obs: SensorObservation) -> Tuple[float, float, bool, bool]:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        """Clear per-match state. Stateless baselines need not override this."""
        return None


class RandomController(BaseController):
    """Baseline 1: Random random actions with continuous firing."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        # Re-seed per match so each trial is reproducible independently of run order
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def decide(self, obs: SensorObservation) -> Tuple[float, float, bool, bool]:
        turn = float(self.rng.uniform(-180.0, 180.0))
        thrust = float(self.rng.uniform(0.0, 1.0))
        shoot = bool(self.rng.uniform(0.0, 1.0) > 0.3)
        drop_mine = bool(self.rng.uniform(0.0, 1.0) > 0.95)
        return turn, thrust, shoot, drop_mine


class NaiveReactiveController(BaseController):
    """
    Baseline 2: Naive reactive controller without Kalman filtering, A*, or CSP constraints.
    Steers directly toward nearest raw radar return and fires continuously whenever ammo is available.
    """
    def __init__(self, arena_width: float = 1000.0, arena_height: float = 800.0):
        self.arena_width = arena_width
        self.arena_height = arena_height

    def decide(self, obs: SensorObservation) -> Tuple[float, float, bool, bool]:
        telemetry = obs.telemetry
        ship_pos = np.array(telemetry.position, dtype=np.float64)

        if not obs.radar_asteroids:
            return 45.0, 0.2, False, False

        # Find closest raw radar detection (with noise)
        closest_det = min(
            obs.radar_asteroids,
            key=lambda d: wrapped_distance(ship_pos, np.array(d.measured_position), self.arena_width, self.arena_height)[0]
        )
        _, rel_vec = wrapped_distance(ship_pos, np.array(closest_det.measured_position), self.arena_width, self.arena_height)
        desired_heading = float(np.degrees(np.arctan2(rel_vec[0], -rel_vec[1])) % 360.0)

        angle_diff = (desired_heading - telemetry.heading + 180.0) % 360.0 - 180.0
        turn_cmd = float(np.clip(angle_diff * 4.0, -180.0, 180.0))
        thrust_cmd = 0.5 if abs(angle_diff) < 45.0 else 0.0
        # Continuous unconstrained shooting
        shoot_cmd = True
        drop_mine_cmd = False
        return turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd


class RivalBot:
    """Rival bot implementation for benchmark scenarios."""
    def __init__(self, behavior: str = "passive", seed: int = 42):
        self.behavior = behavior
        self.rng = np.random.default_rng(seed)

    def decide(self, ship_state: dict, game_state: dict) -> Tuple[float, float, bool, bool]:
        if self.behavior == "none":
            return 0.0, 0.0, False, False

        pos = np.array(ship_state["position"])
        heading = ship_state["heading"]

        if self.behavior == "passive":
            # Gentle wandering and intermittent shooting
            turn = 25.0
            thrust = 0.4 if (int(game_state["time"] * 30) % 70 < 25) else 0.0
            shoot = (int(game_state["time"] * 30) % 18 == 0)
            return turn, thrust, shoot, False

        if self.behavior == "aggressive":
            # Actively track and hunt the player ship
            player = next((s for s in game_state["ships"] if s["id"] == 1), None)
            if player and player["is_alive"]:
                p_pos = np.array(player["position"])
                _, rel_vec = wrapped_distance(pos, p_pos, 1000.0, 800.0)
                desired_h = float(np.degrees(np.arctan2(rel_vec[0], -rel_vec[1])) % 360.0)
                angle_diff = (desired_h - heading + 180.0) % 360.0 - 180.0
                turn = float(np.clip(angle_diff * 4.0, -180.0, 180.0))
                thrust = 0.8 if abs(angle_diff) < 40.0 else 0.2
                shoot = abs(angle_diff) < 20.0
                return turn, thrust, shoot, False

        return 0.0, 0.0, False, False


def setup_scenario_engine(scenario: BenchmarkScenario, seed: int = 42) -> Tuple[GameEngine, RadarSensor, RivalBot]:
    """Builds and populates a GameEngine instance matching scenario specifications."""
    config = GameConfig()
    engine = GameEngine(config=config, seed=seed)
    engine.time_limit = scenario.time_limit
    engine.asteroids.clear()

    # Spawn designated asteroids
    rng = np.random.default_rng(seed)
    for size in scenario.asteroid_sizes:
        ast = engine.spawn_random_asteroid(size=size, min_dist_from_ships=160.0)
        ast.velocity *= scenario.asteroid_speed_multiplier

    # Rival configuration
    rival_bot = None
    if scenario.has_rival:
        engine.add_rival_ship(position=(750.0, 400.0), heading=180.0)
        rival_bot = RivalBot(behavior=scenario.rival_behavior, seed=seed + 10)

    sensor = RadarSensor(config=config.sensor, seed=seed + 20)
    return engine, sensor, rival_bot
