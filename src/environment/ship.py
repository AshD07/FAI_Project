"""
Spacecraft entity, physics, health, weapons, and control actuators.
"""

from typing import Optional, Tuple
import numpy as np
from src.config import ShipConfig
from src.environment.bullet import Bullet
from src.environment.mine import Mine


def heading_to_vector(heading_deg: float) -> np.ndarray:
    """
    Convert heading in degrees to a 2D unit vector.
    0 deg = Up (North, [0, -1])
    90 deg = Right (East, [1, 0])
    180 deg = Down (South, [0, 1])
    270 deg = Left (West, [-1, 0])
    """
    rad = np.radians(heading_deg)
    return np.array([np.sin(rad), -np.cos(rad)], dtype=np.float64)


def vector_to_heading(vec: np.ndarray) -> float:
    """Convert a 2D direction vector [dx, dy] into heading in degrees [0, 360)."""
    deg = np.degrees(np.arctan2(vec[0], -vec[1]))
    return float((deg + 360.0) % 360.0)


class Ship:
    """
    Controllable spacecraft executing:
      1. Turn (deg/s)
      2. Thrust (acceleration along heading)
      3. Shoot (fire projectile from nose)
      4. Drop Mine (deploy explosive mine)
    """

    def __init__(
        self,
        ship_id: int,
        name: str,
        team: int,
        position: Tuple[float, float],
        heading: float = 0.0,
        lives: int = ShipConfig.initial_lives,
        ammo: int = ShipConfig.initial_ammo,
        mines: int = ShipConfig.initial_mines,
    ):
        self.id = ship_id
        self.name = name
        self.team = team

        self.spawn_position = np.array(position, dtype=np.float64)
        self.spawn_heading = float(heading)

        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.heading = float(heading % 360.0)

        self.radius = ShipConfig.radius
        self.mass = ShipConfig.mass
        self.max_speed = ShipConfig.max_speed
        self.thrust_power = ShipConfig.thrust_power
        self.max_turn_rate = ShipConfig.turn_rate
        self.linear_drag = ShipConfig.linear_drag

        # Health & State
        self.max_lives = ShipConfig.initial_lives
        self.lives = lives
        self.is_alive = True
        self.invulnerable_timer = ShipConfig.respawn_invulnerability

        # Ammo & Ordnance
        self.ammo = ammo
        self.mines_remaining = mines
        self.bullet_cooldown_timer = 0.0
        self.mine_cooldown_timer = 0.0

        # Performance Metrics (PEAS)
        self.score = 0
        self.asteroids_hit = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.deaths = 0

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def accuracy(self) -> float:
        if self.shots_fired == 0:
            return 0.0
        return float(self.shots_hit / self.shots_fired)

    @property
    def is_invulnerable(self) -> bool:
        return self.invulnerable_timer > 0.0

    def apply_control(
        self,
        turn_rate: float,
        thrust: float,
        shoot: bool,
        drop_mine: bool,
        dt: float,
        arena_width: float,
        arena_height: float,
        wrap: bool = True,
    ) -> Tuple[Optional[Bullet], Optional[Mine]]:
        """
        Execute the 4 actuators for one timestep dt:
        - turn_rate: commanded angular velocity in deg/s (clamped to [-max_turn_rate, max_turn_rate])
        - thrust: commanded propulsion in [0.0, 1.0]
        - shoot: bool request to fire cannon
        - drop_mine: bool request to deploy mine
        Returns (new_bullet_or_None, new_mine_or_None).
        """
        if not self.is_alive:
            return None, None

        # 1. Update timers
        if self.bullet_cooldown_timer > 0.0:
            self.bullet_cooldown_timer = max(0.0, self.bullet_cooldown_timer - dt)
        if self.mine_cooldown_timer > 0.0:
            self.mine_cooldown_timer = max(0.0, self.mine_cooldown_timer - dt)
        if self.invulnerable_timer > 0.0:
            self.invulnerable_timer = max(0.0, self.invulnerable_timer - dt)

        # 2. Turn
        clamped_turn = float(np.clip(turn_rate, -self.max_turn_rate, self.max_turn_rate))
        self.heading = (self.heading + clamped_turn * dt) % 360.0

        # 3. Thrust & Momentum
        forward = heading_to_vector(self.heading)
        clamped_thrust = float(np.clip(thrust, 0.0, 1.0))

        # Linear acceleration
        acceleration = forward * (clamped_thrust * self.thrust_power)
        self.velocity += acceleration * dt

        # Realistic space drift drag (very subtle)
        if self.linear_drag > 0.0:
            self.velocity *= (1.0 - self.linear_drag * dt)

        # Clamp max speed
        curr_speed = self.speed
        if curr_speed > self.max_speed:
            self.velocity = (self.velocity / curr_speed) * self.max_speed

        # Position update
        self.position += self.velocity * dt

        # Screen boundary handling:
        # If wrap is enabled, wrap around edges.
        # If wrap is disabled, hard screen boundary: ship cannot leave the screen.
        if wrap:
            if self.position[0] < 0:
                self.position[0] += arena_width
            elif self.position[0] >= arena_width:
                self.position[0] -= arena_width

            if self.position[1] < 0:
                self.position[1] += arena_height
            elif self.position[1] >= arena_height:
                self.position[1] -= arena_height
        else:
            # Hard screen boundaries: enforce position within screen view
            min_x = self.radius
            max_x = arena_width - self.radius
            min_y = self.radius
            max_y = arena_height - self.radius

            if self.position[0] < min_x:
                self.position[0] = min_x
                if self.velocity[0] < 0:
                    self.velocity[0] = -self.velocity[0] * 0.3  # Damped rebound
            elif self.position[0] > max_x:
                self.position[0] = max_x
                if self.velocity[0] > 0:
                    self.velocity[0] = -self.velocity[0] * 0.3

            if self.position[1] < min_y:
                self.position[1] = min_y
                if self.velocity[1] < 0:
                    self.velocity[1] = -self.velocity[1] * 0.3
            elif self.position[1] > max_y:
                self.position[1] = max_y
                if self.velocity[1] > 0:
                    self.velocity[1] = -self.velocity[1] * 0.3

        # 4. Weapons: Shoot
        new_bullet = None
        if shoot and self.bullet_cooldown_timer <= 0.0 and self.ammo > 0:
            self.bullet_cooldown_timer = ShipConfig.bullet_cooldown
            self.ammo -= 1
            self.shots_fired += 1

            # Spawn bullet at the tip/nose of the ship
            nose_pos = self.position + forward * (self.radius + 4.0)
            # Bullet velocity = ship velocity + muzzle speed along heading
            bullet_vel = self.velocity + forward * ShipConfig.bullet_speed
            new_bullet = Bullet(
                position=(nose_pos[0], nose_pos[1]),
                velocity=(bullet_vel[0], bullet_vel[1]),
                owner_id=self.id,
            )

        # 5. Weapons: Drop Mine
        new_mine = None
        if drop_mine and self.mine_cooldown_timer <= 0.0 and self.mines_remaining > 0:
            self.mine_cooldown_timer = ShipConfig.mine_deploy_cooldown
            self.mines_remaining -= 1

            # Deploy mine slightly behind ship
            rear_pos = self.position - forward * (self.radius + 6.0)
            new_mine = Mine(
                position=(rear_pos[0], rear_pos[1]),
                owner_id=self.id,
                velocity=(self.velocity[0], self.velocity[1]),
            )

        return new_bullet, new_mine

    def hit_by_hazard(self) -> bool:
        """
        Ship is struck by asteroid or explosion.
        Decrements lives, triggers respawn if lives remain.
        Returns True if ship lost a life.
        """
        if self.is_invulnerable or not self.is_alive:
            return False

        self.lives -= 1
        self.deaths += 1

        if self.lives <= 0:
            self.is_alive = False
            return True

        # Respawn at spawn point with safe heading
        self.position = self.spawn_position.copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.heading = self.spawn_heading
        self.invulnerable_timer = ShipConfig.respawn_invulnerability
        return True

    def reset(self) -> None:
        """Reset ship state for a new game scenario."""
        self.position = self.spawn_position.copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.heading = self.spawn_heading
        self.lives = self.max_lives
        self.ammo = ShipConfig.initial_ammo
        self.mines_remaining = ShipConfig.initial_mines
        self.bullet_cooldown_timer = 0.0
        self.mine_cooldown_timer = 0.0
        self.invulnerable_timer = ShipConfig.respawn_invulnerability
        self.is_alive = True
        self.score = 0
        self.asteroids_hit = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.deaths = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "team": self.team,
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "heading": self.heading,
            "speed": self.speed,
            "lives": self.lives,
            "ammo": self.ammo,
            "mines_remaining": self.mines_remaining,
            "is_alive": self.is_alive,
            "is_invulnerable": self.is_invulnerable,
            "score": self.score,
            "asteroids_hit": self.asteroids_hit,
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "deaths": self.deaths,
            "accuracy": self.accuracy,
        }
