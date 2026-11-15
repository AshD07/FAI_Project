"""
Asteroid entity and splitting mechanics.
"""

from typing import List, Optional, Tuple
import numpy as np
from src.config import AsteroidConfig, ArenaConfig


class Asteroid:
    """
    Represents an asteroid drifting through the arena.
    Supports splitting into smaller fragments when struck.
    """
    _next_id = 1

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        size: int = 4,
        asteroid_id: Optional[int] = None,
    ):
        if asteroid_id is None:
            self.id = Asteroid._next_id
            Asteroid._next_id += 1
        else:
            self.id = asteroid_id
            if asteroid_id >= Asteroid._next_id:
                Asteroid._next_id = asteroid_id + 1

        self.size = int(np.clip(size, 1, 4))
        # XFC Guidelines: radius = size * 8.0, mass = 0.25 * pi * radius^2
        self.radius = AsteroidConfig.radii.get(self.size, float(self.size * 8.0))
        self.mass = AsteroidConfig.masses.get(self.size, float(0.25 * np.pi * self.radius * self.radius))
        self.score_value = AsteroidConfig.score_values.get(self.size, 50)
        
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.is_alive = True

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def update(self, dt: float, arena_width: float, arena_height: float, wrap: bool = True) -> None:
        """Advance position by velocity * dt, wrapping across arena boundaries if enabled."""
        if not self.is_alive:
            return

        self.position += self.velocity * dt

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
            # Elastic boundary reflection keeps asteroids inside the arena
            if self.position[0] < self.radius:
                self.position[0] = self.radius
                self.velocity[0] *= -1.0
            elif self.position[0] > arena_width - self.radius:
                self.position[0] = arena_width - self.radius
                self.velocity[0] *= -1.0

            if self.position[1] < self.radius:
                self.position[1] = self.radius
                self.velocity[1] *= -1.0
            elif self.position[1] > arena_height - self.radius:
                self.position[1] = arena_height - self.radius
                self.velocity[1] *= -1.0

    def split(self, rng: Optional[np.random.Generator] = None, bullet: Optional[object] = None) -> List["Asteroid"]:
        """
        Split this asteroid upon destruction according to XFC guidelines:
        - If size > 1, generates 3 smaller child fragments (size - 1)
        - Inelastic collision momentum transfer when impacted by a bullet
        - If size == 1, returns empty list (completely obliterated).
        """
        self.is_alive = False
        if self.size <= 1:
            return []

        if rng is None:
            rng = np.random.default_rng()

        new_size = self.size - 1
        speed_range = AsteroidConfig.speeds.get(new_size, (50.0, 150.0))

        # Inelastic collision momentum transfer if bullet provided (XFC guideline)
        if bullet is not None and hasattr(bullet, "mass") and hasattr(bullet, "velocity"):
            b_mass = float(bullet.mass)
            vfx = (1.0 / (b_mass + self.mass)) * (b_mass * bullet.velocity[0] + self.mass * self.velocity[0])
            vfy = (1.0 / (b_mass + self.mass)) * (b_mass * bullet.velocity[1] + self.mass * self.velocity[1])
            res_speed = float(np.sqrt(vfx * vfx + vfy * vfy))
            speed = float(np.clip(res_speed * 1.2, speed_range[0], speed_range[1]))
            base_angle = float(np.arctan2(vfy, vfx))
        else:
            speed = rng.uniform(speed_range[0], speed_range[1])
            base_angle = float(np.arctan2(self.velocity[1], self.velocity[0])) if self.speed > 1e-4 else rng.uniform(0, 2 * np.pi)

        # XFC Guidelines: 3 children generated on split
        deflections = [-np.radians(25.0), 0.0, np.radians(25.0)]
        children = []

        for defl in deflections:
            angle = base_angle + defl + rng.uniform(-0.08, 0.08)
            child_speed = float(np.clip(speed * rng.uniform(0.9, 1.1), speed_range[0], speed_range[1]))
            vel = np.array([np.cos(angle) * child_speed, np.sin(angle) * child_speed], dtype=np.float64)
            # Offset position slightly along fragment velocity to prevent immediate overlap
            offset_pos = self.position + (vel / (child_speed + 1e-6)) * (self.radius * 0.5)
            children.append(Asteroid(position=(offset_pos[0], offset_pos[1]), velocity=(vel[0], vel[1]), size=new_size))

        return children

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "size": self.size,
            "radius": self.radius,
            "mass": self.mass,
            "speed": self.speed,
            "is_alive": self.is_alive,
        }
