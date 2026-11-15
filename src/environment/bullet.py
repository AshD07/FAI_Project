"""
Projectile (Bullet) entity fired by spacecraft.
"""

from typing import Optional, Tuple
import numpy as np
from src.config import ShipConfig


class Bullet:
    """
    High-velocity projectile fired from ship's nose.
    Decays after reaching its maximum lifetime.
    """
    _next_id = 1

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        owner_id: int,
        bullet_id: Optional[int] = None,
        lifetime: float = ShipConfig.bullet_lifetime,
        radius: float = ShipConfig.bullet_radius,
    ):
        if bullet_id is None:
            self.id = Bullet._next_id
            Bullet._next_id += 1
        else:
            self.id = bullet_id
            if bullet_id >= Bullet._next_id:
                Bullet._next_id = bullet_id + 1

        self.owner_id = owner_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.mass = getattr(ShipConfig, "bullet_mass", 1.0)
        self.lifetime = lifetime
        self.time_alive = 0.0
        self.is_alive = True

    def update(self, dt: float, arena_width: float, arena_height: float, wrap: bool = True) -> None:
        if not self.is_alive:
            return

        self.time_alive += dt
        if self.time_alive >= self.lifetime:
            self.is_alive = False
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
            # Bullet impacts screen perimeter wall
            if (self.position[0] < 0 or self.position[0] >= arena_width or
                self.position[1] < 0 or self.position[1] >= arena_height):
                self.is_alive = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "radius": self.radius,
            "time_alive": self.time_alive,
            "is_alive": self.is_alive,
        }
