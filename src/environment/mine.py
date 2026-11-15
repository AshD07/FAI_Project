"""
Explosive Mine entity with fuse timer and area-of-effect blast.
"""

from typing import Optional, Tuple
import numpy as np
from src.config import ShipConfig


class Mine:
    """
    Stationary or slowly drifting proximity/timer mine.
    Detonates when fuse expires, clearing an area of effect (blast radius).
    """
    _next_id = 1

    def __init__(
        self,
        position: Tuple[float, float],
        owner_id: int,
        velocity: Tuple[float, float] = (0.0, 0.0),
        mine_id: Optional[int] = None,
        fuse_time: float = ShipConfig.mine_fuse_time,
        blast_radius: float = ShipConfig.mine_blast_radius,
        radius: float = ShipConfig.mine_radius,
    ):
        if mine_id is None:
            self.id = Mine._next_id
            Mine._next_id += 1
        else:
            self.id = mine_id
            if mine_id >= Mine._next_id:
                Mine._next_id = mine_id + 1

        self.owner_id = owner_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64) * 0.2  # subtle drift
        self.radius = radius
        self.blast_radius = blast_radius
        self.fuse_time = fuse_time
        self.time_alive = 0.0
        self.is_armed = False          # 0.5s arming grace period
        self.is_detonated = False
        self.is_alive = True

    def update(self, dt: float, arena_width: float, arena_height: float, wrap: bool = True) -> bool:
        """
        Advance mine countdown. Returns True if mine just detonated on this tick.
        """
        if not self.is_alive:
            return False

        self.time_alive += dt
        if self.time_alive >= 0.5:
            self.is_armed = True

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
            self.position[0] = float(np.clip(self.position[0], self.radius, arena_width - self.radius))
            self.position[1] = float(np.clip(self.position[1], self.radius, arena_height - self.radius))

        if self.time_alive >= self.fuse_time:
            self.detonate()
            return True

        return False

    def detonate(self) -> Tuple[np.ndarray, float]:
        """Triggers the mine explosion, returning (position, blast_radius)."""
        self.is_detonated = True
        self.is_alive = False
        return self.position.copy(), self.blast_radius

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "position": tuple(self.position),
            "radius": self.radius,
            "blast_radius": self.blast_radius,
            "time_alive": self.time_alive,
            "is_armed": self.is_armed,
            "is_detonated": self.is_detonated,
            "is_alive": self.is_alive,
        }
