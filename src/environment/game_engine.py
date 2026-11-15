"""
Core Game Engine: world physics, collision resolution, entity lifecycle, and Pygame renderer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pygame

from src.config import ArenaConfig, AsteroidConfig, GameConfig, ShipConfig
from src.environment.asteroid import Asteroid
from src.environment.bullet import Bullet
from src.environment.mine import Mine
from src.environment.ship import Ship, heading_to_vector


def wrapped_distance(p1: np.ndarray, p2: np.ndarray, width: float, height: float, wrap: bool = False) -> Tuple[float, np.ndarray]:
    """
    Computes Euclidean distance between two points (with optional toroidal wrap).
    Returns (distance, relative_vector_from_p1_to_p2).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if wrap:
        if dx > width / 2.0:
            dx -= width
        elif dx < -width / 2.0:
            dx += width

        if dy > height / 2.0:
            dy -= height
        elif dy < -height / 2.0:
            dy += height

    dist = float(np.sqrt(dx * dx + dy * dy))
    return dist, np.array([dx, dy], dtype=np.float64)


@dataclass
class ExplosionVisual:
    position: np.ndarray
    max_radius: float
    current_radius: float = 0.0
    lifetime: float = 0.4
    time_alive: float = 0.0

    @property
    def is_alive(self) -> bool:
        return self.time_alive < self.lifetime


class GameEngine:
    """
    Master game environment for the AI Piloted Asteroids game.
    Runs headless for fast training or with Pygame for human visualization.
    """

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None, auto_replenish: bool = False):
        self.config = config or GameConfig()
        self.arena = self.config.arena
        self.rng = np.random.default_rng(seed)
        self.auto_replenish = auto_replenish

        self.time: float = 0.0
        self.time_limit: float = 60.0
        self.dt: float = self.config.dt

        self.ships: Dict[int, Ship] = {}
        self.asteroids: List[Asteroid] = []
        self.bullets: List[Bullet] = []
        self.mines: List[Mine] = []
        self.explosions: List[ExplosionVisual] = []

        self.initial_num_asteroids: int = 5
        self._init_default_scenario()

    def _init_default_scenario(self) -> None:
        """Set up standard player ship, optional rival ship, and initial asteroid field."""
        self.ships.clear()
        self.asteroids.clear()
        self.bullets.clear()
        self.mines.clear()
        self.explosions.clear()
        self.time = 0.0

        # Single player AI controlled spacecraft spawned in arena center
        player_pos = (self.arena.width * 0.5, self.arena.height * 0.5)
        self.ships[1] = Ship(
            ship_id=1,
            name="AI_Agent",
            team=1,
            position=player_pos,
            heading=0.0,
        )

        # Spawn initial large asteroids (size 4) safely away from player per XFC guidelines
        for _ in range(self.initial_num_asteroids):
            self.spawn_random_asteroid(size=4)

    def add_rival_ship(self, position: Optional[Tuple[float, float]] = None, heading: float = 180.0) -> Ship:
        """Add a rival spacecraft (Team 2) for adversarial search / Minimax."""
        rival_pos = position or (self.arena.width * 0.7, self.arena.height * 0.5)
        rival = Ship(
            ship_id=2,
            name="Rival_Pilot",
            team=2,
            position=rival_pos,
            heading=heading,
        )
        self.ships[2] = rival
        return rival

    def spawn_random_asteroid(self, size: int = 4, min_dist_from_ships: float = 180.0) -> Asteroid:
        """
        Spawn an asteroid according to XFC guidelines:
        - Safe clearance from ships (min_dist_from_ships)
        - Boundary margin padding
        - Speed governed by XFC formula: max_speed = 60.0 * (2.0 + (4.0 - size) / 4.0)
        - Collision radius = size * 8.0, mass = 0.25 * pi * radius^2
        """
        for _ in range(100):
            pos = np.array([
                self.rng.uniform(60, self.arena.width - 60),
                self.rng.uniform(60, self.arena.height - 60),
            ])
            # Check safe clearance from all ships
            safe = True
            for ship in self.ships.values():
                dist, _ = wrapped_distance(pos, ship.position, self.arena.width, self.arena.height)
                if dist < min_dist_from_ships:
                    safe = False
                    break
            if safe:
                break

        # XFC Speed Formula: max_speed = 60.0 * (2.0 + (4.0 - size) / 4.0)
        speed_scaler = 2.0 + (4.0 - size) / 4.0
        max_speed = 60.0 * speed_scaler
        speed = self.rng.uniform(0.35 * max_speed, max_speed)
        angle = self.rng.uniform(0, 2 * np.pi)
        vel = (np.cos(angle) * speed, np.sin(angle) * speed)

        ast = Asteroid(position=(pos[0], pos[1]), velocity=vel, size=size)
        self.asteroids.append(ast)
        return ast

    def step(self, actions: Dict[int, Tuple[float, float, bool, bool]]) -> Dict:
        """
        Advance game physics by 1 tick (dt).
        actions: dict mapping ship_id to (turn_rate, thrust, shoot, drop_mine).
        Returns a summary observation dict for this tick.
        """
        self.time += self.dt

        # 1. Apply controls to each ship
        for ship_id, ship in self.ships.items():
            if ship.is_alive and ship_id in actions:
                turn, thrust, shoot, drop_mine = actions[ship_id]
                new_bullet, new_mine = ship.apply_control(
                    turn_rate=turn,
                    thrust=thrust,
                    shoot=shoot,
                    drop_mine=drop_mine,
                    dt=self.dt,
                    arena_width=self.arena.width,
                    arena_height=self.arena.height,
                    wrap=self.arena.toroidal_wrap,
                )
                if new_bullet:
                    self.bullets.append(new_bullet)
                if new_mine:
                    self.mines.append(new_mine)

        # 2. Update asteroids
        for ast in self.asteroids:
            ast.update(self.dt, self.arena.width, self.arena.height, wrap=self.arena.toroidal_wrap)

        # 3. Update bullets
        for b in self.bullets:
            b.update(self.dt, self.arena.width, self.arena.height, wrap=self.arena.toroidal_wrap)

        # 4. Update mines & handle detonation
        detonated_mines: List[Mine] = []
        for m in self.mines:
            did_explode = m.update(self.dt, self.arena.width, self.arena.height, wrap=self.arena.toroidal_wrap)
            if did_explode:
                detonated_mines.append(m)

        for m in detonated_mines:
            pos, blast_r = m.detonate()
            self.explosions.append(ExplosionVisual(position=pos, max_radius=blast_r))
            self._handle_mine_blast(pos, blast_r, m.owner_id)

        # 5. Collision: Bullets vs Asteroids
        new_split_asteroids: List[Asteroid] = []
        for b in self.bullets:
            if not b.is_alive:
                continue
            for ast in self.asteroids:
                if not ast.is_alive:
                    continue
                dist, _ = wrapped_distance(b.position, ast.position, self.arena.width, self.arena.height)
                if dist <= (b.radius + ast.radius):
                    # Direct hit
                    b.is_alive = False
                    shooter = self.ships.get(b.owner_id)
                    if shooter:
                        shooter.score += ast.score_value
                        shooter.shots_hit += 1
                        shooter.asteroids_hit += 1

                    self.explosions.append(ExplosionVisual(position=ast.position.copy(), max_radius=ast.radius * 1.5))
                    child_frags = ast.split(self.rng, bullet=b)
                    new_split_asteroids.extend(child_frags)
                    break

        self.asteroids.extend(new_split_asteroids)

        # 6. Collision: Bullets vs Rival Ships
        for b in self.bullets:
            if not b.is_alive:
                continue
            for ship in self.ships.values():
                if not ship.is_alive or ship.id == b.owner_id or ship.is_invulnerable:
                    continue
                dist, _ = wrapped_distance(b.position, ship.position, self.arena.width, self.arena.height)
                if dist <= (b.radius + ship.radius):
                    b.is_alive = False
                    ship.hit_by_hazard()
                    shooter = self.ships.get(b.owner_id)
                    if shooter:
                        shooter.score += 200
                        shooter.shots_hit += 1
                    self.explosions.append(ExplosionVisual(position=ship.position.copy(), max_radius=ship.radius * 2.0))

        # 7. Collision: Ships vs Asteroids
        for ship in self.ships.values():
            if not ship.is_alive or ship.is_invulnerable:
                continue
            for ast in self.asteroids:
                if not ast.is_alive:
                    continue
                dist, _ = wrapped_distance(ship.position, ast.position, self.arena.width, self.arena.height)
                if dist <= (ship.radius + ast.radius):
                    ship.hit_by_hazard()
                    self.explosions.append(ExplosionVisual(position=ship.position.copy(), max_radius=ship.radius * 2.5))
                    break

        # 8. Clean up dead entities
        self.asteroids = [a for a in self.asteroids if a.is_alive]
        self.bullets = [b for b in self.bullets if b.is_alive]
        self.mines = [m for m in self.mines if m.is_alive]

        # In continuous survival mode, replenish asteroid field with large asteroids (XFC size 4)
        if self.auto_replenish and len(self.asteroids) < 4:
            self.spawn_random_asteroid(size=4)

        # 9. Update visual explosions
        for exp in self.explosions:
            exp.time_alive += self.dt
            exp.current_radius = exp.max_radius * (exp.time_alive / exp.lifetime)
        self.explosions = [e for e in self.explosions if e.is_alive]

        return self.get_true_state()

    def _handle_mine_blast(self, blast_pos: np.ndarray, blast_radius: float, owner_id: int) -> None:
        """Applies AOE blast damage to all asteroids and vulnerable ships."""
        new_splits = []
        for ast in self.asteroids:
            if not ast.is_alive:
                continue
            dist, _ = wrapped_distance(blast_pos, ast.position, self.arena.width, self.arena.height)
            if dist <= blast_radius + ast.radius:
                shooter = self.ships.get(owner_id)
                if shooter:
                    shooter.score += ast.score_value
                    shooter.asteroids_hit += 1
                child_frags = ast.split(self.rng)
                new_splits.extend(child_frags)

        self.asteroids.extend(new_splits)

        for ship in self.ships.values():
            if not ship.is_alive or ship.is_invulnerable:
                continue
            dist, _ = wrapped_distance(blast_pos, ship.position, self.arena.width, self.arena.height)
            if dist <= blast_radius + ship.radius:
                ship.hit_by_hazard()

    def is_game_over(self) -> bool:
        """Returns True if player is dead, time expired, or all asteroids eliminated."""
        player = self.ships.get(1)
        if player and not player.is_alive:
            return True
        if self.time >= self.time_limit:
            return True
        if len(self.asteroids) == 0:
            return True
        return False

    def reset(self, num_asteroids: int = 5) -> Dict:
        """Reset game world to fresh initial state."""
        self.initial_num_asteroids = num_asteroids
        self._init_default_scenario()
        return self.get_true_state()

    def get_true_state(self) -> Dict:
        """Returns exact ground-truth state of all objects (internal reference)."""
        return {
            "time": self.time,
            "arena_size": (self.arena.width, self.arena.height),
            "ships": [ship.to_dict() for ship in self.ships.values()],
            "asteroids": [ast.to_dict() for ast in self.asteroids if ast.is_alive],
            "bullets": [b.to_dict() for b in self.bullets if b.is_alive],
            "mines": [m.to_dict() for m in self.mines if m.is_alive],
            "is_game_over": self.is_game_over(),
        }

    # -------------------------------------------------------------------------
    # Pygame Renderer
    # -------------------------------------------------------------------------
    def render(self, surface: pygame.Surface) -> None:
        """Render arena, ships, asteroids, ordnance, and telemetry HUD onto surface."""
        # Deep space background
        surface.fill((10, 12, 24))

        # Subtle starfield grid lines
        grid_color = (20, 24, 40)
        for x in range(0, int(self.arena.width), 100):
            pygame.draw.line(surface, grid_color, (x, 0), (x, int(self.arena.height)), 1)
        for y in range(0, int(self.arena.height), 100):
            pygame.draw.line(surface, grid_color, (0, y), (int(self.arena.width), y), 1)

        # Screen perimeter boundary barrier (hard walls: ship cannot exit screen)
        if not self.arena.toroidal_wrap:
            pygame.draw.rect(surface, (50, 120, 200), (2, 2, int(self.arena.width) - 4, int(self.arena.height) - 4), 2)
            pygame.draw.rect(surface, (25, 60, 110), (0, 0, int(self.arena.width), int(self.arena.height)), 1)

        # Draw Mines
        for m in self.mines:
            color = (255, 60, 60) if m.is_armed else (200, 180, 50)
            pygame.draw.circle(surface, color, (int(m.position[0]), int(m.position[1])), int(m.radius))
            # Pulsing trigger halo
            pygame.draw.circle(surface, (120, 40, 40), (int(m.position[0]), int(m.position[1])), int(m.radius + 3), 1)

        # Draw Asteroids (XFC Sizes 1-4)
        for ast in self.asteroids:
            # Wireframe rock styling
            color = (195, 205, 225) if ast.size >= 4 else ((165, 180, 200) if ast.size == 3 else ((135, 150, 170) if ast.size == 2 else (105, 120, 140)))
            pygame.draw.circle(surface, color, (int(ast.position[0]), int(ast.position[1])), int(ast.radius), 2)
            # Velocity vector indicator
            vel_endpoint = ast.position + ast.velocity * 0.3
            pygame.draw.line(surface, (70, 85, 105), (int(ast.position[0]), int(ast.position[1])), (int(vel_endpoint[0]), int(vel_endpoint[1])), 1)

        # Draw Bullets
        for b in self.bullets:
            color = (255, 220, 60) if b.owner_id == 1 else (255, 90, 90)
            pygame.draw.circle(surface, color, (int(b.position[0]), int(b.position[1])), int(b.radius))

        # Draw Explosions
        for exp in self.explosions:
            alpha_ratio = 1.0 - (exp.time_alive / exp.lifetime)
            color = (int(255 * alpha_ratio), int(160 * alpha_ratio), 40)
            if exp.current_radius > 1:
                pygame.draw.circle(surface, color, (int(exp.position[0]), int(exp.position[1])), int(exp.current_radius), 2)

        # Draw Ships
        for ship in self.ships.values():
            if not ship.is_alive:
                continue

            # Triangular ship vertices based on heading
            forward = heading_to_vector(ship.heading)
            right = np.array([-forward[1], forward[0]])

            nose = ship.position + forward * ship.radius
            left_wing = ship.position - forward * (ship.radius * 0.7) - right * (ship.radius * 0.6)
            right_wing = ship.position - forward * (ship.radius * 0.7) + right * (ship.radius * 0.6)

            ship_color = (80, 210, 255) if ship.id == 1 else (255, 110, 110)
            points = [(int(p[0]), int(p[1])) for p in [nose, left_wing, ship.position, right_wing]]
            pygame.draw.polygon(surface, ship_color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 1)

            # Shield ring when invulnerable
            if ship.is_invulnerable:
                pygame.draw.circle(surface, (100, 200, 255), (int(ship.position[0]), int(ship.position[1])), int(ship.radius + 6), 1)

        # HUD Overlay
        self._render_hud(surface)

    def _render_hud(self, surface: pygame.Surface) -> None:
        """Render score, lives, ammo, and time on top banner."""
        font = pygame.font.SysFont("Consolas", 16)
        player = self.ships.get(1)
        if player:
            lives_str = "<3 [1/1 SINGLE LIFE]" if player.lives > 0 else "DESTROYED"
            hud_text = (
                f"TIME: {self.time:4.1f}s | "
                f"LIFE: {lives_str} | "
                f"SCORE: {player.score:05d} | "
                f"AMMO: {player.ammo} | "
                f"ACC: {player.accuracy * 100:3.0f}% | "
                f"ASTEROIDS: {len(self.asteroids)}"
            )
            text_surf = font.render(hud_text, True, (220, 230, 245))
            surface.blit(text_surf, (15, 12))
