"""
Stage 5: Local A* Pathfinding using dynamic egocentric obstacle grids
and Kalman-predicted hazard trajectories.
"""

from dataclasses import dataclass, field
import heapq
from typing import List, Optional, Tuple
import numpy as np

from src.environment.game_engine import wrapped_distance
from src.environment.ship import heading_to_vector
from src.sensors.radar_sensor import ShipTelemetry
from src.tracking.object_tracker import TrackedEntity


@dataclass
class AStarPlanResult:
    """Output plan from local A* search."""
    turn_rate: float
    thrust: float
    path_grid: List[Tuple[int, int]] = field(default_factory=list)
    path_world: List[Tuple[float, float]] = field(default_factory=list)
    subgoal_world: Optional[Tuple[float, float]] = None
    is_path_found: bool = False
    grid_cost_map: Optional[np.ndarray] = None


class AStarLocalPlanner:
    """
    Local egocentric A* pathfinding.
    Discretizes a rolling window around the ship, inflates obstacles using
    Kalman-predicted trajectories, and plans collision-free waypoints.
    """

    def __init__(
        self,
        grid_width: float = 400.0,
        grid_height: float = 400.0,
        cell_size: float = 20.0,
        safety_margin: float = 22.0,
        max_turn_rate: float = 180.0,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.safety_margin = safety_margin
        self.max_turn_rate = max_turn_rate
        self.arena_width = arena_width
        self.arena_height = arena_height

        self.cols = int(grid_width / cell_size)
        self.rows = int(grid_height / cell_size)
        self.center_col = self.cols // 2
        self.center_row = self.rows // 2

    def world_to_local_grid(self, world_pos: np.ndarray, ship_pos: np.ndarray) -> Tuple[int, int]:
        """Convert a world position into local (col, row) relative to the ship center."""
        _, rel_vec = wrapped_distance(ship_pos, world_pos, self.arena_width, self.arena_height)
        col = int((rel_vec[0] + self.grid_width / 2.0) / self.cell_size)
        row = int((rel_vec[1] + self.grid_height / 2.0) / self.cell_size)
        return col, row

    def local_grid_to_world(self, col: int, row: int, ship_pos: np.ndarray, wrap: bool = False) -> np.ndarray:
        """Convert local (col, row) back to global world coordinates, bounded within screen."""
        rel_x = (col + 0.5) * self.cell_size - (self.grid_width / 2.0)
        rel_y = (row + 0.5) * self.cell_size - (self.grid_height / 2.0)
        if wrap:
            wx = (ship_pos[0] + rel_x) % self.arena_width
            wy = (ship_pos[1] + rel_y) % self.arena_height
        else:
            wx = float(np.clip(ship_pos[0] + rel_x, 18.0, self.arena_width - 18.0))
            wy = float(np.clip(ship_pos[1] + rel_y, 18.0, self.arena_height - 18.0))
        return np.array([wx, wy], dtype=np.float64)

    def is_in_grid(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def build_occupancy_grid(
        self,
        ship_pos: np.ndarray,
        tracked_hazards: List[TrackedEntity],
    ) -> np.ndarray:
        """
        Construct a 2D cost grid.
        0 = clear space, 1 = obstacle (impassable).
        Inflates obstacle footprints using current and predicted Kalman trajectories,
        and marks screen borders as impassable walls so the ship cannot go out of the screen.
        """
        grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        # 1. Screen wall boundary barrier: any cell outside visible arena is an obstacle
        wall_margin = 40.0
        for r in range(self.rows):
            for c in range(self.cols):
                rel_x = (c + 0.5) * self.cell_size - (self.grid_width / 2.0)
                rel_y = (r + 0.5) * self.cell_size - (self.grid_height / 2.0)
                wx = ship_pos[0] + rel_x
                wy = ship_pos[1] + rel_y
                if wx < wall_margin or wx > self.arena_width - wall_margin or wy < wall_margin or wy > self.arena_height - wall_margin:
                    grid[r, c] = 1

        # 2. Dynamic asteroid obstacle rasterization
        for hazard in tracked_hazards:
            inflation_radius = hazard.estimated_radius + 16.0 + self.safety_margin
            rad_cells = int(np.ceil(inflation_radius / self.cell_size))

            # Points to rasterize: current position + future 0.5s & 1.0s lookaheads
            positions_to_rasterize = [hazard.position]
            if len(hazard.future_trajectory) >= 10:
                # 0.5s lookahead (~step 5) and 1.0s lookahead (~step 10)
                positions_to_rasterize.append(np.array(hazard.future_trajectory[4]))
                positions_to_rasterize.append(np.array(hazard.future_trajectory[9]))

            for pos in positions_to_rasterize:
                c, r = self.world_to_local_grid(pos, ship_pos)
                if -rad_cells <= c < self.cols + rad_cells and -rad_cells <= r < self.rows + rad_cells:
                    # Inflate circle in grid
                    for dr in range(-rad_cells, rad_cells + 1):
                        for dc in range(-rad_cells, rad_cells + 1):
                            dist_px = np.sqrt((dc * self.cell_size) ** 2 + (dr * self.cell_size) ** 2)
                            if dist_px <= inflation_radius:
                                nr = r + dr
                                nc = c + dc
                                if self.is_in_grid(nc, nr):
                                    grid[nr, nc] = 1

        # Center cell (ship position) must never be permanently blocked for search initiation
        grid[self.center_row, self.center_col] = 0
        return grid

    def plan(
        self,
        telemetry: ShipTelemetry,
        tracked_hazards: List[TrackedEntity],
        target_world_pos: Optional[np.ndarray] = None,
    ) -> AStarPlanResult:
        """
        Run A* search on the local grid to find an optimal collision-free path.
        Extracts recommended turn_rate and thrust commands.
        """
        ship_pos = np.array(telemetry.position, dtype=np.float64)
        grid = self.build_occupancy_grid(ship_pos, tracked_hazards)

        start_col = self.center_col
        start_row = self.center_row

        # 1. Determine Goal Cell
        if target_world_pos is not None:
            # Map target world pos to local grid
            t_col, t_row = self.world_to_local_grid(target_world_pos, ship_pos)
            # If target is outside grid window, project towards perimeter
            if not self.is_in_grid(t_col, t_row):
                d_col = t_col - start_col
                d_row = t_row - start_row
                scale = max(abs(d_col) / (self.cols / 2 - 2), abs(d_row) / (self.rows / 2 - 2), 1e-4)
                goal_col = int(np.clip(start_col + d_col / scale, 1, self.cols - 2))
                goal_row = int(np.clip(start_row + d_row / scale, 1, self.rows - 2))
            else:
                goal_col, goal_row = t_col, t_row
        else:
            # Default goal: cruise along current heading forward into open space
            forward = heading_to_vector(telemetry.heading)
            f_col = int(np.clip(start_col + forward[0] * (self.cols // 3), 1, self.cols - 2))
            f_row = int(np.clip(start_row + forward[1] * (self.rows // 3), 1, self.rows - 2))
            goal_col, goal_row = f_col, f_row

        # If goal cell is blocked, find closest open cell to goal
        if grid[goal_row, goal_col] == 1:
            best_open = None
            min_open_dist = float("inf")
            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r, c] == 0:
                        d = (c - goal_col) ** 2 + (r - goal_row) ** 2
                        if d < min_open_dist:
                            min_open_dist = d
                            best_open = (c, r)
            if best_open:
                goal_col, goal_row = best_open

        # 2. A* Priority Queue Search
        # 8-connected neighbors: (dcol, drow, cost)
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
        ]

        open_set = []
        # Item: (f_score, g_score, col, row)
        heapq.heappush(open_set, (0.0, 0.0, start_col, start_row))

        came_from = {}
        g_score = {(start_col, start_row): 0.0}
        closed_set = set()

        closest_visited_node = (start_col, start_row)
        min_heuristic = float("inf")

        found_path = False

        while open_set:
            _, current_g, col, row = heapq.heappop(open_set)

            if (col, row) == (goal_col, goal_row):
                found_path = True
                break

            if (col, row) in closed_set:
                continue
            closed_set.add((col, row))

            # Track closest visited node as fallback if goal is unreachable
            h = np.sqrt((col - goal_col) ** 2 + (row - goal_row) ** 2)
            if h < min_heuristic:
                min_heuristic = h
                closest_visited_node = (col, row)

            for dc, dr, move_cost in moves:
                nc, nr = col + dc, row + dr
                if not self.is_in_grid(nc, nr) or grid[nr, nc] == 1:
                    continue

                tentative_g = current_g + move_cost
                if tentative_g < g_score.get((nc, nr), float("inf")):
                    g_score[(nc, nr)] = tentative_g
                    priority = tentative_g + np.sqrt((nc - goal_col) ** 2 + (nr - goal_row) ** 2)
                    heapq.heappush(open_set, (priority, tentative_g, nc, nr))
                    came_from[(nc, nr)] = (col, row)

        # 3. Path Reconstruction
        end_node = (goal_col, goal_row) if found_path else closest_visited_node
        path = []
        curr = end_node
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()

        # 4. Convert Waypoint Path to Turn & Thrust Actuators
        path_world = [tuple(self.local_grid_to_world(c, r, ship_pos)) for c, r in path]

        if len(path) > 0:
            # Look at waypoint 1 or 2 steps ahead for smooth steering
            waypoint_idx = min(2, len(path) - 1)
            target_cell = path[waypoint_idx]
            subgoal_world = tuple(self.local_grid_to_world(target_cell[0], target_cell[1], ship_pos))

            # Direction vector to waypoint
            _, rel_to_wp = wrapped_distance(ship_pos, np.array(subgoal_world), self.arena_width, self.arena_height)
            desired_heading = float(np.degrees(np.arctan2(rel_to_wp[0], -rel_to_wp[1])) % 360.0)

            # Heading difference [-180, 180]
            angle_diff = (desired_heading - telemetry.heading + 180.0) % 360.0 - 180.0

            # Proportional turn rate
            turn_cmd = float(np.clip(angle_diff * 4.0, -self.max_turn_rate, self.max_turn_rate))

            # Thrust: full thrust if pointing near waypoint, reduced if turning sharply
            if abs(angle_diff) < 35.0:
                thrust_cmd = 1.0
            elif abs(angle_diff) < 70.0:
                thrust_cmd = 0.4
            else:
                thrust_cmd = 0.0
        else:
            # Fallback if trapped
            turn_cmd = 0.0
            thrust_cmd = 0.0
            subgoal_world = None

        return AStarPlanResult(
            turn_rate=turn_cmd,
            thrust=thrust_cmd,
            path_grid=path,
            path_world=path_world,
            subgoal_world=subgoal_world,
            is_path_found=found_path,
            grid_cost_map=grid,
        )
