"""
Stage 6: Constraint Satisfaction Problem (CSP) Firing Controller.
Enforces rigorous firing constraints before approving weapon discharge.
Calculates quadratic ballistic lead intercept for accurate deflection shooting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import ArenaConfig, ShipConfig
from src.environment.game_engine import wrapped_distance
from src.sensors.radar_sensor import ShipTelemetry
from src.tracking.object_tracker import TrackedEntity


@dataclass
class CSPFiringDecision:
    """Decision output produced by the CSP firing solver."""
    should_shoot: bool
    aim_heading: Optional[float] = None
    target_track_id: Optional[int] = None
    intercept_time: Optional[float] = None
    intercept_point: Optional[Tuple[float, float]] = None
    constraint_audit: Dict[str, bool] = field(default_factory=dict)
    explanation: str = ""


class CSPShooter:
    """
    Constraint Satisfaction Problem (CSP) Firing Controller.
    Verifies 5 essential constraints before shooting:
      1. Cooldown ready (weapon not recharging)
      2. Ammo available (> 0)
      3. Angular alignment within firing tolerance
      4. Target intercept reachable within projectile lifetime
      5. Safe firing geometry (no immediate split-hazard blowback)
    """

    def __init__(
        self,
        firing_tolerance_deg: float = 7.0,
        bullet_speed: float = ShipConfig.bullet_speed,
        bullet_lifetime: float = ShipConfig.bullet_lifetime,
        min_safe_target_distance: float = 50.0,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
    ):
        self.firing_tolerance_deg = firing_tolerance_deg
        self.bullet_speed = bullet_speed
        self.bullet_lifetime = bullet_lifetime
        self.min_safe_target_distance = min_safe_target_distance
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.wrap = ArenaConfig().toroidal_wrap

    def solve_ballistic_intercept(
        self,
        ship_pos: np.ndarray,
        ship_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> Optional[Tuple[float, np.ndarray, float]]:
        """
        Solves the quadratic intercept equation for bullet travel time t > 0.
        Returns (t_intercept, intercept_world_pos, required_aim_heading) or None if unsolvable.
        """
        # Vector from ship to target accounting for toroidal wrap
        _, r_vec = wrapped_distance(ship_pos, target_pos, self.arena_width, self.arena_height)
        v_rel = target_vel - ship_vel
        vb = self.bullet_speed

        # Quadratic coefficients: a*t^2 + b*t + c = 0
        a = float(np.dot(v_rel, v_rel) - vb * vb)
        b = float(2.0 * np.dot(r_vec, v_rel))
        c = float(np.dot(r_vec, r_vec))

        # Check discriminant
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Select the smallest positive intercept time
        valid_times = [t for t in (t1, t2) if t > 0.0]
        if not valid_times:
            return None

        t_int = min(valid_times)
        if t_int > self.bullet_lifetime:
            return None

        # Predicted intercept position.
        # The arena is bounded: asteroids bounce off walls and bullets expire at them, so a
        # straight-line intercept that lies outside the arena is not physically achievable.
        # (The previous version took a modulo here, which aimed at a point on the opposite
        # side of the screen whenever the target was about to bounce.)
        int_x = target_pos[0] + target_vel[0] * t_int
        int_y = target_pos[1] + target_vel[1] * t_int
        if self.wrap:
            int_x %= self.arena_width
            int_y %= self.arena_height
        elif not (0.0 <= int_x <= self.arena_width and 0.0 <= int_y <= self.arena_height):
            return None
        int_pos = np.array([int_x, int_y], dtype=np.float64)

        # Required bullet direction vector
        _, req_vec = wrapped_distance(ship_pos, int_pos, self.arena_width, self.arena_height)
        aim_heading = float(np.degrees(np.arctan2(req_vec[0], -req_vec[1])) % 360.0)

        return t_int, int_pos, aim_heading

    def evaluate(
        self,
        telemetry: ShipTelemetry,
        targets: List[TrackedEntity],
    ) -> CSPFiringDecision:
        """
        Evaluates shooting candidate targets against all 5 CSP constraints.
        Returns a CSPFiringDecision with detailed constraint audit.
        """
        ship_pos = np.array(telemetry.position, dtype=np.float64)
        ship_vel = np.array(telemetry.velocity, dtype=np.float64)

        # Basic constraints check upfront
        c1_cooldown = (telemetry.bullet_cooldown <= 0.0)
        c2_ammo = (telemetry.ammo > 0)

        if not c1_cooldown:
            return CSPFiringDecision(
                should_shoot=False,
                constraint_audit={
                    "C1_cooldown_ready": False,
                    "C2_ammo_available": c2_ammo,
                    "C3_angular_alignment": False,
                    "C4_range_suitable": False,
                    "C5_safety_satisfied": False,
                },
                explanation=f"Weapon charging (cooldown: {telemetry.bullet_cooldown:.2f}s).",
            )

        if not c2_ammo:
            return CSPFiringDecision(
                should_shoot=False,
                constraint_audit={
                    "C1_cooldown_ready": True,
                    "C2_ammo_available": False,
                    "C3_angular_alignment": False,
                    "C4_range_suitable": False,
                    "C5_safety_satisfied": False,
                },
                explanation="Ammunition reservoir depleted.",
            )

        if not targets:
            return CSPFiringDecision(
                should_shoot=False,
                constraint_audit={
                    "C1_cooldown_ready": True,
                    "C2_ammo_available": True,
                    "C3_angular_alignment": False,
                    "C4_range_suitable": False,
                    "C5_safety_satisfied": False,
                },
                explanation="No tracked targets in range.",
            )

        # Sort targets by priority: favor smaller targets (clean up debris) and close threats
        sorted_targets = sorted(
            targets,
            key=lambda t: (
                wrapped_distance(ship_pos, t.position, self.arena_width, self.arena_height)[0]
                - (4 - getattr(t, "size_category", 2)) * 35.0
            )
        )

        best_decision: Optional[CSPFiringDecision] = None
        closest_aim_heading: Optional[float] = None

        for target in sorted_targets:
            dist, _ = wrapped_distance(ship_pos, target.position, self.arena_width, self.arena_height)

            # Solve ballistic intercept
            intercept_res = self.solve_ballistic_intercept(ship_pos, ship_vel, target.position, target.velocity)
            if intercept_res is None:
                continue

            t_int, int_pos, aim_heading = intercept_res
            if closest_aim_heading is None:
                closest_aim_heading = aim_heading

            # C3: Angular Alignment Constraint
            angle_diff = abs((aim_heading - telemetry.heading + 180.0) % 360.0 - 180.0)
            c3_angle = (angle_diff <= self.firing_tolerance_deg)

            # C4: Predicted Target Range & Intercept Time Suitability
            c4_range = (0.0 < t_int <= self.bullet_lifetime) and (dist < self.bullet_speed * self.bullet_lifetime)

            # C5: Safety Conditions (avoid shooting if too close, preventing immediate fragment blowback)
            target_size = getattr(target, "size_category", 2)
            if target_size > 2:
                # Large splitting asteroids (Sizes 3 and 4) produce multiple fragments; require safe standoff
                min_safe_dist = max(self.min_safe_target_distance, 145.0)
            elif target_size == 2:
                min_safe_dist = max(self.min_safe_target_distance, 100.0)
            else:
                # Size 1 asteroids are vaporized completely with zero fragments
                min_safe_dist = max(self.min_safe_target_distance * 0.7, 50.0)

            c5_safety = (dist >= min_safe_dist)

            audit = {
                "C1_cooldown_ready": c1_cooldown,
                "C2_ammo_available": c2_ammo,
                "C3_angular_alignment": c3_angle,
                "C4_range_suitable": c4_range,
                "C5_safety_satisfied": c5_safety,
            }

            all_satisfied = all(audit.values())

            if all_satisfied:
                return CSPFiringDecision(
                    should_shoot=True,
                    aim_heading=aim_heading,
                    target_track_id=target.track_id,
                    intercept_time=t_int,
                    intercept_point=(float(int_pos[0]), float(int_pos[1])),
                    constraint_audit=audit,
                    explanation=f"Approved shot on Track T{target.track_id} (err={angle_diff:.1f} deg <= {self.firing_tolerance_deg:.1f} deg, t_int={t_int:.2f}s).",
                )

            # Keep track of best candidate for aiming guidance
            if best_decision is None:
                reasons = []
                if not c3_angle:
                    reasons.append(f"Angle error {angle_diff:.1f} deg > {self.firing_tolerance_deg:.1f} deg")
                if not c4_range:
                    reasons.append(f"Intercept time {t_int:.2f}s > {self.bullet_lifetime:.2f}s")
                if not c5_safety:
                    reasons.append(f"Target dist {dist:.1f}px < min safe {min_safe_dist:.1f}px")

                best_decision = CSPFiringDecision(
                    should_shoot=False,
                    aim_heading=aim_heading,
                    target_track_id=target.track_id,
                    intercept_time=t_int,
                    intercept_point=(float(int_pos[0]), float(int_pos[1])),
                    constraint_audit=audit,
                    explanation=f"Denied shot on Track T{target.track_id}: {'; '.join(reasons)}.",
                )

        if best_decision is not None:
            return best_decision

        return CSPFiringDecision(
            should_shoot=False,
            aim_heading=closest_aim_heading,
            constraint_audit={
                "C1_cooldown_ready": c1_cooldown,
                "C2_ammo_available": c2_ammo,
                "C3_angular_alignment": False,
                "C4_range_suitable": False,
                "C5_safety_satisfied": False,
            },
            explanation="No valid ballistic intercept solution found for active targets.",
        )
