"""
Stage 4: Forward Chaining Rule Engine and Explainability System.
Implements reactive pilot reflexes (e.g. emergency collision avoidance, defensive ordnance),
priority inference, and auditable human-readable decision logs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.environment.game_engine import wrapped_distance
from src.sensors.radar_sensor import ShipTelemetry
from src.tracking.object_tracker import TrackedEntity


class RuleSeverity(Enum):
    EMERGENCY = "EMERGENCY"   # Critical safety override (e.g. imminent collision dodge)
    WARNING = "WARNING"       # Defensive positioning / hazard avoidance
    TACTICAL = "TACTICAL"     # Target engagement / mine deployment
    CRUISE = "CRUISE"         # Normal operation, delegate to pathfinder


@dataclass
class ExplanationRecord:
    """
    Human-readable auditable record of an active rule triggering an action.
    """
    timestamp: float
    rule_name: str
    severity: RuleSeverity
    trigger_facts: Dict[str, Any]
    action_directive: str
    message: str

    def format_log(self) -> str:
        facts_str = ", ".join(f"{k}={v}" for k, v in self.trigger_facts.items())
        return f"[{self.timestamp:5.2f}s] [{self.severity.value}] {self.rule_name}: {self.message} ({facts_str}) -> Action: {self.action_directive}"


@dataclass
class RuleActionDirective:
    """Action directive produced by the rule engine."""
    override_movement: bool = False
    turn_rate: float = 0.0
    thrust: float = 0.0
    drop_mine: bool = False
    target_heading: Optional[float] = None
    evade_vector: Optional[np.ndarray] = None
    delegate_to_astar: bool = True


@dataclass
class WorldFacts:
    """Structured situational facts extracted from telemetry and Kalman tracks."""
    timestamp: float
    ship_pos: np.ndarray
    ship_vel: np.ndarray
    ship_speed: float
    ship_heading: float
    ship_lives: int
    ship_ammo: int
    ship_mines: int

    # Hazard assessments
    nearest_asteroid_dist: float
    nearest_asteroid: Optional[TrackedEntity]
    imminent_collision: bool
    imminent_hazard: Optional[TrackedEntity]
    time_to_impact: float
    closure_rate: float
    hazard_relative_bearing: float

    # Pursuit / tailgating assessment
    pursuer_behind: bool
    pursuer_dist: float

    # Screen wall boundary assessment (ship cannot go out of screen)
    wall_proximity: bool = False
    wall_distance: float = 1000.0
    wall_repulsion: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    # Ammo status
    ammo_depleted: bool = False
    mines_depleted: bool = False


class ForwardChainingRuleEngine:
    """
    Forward Chaining Production System.
    Evaluates situation facts against priority-ranked rules to execute
    immediate reactive decisions and record explainability logs.
    """

    def __init__(
        self,
        safe_distance: float = 150.0,
        critical_tti: float = 2.2,
        high_closure_rate: float = 15.0,
        rear_mine_distance: float = 90.0,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
    ):
        self.safe_distance = safe_distance
        self.critical_tti = critical_tti
        self.high_closure_rate = high_closure_rate
        self.rear_mine_distance = rear_mine_distance
        self.arena_width = arena_width
        self.arena_height = arena_height

        self.explanation_log: List[ExplanationRecord] = []
        self.active_rule_name: str = "Rule_DefaultCruise"
        self.active_explanation: Optional[ExplanationRecord] = None
        self.last_evade_vector: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear per-episode memory (evasion hysteresis and explanation history)."""
        self.explanation_log.clear()
        self.active_rule_name = "Rule_DefaultCruise"
        self.active_explanation = None
        self.last_evade_vector = None

    def extract_facts(
        self,
        telemetry: ShipTelemetry,
        tracked_hazards: List[TrackedEntity],
        timestamp: float,
    ) -> WorldFacts:
        """Derive symbolic and quantitative situational facts from sensor & track data."""
        ship_pos = np.array(telemetry.position, dtype=np.float64)
        ship_vel = np.array(telemetry.velocity, dtype=np.float64)

        nearest_ast = None
        min_dist = float("inf")
        imminent_hazard = None
        min_tti = float("inf")
        max_closure = 0.0
        hazard_bearing = 0.0

        pursuer_behind = False
        min_rear_dist = float("inf")

        # Ship forward and rear unit vectors
        rad = np.radians(telemetry.heading)
        forward = np.array([np.sin(rad), -np.cos(rad)], dtype=np.float64)
        rear = -forward

        for hazard in tracked_hazards:
            dist, rel_vec = wrapped_distance(ship_pos, hazard.position, self.arena_width, self.arena_height)
            if dist < min_dist:
                min_dist = dist
                nearest_ast = hazard

            # Calculate relative velocity: v_rel = v_hazard - v_ship
            v_rel = hazard.velocity - ship_vel
            rel_speed = float(np.linalg.norm(v_rel))

            # Closure rate: projection of relative velocity onto distance vector towards ship
            # rel_vec points from ship to hazard. Direction towards ship is -rel_vec
            if dist > 1e-4:
                dir_to_ship = -rel_vec / dist
                closure = float(np.dot(hazard.velocity - ship_vel, dir_to_ship))
            else:
                closure = 0.0

            # Time to impact (TTI) approximation
            combined_radius = telemetry.speed * 0.05 + hazard.estimated_radius + 16.0
            effective_dist = max(0.0, dist - combined_radius)
            if closure > 5.0:
                tti = effective_dist / closure
                if tti < min_tti:
                    min_tti = tti
                    imminent_hazard = hazard
                    max_closure = closure
                    # Bearing from ship to hazard
                    hazard_bearing = float(np.degrees(np.arctan2(rel_vec[0], -rel_vec[1])) % 360.0)

            # Check if hazard is trailing directly behind the ship
            if dist < min_rear_dist and dist > 1e-4:
                dot_rear = float(np.dot(rel_vec / dist, rear))
                if dot_rear > 0.7:  # Within ~45 deg of directly behind
                    min_rear_dist = dist
                    if dist <= self.rear_mine_distance and closure > 10.0:
                        pursuer_behind = True

        # Screen wall boundary assessment: ensure ship never goes out of the screen
        wall_buffer = max(70.0, telemetry.speed * 0.65)
        d_left = ship_pos[0]
        d_right = self.arena_width - ship_pos[0]
        d_top = ship_pos[1]
        d_bottom = self.arena_height - ship_pos[1]
        min_wall_dist = min(d_left, d_right, d_top, d_bottom)

        rep_x = 0.0
        rep_y = 0.0
        if d_left < wall_buffer:
            rep_x += (wall_buffer - d_left) / wall_buffer
        elif d_right < wall_buffer:
            rep_x -= (wall_buffer - d_right) / wall_buffer

        if d_top < wall_buffer:
            rep_y += (wall_buffer - d_top) / wall_buffer
        elif d_bottom < wall_buffer:
            rep_y -= (wall_buffer - d_bottom) / wall_buffer

        wall_rep = np.array([rep_x, rep_y], dtype=np.float64)
        wall_prox = (min_wall_dist < wall_buffer) and (np.linalg.norm(wall_rep) > 0.05)

        return WorldFacts(
            timestamp=timestamp,
            ship_pos=ship_pos,
            ship_vel=ship_vel,
            ship_speed=telemetry.speed,
            ship_heading=telemetry.heading,
            ship_lives=telemetry.lives,
            ship_ammo=telemetry.ammo,
            ship_mines=telemetry.mines_remaining,
            nearest_asteroid_dist=min_dist,
            nearest_asteroid=nearest_ast,
            # Closing-speed threshold is the GA-tunable `closing_speed_threshold` gene
            # (previously hard-coded to 15.0, so the gene had no effect).
            imminent_collision=((min_tti < self.critical_tti and max_closure > self.high_closure_rate) or (min_dist < 65.0 and max_closure > 5.0)),
            imminent_hazard=imminent_hazard if imminent_hazard is not None else nearest_ast,
            time_to_impact=min_tti,
            closure_rate=max_closure,
            hazard_relative_bearing=hazard_bearing,
            pursuer_behind=pursuer_behind,
            pursuer_dist=min_rear_dist,
            wall_proximity=wall_prox,
            wall_distance=min_wall_dist,
            wall_repulsion=wall_rep,
            ammo_depleted=(telemetry.ammo <= 0),
            mines_depleted=(telemetry.mines_remaining <= 0),
        )

    def evaluate(self, facts: WorldFacts) -> Tuple[RuleActionDirective, ExplanationRecord]:
        """
        Execute forward chaining inference across rule set in priority order.
        Returns the action directive and audit explanation.
        """
        # =========================================================================
        # Rule 1: Defensive Mine Deployment on Rear Pursuer
        # IF hazard is in close pursuit directly behind the ship AND mines remain
        # THEN drop an explosive mine to neutralize the pursuer and boost forward
        # =========================================================================
        if facts.pursuer_behind and not facts.mines_depleted:
            record = ExplanationRecord(
                timestamp=facts.timestamp,
                rule_name="Rule_DefensiveMineDrop",
                severity=RuleSeverity.TACTICAL,
                trigger_facts={
                    "pursuer_dist": round(facts.pursuer_dist, 1),
                    "rear_threshold": self.rear_mine_distance,
                    "mines_remaining": facts.ship_mines,
                },
                action_directive="Deploy Defensive Mine & Boost Forward",
                message="Hazard in close tail pursuit. Deploying explosive rear mine.",
            )
            self._log(record)
            return RuleActionDirective(
                override_movement=True,
                turn_rate=0.0,
                thrust=1.0,  # Burn forward away from rear mine
                drop_mine=True,
                delegate_to_astar=False,
            ), record

        # =========================================================================
        # Rule 2: Emergency Collision Avoidance (Highest Priority for Head-on / Flank Hazards)
        # IF asteroid is on imminent collision course (TTI < threshold, high closure)
        # THEN perform immediate emergency evasive burn perpendicular/away from hazard
        # =========================================================================
        if facts.imminent_collision and facts.imminent_hazard is not None:
            # Compute evasive vectors relative to the collision course
            _, rel_vec = wrapped_distance(facts.ship_pos, facts.imminent_hazard.position, self.arena_width, self.arena_height)
            dist = float(np.linalg.norm(rel_vec))
            collision_axis = rel_vec / (dist + 1e-6)

            # Candidates: lateral dodge (perp1, perp2), radial push away, diagonal escapes, and retrograde brake
            perp1 = np.array([-collision_axis[1], collision_axis[0]])
            perp2 = np.array([collision_axis[1], -collision_axis[0]])
            away = -collision_axis
            diag1 = (perp1 + away * 0.7) / np.linalg.norm(perp1 + away * 0.7)
            diag2 = (perp2 + away * 0.7) / np.linalg.norm(perp2 + away * 0.7)

            candidate_vectors = [perp1, perp2, diag1, diag2, away]
            if facts.ship_speed > 30.0:
                retro = - facts.ship_vel / facts.ship_speed
                candidate_vectors.append(retro)

            best_choice = perp1
            best_score = -float("inf")

            for cand in candidate_vectors:
                # Target heading for candidate vector
                cand_heading = float(np.degrees(np.arctan2(cand[0], -cand[1])) % 360.0)
                turn_cost = abs((cand_heading - facts.ship_heading + 180.0) % 360.0 - 180.0)

                # Projected lookahead position (120px)
                proj_pos = facts.ship_pos + cand * 120.0

                # 1. Screen wall boundary clearance penalty
                wall_d = min(proj_pos[0], self.arena_width - proj_pos[0], proj_pos[1], self.arena_height - proj_pos[1])
                wall_score = 0.0
                if wall_d < 120.0:
                    wall_score = -900.0 * (120.0 - wall_d) / 120.0

                # 2. Radial clearance bonus (moving away from collision axis)
                rad_clearance = float(np.dot(cand, away)) * 50.0

                # 3. Kinetic drift continuity / speed relief bonus
                drift_score = 0.0
                if facts.ship_speed > 10.0:
                    drift_score = float(np.dot(facts.ship_vel / facts.ship_speed, cand)) * 25.0
                    # If moving too fast, reward braking vectors that shed speed
                    if facts.ship_speed > 80.0 and np.dot(facts.ship_vel, cand) < 0:
                        drift_score += ((facts.ship_speed - 80.0) / 30.0) * 70.0

                # 4. Turn cost penalty: strongly favor directions reachable quickly with 180 deg/s max turn rate
                turn_weight = 100.0 if facts.nearest_asteroid_dist < 120.0 else 60.0
                turn_score = - (turn_cost / 180.0) * turn_weight
                # If turn cannot be completed before collision impact, heavily penalize impossible evasion vector
                if facts.time_to_impact > 0.0 and turn_cost > facts.time_to_impact * 180.0:
                    turn_score -= 300.0 * (turn_cost / (facts.time_to_impact * 180.0))

                # 5. Hysteresis commitment bonus (prevents alternating left/right oscillation)
                hysteresis_score = 0.0
                if self.last_evade_vector is not None:
                    hysteresis_score = float(np.dot(cand, self.last_evade_vector)) * 75.0

                # 6. Relative Lateral Opening Rate (COLREGs Anti-Collision):
                # Never dodge across the bow of an oncoming hazard. Dodge in the opposite lateral
                # direction so the hazard passes safely behind the ship.
                hazard_vel = facts.imminent_hazard.velocity
                h_lat = hazard_vel - float(np.dot(hazard_vel, collision_axis)) * collision_axis
                h_lat_norm = float(np.linalg.norm(h_lat))
                lateral_score = 0.0
                if h_lat_norm > 15.0:
                    cand_lat = cand - float(np.dot(cand, collision_axis)) * collision_axis
                    cand_lat_norm = float(np.linalg.norm(cand_lat))
                    if cand_lat_norm > 1e-3:
                        lat_align = float(np.dot(cand_lat / cand_lat_norm, h_lat / h_lat_norm))
                        # Strong penalty if dodging in the same direction the hazard is traveling
                        lateral_score = - lat_align * 90.0

                total_score = wall_score + rad_clearance + drift_score + turn_score + hysteresis_score + lateral_score
                if total_score > best_score:
                    best_score = total_score
                    best_choice = cand

            choice = best_choice
            self.last_evade_vector = choice
            target_heading = float(np.degrees(np.arctan2(choice[0], -choice[1])) % 360.0)

            # Angle difference from current heading
            angle_diff = (target_heading - facts.ship_heading + 180.0) % 360.0 - 180.0
            turn_cmd = 180.0 if angle_diff > 0 else -180.0

            # Proportional Propulsion Modulation:
            # - If burn actively opposes velocity (retro-braking): full 1.0 braking power
            # - If point-blank emergency (dist < 65px): full 1.0 thrust to survive
            # - If speed is elevated, scale thrust proportionally to prevent runaway while maintaining evasive authority
            rad = np.radians(facts.ship_heading)
            ship_forward = np.array([np.sin(rad), -np.cos(rad)], dtype=np.float64)
            vel_dir = facts.ship_vel / (facts.ship_speed + 1e-5)
            is_braking = float(np.dot(ship_forward, vel_dir)) < -0.2

            if is_braking or facts.nearest_asteroid_dist < 65.0:
                base_thrust = 1.0
            elif facts.ship_speed > 90.0:
                base_thrust = 0.0
            elif facts.ship_speed > 70.0:
                base_thrust = max(0.2, (90.0 - facts.ship_speed) / 20.0)
            else:
                base_thrust = 1.0

            max_align = 65.0 if facts.nearest_asteroid_dist < 75.0 else 45.0
            thrust_cmd = base_thrust if abs(angle_diff) < max_align else 0.0

            record = ExplanationRecord(
                timestamp=facts.timestamp,
                rule_name="Rule_EmergencyCollisionAvoidance",
                severity=RuleSeverity.EMERGENCY,
                trigger_facts={
                    "distance": round(facts.nearest_asteroid_dist, 1),
                    "safe_dist": round(self.safe_distance, 1),
                    "closure_rate": round(facts.closure_rate, 1),
                    "tti": round(facts.time_to_impact, 2),
                },
                action_directive=f"Evasive Burn (Turn: {turn_cmd:.0f} deg/s, Thrust: {thrust_cmd:.1f})",
                message="Imminent asteroid impact detected. Emergency evasive maneuver triggered.",
            )
            self._log(record)
            return RuleActionDirective(
                override_movement=True,
                turn_rate=turn_cmd,
                thrust=thrust_cmd,
                drop_mine=False,
                target_heading=target_heading,
                evade_vector=choice,
                delegate_to_astar=False,
            ), record

        # If imminent collision has passed, reset evasion vector memory
        self.last_evade_vector = None

        # =========================================================================
        # Rule 2B: Screen Wall Boundary Repulsion (Ship cannot go out of screen)
        # IF ship is within safety buffer of screen borders and heading toward walls
        # THEN steer inward toward arena center and apply propulsion
        # =========================================================================
        if facts.wall_proximity:
            inward_heading = float(np.degrees(np.arctan2(facts.wall_repulsion[0], -facts.wall_repulsion[1])) % 360.0)
            angle_diff = (inward_heading - facts.ship_heading + 180.0) % 360.0 - 180.0
            turn_cmd = 180.0 if angle_diff > 0 else -180.0
            # Apply strong inward propulsion when pointing away from wall; zero thrust outward towards boundary
            thrust_cmd = 1.0 if abs(angle_diff) < 50.0 else 0.0

            record = ExplanationRecord(
                timestamp=facts.timestamp,
                rule_name="Rule_WallBoundaryRepulsion",
                severity=RuleSeverity.WARNING,
                trigger_facts={
                    "wall_distance": round(facts.wall_distance, 1),
                    "wall_buffer": max(70.0, facts.ship_speed * 0.65),
                },
                action_directive=f"Inward Boundary Repulsion (Turn: {turn_cmd:.0f} deg/s, Thrust: {thrust_cmd:.1f})",
                message="Ship approaching screen boundary. Steering inward to maintain safe arena positioning.",
            )
            self._log(record)
            return RuleActionDirective(
                override_movement=True,
                turn_rate=turn_cmd,
                thrust=thrust_cmd,
                drop_mine=False,
                target_heading=inward_heading,
                delegate_to_astar=False,
            ), record

        # =========================================================================
        # Rule 3: Proximity Hazard Caution
        # IF nearest asteroid is closer than safe distance but not on direct collision
        # THEN steer gently away and delegate pathfinding to A*
        # =========================================================================
        if facts.nearest_asteroid_dist < self.safe_distance:
            record = ExplanationRecord(
                timestamp=facts.timestamp,
                rule_name="Rule_ProximityHazardCaution",
                severity=RuleSeverity.WARNING,
                trigger_facts={
                    "distance": round(facts.nearest_asteroid_dist, 1),
                    "safe_threshold": round(self.safe_distance, 1),
                },
                action_directive="Local A* Repulsion Path",
                message="Asteroid inside proximity safety perimeter. Relying on A* hazard clearance.",
            )
            self._log(record)
            return RuleActionDirective(
                override_movement=False,
                drop_mine=False,
                delegate_to_astar=True,
            ), record

        # =========================================================================
        # Rule 4: Normal Cruise & Tactical Hunt (Default Fallthrough)
        # IF no immediate threats or overrides
        # THEN delegate navigation to A* search and firing to CSP
        # =========================================================================
        record = ExplanationRecord(
            timestamp=facts.timestamp,
            rule_name="Rule_DefaultCruise",
            severity=RuleSeverity.CRUISE,
            trigger_facts={
                "nearest_dist": round(facts.nearest_asteroid_dist, 1),
                "ammo": facts.ship_ammo,
            },
            action_directive="Normal Cruise / A* Navigation",
            message="No immediate collision hazard. Executing standard autonomous pathfinding.",
        )
        self._log(record)
        return RuleActionDirective(
            override_movement=False,
            drop_mine=False,
            delegate_to_astar=True,
        ), record

    def _log(self, record: ExplanationRecord) -> None:
        self.active_rule_name = record.rule_name
        self.active_explanation = record
        # Keep log size bounded
        if len(self.explanation_log) > 500:
            self.explanation_log.pop(0)
        self.explanation_log.append(record)

    def get_recent_explanations(self, n: int = 5) -> List[ExplanationRecord]:
        return self.explanation_log[-n:]
