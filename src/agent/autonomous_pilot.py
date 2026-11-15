"""
Integrated Autonomous Spacecraft Pilot.
Unifies Stages 1-9 into a single real-time decision loop:
Sensors -> Kalman MOT Tracking -> Rule Engine -> Minimax Adversarial Search -> A* Navigation -> CSP Firing -> Actuators -> Explainability Logger.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.environment.game_engine import wrapped_distance
from src.explainability.decision_logger import DecisionLogEntry, DecisionLogger
from src.firing.csp_shooter import CSPFiringDecision, CSPShooter
from src.navigation.astar_planner import AStarLocalPlanner, AStarPlanResult
from src.optimization.genetic_algorithm import AgentGenome
from src.reasoning.rule_engine import (
    ExplanationRecord,
    ForwardChainingRuleEngine,
    RuleActionDirective,
    WorldFacts,
)
from src.sensors.radar_sensor import SensorObservation
from src.strategy.minimax_tactics import (
    MinimaxStrategicDecision,
    MinimaxTacticalPlanner,
)
from src.tracking.object_tracker import MultiObjectTracker, TrackedEntity


@dataclass
class AgentDecisionOutput:
    """Complete output produced by the autonomous pilot on each tick."""
    turn_rate: float
    thrust: float
    shoot: bool
    drop_mine: bool
    active_rule: str
    active_explanation: str
    csp_decision: CSPFiringDecision
    minimax_decision: Optional[MinimaxStrategicDecision] = None
    astar_plan: Optional[AStarPlanResult] = None
    facts: Optional[WorldFacts] = None
    log_entry: Optional[DecisionLogEntry] = None


class AutonomousPilot:
    """
    Main Autonomous Agent Class implementing Stages 1 through 9.
    """

    def __init__(
        self,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
        genome: Optional[AgentGenome] = None,
        safe_distance: float = 150.0,
        critical_tti: float = 2.2,
        firing_tolerance_deg: float = 7.5,
    ):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.tick_count = 0
        # Agility speed governor (px/s). Cruise limit applies to tactical (A*) motion;
        # emergency limit caps forward thrust during reflexive evasion.
        self.cruise_speed_limit = 70.0
        # Was 70 px/s (same as cruise). Asteroids move at 30-165 px/s, so capping the
        # escape burn at 70 px/s made many hazards impossible to outrun: in a 15-seed
        # probe every death happened mid-evasion at ~70-78 px/s. Raising the emergency
        # cap to 200 px/s raised mean solo survival from 16.2 s to 18.2 s (median
        # 13.4 s -> 16.9 s) over 70 seeds while cruise speed stays governed at 70 px/s.
        self.emergency_speed_limit = 200.0

        # Load genome if provided, or default configuration
        self.genome = genome or AgentGenome(
            safe_distance=safe_distance,
            firing_tolerance_deg=firing_tolerance_deg,
        )

        # Stage 3: Kalman Multi-Object Tracker
        self.tracker = MultiObjectTracker(arena_width=arena_width, arena_height=arena_height)

        # Stage 4: Forward Chaining Rule Engine
        self.rule_engine = ForwardChainingRuleEngine(
            safe_distance=self.genome.safe_distance,
            critical_tti=critical_tti,
            high_closure_rate=self.genome.closing_speed_threshold,
            rear_mine_distance=90.0,
            arena_width=arena_width,
            arena_height=arena_height,
        )

        # Stage 5: Local Egocentric A* Planner
        self.planner = AStarLocalPlanner(
            safety_margin=self.genome.astar_safety_margin,
            arena_width=arena_width,
            arena_height=arena_height,
        )

        # Stage 6: Constraint Satisfaction Problem Firing Controller
        self.shooter = CSPShooter(
            firing_tolerance_deg=self.genome.firing_tolerance_deg,
            min_safe_target_distance=self.genome.shooting_safety_margin,
            arena_width=arena_width,
            arena_height=arena_height,
        )

        # Stage 7: Minimax Adversarial Strategic Planner
        self.minimax_planner = MinimaxTacticalPlanner(
            weight_lives=3.0,
            weight_contest=self.genome.minimax_weight_contest,
            weight_safety=self.genome.minimax_weight_safety,
            weight_rival=self.genome.minimax_weight_rival,
            arena_width=arena_width,
            arena_height=arena_height,
        )

        # Stage 9: Decision Logger
        self.logger = DecisionLogger()

    def reset(self) -> None:
        """
        Clear all per-episode state (tracks, tick counter, rule hysteresis, decision log).
        Must be called between independent matches; reusing a pilot without a reset
        carries stale Kalman tracks and a backwards clock into the next game.
        """
        self.tick_count = 0
        self.tracker.reset()
        self.logger = DecisionLogger()
        if hasattr(self.rule_engine, "reset"):
            self.rule_engine.reset()

    def decide(self, obs: SensorObservation) -> Tuple[float, float, bool, bool, AgentDecisionOutput]:
        """
        Executes complete cognitive cycle:
        Sensors -> Kalman -> Rules -> Minimax -> A* -> CSP -> Actuators -> Logger.
        """
        self.tick_count += 1
        telemetry = obs.telemetry
        ship_pos = np.array(telemetry.position, dtype=np.float64)

        # 1. State Estimation & Tracking (Stage 3)
        tracked_entities = self.tracker.update(obs)
        tracked_asteroids = self.tracker.get_asteroids(tracked_entities)
        tracked_rivals = self.tracker.get_rivals(tracked_entities)

        # 2. Logic & Knowledge Representation - Rule Engine (Stage 4)
        facts = self.rule_engine.extract_facts(telemetry, tracked_asteroids, obs.timestamp)
        rule_directive, explanation = self.rule_engine.evaluate(facts)

        # 3. Adversarial Search - Minimax with Alpha-Beta Pruning (Stage 7)
        minimax_decision = self.minimax_planner.plan_strategy(telemetry, tracked_rivals, tracked_asteroids)

        # 4. Constraint Satisfaction Problem - Firing (Stage 6)
        # Select target pool based on Minimax tactic
        target_candidates = tracked_rivals + tracked_asteroids if minimax_decision.chosen_tactic.value == "ATTACK_RIVAL" else tracked_asteroids
        csp_decision = self.shooter.evaluate(telemetry, target_candidates)

        # 5. Movement Planning Arbitration (Stages 4 & 5)
        astar_res = None
        turn_cmd = 0.0
        thrust_cmd = 0.0
        
        # In single-player survival mode, strictly never shoot while evading an emergency hazard or repelling from screen walls
        if rule_directive.override_movement or facts.imminent_collision or facts.wall_proximity:
            shoot_cmd = False
        else:
            shoot_cmd = csp_decision.should_shoot

        # In single-player survival mode without mines, drop_mine is strictly False
        drop_mine_cmd = False if telemetry.mines_remaining == 0 else rule_directive.drop_mine

        speed = telemetry.speed
        cruise_speed_limit = self.cruise_speed_limit

        if rule_directive.override_movement:
            # Reflexive emergency dodge overrides tactical planning
            turn_cmd = rule_directive.turn_rate
            thrust_cmd = rule_directive.thrust
            # In single life survival, strictly cap forward thrust to preserve maneuverability
            if speed > self.emergency_speed_limit:
                rad = np.radians(telemetry.heading)
                ship_forward = np.array([np.sin(rad), -np.cos(rad)], dtype=np.float64)
                vel_dir = np.array(telemetry.velocity, dtype=np.float64) / (speed + 1e-5)
                if np.dot(ship_forward, vel_dir) > 0.0:
                    thrust_cmd = 0.0
        else:
            # Active Retro-Braking: if excessive kinetic momentum builds up, execute counter-burn to regain maneuverability
            if speed > 85.0 and facts.nearest_asteroid_dist > self.rule_engine.safe_distance * 0.7:
                vel = np.array(telemetry.velocity, dtype=np.float64)
                # Heading pointing opposite to current velocity vector
                retro_heading = float(np.degrees(np.arctan2(-vel[0], vel[1])) % 360.0)
                retro_diff = (retro_heading - telemetry.heading + 180.0) % 360.0 - 180.0
                turn_cmd = float(np.clip(retro_diff * 5.0, -180.0, 180.0))
                thrust_cmd = 1.0 if abs(retro_diff) < 30.0 else 0.0
            # Orient towards ballistic intercept shot only when safe from close proximity hazards
            elif not rule_directive.delegate_to_astar and csp_decision.aim_heading is not None and facts.nearest_asteroid_dist > self.rule_engine.safe_distance:
                aim_diff = (csp_decision.aim_heading - telemetry.heading + 180.0) % 360.0 - 180.0
                turn_cmd = float(np.clip(aim_diff * 4.0, -180.0, 180.0))
                # Rotate in place to fire; strictly zero forward thrust while aiming at asteroids
                thrust_cmd = 0.0
            else:
                # Delegate navigation to Local A* Pathfinding using Minimax tactical waypoint
                target_waypoint = np.array(minimax_decision.target_waypoint, dtype=np.float64)
                astar_res = self.planner.plan(telemetry, tracked_asteroids, target_world_pos=target_waypoint)
                turn_cmd = astar_res.turn_rate
                # Apply agility speed governor to A* thrust
                if speed > cruise_speed_limit:
                    thrust_cmd = 0.0  # Coast and allow natural drag to decelerate
                elif speed > cruise_speed_limit * 0.75:
                    thrust_cmd = min(astar_res.thrust * 0.35, 0.3)
                else:
                    thrust_cmd = min(astar_res.thrust * 0.7, 0.6)

        # 6. Stage 9: Explainability Decision Logging
        log_entry = DecisionLogEntry(
            tick=self.tick_count,
            timestamp=obs.timestamp,
            ship_pos=telemetry.position,
            ship_speed=telemetry.speed,
            ship_heading=telemetry.heading,
            ship_lives=telemetry.lives,
            ship_ammo=telemetry.ammo,
            active_rule=explanation.rule_name,
            rule_severity=explanation.severity.value,
            rule_facts=explanation.trigger_facts,
            rule_action=explanation.action_directive,
            rule_explanation=explanation.message,
            minimax_tactic=minimax_decision.chosen_tactic.value,
            minimax_utility=minimax_decision.utility_score,
            rival_anticipated_tactic=minimax_decision.best_rival_response.value,
            csp_shoot=csp_decision.should_shoot,
            csp_target_id=csp_decision.target_track_id,
            csp_audit=csp_decision.constraint_audit,
            csp_explanation=csp_decision.explanation,
            actuator_turn=turn_cmd,
            actuator_thrust=thrust_cmd,
            actuator_shoot=shoot_cmd,
            actuator_drop_mine=drop_mine_cmd,
        )
        self.logger.log(log_entry)

        decision_output = AgentDecisionOutput(
            turn_rate=turn_cmd,
            thrust=thrust_cmd,
            shoot=shoot_cmd,
            drop_mine=drop_mine_cmd,
            active_rule=explanation.rule_name,
            active_explanation=explanation.message,
            csp_decision=csp_decision,
            minimax_decision=minimax_decision,
            astar_plan=astar_res,
            facts=facts,
            log_entry=log_entry,
        )

        return turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd, decision_output
