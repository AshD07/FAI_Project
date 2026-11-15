"""
Stage 7: Adversarial Search using Minimax with Alpha-Beta Pruning.
Evaluates strategic macro-tactics (Attack Rival, Contest Asteroid, Retreat to Safety)
against an opposing spacecraft 2 steps ahead.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import ShipConfig
from src.environment.game_engine import wrapped_distance
from src.sensors.radar_sensor import ShipTelemetry
from src.tracking.object_tracker import TrackedEntity


class Tactic(Enum):
    ATTACK_RIVAL = "ATTACK_RIVAL"           # Move towards and engage opponent spacecraft
    CONTEST_ASTEROID = "CONTEST_ASTEROID"   # Race to secure & destroy high-value asteroid
    RETREAT_SAFETY = "RETREAT_SAFETY"       # Move away from high-density danger towards safe sector


@dataclass
class MinimaxStrategicDecision:
    """Output decision from Minimax with Alpha-Beta Pruning."""
    chosen_tactic: Tactic
    target_waypoint: Tuple[float, float]
    utility_score: float
    best_rival_response: Tactic
    target_description: str
    alpha_beta_pruned_nodes: int
    evaluated_nodes: int
    explanation: str


@dataclass
class StrategicState:
    """Lightweight abstraction of the game state for 2-ply adversarial tree search."""
    agent_pos: np.ndarray
    agent_lives: int
    agent_ammo: int
    rival_pos: np.ndarray
    rival_lives: int
    asteroids: List[Tuple[np.ndarray, float, int]]  # (pos, radius, score_value)


class MinimaxTacticalPlanner:
    """
    Minimax Search with Alpha-Beta Pruning for macro-tactical decision making.
    Looks 2 steps ahead to pick the dominant strategy against a rival pilot.
    """

    def __init__(
        self,
        weight_score: float = 1.5,
        weight_lives: float = 3.0,
        weight_contest: float = 2.0,
        weight_safety: float = 2.5,
        arena_width: float = 1000.0,
        arena_height: float = 800.0,
        weight_rival: float = 1.5,
    ):
        self.weight_score = weight_score
        # Scales the ATTACK_RIVAL utility (GA gene `minimax_weight_rival`; previously unused)
        self.weight_rival = weight_rival
        self.weight_lives = weight_lives
        self.weight_contest = weight_contest
        self.weight_safety = weight_safety
        self.arena_width = arena_width
        self.arena_height = arena_height

    def evaluate_state(
        self,
        agent_tactic: Tactic,
        rival_tactic: Tactic,
        state: StrategicState,
    ) -> float:
        """
        Static evaluation function U(s) assessing strategic outcome:
        Rewards high-value asteroid control, combat superiority, and hazard avoidance.
        """
        utility = 0.0

        # Distance between agent and rival
        agent_rival_dist, _ = wrapped_distance(state.agent_pos, state.rival_pos, self.arena_width, self.arena_height)

        # 1. Lives advantage
        utility += self.weight_lives * (state.agent_lives - state.rival_lives) * 20.0

        # 2. Combat tactical utility (Attack vs Rival)
        if agent_tactic == Tactic.ATTACK_RIVAL:
            if state.agent_ammo > 10 and state.agent_lives >= state.rival_lives:
                # Advantageous offensive posture
                attack_value = (800.0 - min(agent_rival_dist, 800.0)) * 0.1
                if rival_tactic == Tactic.RETREAT_SAFETY:
                    attack_value += 30.0  # Successfully pressing retreating rival
                utility += self.weight_rival * attack_value
            else:
                # Risky attack with low ammo or life deficit
                utility -= 40.0

        # 3. Asteroid Contestation utility
        if agent_tactic == Tactic.CONTEST_ASTEROID and state.asteroids:
            # Find closest asteroid to agent
            dists_agent = [wrapped_distance(state.agent_pos, a[0], self.arena_width, self.arena_height)[0] for a in state.asteroids]
            dists_rival = [wrapped_distance(state.rival_pos, a[0], self.arena_width, self.arena_height)[0] for a in state.asteroids]

            min_idx = int(np.argmin(dists_agent))
            target_val = state.asteroids[min_idx][2]

            # Agent closer than rival to asteroid -> favorable contest
            if dists_agent[min_idx] < dists_rival[min_idx]:
                utility += self.weight_contest * (target_val + 20.0)
            else:
                utility += target_val * 0.5

        # 4. Safety & Retreat utility
        if agent_tactic == Tactic.RETREAT_SAFETY:
            # Desirable if low on lives or surrounded by hazards
            if state.agent_lives <= 1 or state.agent_ammo <= 5:
                utility += self.weight_safety * 35.0
            else:
                utility -= 15.0  # Penalize passive retreat when healthy

        # Rival counter-tactic penalties
        if rival_tactic == Tactic.ATTACK_RIVAL and agent_tactic == Tactic.ATTACK_RIVAL:
            # Mutual dogfight -> higher risk
            utility -= 10.0

        return float(utility)

    def plan_strategy(
        self,
        telemetry: ShipTelemetry,
        rival_entities: List[TrackedEntity],
        asteroid_entities: List[TrackedEntity],
    ) -> MinimaxStrategicDecision:
        """
        Executes 2-ply Minimax with Alpha-Beta pruning over tactics:
        Max Node (Agent): ATTACK_RIVAL, CONTEST_ASTEROID, RETREAT_SAFETY
        Min Node (Rival): ATTACK_RIVAL, CONTEST_ASTEROID, RETREAT_SAFETY
        """
        agent_pos = np.array(telemetry.position, dtype=np.float64)

        # Fallback if no rival exists in environment
        if not rival_entities:
            # Check hazard pressure: if nearby asteroids are close or density is high, retreat to safe quadrant
            min_dist = float("inf")
            close_hazard_count = 0
            for a in asteroid_entities:
                d, _ = wrapped_distance(agent_pos, a.position, self.arena_width, self.arena_height)
                if d < min_dist:
                    min_dist = d
                if d < 180.0:
                    close_hazard_count += 1

            if close_hazard_count >= 2 or min_dist < 140.0:
                best_tactic = Tactic.RETREAT_SAFETY
                target_pt = self._get_safest_sector(agent_pos, agent_pos, asteroid_entities)
                desc = f"Safest Sector Waypoint at ({target_pt[0]:.0f}, {target_pt[1]:.0f}) (solo mode)"
                expl = "High hazard density detected in solo arena. Relocating to low-density sector."
            else:
                best_tactic = Tactic.CONTEST_ASTEROID
                target_pt = self._get_best_asteroid_pos(agent_pos, asteroid_entities)
                desc = f"Standoff Waypoint at ({target_pt[0]:.0f}, {target_pt[1]:.0f}) (solo mode)"
                expl = "Clear tactical envelope. Standoff targeting active."

            return MinimaxStrategicDecision(
                chosen_tactic=best_tactic,
                target_waypoint=target_pt,
                utility_score=50.0,
                best_rival_response=Tactic.RETREAT_SAFETY,
                target_description=desc,
                alpha_beta_pruned_nodes=0,
                evaluated_nodes=1,
                explanation=expl,
            )

        rival = rival_entities[0]
        rival_pos = rival.position.copy()

        # Build simplified state
        ast_tuples = [(a.position.copy(), a.estimated_radius, a.size_category * 30) for a in asteroid_entities]
        state = StrategicState(
            agent_pos=agent_pos,
            agent_lives=telemetry.lives,
            agent_ammo=telemetry.ammo,
            rival_pos=rival_pos,
            rival_lives=ShipConfig.initial_lives,  # Rival is assumed to start with the same lives as us
            asteroids=ast_tuples,
        )

        tactics = [Tactic.ATTACK_RIVAL, Tactic.CONTEST_ASTEROID, Tactic.RETREAT_SAFETY]

        best_agent_tactic = Tactic.CONTEST_ASTEROID
        best_rival_response = Tactic.RETREAT_SAFETY
        max_eval = -float("inf")
        alpha = -float("inf")

        nodes_evaluated = 0
        nodes_pruned = 0

        # MAX PLAYER: Agent chooses tactic
        for a_tac in tactics:
            min_eval = float("inf")
            current_rival_best = None
            # Each MIN node starts with its own beta (= +inf passed down from the root).
            # Previously beta was shared across sibling MIN nodes, so after the first
            # sibling every later one was cut off after a single child and returned an
            # over-optimistic value, which could select a dominated tactic.
            beta = float("inf")

            # MIN PLAYER: Anticipate rival's counter-tactic
            for r_tac in tactics:
                nodes_evaluated += 1
                eval_val = self.evaluate_state(a_tac, r_tac, state)

                if eval_val < min_eval:
                    min_eval = eval_val
                    current_rival_best = r_tac

                # Alpha-Beta pruning condition for Min node
                beta = min(beta, min_eval)
                if beta <= alpha:
                    nodes_pruned += 1
                    break  # Prune remaining rival branches

            if min_eval > max_eval:
                max_eval = min_eval
                best_agent_tactic = a_tac
                best_rival_response = current_rival_best or Tactic.RETREAT_SAFETY

            alpha = max(alpha, max_eval)

        # Compute specific target waypoint based on chosen tactic
        if best_agent_tactic == Tactic.ATTACK_RIVAL:
            target_pt = (float(rival_pos[0]), float(rival_pos[1]))
            desc = f"Rival Spacecraft at ({rival_pos[0]:.0f}, {rival_pos[1]:.0f})"
            expl = f"Minimax selected ATTACK_RIVAL (U={max_eval:.1f}). Pressing rival, anticipating rival {best_rival_response.value}."
        elif best_agent_tactic == Tactic.CONTEST_ASTEROID:
            target_pt = self._get_best_asteroid_pos(agent_pos, asteroid_entities)
            desc = f"Contested Asteroid at ({target_pt[0]:.0f}, {target_pt[1]:.0f})"
            expl = f"Minimax selected CONTEST_ASTEROID (U={max_eval:.1f}). Securing high-value target ahead of rival."
        else:
            target_pt = self._get_safest_sector(agent_pos, rival_pos, asteroid_entities)
            desc = f"Safe Quadrant Waypoint at ({target_pt[0]:.0f}, {target_pt[1]:.0f})"
            expl = f"Minimax selected RETREAT_SAFETY (U={max_eval:.1f}). Tactical relocation away from threat envelope."

        return MinimaxStrategicDecision(
            chosen_tactic=best_agent_tactic,
            target_waypoint=target_pt,
            utility_score=max_eval,
            best_rival_response=best_rival_response,
            target_description=desc,
            alpha_beta_pruned_nodes=nodes_pruned,
            evaluated_nodes=nodes_evaluated,
            explanation=expl,
        )

    def _get_best_asteroid_pos(self, ship_pos: np.ndarray, asteroids: List[TrackedEntity]) -> Tuple[float, float]:
        """Finds highest-priority asteroid and computes a safe standoff firing waypoint."""
        if not asteroids:
            return (self.arena_width * 0.5, self.arena_height * 0.5)

        # Prioritize smaller asteroids (1 hit kill) and threatening nearby asteroids
        def asteroid_priority(a: TrackedEntity) -> float:
            d, _ = wrapped_distance(ship_pos, a.position, self.arena_width, self.arena_height)
            # Smaller size gets higher elimination priority to clear screen clutter
            size_bonus = {1: 80.0, 2: 50.0, 3: 20.0, 4: 5.0}.get(a.size_category, 10.0)
            proximity_score = 250.0 / (d + 30.0)
            return size_bonus + proximity_score

        best_ast = max(asteroids, key=asteroid_priority)

        # Standoff waypoint: maintain ~230px standoff distance along line from asteroid to ship
        dist, rel_vec = wrapped_distance(best_ast.position, ship_pos, self.arena_width, self.arena_height)
        standoff_dist = 230.0
        if dist > 1e-4:
            unit_dir = rel_vec / dist
        else:
            unit_dir = np.array([0.0, -1.0])

        standoff_pt = best_ast.position + unit_dir * standoff_dist
        # Clamp standoff waypoint safely within screen boundaries
        margin = 100.0
        clamped_x = float(np.clip(standoff_pt[0], margin, self.arena_width - margin))
        clamped_y = float(np.clip(standoff_pt[1], margin, self.arena_height - margin))
        return (clamped_x, clamped_y)

    def _get_safest_sector(self, ship_pos: np.ndarray, rival_pos: np.ndarray, asteroids: List[TrackedEntity]) -> Tuple[float, float]:
        """Calculates a waypoint in the sector with lowest hazard density and distance from rival, keeping well clear of boundaries."""
        sectors = [
            np.array([self.arena_width * 0.35, self.arena_height * 0.35]),
            np.array([self.arena_width * 0.65, self.arena_height * 0.35]),
            np.array([self.arena_width * 0.35, self.arena_height * 0.65]),
            np.array([self.arena_width * 0.65, self.arena_height * 0.65]),
            np.array([self.arena_width * 0.50, self.arena_height * 0.50]),
        ]

        best_sector = sectors[4]
        min_density = float("inf")

        for s in sectors:
            # Density = sum of inverse distances to hazards + proximity to rival
            rival_d, _ = wrapped_distance(s, rival_pos, self.arena_width, self.arena_height)
            density = 100.0 / (rival_d + 10.0)

            for a in asteroids:
                ad, _ = wrapped_distance(s, a.position, self.arena_width, self.arena_height)
                density += 200.0 / (ad + 10.0)

            if density < min_density:
                min_density = density
                best_sector = s

        return (float(best_sector[0]), float(best_sector[1]))
