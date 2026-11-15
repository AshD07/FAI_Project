"""
Unit and integration tests for Stages 7, 8, and 9:
- Stage 7: Minimax Adversarial Search with Alpha-Beta Pruning
- Stage 8: Genetic Algorithm Offline Parameter Optimizer
- Stage 9: Decision Logging, JSON/CSV Export, and Explainability Summary Reporting
"""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.append(".")

from src.explainability.decision_logger import DecisionLogEntry, DecisionLogger
from src.optimization.genetic_algorithm import AgentGenome, GeneticOptimizer
from src.sensors.radar_sensor import ShipTelemetry
from src.strategy.minimax_tactics import (
    MinimaxTacticalPlanner,
    StrategicState,
    Tactic,
)
from src.tracking.object_tracker import TrackedEntity


def make_dummy_telemetry(
    pos=(400.0, 400.0),
    vel=(0.0, 0.0),
    heading=0.0,
    lives=3,
    ammo=100,
) -> ShipTelemetry:
    return ShipTelemetry(
        position=pos,
        velocity=vel,
        heading=heading,
        speed=float(np.linalg.norm(vel)),
        lives=lives,
        ammo=ammo,
        mines_remaining=3,
        bullet_cooldown=0.0,
        mine_cooldown=0.0,
        is_invulnerable=False,
    )


class TestStage7Minimax(unittest.TestCase):
    """Test Minimax Strategic Search, Alpha-Beta Pruning, and Macro-Tactics."""

    def setUp(self):
        self.planner = MinimaxTacticalPlanner()

    def test_minimax_solo_fallback(self):
        telemetry = make_dummy_telemetry()
        # No rival present
        decision = self.planner.plan_strategy(telemetry, rival_entities=[], asteroid_entities=[])
        self.assertEqual(decision.chosen_tactic, Tactic.CONTEST_ASTEROID)
        self.assertIn("solo", decision.target_description)

    def test_minimax_attack_superiority(self):
        # Agent has 3 lives and 100 ammo; rival has 1 life
        telemetry = make_dummy_telemetry(lives=3, ammo=100)
        rival = TrackedEntity(
            track_id=2,
            entity_type="ship",
            position=np.array([500.0, 400.0]),
            velocity=np.array([0.0, 0.0]),
            speed=0.0,
            estimated_radius=16.0,
            size_category=0,
            is_confirmed=True,
        )

        decision = self.planner.plan_strategy(telemetry, [rival], [])
        # With clear superiority and no asteroids, agent should choose ATTACK_RIVAL
        self.assertEqual(decision.chosen_tactic, Tactic.ATTACK_RIVAL)
        self.assertGreater(decision.evaluated_nodes, 0)
        self.assertIn("ATTACK_RIVAL", decision.explanation)

    def test_minimax_retreat_when_critical(self):
        # Agent has only 1 life and 0 ammo; rival is full strength
        telemetry = make_dummy_telemetry(lives=1, ammo=0)
        rival = TrackedEntity(
            track_id=2,
            entity_type="ship",
            position=np.array([450.0, 400.0]),
            velocity=np.array([0.0, 0.0]),
            speed=0.0,
            estimated_radius=16.0,
            size_category=0,
            is_confirmed=True,
        )

        decision = self.planner.plan_strategy(telemetry, [rival], [])
        self.assertEqual(decision.chosen_tactic, Tactic.RETREAT_SAFETY)


class TestStage8GeneticAlgorithm(unittest.TestCase):
    """Test Genetic Algorithm Genome Encoding/Decoding, Population Evolution, and Elitism."""

    def test_genome_encoding_and_decoding(self):
        orig = AgentGenome(
            safe_distance=150.0,
            closing_speed_threshold=50.0,
            shooting_safety_margin=65.0,
            astar_safety_margin=30.0,
            minimax_weight_rival=2.2,
            minimax_weight_contest=1.8,
            minimax_weight_safety=3.1,
            firing_tolerance_deg=8.5,
        )
        chrom = orig.to_chromosome()
        self.assertEqual(len(chrom), 8)
        self.assertTrue(np.all((chrom >= 0.0) & (chrom <= 1.0)))

        # Decode back
        decoded = AgentGenome.from_chromosome(chrom)
        self.assertAlmostEqual(decoded.safe_distance, orig.safe_distance, delta=1e-4)
        self.assertAlmostEqual(decoded.firing_tolerance_deg, orig.firing_tolerance_deg, delta=1e-4)
        self.assertAlmostEqual(decoded.astar_safety_margin, orig.astar_safety_margin, delta=1e-4)

    def test_optimizer_single_generation(self):
        optimizer = GeneticOptimizer(population_size=4, seed=42)
        summary = optimizer.evolve_generation(ticks_per_eval=30)

        self.assertEqual(summary["generation"], 1)
        self.assertIn("best_fitness", summary)
        self.assertIn("best_genome", summary)
        self.assertIsNotNone(optimizer.best_individual)


class TestStage9Explainability(unittest.TestCase):
    """Test Decision Logger, JSON/CSV Persistence, and Audit Reporting."""

    def test_decision_logger_export_and_summary(self):
        logger = DecisionLogger()

        entry = DecisionLogEntry(
            tick=1,
            timestamp=0.033,
            ship_pos=(400.0, 400.0),
            ship_speed=25.0,
            ship_heading=45.0,
            ship_lives=3,
            ship_ammo=100,
            active_rule="Rule_EmergencyCollisionAvoidance",
            rule_severity="EMERGENCY",
            rule_facts={"distance": 42.0, "safe_dist": 60.0, "closure_rate": 58.2, "tti": 0.72},
            rule_action="Evasive Burn",
            rule_explanation="Imminent asteroid impact detected.",
            minimax_tactic="CONTEST_ASTEROID",
            minimax_utility=75.4,
            rival_anticipated_tactic="RETREAT_SAFETY",
            csp_shoot=True,
            csp_target_id=1,
            csp_audit={"C1": True, "C2": True, "C3": True, "C4": True, "C5": True},
            csp_explanation="Approved shot.",
            actuator_turn=180.0,
            actuator_thrust=1.0,
            actuator_shoot=True,
            actuator_drop_mine=False,
        )
        logger.log(entry)

        summary_text = logger.generate_summary_report()
        self.assertIn("EXPLAINABILITY DECISION AUDIT SUMMARY REPORT", summary_text)
        self.assertIn("Rule_EmergencyCollisionAvoidance", summary_text)
        self.assertIn("CONTEST_ASTEROID", summary_text)

        # Test JSON & CSV export
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "audit.json")
            csv_file = os.path.join(tmpdir, "audit.csv")

            logger.export_json(json_file)
            logger.export_csv(csv_file)

            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(csv_file))
            self.assertGreater(os.path.getsize(json_file), 50)
            self.assertGreater(os.path.getsize(csv_file), 50)


if __name__ == "__main__":
    unittest.main()
