"""
Unit and integration tests for Stages 4, 5, and 6:
- Stage 4: Forward Chaining Rule Engine, Collision Avoidance, and Explainability Logs
- Stage 5: Local Egocentric A* Pathfinding with Dynamic Obstacle Inflation
- Stage 6: Constraint Satisfaction Problem (CSP) Firing Controller with Quadratic Intercept
"""

import sys
import unittest
import numpy as np

sys.path.append(".")

from src.firing.csp_shooter import CSPShooter
from src.navigation.astar_planner import AStarLocalPlanner
from src.reasoning.rule_engine import (
    ForwardChainingRuleEngine,
    RuleSeverity,
)
from src.sensors.radar_sensor import ShipTelemetry
from src.tracking.object_tracker import TrackedEntity
from src.agent.autonomous_pilot import AutonomousPilot
from src.sensors.radar_sensor import SensorObservation


def make_dummy_telemetry(
    pos=(400.0, 400.0),
    vel=(0.0, 0.0),
    heading=0.0,
    lives=3,
    ammo=100,
    mines=3,
    cooldown=0.0,
) -> ShipTelemetry:
    speed = float(np.linalg.norm(vel))
    return ShipTelemetry(
        position=pos,
        velocity=vel,
        heading=heading,
        speed=speed,
        lives=lives,
        ammo=ammo,
        mines_remaining=mines,
        bullet_cooldown=cooldown,
        mine_cooldown=0.0,
        is_invulnerable=False,
    )


class TestStage4RuleEngine(unittest.TestCase):
    """Test Forward Chaining Rule Engine, Reflexes, and Explainability."""

    def setUp(self):
        self.engine = ForwardChainingRuleEngine(safe_distance=120.0, critical_tti=1.2)

    def test_emergency_collision_avoidance_rule(self):
        # Ship at (400, 400), asteroid at (400, 350) moving down towards ship at 60 px/s
        telemetry = make_dummy_telemetry(pos=(400.0, 400.0), vel=(0.0, 0.0), heading=0.0)
        hazard = TrackedEntity(
            track_id=1,
            entity_type="asteroid",
            position=np.array([400.0, 350.0]),
            velocity=np.array([0.0, 60.0]),  # Moving down directly toward ship
            speed=60.0,
            estimated_radius=30.0,
            size_category=3,
            is_confirmed=True,
        )

        facts = self.engine.extract_facts(telemetry, [hazard], timestamp=1.5)
        self.assertTrue(facts.imminent_collision)
        self.assertLess(facts.time_to_impact, 1.2)
        self.assertGreater(facts.closure_rate, 40.0)

        directive, record = self.engine.evaluate(facts)

        # Must trigger emergency avoidance
        self.assertTrue(directive.override_movement)
        self.assertEqual(record.rule_name, "Rule_EmergencyCollisionAvoidance")
        self.assertEqual(record.severity, RuleSeverity.EMERGENCY)
        self.assertIn("distance", record.trigger_facts)
        self.assertIn("closure_rate", record.trigger_facts)
        self.assertIn("tti", record.trigger_facts)

    def test_defensive_mine_rule(self):
        # Ship moving up with heading=0. Hazard trailing directly behind at (400, 470)
        telemetry = make_dummy_telemetry(pos=(400.0, 400.0), vel=(0.0, -50.0), heading=0.0, mines=3)
        pursuer = TrackedEntity(
            track_id=2,
            entity_type="asteroid",
            position=np.array([400.0, 460.0]),  # 60px directly behind
            velocity=np.array([0.0, -80.0]),   # Closing in from behind
            speed=80.0,
            estimated_radius=20.0,
            size_category=2,
            is_confirmed=True,
        )

        facts = self.engine.extract_facts(telemetry, [pursuer], timestamp=2.0)
        self.assertTrue(facts.pursuer_behind)

        directive, record = self.engine.evaluate(facts)
        self.assertTrue(directive.drop_mine)
        self.assertEqual(record.rule_name, "Rule_DefensiveMineDrop")

    def test_explainability_log_formatting(self):
        telemetry = make_dummy_telemetry()
        facts = self.engine.extract_facts(telemetry, [], timestamp=0.5)
        _, record = self.engine.evaluate(facts)

        log_str = record.format_log()
        self.assertIn("[ 0.50s]", log_str)
        self.assertIn("Rule_DefaultCruise", log_str)
        self.assertIn("Action:", log_str)


class TestStage5AStarPathfinding(unittest.TestCase):
    """Test Local A* Planner, Dynamic Grid Inflation, and Waypoint Navigation."""

    def setUp(self):
        self.planner = AStarLocalPlanner(grid_width=400.0, grid_height=400.0, cell_size=20.0)

    def test_coordinate_conversions(self):
        ship_pos = np.array([500.0, 400.0])
        # Ship itself should be at center cell
        c, r = self.planner.world_to_local_grid(ship_pos, ship_pos)
        self.assertEqual(c, self.planner.center_col)
        self.assertEqual(r, self.planner.center_row)

        world_back = self.planner.local_grid_to_world(c, r, ship_pos)
        self.assertAlmostEqual(world_back[0], ship_pos[0], delta=20.0)
        self.assertAlmostEqual(world_back[1], ship_pos[1], delta=20.0)

    def test_occupancy_grid_inflation(self):
        ship_pos = np.array([500.0, 400.0])
        # Hazard 60 px to the right
        hazard = TrackedEntity(
            track_id=1,
            entity_type="asteroid",
            position=np.array([560.0, 400.0]),
            velocity=np.array([0.0, 0.0]),
            speed=0.0,
            estimated_radius=20.0,
            size_category=2,
            is_confirmed=True,
        )

        grid = self.planner.build_occupancy_grid(ship_pos, [hazard])
        # Cell to the right should be marked as an obstacle
        hz_col, hz_row = self.planner.world_to_local_grid(hazard.position, ship_pos)
        self.assertEqual(grid[hz_row, hz_col], 1)
        # Ship center cell must remain 0 (free to move)
        self.assertEqual(grid[self.planner.center_row, self.planner.center_col], 0)

    def test_astar_plans_around_obstacle(self):
        ship_pos = np.array([500.0, 400.0])
        telemetry = make_dummy_telemetry(pos=(500.0, 400.0), heading=90.0)  # Facing Right

        # Obstacle placed directly in front of ship
        hazard = TrackedEntity(
            track_id=1,
            entity_type="asteroid",
            position=np.array([550.0, 400.0]),
            velocity=np.array([0.0, 0.0]),
            speed=0.0,
            estimated_radius=25.0,
            size_category=3,
            is_confirmed=True,
        )

        # Target point is past the obstacle at (650, 400)
        target = np.array([650.0, 400.0])
        res = self.planner.plan(telemetry, [hazard], target_world_pos=target)

        self.assertTrue(res.is_path_found)
        self.assertGreater(len(res.path_grid), 0)
        # Result should recommend non-zero steering to circumvent the obstacle
        self.assertNotEqual(res.turn_rate, 0.0)


class TestStage6CSPShooting(unittest.TestCase):
    """Test CSP Firing Constraints, Quadratic Intercept, and Audit Reporting."""

    def setUp(self):
        self.shooter = CSPShooter(firing_tolerance_deg=8.0, bullet_speed=750.0)

    def test_quadratic_ballistic_intercept_math(self):
        # Ship at (400, 400), target at (400, 200) moving right at 100 px/s
        ship_pos = np.array([400.0, 400.0])
        ship_vel = np.array([0.0, 0.0])
        target_pos = np.array([400.0, 200.0])
        target_vel = np.array([100.0, 0.0])

        res = self.shooter.solve_ballistic_intercept(ship_pos, ship_vel, target_pos, target_vel)
        self.assertIsNotNone(res)
        t_int, int_pos, aim_heading = res

        # Distance is ~200px, bullet speed is 750px/s, intercept should be ~0.27s
        self.assertGreater(t_int, 0.2)
        self.assertLess(t_int, 0.4)
        # Intercept x should be lead ahead to the right
        self.assertGreater(int_pos[0], 400.0)

    def test_csp_firing_constraints_validation(self):
        telemetry = make_dummy_telemetry(pos=(400.0, 400.0), heading=0.0, cooldown=0.0, ammo=50)

        # 1. Target directly north aligned with heading (heading=0 is North)
        aligned_target = TrackedEntity(
            track_id=1,
            entity_type="asteroid",
            position=np.array([400.0, 250.0]),
            velocity=np.array([0.0, 0.0]),
            speed=0.0,
            estimated_radius=20.0,
            size_category=2,
            is_confirmed=True,
        )

        decision = self.shooter.evaluate(telemetry, [aligned_target])
        self.assertTrue(decision.should_shoot)
        self.assertTrue(all(decision.constraint_audit.values()))
        self.assertIn("Approved shot", decision.explanation)

        # 2. Denied when weapon on cooldown
        telemetry_cooldown = make_dummy_telemetry(pos=(400.0, 400.0), heading=0.0, cooldown=0.15)
        dec_cooldown = self.shooter.evaluate(telemetry_cooldown, [aligned_target])
        self.assertFalse(dec_cooldown.should_shoot)
        self.assertFalse(dec_cooldown.constraint_audit["C1_cooldown_ready"])

        # 3. Denied when ammo is 0
        telemetry_no_ammo = make_dummy_telemetry(pos=(400.0, 400.0), heading=0.0, ammo=0)
        dec_no_ammo = self.shooter.evaluate(telemetry_no_ammo, [aligned_target])
        self.assertFalse(dec_no_ammo.should_shoot)
        self.assertFalse(dec_no_ammo.constraint_audit["C2_ammo_available"])

        # 4. Denied when target is outside angular alignment tolerance (facing North, target to East)
        telemetry_misaligned = make_dummy_telemetry(pos=(400.0, 400.0), heading=90.0)  # Facing East
        dec_angle = self.shooter.evaluate(telemetry_misaligned, [aligned_target])
        self.assertFalse(dec_angle.should_shoot)
        self.assertFalse(dec_angle.constraint_audit["C3_angular_alignment"])


class TestIntegratedPilot(unittest.TestCase):
    """Test full integration of Stages 1-6 in AutonomousPilot."""

    def test_pilot_decide_loop(self):
        pilot = AutonomousPilot()
        telemetry = make_dummy_telemetry()
        obs = SensorObservation(timestamp=1.0, telemetry=telemetry)

        turn, thrust, shoot, drop_mine, output = pilot.decide(obs)

        # Must return valid actuator ranges
        self.assertTrue(-180.0 <= turn <= 180.0)
        self.assertTrue(0.0 <= thrust <= 1.0)
        self.assertIsInstance(shoot, (bool, np.bool_))
        self.assertIsInstance(drop_mine, (bool, np.bool_))
        self.assertIsNotNone(output.active_rule)
        self.assertIsNotNone(output.csp_decision)


if __name__ == "__main__":
    unittest.main()
