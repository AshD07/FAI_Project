"""
Unit and integration tests for Stages 1 to 3:
- Stage 1: Game Environment, Physics, Collisions, Splitting, Actuators
- Stage 2: Sensors, Telemetry, and Noisy Radar Observations
- Stage 3: Kalman Filter State Estimation, MOT Tracking, and Trajectory Prediction
"""

import sys
import unittest
import numpy as np

# Ensure root path is accessible
sys.path.append(".")

from src.config import GameConfig
from src.environment.asteroid import Asteroid
from src.environment.bullet import Bullet
from src.environment.game_engine import GameEngine, wrapped_distance
from src.environment.mine import Mine
from src.environment.ship import Ship, heading_to_vector
from src.sensors.radar_sensor import RadarSensor
from src.tracking.kalman_filter import KalmanFilter2D
from src.tracking.object_tracker import MultiObjectTracker


class TestStage1Environment(unittest.TestCase):
    """Test Game Environment, Spacecraft Physics, Ordnance, Collisions & Splitting."""

    def setUp(self):
        self.config = GameConfig()
        self.engine = GameEngine(config=self.config, seed=42)

    def test_ship_turning_and_thrusting(self):
        ship = self.engine.ships[1]
        initial_heading = ship.heading
        initial_pos = ship.position.copy()

        # Turn right by 90 deg/s for 1 second (30 ticks)
        for _ in range(30):
            ship.apply_control(
                turn_rate=90.0,
                thrust=0.0,
                shoot=False,
                drop_mine=False,
                dt=1.0 / 30.0,
                arena_width=1000.0,
                arena_height=800.0,
            )

        self.assertAlmostEqual(ship.heading, (initial_heading + 90.0) % 360.0, delta=1.0)

        # Now apply full thrust for 1 second
        for _ in range(30):
            ship.apply_control(
                turn_rate=0.0,
                thrust=1.0,
                shoot=False,
                drop_mine=False,
                dt=1.0 / 30.0,
                arena_width=1000.0,
                arena_height=800.0,
            )

        self.assertGreater(ship.speed, 50.0)
        self.assertFalse(np.array_equal(ship.position, initial_pos))

    def test_weapons_shooting_and_cooldown(self):
        ship = self.engine.ships[1]
        initial_ammo = ship.ammo

        # Request shoot
        bullet, _ = ship.apply_control(
            turn_rate=0.0,
            thrust=0.0,
            shoot=True,
            drop_mine=False,
            dt=1.0 / 30.0,
            arena_width=1000.0,
            arena_height=800.0,
        )

        self.assertIsNotNone(bullet)
        self.assertEqual(ship.ammo, initial_ammo - 1)
        self.assertGreater(ship.bullet_cooldown_timer, 0.0)

        # Immediate second shot should be blocked by cooldown
        bullet_blocked, _ = ship.apply_control(
            turn_rate=0.0,
            thrust=0.0,
            shoot=True,
            drop_mine=False,
            dt=1.0 / 30.0,
            arena_width=1000.0,
            arena_height=800.0,
        )
        self.assertIsNone(bullet_blocked)

    def test_weapons_drop_mine(self):
        ship = self.engine.ships[1]
        ship.mines_remaining = 3  # Set test mines to verify ordnance logic
        initial_mines = ship.mines_remaining

        _, mine = ship.apply_control(
            turn_rate=0.0,
            thrust=0.0,
            shoot=False,
            drop_mine=True,
            dt=1.0 / 30.0,
            arena_width=1000.0,
            arena_height=800.0,
        )

        self.assertIsNotNone(mine)
        self.assertEqual(ship.mines_remaining, initial_mines - 1)
        self.assertGreater(ship.mine_cooldown_timer, 0.0)

    def test_asteroid_splitting(self):
        large_ast = Asteroid(position=(500, 400), velocity=(40, 0), size=3)
        self.assertEqual(large_ast.size, 3)

        fragments = large_ast.split()
        self.assertFalse(large_ast.is_alive)
        self.assertEqual(len(fragments), 3)  # 3 fragments per XFC guidelines
        for frag in fragments:
            self.assertEqual(frag.size, 2)
            self.assertGreater(frag.speed, 0.0)

        # Small asteroid obliteration
        small_ast = Asteroid(position=(500, 400), velocity=(80, 0), size=1)
        small_fragments = small_ast.split()
        self.assertEqual(len(small_fragments), 0)

    def test_bullet_asteroid_collision_in_game_loop(self):
        self.engine.asteroids.clear()
        target_ast = Asteroid(position=(500, 300), velocity=(0, 0), size=3)
        self.engine.asteroids.append(target_ast)

        # Spawn a bullet directly heading into the asteroid
        bullet = Bullet(
            position=(500, 315),
            velocity=(0, -700),
            owner_id=1,
        )
        self.engine.bullets.append(bullet)

        # Run 1 tick
        self.engine.step({})

        # Bullet should have hit and triggered split (3 fragments per XFC guidelines)
        self.assertEqual(len(self.engine.bullets), 0)
        self.assertEqual(len(self.engine.asteroids), 3)
        player = self.engine.ships[1]
        self.assertGreater(player.score, 0)
        self.assertEqual(player.shots_hit, 1)

    def test_ship_lives_and_hazard_collision(self):
        ship = self.engine.ships[1]
        ship.invulnerable_timer = 0.0  # Remove starting invulnerability
        initial_lives = ship.lives

        # With single life, taking a hit reduces lives to 0 and marks ship dead
        hit = ship.hit_by_hazard()
        self.assertTrue(hit)
        self.assertEqual(ship.lives, initial_lives - 1)
        self.assertFalse(ship.is_alive)

        # Multi-life respawn verification (if extra lives are granted)
        ship.lives = 2
        ship.is_alive = True
        ship.invulnerable_timer = 0.0
        hit2 = ship.hit_by_hazard()
        self.assertTrue(hit2)
        self.assertEqual(ship.lives, 1)
        self.assertTrue(ship.is_invulnerable)


class TestStage2Sensors(unittest.TestCase):
    """Test Radar Sensor, Noise Generation, and Telemetry Percepts."""

    def setUp(self):
        self.engine = GameEngine(seed=123)
        self.sensor = RadarSensor(seed=456)

    def test_sensor_telemetry_accuracy(self):
        obs = self.sensor.observe(self.engine, observing_ship_id=1)
        ship = self.engine.ships[1]

        self.assertAlmostEqual(obs.telemetry.position[0], ship.position[0], delta=1e-5)
        self.assertAlmostEqual(obs.telemetry.position[1], ship.position[1], delta=1e-5)
        self.assertEqual(obs.telemetry.lives, ship.lives)
        self.assertEqual(obs.telemetry.ammo, ship.ammo)

    def test_radar_noisy_measurements(self):
        self.engine.asteroids.clear()
        true_pos = (600.0, 400.0)
        true_vel = (50.0, -20.0)
        ast = Asteroid(position=true_pos, velocity=true_vel, size=3)
        self.engine.asteroids.append(ast)

        errors = []
        detections_count = 0
        total_attempts = 100

        for _ in range(total_attempts):
            obs = self.sensor.observe(self.engine, observing_ship_id=1)
            if len(obs.radar_asteroids) > 0:
                detections_count += 1
                det = obs.radar_asteroids[0]
                err = np.linalg.norm(np.array(det.measured_position) - np.array(true_pos))
                errors.append(err)

        # Detection probability should be close to 0.98
        self.assertGreaterEqual(detections_count, 85)

        mean_err = np.mean(errors)
        # Radar noise std is 12px, so 2D Euclidean error average is expected ~15px
        self.assertGreater(mean_err, 4.0)
        self.assertLess(mean_err, 35.0)


class TestStage3KalmanTracking(unittest.TestCase):
    """Test Kalman Filter state estimation, boundary wrapping, and trajectory prediction."""

    def test_kalman_filter_noise_filtering(self):
        """Verify that Kalman filter reduces positional measurement noise over time."""
        true_pos = np.array([200.0, 200.0])
        true_vel = np.array([40.0, 30.0])
        dt = 1.0 / 30.0

        kf = KalmanFilter2D(initial_pos=(true_pos[0], true_pos[1]), initial_vel=(0.0, 0.0))
        rng = np.random.default_rng(789)

        raw_errors = []
        kf_errors = []

        # Run 60 ticks (2 seconds)
        for _ in range(60):
            true_pos = true_pos + true_vel * dt
            noisy_pos = true_pos + rng.normal(0.0, 15.0, size=2)

            kf.predict(dt)
            kf.update((noisy_pos[0], noisy_pos[1]))

            raw_errors.append(np.linalg.norm(noisy_pos - true_pos))
            kf_errors.append(np.linalg.norm(kf.position - true_pos))

        # Kalman filtered error on the second half should be significantly lower than raw noise
        self.assertLess(np.mean(kf_errors[30:]), np.mean(raw_errors[30:]))
        # Velocity estimation should converge close to ground truth [40, 30]
        self.assertAlmostEqual(kf.velocity[0], true_vel[0], delta=8.0)
        self.assertAlmostEqual(kf.velocity[1], true_vel[1], delta=8.0)

    def test_kalman_filter_toroidal_boundary_wrap(self):
        """Verify that crossing arena borders does not cause a velocity explosion."""
        # Start near right edge (x = 990, width = 1000) moving right at +60 px/s
        kf = KalmanFilter2D(
            initial_pos=(990.0, 400.0),
            initial_vel=(60.0, 0.0),
            arena_width=1000.0,
            arena_height=800.0,
            wrap=True,  # toroidal mode is still supported when explicitly requested
        )
        dt = 0.5  # after 0.5s, moves +30px to x=1020 -> wraps to x=20

        kf.predict(dt)
        # Sensor measurement wrapped to x=20
        kf.update((20.0, 400.0))

        # Filtered velocity must remain positive and stable (~60), not huge negative (-970)
        self.assertGreater(kf.velocity[0], 20.0)
        self.assertLess(kf.velocity[0], 120.0)

    def test_kalman_filter_bounded_wall_reflection(self):
        """In the bounded arena a track heading into a wall must bounce, not teleport."""
        kf = KalmanFilter2D(
            initial_pos=(990.0, 400.0),
            initial_vel=(60.0, 0.0),
            arena_width=1000.0,
            arena_height=800.0,
            wrap=False,
        )
        kf.predict(0.5)  # 990 + 30 = 1020 -> reflected to 980, velocity flipped
        self.assertAlmostEqual(kf.position[0], 980.0, delta=1e-6)
        self.assertLess(kf.velocity[0], 0.0)
        fut, _ = kf.predict_future(1.0)  # 980 - 60 = 920, stays on the right side
        self.assertAlmostEqual(fut[0], 920.0, delta=1e-6)

    def test_tracker_reset_on_new_episode(self):
        """A backwards clock (new match without reset) must not leave stale tracks."""
        engine = GameEngine(seed=5)
        sensor = RadarSensor(seed=6)
        tracker = MultiObjectTracker()
        for _ in range(5):
            engine.step({})
            tracker.update(sensor.observe(engine, observing_ship_id=1))
        self.assertGreater(len(tracker.tracks), 0)
        fresh = GameEngine(seed=7)
        fresh.step({})
        tracker.update(sensor.observe(fresh, observing_ship_id=1))
        self.assertTrue(all(t.hits == 1 for t in tracker.tracks.values()))

    def test_multi_object_tracker_lifecycle_and_predictions(self):
        """Test MultiObjectTracker track confirmation and 2.0s trajectory prediction."""
        engine = GameEngine(seed=999)
        sensor = RadarSensor(seed=888)
        tracker = MultiObjectTracker()

        # Step engine and tracker for 10 frames
        for _ in range(10):
            engine.step({})
            obs = sensor.observe(engine, observing_ship_id=1)
            tracked_entities = tracker.update(obs)

        # Should have tracked confirmed asteroids
        confirmed = tracker.get_asteroids(tracked_entities)
        self.assertGreater(len(confirmed), 0)

        # Verify trajectory prediction
        first_ast = confirmed[0]
        self.assertEqual(len(first_ast.future_trajectory), 20)
        pred_2s = first_ast.predict_at(seconds_ahead=2.0)
        self.assertEqual(len(pred_2s), 2)
        # Future position must be in arena bounds [0, 1000] x [0, 800]
        self.assertTrue(0 <= pred_2s[0] <= 1000.0)
        self.assertTrue(0 <= pred_2s[1] <= 800.0)


if __name__ == "__main__":
    unittest.main()
