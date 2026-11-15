"""
Interactive & Visual Demonstration for Stages 1 to 3:
- Stage 1: Game Environment, Physics, Collisions, Splitting, Mines, Bullets
- Stage 2: Telemetry + Noisy Radar Sensor Readings (Yellow dots)
- Stage 3: Kalman Filter State Estimates (Green circles) + 2.0s Trajectory Lookaheads (Cyan paths)

Usage:
  python demo_stage1_3.py           # Runs interactive Pygame visualizer
  python demo_stage1_3.py --headless # Runs 300 ticks in headless mode and reports tracking stats
"""

import argparse
import sys
import numpy as np
import pygame

from src.config import GameConfig
from src.environment.game_engine import GameEngine, wrapped_distance
from src.environment.ship import heading_to_vector
from src.sensors.radar_sensor import RadarSensor
from src.tracking.object_tracker import MultiObjectTracker


def run_demo(visual: bool = True, max_ticks: int = 600) -> None:
    config = GameConfig()
    engine = GameEngine(config=config, seed=42)
    # Add a rival ship for demonstration
    engine.add_rival_ship(position=(750.0, 400.0), heading=180.0)

    sensor = RadarSensor(config=config.sensor, seed=101)
    tracker = MultiObjectTracker(
        config=config.kalman,
        arena_width=config.arena.width,
        arena_height=config.arena.height,
    )

    if visual:
        pygame.init()
        screen = pygame.display.set_mode((int(config.arena.width), int(config.arena.height)))
        pygame.display.set_caption("AI Piloted Asteroid Game - Stages 1-3 Demo (Environment, Sensors, Kalman Tracking)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Consolas", 14)
        title_font = pygame.font.SysFont("Consolas", 18, bold=True)

    print("=" * 70)
    print("AI Piloted Asteroid Game - Stages 1 to 3 Demonstration")
    print("Stage 1: Physics Engine, Momentum, Asteroid Splitting, Bullets, Mines")
    print("Stage 2: On-board Radar with Synthetic Gaussian Noise (sigma = 12px)")
    print("Stage 3: Multi-Target Kalman Filter with 2.0s Lookahead Trajectory Projection")
    print("=" * 70)
    if visual:
        print("Controls:")
        print("  UP / W     : Thrust")
        print("  LEFT/RIGHT : Turn")
        print("  SPACE      : Shoot")
        print("  M          : Drop Mine")
        print("  R          : Reset Environment")
        print("  ESC        : Exit")

    tick = 0
    running = True

    # Error accumulation for benchmark metrics
    radar_errors = []
    kalman_errors = []

    while running and tick < max_ticks:
        turn_rate = 0.0
        thrust = 0.0
        shoot = False
        drop_mine = False

        if visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        engine.reset()
                        tracker.reset()  # clear stale tracks / per-episode state
                    elif event.key == pygame.K_m:
                        drop_mine = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                turn_rate = -config.ship.turn_rate
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                turn_rate = config.ship.turn_rate

            if keys[pygame.K_UP] or keys[pygame.K_w]:
                thrust = 1.0

            if keys[pygame.K_SPACE]:
                shoot = True
        else:
            # Headless autonomous gentle rotation and test firing
            turn_rate = 45.0
            thrust = 0.3 if tick % 60 < 20 else 0.0
            shoot = (tick % 15 == 0)

        # Simple rival baseline AI (turns and drifts)
        rival_actions = (30.0, 0.4 if tick % 90 < 30 else 0.0, tick % 25 == 0, False)

        actions = {
            1: (turn_rate, thrust, shoot, drop_mine),
            2: rival_actions,
        }

        # Step 1: Environment Physics
        true_state = engine.step(actions)

        # Step 2: Sensors (Noisy Radar Scan)
        sensor_obs = sensor.observe(engine, observing_ship_id=1)

        # Step 3: Kalman Filter & Multi-Object Tracking
        tracked_entities = tracker.update(sensor_obs)

        # Compute error comparison: Raw Radar Noise vs Kalman Filtered State (accounting for boundary wrap)
        for det in sensor_obs.radar_asteroids:
            if det.true_id is not None:
                for true_ast in engine.asteroids:
                    if true_ast.id == det.true_id:
                        r_err, _ = wrapped_distance(np.array(det.measured_position), true_ast.position, config.arena.width, config.arena.height)
                        radar_errors.append(r_err)
                        break

        for entity in tracked_entities:
            if entity.entity_type == "asteroid" and entity.true_id is not None and entity.is_confirmed:
                for true_ast in engine.asteroids:
                    if true_ast.id == entity.true_id:
                        k_err, _ = wrapped_distance(entity.position, true_ast.position, config.arena.width, config.arena.height)
                        kalman_errors.append(k_err)
                        break

        # Visual Rendering
        if visual:
            # 1. Base Game Engine Render
            engine.render(screen)

            # 2. Overlay Stage 2: Raw Noisy Radar Detections (Yellow dots with jitter rings)
            for det in sensor_obs.radar_asteroids:
                pos = (int(det.measured_position[0]), int(det.measured_position[1]))
                pygame.draw.circle(screen, (255, 230, 80), pos, 4)
                # Dotted noise radius
                pygame.draw.circle(screen, (160, 140, 40), pos, int(config.sensor.radar_noise_std_pos), 1)

            # 3. Overlay Stage 3: Kalman Filter Tracks & Future Trajectory Lookahead
            for entity in tracked_entities:
                if not entity.is_confirmed:
                    continue
                k_pos = (int(entity.position[0]), int(entity.position[1]))
                # Filtered bounding box / circle (Green)
                pygame.draw.circle(screen, (50, 255, 120), k_pos, int(entity.estimated_radius + 4), 2)

                # Filtered Velocity Vector (Green arrow)
                vel_end = entity.position + entity.velocity * 0.5
                pygame.draw.line(screen, (50, 255, 120), k_pos, (int(vel_end[0]), int(vel_end[1])), 2)

                # 2.0-Second Future Trajectory Lookahead (Cyan line with future nodes)
                if len(entity.future_trajectory) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in entity.future_trajectory]
                    # Draw future trajectory steps
                    for idx, pt in enumerate(pts):
                        # Node dot
                        node_color = (0, 220, 255) if idx == len(pts) - 1 else (0, 160, 200)
                        pygame.draw.circle(screen, node_color, pt, 2 if idx < len(pts) - 1 else 4)

                # Label with track ID & speed
                label_text = f"T{entity.track_id} v:{entity.speed:3.0f}"
                lbl_surf = font.render(label_text, True, (120, 255, 180))
                screen.blit(lbl_surf, (k_pos[0] + 12, k_pos[1] - 12))

            # HUD Telemetry & Statistics
            avg_radar_err = np.mean(radar_errors[-100:]) if radar_errors else 0.0
            avg_kalman_err = np.mean(kalman_errors[-100:]) if kalman_errors else 0.0

            diag_y = config.arena.height - 85
            # Semi-transparent bottom bar
            bottom_surf = pygame.Surface((int(config.arena.width), 80))
            bottom_surf.set_alpha(200)
            bottom_surf.fill((15, 20, 35))
            screen.blit(bottom_surf, (0, diag_y))

            t1 = title_font.render("STAGES 1-3 PIPELINE STATUS", True, (100, 220, 255))
            screen.blit(t1, (15, diag_y + 8))

            info_line1 = (
                f"STAGE 1 [Physics]: Entities={len(engine.asteroids)} Ast, {len(engine.bullets)} Bullets, {len(engine.mines)} Mines | "
                f"Ship Pos=({sensor_obs.telemetry.position[0]:.0f}, {sensor_obs.telemetry.position[1]:.0f}) Speed={sensor_obs.telemetry.speed:.1f} px/s"
            )
            screen.blit(font.render(info_line1, True, (210, 225, 240)), (15, diag_y + 32))

            info_line2 = (
                f"STAGE 2 [Sensors]: Radar Dets={len(sensor_obs.radar_asteroids)} (Yellow) [sigma={config.sensor.radar_noise_std_pos:.1f}px, Avg Err={avg_radar_err:4.1f}px] | "
                f"STAGE 3 [Kalman]: Active Tracks={len(tracked_entities)} (Green) [Avg Err={avg_kalman_err:4.1f}px, Lookahead=+2.0s (Cyan)]"
            )
            screen.blit(font.render(info_line2, True, (80, 255, 170)), (15, diag_y + 52))

            pygame.display.flip()
            clock.tick(config.fps)

        tick += 1

    if visual:
        pygame.quit()

    avg_radar_err = np.mean(radar_errors) if radar_errors else 0.0
    avg_kalman_err = np.mean(kalman_errors) if kalman_errors else 0.0

    print("\n" + "=" * 70)
    print("DEMO RUN FINISHED - SUMMARY STATISTICS")
    print(f"Total Ticks Executed: {tick}")
    print(f"Stage 1 Score:        {engine.ships[1].score}")
    print(f"Stage 1 Shots Fired:  {engine.ships[1].shots_fired} (Accuracy: {engine.ships[1].accuracy * 100:.1f}%)")
    print(f"Stage 2 Radar Noise:  Mean Euclidean Error = {avg_radar_err:.2f} px")
    print(f"Stage 3 Kalman Track: Mean Estimation Error = {avg_kalman_err:.2f} px")
    if avg_radar_err > 0:
        improvement = ((avg_radar_err - avg_kalman_err) / avg_radar_err) * 100.0
        print(f"Noise Reduction Rate: {improvement:.1f}% error reduction via Kalman Filter!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Piloted Asteroid Game - Stages 1-3 Demo")
    parser.add_argument("--headless", action="store_true", help="Run without opening Pygame GUI window")
    parser.add_argument("--ticks", type=int, default=300, help="Number of ticks for headless mode")
    args = parser.parse_args()

    run_demo(visual=not args.headless, max_ticks=args.ticks)
