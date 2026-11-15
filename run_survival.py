"""
Single-Player AI Spacecraft Survival & Asteroid Elimination Runner:
- Solo AI pilot (No PvP rival)
- Primary objectives:
  1. Maximize survival time (survive continuous asteroid waves)
  2. Actively target, shoot, and destroy asteroids using Kalman-predicted ballistic intercept
- Strict Physical Constraint: Spacecraft cannot go out of the screen (enforced by triple-layer boundary protection:
  physics clamping/rebound, rule-based wall repulsion steering, and A* obstacle wall borders).

Usage:
  python run_survival.py                 # Live Pygame visualization at 60 FPS
  python run_survival.py --headless      # Fast-as-possible simulation (headless)
  python run_survival.py --ticks 1000    # Run for 1000 ticks (~33.3 seconds)
"""

import argparse
import json
import os
import sys
from typing import Optional
import numpy as np
import pygame

from src.agent.autonomous_pilot import AutonomousPilot
from src.config import GameConfig
from src.environment.game_engine import GameEngine
from src.explainability.hud_overlay import ExplainabilityHUD
from src.optimization.genetic_algorithm import AgentGenome
from src.sensors.radar_sensor import RadarSensor


def run_survival(
    visual: bool = True,
    max_ticks: int = 1800,
    seed: int = 42,
    genome_file: Optional[str] = "best_optimized_genome.json",
) -> None:
    config = GameConfig()
    w = config.arena.width
    h = config.arena.height

    # Single-player environment with continuous asteroid replenishment
    engine = GameEngine(config=config, seed=seed, auto_replenish=True)
    sensor = RadarSensor(config=config.sensor, seed=seed + 100)

    # Load GA-optimized parameters if present
    genome = None
    if genome_file and os.path.exists(genome_file):
        with open(genome_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            genome = AgentGenome(**params)
            print(f"[Survival Mode] Loaded GA-optimized parameters from {genome_file} (Fitness: {data.get('fitness', 0):.1f})")

    pilot = AutonomousPilot(
        arena_width=w,
        arena_height=h,
        genome=genome,
    )

    hud = ExplainabilityHUD(width=w, height=h)

    if visual:
        pygame.init()
        screen = pygame.display.set_mode((int(w), int(h)))
        pygame.display.set_caption("AI Asteroids: Solo Survival & Asteroid Destruction (Bounded Arena)")
        clock = pygame.time.Clock()
        font_sm = pygame.font.SysFont("Consolas", 13)
        font_md = pygame.font.SysFont("Consolas", 15, bold=True)
        font_lg = pygame.font.SysFont("Consolas", 22, bold=True)

    print("=" * 80)
    print("AI SPACECRAFT SOLO SURVIVAL & ASTEROID ELIMINATION")
    print("Mode: Single AI Pilot (Single Life Permadeath, No Mines)")
    print(f"Arena: {w:.0f} x {h:.0f} px (Impenetrable Physical Screen Perimeter)")
    print("Spawning: XFC Guidelines (Sizes 1-4, Inelastic Momentum 3-Way Split)")
    print("Objective 1: Maximize survival time under continuous hazard pressure")
    print("Objective 2: Destroy asteroids via Kalman tracking & CSP ballistic interception")
    print("Constraint: Spacecraft strictly cannot exit the screen")
    print("=" * 80)
    if visual:
        print("Controls:")
        print("  SPACE / P : Pause / Resume")
        print("  G         : Toggle Local A* Grid & Waypoint Overlay")
        print("  E         : Export Decision Logs (JSON & CSV)")
        print("  R         : Reset Simulation")
        print("  ESC / Q   : Quit and Output Detailed Survival Report")
        print("=" * 80)

    tick = 0
    running = True
    paused = False
    show_grid_overlay = True
    wall_breaches = 0      # can never be > 0: Ship.apply_control clamps position (physics invariant)
    wall_contact_ticks = 0 # ticks the ship was pressed against a wall (measures the AI's wall avoidance)

    while running and tick < max_ticks:
        if visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        engine.reset()
                        pilot.reset()  # clear stale tracks / per-episode state
                        wall_breaches = 0
                        wall_contact_ticks = 0
                        tick = 0
                    elif event.key in (pygame.K_SPACE, pygame.K_p):
                        paused = not paused
                    elif event.key == pygame.K_g:
                        show_grid_overlay = not show_grid_overlay
                    elif event.key == pygame.K_e:
                        pilot.logger.export_json("survival_decision_log.json")
                        pilot.logger.export_csv("survival_decision_log.csv")
                        print("[Logging] Saved logs to survival_decision_log.json and .csv")

            if paused:
                clock.tick(15)
                continue

        # 1. Sense Environment
        sensor_obs = sensor.observe(engine, observing_ship_id=1)

        # 2. Autonomous Decision Cycle
        turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd, decision = pilot.decide(sensor_obs)

        # 3. Environment Step (Single AI pilot)
        actions = {
            1: (turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd),
        }
        engine.step(actions)

        # 4. Strict Boundary Verification
        ship = engine.ships[1]
        if ship.is_alive:
            r = ship.radius
            if not (r <= ship.position[0] <= w - r and r <= ship.position[1] <= h - r):
                wall_breaches += 1
            if (ship.position[0] <= r + 0.5 or ship.position[0] >= w - r - 0.5
                    or ship.position[1] <= r + 0.5 or ship.position[1] >= h - r - 0.5):
                wall_contact_ticks += 1

        # Check Game Over
        if engine.is_game_over():
            print(f"[Game Over] AI Spacecraft destroyed at tick {tick} ({engine.time:.1f}s)")
            break

        # 5. Visual Rendering
        if visual:
            engine.render(screen)
            ship_pos = np.array(sensor_obs.telemetry.position)

            # Draw A* Local Occupancy Grid & Planned Route (Stage 5)
            if show_grid_overlay and decision.astar_plan is not None:
                plan = decision.astar_plan
                grid = plan.grid_cost_map
                if grid is not None:
                    g_half_w = pilot.planner.grid_width / 2.0
                    g_half_h = pilot.planner.grid_height / 2.0
                    grid_surf = pygame.Surface(
                        (int(pilot.planner.grid_width), int(pilot.planner.grid_height)),
                        pygame.SRCALPHA,
                    )
                    grid_surf.fill((20, 30, 50, 40))

                    cell_sz = pilot.planner.cell_size
                    for r_idx in range(pilot.planner.rows):
                        for c_idx in range(pilot.planner.cols):
                            if grid[r_idx, c_idx] == 1:
                                cell_rect = pygame.Rect(
                                    int(c_idx * cell_sz), int(r_idx * cell_sz),
                                    int(cell_sz), int(cell_sz)
                                )
                                pygame.draw.rect(grid_surf, (255, 60, 40, 70), cell_rect)

                    screen.blit(grid_surf, (int(ship_pos[0] - g_half_w), int(ship_pos[1] - g_half_h)))

                if len(plan.path_world) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in plan.path_world]
                    pygame.draw.lines(screen, (50, 255, 140), False, pts, 3)

            # Draw Ballistic Intercept Reticle (Stage 6)
            csp = decision.csp_decision
            if csp.intercept_point is not None:
                int_pt = (int(csp.intercept_point[0]), int(csp.intercept_point[1]))
                reticle_color = (255, 240, 60) if csp.should_shoot else (140, 140, 90)
                pygame.draw.circle(screen, reticle_color, int_pt, 7, 2)
                pygame.draw.line(screen, reticle_color, (int_pt[0] - 10, int_pt[1]), (int_pt[0] + 10, int_pt[1]), 1)
                pygame.draw.line(screen, reticle_color, (int_pt[0], int_pt[1] - 10), (int_pt[0], int_pt[1] + 10), 1)

            # Draw Tactical Target Waypoint
            if decision.minimax_decision is not None:
                wp = decision.minimax_decision.target_waypoint
                wp_pt = (int(wp[0]), int(wp[1]))
                pygame.draw.circle(screen, (255, 200, 60), wp_pt, 6, 2)
                pygame.draw.line(screen, (200, 160, 50), (int(ship_pos[0]), int(ship_pos[1])), wp_pt, 1)

            # Render Explainability HUD Panel (Stage 9)
            recent = pilot.logger.entries[-5:]
            hud.render(screen, decision.log_entry, recent)

            # Bottom Status Bar with Boundary Proximity Indicator
            diag_y = int(h - 35)
            b_surf = pygame.Surface((int(w), 35))
            b_surf.set_alpha(210)
            b_surf.fill((10, 14, 24))
            screen.blit(b_surf, (0, diag_y))

            wall_d = min(
                ship.position[0],
                w - ship.position[0],
                ship.position[1],
                h - ship.position[1],
            )
            wall_status = f"WALL CLEARANCE: {wall_d:.0f}px" if wall_d > 65.0 else f"WALL WARNING ({wall_d:.0f}px)"

            status_text = (
                f"SURVIVED: {engine.time:4.1f}s ({tick:04d} ticks) | "
                f"SCORE: {ship.score:05d} | "
                f"RULE: {decision.active_rule} | "
                f"FIRE: {'YES' if decision.shoot else ('HELD' if csp.should_shoot else 'HOLD')} | "
                f"{wall_status}"
            )
            status_color = (180, 215, 245) if wall_d > 65.0 else (255, 180, 80)
            screen.blit(font_sm.render(status_text, True, status_color), (15, diag_y + 10))

            pygame.display.flip()
            clock.tick(config.fps)

        tick += 1

    if visual:
        pygame.quit()

    # Save decision logs
    pilot.logger.export_json("survival_decision_log.json")
    pilot.logger.export_csv("survival_decision_log.csv")

    player = engine.ships[1]
    print("\n" + "=" * 80)
    print("SURVIVAL MISSION AUDIT REPORT")
    print("=" * 80)
    print(f"Total Survival Time:      {engine.time:.2f} seconds ({tick} ticks)")
    print(f"Final Score:              {player.score}")
    print(f"Asteroids Obliterated:    {player.asteroids_hit}")
    print(f"Ballistic Shots Fired:    {player.shots_fired}")
    print(f"Ballistic Hits:           {player.shots_hit} (Accuracy: {player.accuracy * 100:.1f}%)")
    print(f"Lives Remaining:          {player.lives} / 1 (Deaths: {player.deaths})")
    print(f"Boundary Breaches:        {wall_breaches} (guaranteed 0 by the physics clamp)")
    print(f"Wall-Contact Ticks:       {wall_contact_ticks} ({wall_contact_ticks / max(1, tick) * 100:.1f}% of ticks touching a wall)")
    print("=" * 80)
    print(pilot.logger.generate_summary_report())
    print("Logs saved: survival_decision_log.json and survival_decision_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Spacecraft Solo Survival & Asteroid Elimination Runner")
    parser.add_argument("--headless", action="store_true", help="Run simulation headless without Pygame window")
    parser.add_argument("--ticks", type=int, default=1800, help="Max ticks to run (default: 1800 = 60s at 30fps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and sensors")
    parser.add_argument("--genome", type=str, default="best_optimized_genome.json", help="GA genome parameters JSON")
    args = parser.parse_args()

    run_survival(visual=not args.headless, max_ticks=args.ticks, seed=args.seed, genome_file=args.genome)
