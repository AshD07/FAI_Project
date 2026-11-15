"""
Interactive Demonstration for Stages 7, 8, and 9:
- Stage 7: Minimax Adversarial Strategic Search with Alpha-Beta Pruning
- Stage 8: Integration with GA-Optimized Parameters (from best_optimized_genome.json)
- Stage 9: Real-Time On-Screen Explainability HUD & Auditable Decision Logging

Usage:
  python demo_stage7_9.py             # Runs interactive Pygame visualizer
  python demo_stage7_9.py --headless  # Runs automated simulation, exports JSON/CSV logs
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


def run_demo(visual: bool = True, max_ticks: int = 900, genome_file: Optional[str] = None) -> None:
    config = GameConfig()
    # Single AI controlled spacecraft (no PvP rival), with auto asteroid replenishment
    engine = GameEngine(config=config, seed=42, auto_replenish=True)

    sensor = RadarSensor(config=config.sensor, seed=303)

    # Load GA-optimized genome if available
    genome = None
    if genome_file and os.path.exists(genome_file):
        with open(genome_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            genome = AgentGenome(**params)
            print(f"[Stage 8] Loaded GA-optimized parameters from {genome_file} (Fitness: {data.get('fitness', 0):.1f})")

    pilot = AutonomousPilot(
        arena_width=config.arena.width,
        arena_height=config.arena.height,
        genome=genome,
    )

    hud = ExplainabilityHUD(width=config.arena.width, height=config.arena.height)

    if visual:
        pygame.init()
        screen = pygame.display.set_mode((int(config.arena.width), int(config.arena.height)))
        pygame.display.set_caption("AI Piloted Asteroid Game - Stages 7-9 (Minimax, GA Optimization, Explainability)")
        clock = pygame.time.Clock()
        font_sm = pygame.font.SysFont("Consolas", 13)
        font_md = pygame.font.SysFont("Consolas", 15, bold=True)

    print("=" * 75)
    print("AI Piloted Asteroid Game - Stages 7 to 9 Demonstration")
    print("Stage 7: Minimax Adversarial Search with Alpha-Beta Pruning (2-Ply)")
    print("Stage 8: Offline Genetic Algorithm Optimization Integration")
    print("Stage 9: Live On-Screen Explainability HUD & Auditable Decision Logging")
    print("=" * 75)
    if visual:
        print("Keys:")
        print("  P / SPACE : Pause / Resume")
        print("  G         : Toggle Local A* Grid Overlay")
        print("  E         : Export Current Decision Logs to JSON & CSV")
        print("  R         : Reset Scenario")
        print("  ESC / Q   : Quit & Print Audit Summary")

    tick = 0
    running = True
    paused = False
    show_grid_overlay = True

    while running and tick < max_ticks:
        if visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        engine.reset()
                        pilot.reset()  # clear stale tracks / per-episode state
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_g:
                        show_grid_overlay = not show_grid_overlay
                    elif event.key == pygame.K_e:
                        pilot.logger.export_json("exported_decision_log.json")
                        pilot.logger.export_csv("exported_decision_log.csv")
                        print("[Stage 9] Decision logs exported to exported_decision_log.json and .csv")

            if paused:
                clock.tick(15)
                continue

        # 1. Sense Environment
        sensor_obs = sensor.observe(engine, observing_ship_id=1)

        # 2. Complete Autonomous Agent Decision Cycle (Stages 3 -> 4 -> 7 -> 5 -> 6 -> 9)
        turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd, decision = pilot.decide(sensor_obs)

        # Single autonomous spacecraft control
        actions = {
            1: (turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd),
        }

        # 3. Environment Step
        true_state = engine.step(actions)

        # 4. Visual Rendering
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
                    grid_surf = pygame.Surface((int(pilot.planner.grid_width), int(pilot.planner.grid_height)), pygame.SRCALPHA)
                    grid_surf.fill((20, 30, 50, 40))

                    cell_sz = pilot.planner.cell_size
                    for r in range(pilot.planner.rows):
                        for c in range(pilot.planner.cols):
                            if grid[r, c] == 1:
                                cell_rect = pygame.Rect(int(c * cell_sz), int(r * cell_sz), int(cell_sz), int(cell_sz))
                                pygame.draw.rect(grid_surf, (255, 60, 40, 70), cell_rect)

                    screen.blit(grid_surf, (int(ship_pos[0] - g_half_w), int(ship_pos[1] - g_half_h)))

                if len(plan.path_world) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in plan.path_world]
                    pygame.draw.lines(screen, (50, 255, 140), False, pts, 3)

            # Draw Minimax Waypoint Target Line (Stage 7)
            if decision.minimax_decision is not None:
                wp = decision.minimax_decision.target_waypoint
                wp_color = (255, 80, 80) if decision.minimax_decision.chosen_tactic.value == "ATTACK_RIVAL" else (
                    (255, 220, 80) if decision.minimax_decision.chosen_tactic.value == "CONTEST_ASTEROID" else (80, 255, 120)
                )
                pygame.draw.circle(screen, wp_color, (int(wp[0]), int(wp[1])), 8, 2)
                # Dotted target indicator
                pygame.draw.line(screen, wp_color, (int(ship_pos[0]), int(ship_pos[1])), (int(wp[0]), int(wp[1])), 1)

            # Draw CSP Ballistic Intercept Target (Stage 6)
            csp = decision.csp_decision
            if csp.intercept_point is not None:
                int_pt = (int(csp.intercept_point[0]), int(csp.intercept_point[1]))
                color = (255, 235, 50) if csp.should_shoot else (150, 145, 80)
                pygame.draw.circle(screen, color, int_pt, 7, 2)
                pygame.draw.line(screen, color, (int_pt[0] - 10, int_pt[1]), (int_pt[0] + 10, int_pt[1]), 1)
                pygame.draw.line(screen, color, (int_pt[0], int_pt[1] - 10), (int_pt[0], int_pt[1] + 10), 1)

            # Render Stage 9 Explainability HUD Dashboard
            recent = pilot.logger.entries[-5:]
            hud.render(screen, decision.log_entry, recent)

            # Bottom Controls & Status Line
            diag_y = config.arena.height - 35
            b_surf = pygame.Surface((int(config.arena.width), 35))
            b_surf.set_alpha(200)
            b_surf.fill((12, 16, 26))
            screen.blit(b_surf, (0, diag_y))

            tactic_name = decision.minimax_decision.chosen_tactic.value if decision.minimax_decision else "N/A"
            status_text = (
                f"TICK: {tick:04d} | "
                f"MINIMAX: {tactic_name} | "
                f"RULE: {decision.active_rule} | "
                f"CSP: {'SHOOT' if decision.csp_decision.should_shoot else 'HOLD'} | "
                f"KEYS: [P] Pause, [G] Grid Overlay, [E] Export Logs, [ESC] Quit"
            )
            screen.blit(font_sm.render(status_text, True, (180, 205, 235)), (15, diag_y + 10))

            pygame.display.flip()
            clock.tick(config.fps)

        tick += 1

    if visual:
        pygame.quit()

    # Export decision logs
    pilot.logger.export_json("decision_log.json")
    pilot.logger.export_csv("decision_log.csv")

    print("\n" + pilot.logger.generate_summary_report())
    player = engine.ships[1]
    print(f"Final Score:       {player.score}")
    print(f"Shots Fired:       {player.shots_fired} (Hits: {player.shots_hit}, Accuracy: {player.accuracy * 100:.1f}%)")
    print(f"Lives Remaining:   {player.lives} / 1 (Deaths: {player.deaths})")
    print(f"Decision Logs Saved to: decision_log.json and decision_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Piloted Asteroid Game - Stages 7-9 Demo")
    parser.add_argument("--headless", action="store_true", help="Run in headless simulation mode")
    parser.add_argument("--ticks", type=int, default=300, help="Number of ticks to run")
    parser.add_argument("--genome", type=str, default="best_optimized_genome.json", help="Path to GA genome JSON")
    args = parser.parse_args()

    run_demo(visual=not args.headless, max_ticks=args.ticks, genome_file=args.genome)
