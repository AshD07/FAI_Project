"""
Interactive & Visual Demonstration for Stages 4 to 6:
- Stage 4: Forward Chaining Rule Engine & Real-Time Explainability HUD
- Stage 5: Local Egocentric A* Pathfinding (Green planned route + Hazard Cost Map)
- Stage 6: Constraint Satisfaction Problem (CSP) Firing & Ballistic Lead Crosshair

Usage:
  python demo_stage4_6.py             # Runs interactive Pygame visualizer
  python demo_stage4_6.py --headless  # Runs automated simulation and outputs decision audit
"""

import argparse
import sys
import numpy as np
import pygame

from src.agent.autonomous_pilot import AutonomousPilot
from src.config import GameConfig
from src.environment.game_engine import GameEngine, wrapped_distance
from src.sensors.radar_sensor import RadarSensor


def run_demo(visual: bool = True, max_ticks: int = 900) -> None:
    config = GameConfig()
    engine = GameEngine(config=config, seed=42)
    # Add rival pilot for dynamic environment
    engine.add_rival_ship(position=(750.0, 400.0), heading=180.0)

    sensor = RadarSensor(config=config.sensor, seed=202)
    pilot = AutonomousPilot(
        arena_width=config.arena.width,
        arena_height=config.arena.height,
        safe_distance=130.0,
        critical_tti=1.2,
        firing_tolerance_deg=7.5,
    )

    if visual:
        pygame.init()
        screen = pygame.display.set_mode((int(config.arena.width), int(config.arena.height)))
        pygame.display.set_caption("AI Piloted Asteroid Game - Stages 4-6 (Rules, A* Pathfinding, CSP Firing)")
        clock = pygame.time.Clock()
        font_sm = pygame.font.SysFont("Consolas", 13)
        font_md = pygame.font.SysFont("Consolas", 15, bold=True)
        font_title = pygame.font.SysFont("Consolas", 18, bold=True)

    print("=" * 75)
    print("AI Piloted Asteroid Game - Stages 4 to 6 Demonstration")
    print("Stage 4: Forward Chaining Rule Engine & Auditable Explainability Log")
    print("Stage 5: Local Egocentric A* Pathfinding with Dynamic Obstacle Inflation")
    print("Stage 6: Constraint Satisfaction Problem (CSP) Lead Intercept Firing")
    print("=" * 75)
    if visual:
        print("Keys:")
        print("  P / SPACE : Pause / Resume")
        print("  G         : Toggle A* Local Grid Overlay")
        print("  R         : Reset Scenario")
        print("  ESC / Q   : Quit")

    tick = 0
    running = True
    paused = False
    show_grid_overlay = True

    # Rule and CSP event counters
    rule_counts = {}
    shots_approved = 0
    shots_denied = 0

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

            if paused:
                clock.tick(15)
                continue

        # 1. Sense Environment
        sensor_obs = sensor.observe(engine, observing_ship_id=1)

        # 2. Autonomous Agent Decision Cycle (Stages 3 -> 4 -> 5 -> 6)
        turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd, decision = pilot.decide(sensor_obs)

        # Track statistics
        rule_name = decision.active_rule
        rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        if decision.csp_decision.should_shoot:
            shots_approved += 1
        else:
            shots_denied += 1

        # Simple rival behavior
        rival_actions = (30.0, 0.4 if tick % 90 < 35 else 0.0, tick % 20 == 0, False)

        actions = {
            1: (turn_cmd, thrust_cmd, shoot_cmd, drop_mine_cmd),
            2: rival_actions,
        }

        # 3. Step Environment Physics
        true_state = engine.step(actions)

        # 4. Visual Rendering
        if visual:
            engine.render(screen)
            ship_pos = np.array(sensor_obs.telemetry.position)

            # Draw A* Local Egocentric Grid & Path Overlay (Stage 5)
            if show_grid_overlay and decision.astar_plan is not None:
                plan = decision.astar_plan
                grid = plan.grid_cost_map
                if grid is not None:
                    # Draw semi-transparent bounding box for local grid
                    g_half_w = pilot.planner.grid_width / 2.0
                    g_half_h = pilot.planner.grid_height / 2.0
                    grid_surf = pygame.Surface((int(pilot.planner.grid_width), int(pilot.planner.grid_height)), pygame.SRCALPHA)
                    grid_surf.fill((20, 30, 50, 45))  # Faint blue area

                    # Draw obstacle cells (Red/Orange)
                    cell_sz = pilot.planner.cell_size
                    for r in range(pilot.planner.rows):
                        for c in range(pilot.planner.cols):
                            if grid[r, c] == 1:
                                cell_rect = pygame.Rect(int(c * cell_sz), int(r * cell_sz), int(cell_sz), int(cell_sz))
                                pygame.draw.rect(grid_surf, (255, 60, 40, 75), cell_rect)

                    # Blit local grid surface centered at ship
                    grid_top_left = (int(ship_pos[0] - g_half_w), int(ship_pos[1] - g_half_h))
                    screen.blit(grid_surf, grid_top_left)

                # Draw A* Path Waypoints (Bright Green Route)
                if len(plan.path_world) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in plan.path_world]
                    pygame.draw.lines(screen, (50, 255, 140), False, pts, 3)
                    # Subgoal waypoint dot
                    if plan.subgoal_world is not None:
                        sg = (int(plan.subgoal_world[0]), int(plan.subgoal_world[1]))
                        pygame.draw.circle(screen, (0, 255, 200), sg, 6)

            # Draw CSP Ballistic Intercept & Aiming Crosshair (Stage 6)
            csp = decision.csp_decision
            if csp.intercept_point is not None:
                int_pt = (int(csp.intercept_point[0]), int(csp.intercept_point[1]))
                crosshair_color = (255, 230, 40) if csp.should_shoot else (160, 150, 90)
                # Lead intercept crosshair
                pygame.draw.circle(screen, crosshair_color, int_pt, 8, 2)
                pygame.draw.line(screen, crosshair_color, (int_pt[0] - 12, int_pt[1]), (int_pt[0] + 12, int_pt[1]), 1)
                pygame.draw.line(screen, crosshair_color, (int_pt[0], int_pt[1] - 12), (int_pt[0], int_pt[1] + 12), 1)

                # Firing trajectory line from ship nose to intercept
                nose = (int(ship_pos[0]), int(ship_pos[1]))
                line_color = (255, 240, 80) if csp.should_shoot else (90, 85, 40)
                pygame.draw.line(screen, line_color, nose, int_pt, 1)

            # Draw Explainability HUD Overlay (Stage 4)
            diag_h = 100
            diag_y = config.arena.height - diag_h
            hud_surf = pygame.Surface((int(config.arena.width), diag_h))
            hud_surf.set_alpha(225)
            hud_surf.fill((12, 16, 28))
            screen.blit(hud_surf, (0, diag_y))

            # Title
            t_hdr = font_title.render("AUTONOMOUS PILOT REASONING & AUDIT HUD", True, (80, 220, 255))
            screen.blit(t_hdr, (15, diag_y + 8))

            # Stage 4 Active Rule
            rule_color = (255, 80, 80) if "Emergency" in decision.active_rule else ((255, 200, 60) if "Mine" in decision.active_rule else (100, 255, 160))
            rule_str = f"STAGE 4 [RULE]: {decision.active_rule} -> {decision.active_explanation}"
            screen.blit(font_md.render(rule_str, True, rule_color), (15, diag_y + 32))

            # Stage 5 & 6 Status
            nav_mode = "OVERRIDE BURN" if "Emergency" in decision.active_rule else ("A* PATHFINDING" if decision.astar_plan else "BALLISTIC AIM")
            csp_color = (255, 230, 40) if csp.should_shoot else (180, 190, 205)
            audit_keys = " ".join([f"{k[:2]}:{'✓' if v else '✗'}" for k, v in csp.constraint_audit.items()])
            status_str = (
                f"STAGE 5 [NAV]: {nav_mode} (Turn: {turn_cmd:4.0f}°/s, Thrust: {thrust_cmd:3.1f}) | "
                f"STAGE 6 [CSP FIRING]: {'SHOOT' if csp.should_shoot else 'HOLD'} [{audit_keys}] {csp.explanation}"
            )
            screen.blit(font_sm.render(status_str, True, csp_color), (15, diag_y + 56))

            # Key facts summary
            facts = decision.facts
            if facts:
                facts_str = (
                    f"FACTS: Nearest Ast={facts.nearest_asteroid_dist:4.0f}px | Closure={facts.closure_rate:4.1f}px/s | "
                    f"TTI={facts.time_to_impact:4.2f}s | Ammo={facts.ship_ammo} | Mines={facts.ship_mines} | "
                    f"Controls: [G] Toggle A* Grid, [P/Space] Pause, [R] Reset"
                )
                screen.blit(font_sm.render(facts_str, True, (160, 180, 205)), (15, diag_y + 78))

            pygame.display.flip()
            clock.tick(config.fps)

        tick += 1

    if visual:
        pygame.quit()

    player = engine.ships[1]
    print("\n" + "=" * 75)
    print("DEMO EXECUTION COMPLETED - EVALUATION REPORT")
    print(f"Total Ticks Simulated: {tick}")
    print(f"Final Score:           {player.score}")
    print(f"Asteroids Hit:         {player.asteroids_hit}")
    print(f"Shots Fired:           {player.shots_fired} (Accuracy: {player.accuracy * 100:.1f}%)")
    print(f"Lives Remaining:       {player.lives} / 3 (Deaths: {player.deaths})")
    print("\nStage 4 Rule Trigger Statistics:")
    for r_name, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  - {r_name:32s}: {count:4d} ticks ({count / tick * 100:4.1f}%)")
    print(f"\nStage 6 CSP Shooting Decisions: Approved={shots_approved}, Denied={shots_denied}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Piloted Asteroid Game - Stages 4-6 Demo")
    parser.add_argument("--headless", action="store_true", help="Run without Pygame visual window")
    parser.add_argument("--ticks", type=int, default=300, help="Number of ticks for simulation")
    args = parser.parse_args()

    run_demo(visual=not args.headless, max_ticks=args.ticks)
