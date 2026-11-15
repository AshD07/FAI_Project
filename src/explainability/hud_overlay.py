"""
Stage 9: Pygame On-Screen Explainability HUD.
Renders real-time visual cards displaying active rules, facts, Minimax tactics,
and CSP constraints directly during gameplay.
"""

from typing import List, Optional
import pygame

from src.explainability.decision_logger import DecisionLogEntry


class ExplainabilityHUD:
    """
    Renders the live Explainability Dashboard on the Pygame surface.
    Displays rule triggers, trigger facts, Minimax tactic scores, and CSP audits.
    """

    def __init__(self, width: float = 1000.0, height: float = 800.0):
        self.width = width
        self.height = height

    def render(
        self,
        surface: pygame.Surface,
        latest_entry: Optional[DecisionLogEntry],
        recent_entries: List[DecisionLogEntry],
    ) -> None:
        """Render explainability cards and decision timeline onto the Pygame surface."""
        if latest_entry is None:
            return

        font_sm = pygame.font.SysFont("Consolas", 12)
        font_md = pygame.font.SysFont("Consolas", 14, bold=True)
        font_title = pygame.font.SysFont("Consolas", 16, bold=True)

        panel_w = 340
        panel_h = 240
        panel_x = self.width - panel_w - 15
        panel_y = 40

        # Semi-transparent background card
        bg_surf = pygame.Surface((panel_w, panel_h))
        bg_surf.set_alpha(220)
        bg_surf.fill((14, 18, 30))
        surface.blit(bg_surf, (panel_x, panel_y))
        pygame.draw.rect(surface, (50, 80, 120), (panel_x, panel_y, panel_w, panel_h), 1)

        # Card Title
        surface.blit(font_title.render("AI EXPLAINABILITY PANEL", True, (80, 220, 255)), (panel_x + 12, panel_y + 10))

        # 1. Active Rule Section
        is_emergency = "Emergency" in latest_entry.active_rule
        rule_color = (255, 75, 75) if is_emergency else ((255, 200, 50) if "Mine" in latest_entry.active_rule else (100, 255, 160))
        rule_title = "DODGE RULE ACTIVATED" if is_emergency else f"RULE: {latest_entry.active_rule}"
        surface.blit(font_md.render(rule_title, True, rule_color), (panel_x + 12, panel_y + 35))

        # Rule Facts Display (e.g. Asteroid distance, Safe distance, Closing speed)
        facts_y = panel_y + 55
        for k, v in latest_entry.rule_facts.items():
            fact_line = f" - {k.replace('_', ' ').capitalize()}: {v}"
            surface.blit(font_sm.render(fact_line, True, (210, 225, 240)), (panel_x + 15, facts_y))
            facts_y += 16

        # 2. Minimax Strategy Card
        strat_y = panel_y + 130
        pygame.draw.line(surface, (40, 60, 90), (panel_x + 10, strat_y - 5), (panel_x + panel_w - 10, strat_y - 5), 1)
        surface.blit(font_md.render("MINIMAX TACTICAL REASONING", True, (120, 210, 255)), (panel_x + 12, strat_y))

        tac_line = f"Tactic: {latest_entry.minimax_tactic} (U={latest_entry.minimax_utility:+.1f})"
        surface.blit(font_sm.render(tac_line, True, (255, 220, 100)), (panel_x + 15, strat_y + 20))
        if latest_entry.minimax_tactic == "ATTACK_RIVAL":
            sub_line = f"Anticipating Rival: {latest_entry.rival_anticipated_tactic}"
        else:
            sub_line = "Objective: Solo Survival & Asteroid Elimination"
        surface.blit(font_sm.render(sub_line, True, (180, 200, 225)), (panel_x + 15, strat_y + 36))

        # 3. CSP Firing Gate Card
        csp_y = panel_y + 185
        pygame.draw.line(surface, (40, 60, 90), (panel_x + 10, csp_y - 5), (panel_x + panel_w - 10, csp_y - 5), 1)
        # Show what the actuator actually did: the CSP may approve a shot that the
        # pilot then withholds because an emergency evasion or wall repulsion is active.
        if latest_entry.actuator_shoot:
            fire_status = "WEAPON FIRE: APPROVED"
        elif latest_entry.csp_shoot:
            fire_status = "WEAPON FIRE: CSP OK, HELD (EVADING)"
        else:
            fire_status = "WEAPON FIRE: HOLD"
        fire_color = (255, 230, 40) if latest_entry.actuator_shoot else (160, 175, 195)
        surface.blit(font_md.render(fire_status, True, fire_color), (panel_x + 12, csp_y))

        audit_str = " ".join([f"{k[:2]}:{'✓' if v else '✗'}" for k, v in latest_entry.csp_audit.items()])
        surface.blit(font_sm.render(f"Constraints: [{audit_str}]", True, (200, 215, 230)), (panel_x + 15, csp_y + 20))
        surface.blit(font_sm.render(latest_entry.csp_explanation[:45], True, (150, 170, 190)), (panel_x + 15, csp_y + 36))
