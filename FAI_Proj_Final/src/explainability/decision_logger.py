"""
Stage 9: Comprehensive Decision Logger for Explainable Autonomous Agent Behavior.
Records structured logs of rules, facts, tactics, and constraints for post-game auditing.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class DecisionLogEntry:
    """Complete audit record for a single game tick."""
    tick: int
    timestamp: float
    ship_pos: Tuple[float, float]
    ship_speed: float
    ship_heading: float
    ship_lives: int
    ship_ammo: int

    # Stage 4: Rule Engine Facts & Explanations
    active_rule: str
    rule_severity: str
    rule_facts: Dict[str, Any]
    rule_action: str
    rule_explanation: str

    # Stage 7: Minimax Adversarial Strategy
    minimax_tactic: str
    minimax_utility: float
    rival_anticipated_tactic: str

    # Stage 6: CSP Firing Constraints
    csp_shoot: bool
    csp_target_id: Optional[int]
    csp_audit: Dict[str, bool]
    csp_explanation: str

    # Final Actuators
    actuator_turn: float
    actuator_thrust: float
    actuator_shoot: bool
    actuator_drop_mine: bool


class DecisionLogger:
    """
    Maintains persistent, structured records of the AI agent's decisions.
    Supports export to JSON, CSV, and formatted text summaries.
    """

    def __init__(self):
        self.entries: List[DecisionLogEntry] = []

    def log(self, entry: DecisionLogEntry) -> None:
        self.entries.append(entry)

    def export_json(self, filepath: str = "decision_log.json") -> None:
        """Export all decision logs to formatted JSON."""
        data = [asdict(e) for e in self.entries]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str = "decision_log.csv") -> None:
        """Export tabular summary to CSV."""
        rows = []
        for e in self.entries:
            row = {
                "tick": e.tick,
                "timestamp": e.timestamp,
                "ship_x": e.ship_pos[0],
                "ship_y": e.ship_pos[1],
                "ship_speed": e.ship_speed,
                "ship_heading": e.ship_heading,
                "ship_lives": e.ship_lives,
                "ship_ammo": e.ship_ammo,
                "active_rule": e.active_rule,
                "rule_severity": e.rule_severity,
                "minimax_tactic": e.minimax_tactic,
                "minimax_utility": e.minimax_utility,
                "csp_shoot": e.csp_shoot,
                "actuator_turn": e.actuator_turn,
                "actuator_thrust": e.actuator_thrust,
                "actuator_shoot": e.actuator_shoot,
                "actuator_drop_mine": e.actuator_drop_mine,
            }
            # Flatten rule facts
            for k, v in e.rule_facts.items():
                row[f"fact_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

    def generate_summary_report(self) -> str:
        """Generates a comprehensive explainability summary report."""
        if not self.entries:
            return "No decision entries recorded."

        total_ticks = len(self.entries)
        rule_counts = {}
        tactic_counts = {}
        shots_approved = 0
        emergency_dodges = 0
        mines_deployed = 0

        for e in self.entries:
            rule_counts[e.active_rule] = rule_counts.get(e.active_rule, 0) + 1
            tactic_counts[e.minimax_tactic] = tactic_counts.get(e.minimax_tactic, 0) + 1
            if e.csp_shoot:
                shots_approved += 1
            if "Emergency" in e.active_rule:
                emergency_dodges += 1
            if e.actuator_drop_mine:
                mines_deployed += 1

        lines = [
            "=" * 70,
            "EXPLAINABILITY DECISION AUDIT SUMMARY REPORT",
            "=" * 70,
            f"Total Evaluated Decision Cycles: {total_ticks}",
            f"Emergency Collision Dodges:       {emergency_dodges} ({emergency_dodges / total_ticks * 100:.1f}%)",
            f"Defensive Mines Deployed:        {mines_deployed}",
            f"CSP Approved Firing Decisions:   {shots_approved} ({shots_approved / total_ticks * 100:.1f}%)",
            "\n[Rule Engine Activation Breakdown]:",
        ]
        for r_name, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {r_name:35s}: {count:4d} ticks ({count / total_ticks * 100:4.1f}%)")

        lines.append("\n[Minimax Macro-Tactic Breakdown]:")
        for t_name, count in sorted(tactic_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {t_name:35s}: {count:4d} ticks ({count / total_ticks * 100:4.1f}%)")

        lines.append("=" * 70)
        return "\n".join(lines)
