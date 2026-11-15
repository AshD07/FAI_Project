"""
Stage 10: Multi-Run Benchmark Evaluator, Statistical Analysis, and Visualization.
Runs multi-seeded matches across all benchmark scenarios, evaluates the 5 PEAS performance measures,
and produces tables, JSON/CSV exports, and visual comparison charts.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent.autonomous_pilot import AutonomousPilot
from src.evaluation.benchmark_suite import (
    BENCHMARK_SCENARIOS,
    BaseController,
    BenchmarkScenario,
    NaiveReactiveController,
    RandomController,
    RivalBot,
    setup_scenario_engine,
)
from src.config import ShipConfig
from src.optimization.genetic_algorithm import AgentGenome

# Lives each ship starts with (1 = single-life permadeath in this project)
MAX_LIVES = ShipConfig.initial_lives


@dataclass
class RunResult:
    """Individual match results for a single trial."""
    agent_name: str
    scenario_name: str
    seed: int
    score: int
    asteroids_hit: int
    shots_fired: int
    shots_hit: int
    accuracy: float
    lives_remaining: int
    deaths: int
    survival_time: float
    successful_dodges: int            # ticks on which the emergency-evasion rule was in control
    logs_recorded: int
    field_cleared: bool = False       # True if the match ended because every asteroid was destroyed


@dataclass
class AggregatedMetric:
    """Mean and standard deviation for a performance measure."""
    mean: float
    std: float

    def format(self, is_percent: bool = False) -> str:
        if is_percent:
            return f"{self.mean * 100:.1f}% ± {self.std * 100:.1f}%"
        return f"{self.mean:.1f} ± {self.std:.1f}"


@dataclass
class AgentScenarioSummary:
    """Aggregated statistics across multiple runs of an agent in a scenario."""
    agent_name: str
    scenario_name: str
    runs: int
    score: AggregatedMetric
    asteroids_hit: AggregatedMetric
    accuracy: AggregatedMetric
    lives: AggregatedMetric
    deaths: AggregatedMetric
    survival_time: AggregatedMetric
    successful_dodges: AggregatedMetric


class BenchmarkEvaluator:
    """
    Executes standardized evaluation of autonomous pilot agents vs baselines.
    """

    def __init__(self, runs_per_scenario: int = 5):
        self.runs_per_scenario = runs_per_scenario
        self.raw_results: List[RunResult] = []

    def evaluate_match(
        self,
        agent_name: str,
        controller: BaseController,
        scenario: BenchmarkScenario,
        seed: int,
    ) -> RunResult:
        """Simulate a single benchmark scenario match and collect performance metrics."""
        engine, sensor, rival_bot = setup_scenario_engine(scenario, seed=seed)

        # Every match must start from a clean controller. Previously one AutonomousPilot
        # instance was reused across all 15+ matches: its Kalman tracks, tick counter,
        # rule hysteresis and decision log leaked between games, and the tracker saw the
        # clock jump backwards (negative dt) at the start of every new match.
        if isinstance(controller, AutonomousPilot):
            controller.reset()
        elif hasattr(controller, "reset"):
            controller.reset(seed)

        dodges_count = 0
        logs_count = 0

        max_ticks = int(scenario.time_limit * 30)

        for tick in range(max_ticks):
            if engine.is_game_over():
                break

            sensor_obs = sensor.observe(engine, observing_ship_id=1)

            # Controller action
            if isinstance(controller, AutonomousPilot):
                turn, thrust, shoot, drop_mine, dec = controller.decide(sensor_obs)
                logs_count += 1
                if "Emergency" in dec.active_rule:
                    dodges_count += 1
            else:
                turn, thrust, shoot, drop_mine = controller.decide(sensor_obs)

            # Rival bot action
            rival_turn, rival_thrust, rival_shoot, rival_mine = 0.0, 0.0, False, False
            if rival_bot and 2 in engine.ships and engine.ships[2].is_alive:
                rival_turn, rival_thrust, rival_shoot, rival_mine = rival_bot.decide(
                    engine.ships[2].to_dict(),
                    engine.get_true_state(),
                )

            actions = {
                1: (turn, thrust, shoot, drop_mine),
                2: (rival_turn, rival_thrust, rival_shoot, rival_mine),
            }
            engine.step(actions)

        player = engine.ships[1]
        return RunResult(
            agent_name=agent_name,
            scenario_name=scenario.name,
            seed=seed,
            score=player.score,
            asteroids_hit=player.asteroids_hit,
            shots_fired=player.shots_fired,
            shots_hit=player.shots_hit,
            accuracy=player.accuracy,
            lives_remaining=player.lives,
            deaths=player.deaths,
            survival_time=engine.time,
            successful_dodges=dodges_count,
            logs_recorded=logs_count,
            field_cleared=(player.is_alive and len(engine.asteroids) == 0),
        )

    def run_full_evaluation(
        self,
        agents: Dict[str, BaseController],
        scenarios: Optional[List[BenchmarkScenario]] = None,
    ) -> List[AgentScenarioSummary]:
        """Runs the complete evaluation matrix across all agents and scenarios."""
        scenarios = scenarios or BENCHMARK_SCENARIOS
        self.raw_results.clear()

        total_trials = len(agents) * len(scenarios) * self.runs_per_scenario
        trial_idx = 0

        print(f"Starting Multi-Run Evaluation: {total_trials} total trials ({self.runs_per_scenario} seeds/scenario)...")

        for scenario in scenarios:
            print(f"\n---> Evaluating Scenario: {scenario.name} ({scenario.description[:60]}...)")
            for agent_name, controller in agents.items():
                for run_idx in range(self.runs_per_scenario):
                    seed = 1000 + run_idx * 17
                    res = self.evaluate_match(agent_name, controller, scenario, seed=seed)
                    self.raw_results.append(res)
                    trial_idx += 1

                agent_runs = [r for r in self.raw_results if r.agent_name == agent_name and r.scenario_name == scenario.name]
                avg_score = np.mean([r.score for r in agent_runs])
                avg_acc = np.mean([r.accuracy for r in agent_runs]) * 100
                avg_surv = np.mean([r.survival_time for r in agent_runs])
                print(f"  [{agent_name:31s}] Score: {avg_score:5.1f} | Acc: {avg_acc:4.1f}% | Survival: {avg_surv:4.1f}s")

        return self.aggregate_summaries()

    def aggregate_summaries(self) -> List[AgentScenarioSummary]:
        """Compute mean and standard deviation across trials."""
        summaries = []
        df = pd.DataFrame([asdict(r) for r in self.raw_results])

        for (agent_name, scenario_name), group in df.groupby(["agent_name", "scenario_name"]):
            summaries.append(
                AgentScenarioSummary(
                    agent_name=agent_name,
                    scenario_name=scenario_name,
                    runs=len(group),
                    score=AggregatedMetric(mean=float(group["score"].mean()), std=float(group["score"].std())),
                    asteroids_hit=AggregatedMetric(mean=float(group["asteroids_hit"].mean()), std=float(group["asteroids_hit"].std())),
                    accuracy=AggregatedMetric(mean=float(group["accuracy"].mean()), std=float(group["accuracy"].std())),
                    lives=AggregatedMetric(mean=float(group["lives_remaining"].mean()), std=float(group["lives_remaining"].std())),
                    deaths=AggregatedMetric(mean=float(group["deaths"].mean()), std=float(group["deaths"].std())),
                    survival_time=AggregatedMetric(mean=float(group["survival_time"].mean()), std=float(group["survival_time"].std())),
                    successful_dodges=AggregatedMetric(mean=float(group["successful_dodges"].mean()), std=float(group["successful_dodges"].std())),
                )
            )
        return summaries

    def export_data(self, json_path: str = "evaluation_results.json", csv_path: str = "evaluation_results.csv") -> None:
        """Export raw run trials to JSON and CSV."""
        data = [asdict(r) for r in self.raw_results]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Evaluation data exported to {json_path} and {csv_path}")

    def generate_markdown_report(self, filepath: str = "EVALUATION_REPORT.md") -> str:
        """Generate comprehensive final evaluation report in GitHub-flavored Markdown."""
        summaries = self.aggregate_summaries()
        df = pd.DataFrame([asdict(r) for r in self.raw_results])

        agents_order = list(df["agent_name"].unique())
        overall = {
            ag: {
                "score": df[df["agent_name"] == ag]["score"].mean(),
                "acc": df[df["agent_name"] == ag]["accuracy"].mean() * 100.0,
                "surv": df[df["agent_name"] == ag]["survival_time"].mean(),
            }
            for ag in agents_order
        }
        best_score = max(overall, key=lambda a: overall[a]["score"])
        best_acc = max(overall, key=lambda a: overall[a]["acc"])
        best_surv = max(overall, key=lambda a: overall[a]["surv"])
        n_runs = int(df.groupby(["agent_name", "scenario_name"]).size().iloc[0])

        lines = [
            "# Stage 10: Performance Evaluation Report",
            "",
            f"Scenarios: {df['scenario_name'].nunique()} | Agents: {len(agents_order)} | "
            f"Runs per agent per scenario: {n_runs} | Total trials: {len(df)} | "
            f"Lives per ship: {MAX_LIVES}",
            "",
            "## 1. Headline Results (computed from the data below)",
            "| Measure | Best agent | Value |",
            "| :--- | :--- | :--- |",
            f"| Mean score | {best_score} | {overall[best_score]['score']:.1f} |",
            f"| Mean shooting accuracy | {best_acc} | {overall[best_acc]['acc']:.1f}% |",
            f"| Mean survival time | {best_surv} | {overall[best_surv]['surv']:.1f} s |",
            "",
            "## 2. Scenario Benchmark Results",
            "",
        ]

        # Group by scenario
        for scenario_name in df["scenario_name"].unique():
            lines.append(f"### Scenario: `{scenario_name}`")
            lines.append("| Agent | Score | Hits | Accuracy | Deaths | Survival Time |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

            scen_sums = [s for s in summaries if s.scenario_name == scenario_name]
            for s in scen_sums:
                lines.append(
                    f"| **{s.agent_name}** | {s.score.format()} | {s.asteroids_hit.format()} | {s.accuracy.format(is_percent=True)} | "
                    f"{s.deaths.format()} / {MAX_LIVES} | {s.survival_time.format()}s |"
                )
            lines.append("")

        # Overall Agent Comparison
        lines.append("## 3. Overall Agent Performance Averages")
        lines.append("| Agent Name | Mean Score | Mean Accuracy | Mean Survival (s) | Death Rate | Emergency-Evasion Ticks |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

        for agent_name in agents_order:
            adf = df[df["agent_name"] == agent_name]
            lines.append(
                f"| **{agent_name}** | {adf['score'].mean():.1f} ± {adf['score'].std():.1f} | "
                f"{adf['accuracy'].mean() * 100:.1f}% | "
                f"{adf['survival_time'].mean():.1f} ± {adf['survival_time'].std():.1f} | "
                f"{adf['deaths'].mean() / MAX_LIVES * 100:.0f}% | "
                f"{adf['successful_dodges'].mean():.1f} |"
            )

        lines.append("")
        lines.append("Notes: accuracy = hits / shots fired; survival time is capped at the 30 s scenario limit "
                     "and a match also ends early if every asteroid is destroyed; emergency-evasion ticks count "
                     "frames on which the reflexive dodge rule controlled the ship (not distinct dodges).")

        report_content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Markdown report generated at {filepath}")
        return report_content

    def generate_charts(self, filepath: str = "evaluation_charts.png") -> None:
        """Produces a 4-panel comparison chart (print-friendly light theme)."""
        df = pd.DataFrame([asdict(r) for r in self.raw_results])
        if df.empty:
            return

        agents = list(df["agent_name"].unique())
        # Fixed categorical order (colour follows the agent, never its rank)
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
        colors = [palette[i % len(palette)] for i in range(len(agents))]
        labels = [a.replace("Autonomous Pilot ", "Autonomous Pilot\n") for a in agents]

        panels = [
            ("score", "Mean score (higher is better)", "Score", 1.0, "{:.0f}"),
            ("accuracy", "Shooting accuracy (hits / shots)", "Accuracy (%)", 100.0, "{:.1f}%"),
            ("survival_time", "Mean survival time (30 s cap)", "Seconds", 1.0, "{:.1f}"),
            ("deaths", f"Death rate (lower is better, {MAX_LIVES} life)", "Deaths per match", 1.0, "{:.2f}"),
        ]

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
        fig.patch.set_facecolor("#ffffff")

        for ax, (col, title, ylabel, scale, fmt) in zip(axs.flat, panels):
            means = [df[df["agent_name"] == ag][col].mean() * scale for ag in agents]
            stds = [df[df["agent_name"] == ag][col].std() * scale for ag in agents]
            x = np.arange(len(agents))
            ax.bar(x, means, width=0.62, color=colors, edgecolor="#ffffff", linewidth=2, zorder=3)
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#52514e", elinewidth=1, capsize=3, zorder=4)
            top = max(m + s_ for m, s_ in zip(means, stds)) if means else 1.0
            for xi, m, sd in zip(x, means, stds):
                # Label sits above the error-bar cap so the two never collide
                ax.text(xi, m + (sd if np.isfinite(sd) else 0.0) + top * 0.02, fmt.format(m),
                        ha="center", va="bottom", color="#0b0b0b", fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5, color="#52514e")
            ax.set_title(title, fontsize=11, fontweight="bold", color="#0b0b0b", loc="left")
            ax.set_ylabel(ylabel, color="#52514e")
            ax.set_ylim(0, top * 1.12 if top > 0 else 1.0)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#b5b4ad")
            ax.tick_params(colors="#52514e")

        n_runs = int(df.groupby(["agent_name", "scenario_name"]).size().iloc[0])
        fig.suptitle(
            f"Benchmark evaluation: {df['scenario_name'].nunique()} scenarios x {n_runs} runs per agent "
            f"(error bars = 1 s.d.)",
            fontsize=13, fontweight="bold", color="#0b0b0b",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filepath, dpi=180, facecolor="#ffffff")
        plt.close()
        print(f"Evaluation charts saved to {filepath}")
