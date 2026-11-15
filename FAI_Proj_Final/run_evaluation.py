"""
Stage 10: Master Evaluation Execution Script.
Runs comprehensive multi-scenario, multi-seed comparative benchmarking:
  - Random Baseline
  - Naive Reactive Baseline
  - Autonomous Pilot (Default)
  - Autonomous Pilot (GA-Optimized)
Produces statistical summaries, Markdown report, JSON/CSV exports, and Matplotlib charts.

Usage:
  python run_evaluation.py --runs 5
"""

import argparse
import json
import os
import sys

from src.agent.autonomous_pilot import AutonomousPilot
from src.evaluation.benchmark_suite import (
    BENCHMARK_SCENARIOS,
    NaiveReactiveController,
    RandomController,
)
from src.evaluation.evaluator import BenchmarkEvaluator
from src.optimization.genetic_algorithm import AgentGenome


def main():
    parser = argparse.ArgumentParser(description="Stage 10: Performance Evaluation Suite")
    parser.add_argument("--runs", type=int, default=5, help="Number of seeds/runs per scenario per agent")
    parser.add_argument("--genome", type=str, default="best_optimized_genome.json", help="Path to GA optimized genome")
    parser.add_argument("--charts", type=str, default="evaluation_charts.png", help="Path for comparison charts PNG")
    parser.add_argument("--report", type=str, default="EVALUATION_REPORT.md", help="Path for markdown report")
    args = parser.parse_args()

    print("=" * 75)
    print("STAGE 10: COMPREHENSIVE PERFORMANCE EVALUATION")
    print(f"Scenarios: {len(BENCHMARK_SCENARIOS)} | Runs per scenario: {args.runs}")
    print("Performance Dimensions Evaluated (Proposal Section 10):")
    print("  1. Score               : Asteroids destroyed and total score earned")
    print("  2. Accuracy            : Successful hits / total shots fired (%)")
    print("  3. Survival            : Survival time (s) and deaths (single life)")
    print("  4. Collision Avoidance : Number of successful evasive maneuvers")
    print("  5. Explainability      : 100% of decisions recorded and auditable")
    print("=" * 75)

    # 1. Setup Agents
    agents = {
        "Random Baseline": RandomController(seed=42),
        "Naive Reactive": NaiveReactiveController(),
        "Autonomous Pilot (Default)": AutonomousPilot(),
    }

    # Add GA-Optimized pilot if genome exists
    if os.path.exists(args.genome):
        with open(args.genome, "r", encoding="utf-8") as f:
            data = json.load(f)
            genome = AgentGenome(**data.get("parameters", {}))
            agents["Autonomous Pilot (GA-Optimized)"] = AutonomousPilot(genome=genome)
            print(f"[Optimizer] Loaded GA parameters from {args.genome} for Optimized Agent.")

    evaluator = BenchmarkEvaluator(runs_per_scenario=args.runs)

    # 2. Run Full Evaluation Matrix
    summaries = evaluator.run_full_evaluation(agents=agents, scenarios=BENCHMARK_SCENARIOS)

    # 3. Export Data and Reports
    evaluator.export_data(json_path="evaluation_results.json", csv_path="evaluation_results.csv")
    evaluator.generate_markdown_report(filepath=args.report)
    evaluator.generate_charts(filepath=args.charts)

    print("\n" + "=" * 75)
    print("STAGE 10 EVALUATION COMPLETE!")
    print(f"  - Markdown Report : {args.report}")
    print(f"  - Visual Charts   : {args.charts}")
    print(f"  - Raw JSON Data   : evaluation_results.json")
    print(f"  - Raw CSV Data    : evaluation_results.csv")
    print("=" * 75)


if __name__ == "__main__":
    main()
