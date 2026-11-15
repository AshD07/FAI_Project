"""
Stage 8: Offline Genetic Algorithm Training Script.
Evolves autonomous pilot parameters over simulated games and outputs the best genome.

Usage:
  python train_ga.py --generations 5 --pop 8 --ticks 150
"""

import argparse
import json
import time

from src.optimization.genetic_algorithm import AgentGenome, GeneticOptimizer


def main():
    parser = argparse.ArgumentParser(description="Stage 8: Offline Genetic Algorithm Optimizer")
    parser.add_argument("--generations", type=int, default=5, help="Number of evolutionary generations")
    parser.add_argument("--pop", type=int, default=8, help="Population size")
    parser.add_argument("--ticks", type=int, default=150, help="Simulation ticks per fitness evaluation")
    parser.add_argument("--output", type=str, default="best_optimized_genome.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 8: GENETIC ALGORITHM OFFLINE PARAMETER OPTIMIZATION")
    print(f"Population Size: {args.pop} | Generations: {args.generations} | Ticks/Eval: {args.ticks}")
    print("Parameters being optimized:")
    print("  1. safe_distance (px)")
    print("  2. closing_speed_threshold (px/s)")
    print("  3. shooting_safety_margin (px)")
    print("  4. astar_safety_margin (px)")
    print("  5. minimax_weight_rival, minimax_weight_contest, minimax_weight_safety")
    print("  6. firing_tolerance_deg (deg)")
    print("=" * 70)

    optimizer = GeneticOptimizer(
        population_size=args.pop,
        crossover_rate=0.7,
        mutation_rate=0.25,
        tournament_size=3,
        seed=101,
    )

    t0 = time.time()
    for gen in range(1, args.generations + 1):
        g_t0 = time.time()
        summary = optimizer.evolve_generation(ticks_per_eval=args.ticks)
        g_elapsed = time.time() - g_t0

        print(
            f"Gen {summary['generation']:02d}/{args.generations:02d} | "
            f"Best Fit: {summary['best_fitness']:6.1f} | "
            f"Gen Best: {summary['gen_best_fitness']:6.1f} | "
            f"Mean Fit: {summary['mean_fitness']:6.1f} | "
            f"Time: {g_elapsed:4.1f}s"
        )

    total_time = time.time() - t0
    optimizer.save_best(args.output)

    print("\n" + "=" * 70)
    print(f"OPTIMIZATION FINISHED IN {total_time:.1f}s")
    print(f"Best Overall Fitness: {optimizer.best_fitness:.1f}")
    best_genome = AgentGenome.from_chromosome(optimizer.best_individual)
    print("Best Evolved Parameters:")
    for k, v in best_genome.to_dict().items():
        print(f"  - {k:28s}: {v:6.2f}")
    print(f"Checkpoint saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
