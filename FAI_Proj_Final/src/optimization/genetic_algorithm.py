"""
Stage 8: Genetic Algorithm for Offline Parameter Optimization.
Optimizes critical AI thresholds (safe distance, closing speed, safety margins,
A* parameters, and Minimax weights) across simulated matches.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import GameConfig
from src.environment.game_engine import GameEngine
from src.sensors.radar_sensor import RadarSensor


@dataclass
class AgentGenome:
    """
    Decoded numerical parameter set optimized by the Genetic Algorithm.
    Covers the 5 core parameters identified in Section 4.2 of the specification.
    """
    safe_distance: float = 145.0             # Parameter 1: Safe distance from asteroids (px)
    closing_speed_threshold: float = 15.0    # Parameter 2: Closing speed threshold for dodge (px/s)
    shooting_safety_margin: float = 50.0     # Parameter 3: Shooting safety margin (px)
    astar_safety_margin: float = 22.0        # Parameter 4: A* obstacle inflation clearance (px)
    minimax_weight_rival: float = 1.5        # Parameter 5a: Minimax attacking weight
    minimax_weight_contest: float = 2.0      # Parameter 5b: Minimax asteroid contest weight
    minimax_weight_safety: float = 2.5       # Parameter 5c: Minimax retreat safety weight
    firing_tolerance_deg: float = 7.5        # Parameter 6: Firing angle alignment tolerance (deg)

    # Allowable bounds for normalization [min, max]
    BOUNDS = {
        "safe_distance": (60.0, 180.0),
        "closing_speed_threshold": (10.0, 60.0),
        "shooting_safety_margin": (30.0, 90.0),
        "astar_safety_margin": (10.0, 40.0),
        "minimax_weight_rival": (0.5, 3.5),
        "minimax_weight_contest": (0.5, 3.5),
        "minimax_weight_safety": (0.5, 4.0),
        "firing_tolerance_deg": (4.0, 12.0),
    }

    def to_chromosome(self) -> np.ndarray:
        """Encode physical parameters to normalized values in [0.0, 1.0]."""
        genes = []
        for key, (b_min, b_max) in self.BOUNDS.items():
            val = getattr(self, key)
            norm = (val - b_min) / (b_max - b_min)
            genes.append(float(np.clip(norm, 0.0, 1.0)))
        return np.array(genes, dtype=np.float64)

    @classmethod
    def from_chromosome(cls, chromosome: np.ndarray) -> "AgentGenome":
        """Decode normalized [0.0, 1.0] chromosome back to physical parameters."""
        params = {}
        for i, (key, (b_min, b_max)) in enumerate(cls.BOUNDS.items()):
            norm_val = float(np.clip(chromosome[i], 0.0, 1.0))
            params[key] = b_min + norm_val * (b_max - b_min)
        return cls(**params)

    def to_dict(self) -> Dict:
        return asdict(self)


class GeneticOptimizer:
    """
    Offline Genetic Algorithm optimizing agent parameters over simulated game matches.
    Uses Tournament Selection, Arithmetic Crossover, Gaussian Mutation, and Elitism.
    """

    def __init__(
        self,
        population_size: int = 10,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 0.15,
        tournament_size: int = 3,
        seed: Optional[int] = None,
    ):
        self.pop_size = population_size
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.mut_sigma = mutation_sigma
        self.tourn_size = tournament_size
        self.rng = np.random.default_rng(seed)

        self.num_genes = len(AgentGenome.BOUNDS)
        self.population: List[np.ndarray] = [
            self.rng.uniform(0.0, 1.0, size=self.num_genes) for _ in range(self.pop_size)
        ]
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = -float("inf")
        self.generation: int = 0

    def evaluate_fitness(self, chromosome: np.ndarray, ticks: int = 150, seed: int = 42) -> float:
        """
        Runs headless simulated games to evaluate a candidate genome.

        Two episodes are played so the GA optimises for the conditions the agent is
        actually deployed in:
          1. solo survival with continuous asteroid replenishment (run_survival.py), and
          2. a match against a scripted rival ship (adversarial benchmark scenarios).
        Per-episode fitness = Score + 150*Accuracy + 100*Lives - 100*Deaths + 30*Hits
                              + 10*SurvivalSeconds
        and the genome's fitness is the mean over both episodes.

        Fixes over the original version: the decoded genome is now actually passed to
        the pilot (previously only 4 of the 8 genes were applied, so the minimax weights
        and the closing-speed threshold were never evaluated), and the pilot uses the
        same critical TTI as at deployment (previously 1.2 s in training vs 2.2 s live).
        """
        from src.agent.autonomous_pilot import AutonomousPilot

        genome = AgentGenome.from_chromosome(chromosome)
        episode_fitness = []

        for episode, with_rival in enumerate((False, True)):
            ep_seed = seed + 1000 * episode
            config = GameConfig()
            engine = GameEngine(config=config, seed=ep_seed, auto_replenish=not with_rival)
            if with_rival:
                engine.add_rival_ship(position=(750.0, 400.0), heading=180.0)
            sensor = RadarSensor(config=config.sensor, seed=ep_seed + 1)

            pilot = AutonomousPilot(
                arena_width=config.arena.width,
                arena_height=config.arena.height,
                genome=genome,
            )

            for tick in range(ticks):
                if engine.is_game_over():
                    break

                sensor_obs = sensor.observe(engine, observing_ship_id=1)
                turn, thrust, shoot, drop_mine, _ = pilot.decide(sensor_obs)

                actions = {1: (turn, thrust, shoot, drop_mine)}
                if with_rival:
                    actions[2] = (30.0, 0.4 if tick % 90 < 35 else 0.0, tick % 20 == 0, False)
                engine.step(actions)

            player = engine.ships[1]
            episode_fitness.append(
                player.score
                + 150.0 * player.accuracy
                + 100.0 * player.lives
                - 100.0 * player.deaths
                + 30.0 * player.asteroids_hit
                + 10.0 * engine.time
            )

        return float(np.mean(episode_fitness))

    def evolve_generation(self, ticks_per_eval: int = 150) -> Dict:
        """Run 1 generational cycle of selection, crossover, mutation, and elitism."""
        self.generation += 1

        # 1. Evaluate population
        fitnesses = [
            self.evaluate_fitness(ind, ticks=ticks_per_eval, seed=42 + self.generation)
            for ind in self.population
        ]

        # 2. Track best individual (Elitism with re-evaluation)
        # Every generation is evaluated on a fresh seed, and the elite carried over from
        # the previous generation is part of this population, so it is re-scored here.
        # The champion is the best individual *on this generation's seed*.  (Previously a
        # one-off lucky score on an earlier seed was kept forever and could never be beaten
        # on a fair comparison, so the saved "best" genome was mostly the luckiest one.)
        max_idx = int(np.argmax(fitnesses))
        gen_best_fitness = fitnesses[max_idx]
        self.best_fitness = gen_best_fitness
        self.best_individual = self.population[max_idx].copy()

        # 3. Tournament Selection
        selected = []
        for _ in range(self.pop_size):
            k = min(self.tourn_size, self.pop_size)  # avoid crash when tournament > population
            candidates = self.rng.choice(self.pop_size, size=k, replace=False)
            best_cand = max(candidates, key=lambda c: fitnesses[c])
            selected.append(self.population[best_cand].copy())

        # 4. Crossover & Mutation
        next_pop = []
        # Elitism: carry over the all-time best individual directly
        next_pop.append(self.best_individual.copy() if self.best_individual is not None else selected[0].copy())

        while len(next_pop) < self.pop_size:
            idx1, idx2 = self.rng.choice(len(selected), size=2, replace=False)
            p1, p2 = selected[idx1], selected[idx2]

            # Arithmetic Crossover
            if self.rng.uniform(0.0, 1.0) < self.cx_rate:
                alpha = self.rng.uniform(0.2, 0.8)
                child = alpha * p1 + (1.0 - alpha) * p2
            else:
                child = p1.copy()

            # Gaussian Mutation
            if self.rng.uniform(0.0, 1.0) < self.mut_rate:
                noise = self.rng.normal(0.0, self.mut_sigma, size=self.num_genes)
                child = np.clip(child + noise, 0.0, 1.0)

            next_pop.append(child)

        self.population = next_pop[:self.pop_size]

        summary = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": float(np.mean(fitnesses)),
            "gen_best_fitness": gen_best_fitness,
            "best_genome": AgentGenome.from_chromosome(self.best_individual).to_dict() if self.best_individual is not None else {},
        }
        return summary

    def save_best(self, filepath: str = "best_optimized_genome.json") -> None:
        """Save best discovered parameters to JSON."""
        if self.best_individual is None:
            return

        genome = AgentGenome.from_chromosome(self.best_individual)
        data = {
            "generation": self.generation,
            "fitness": self.best_fitness,
            "parameters": genome.to_dict(),
            "chromosome": [float(x) for x in self.best_individual],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
