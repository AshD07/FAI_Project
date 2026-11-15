"""
Evaluation subpackage: Multi-scenario benchmarking and statistical evaluators.
"""

from src.evaluation.benchmark_suite import (
    BENCHMARK_SCENARIOS,
    BaseController,
    BenchmarkScenario,
    NaiveReactiveController,
    RandomController,
    RivalBot,
)
from src.evaluation.evaluator import BenchmarkEvaluator

__all__ = [
    "BENCHMARK_SCENARIOS",
    "BaseController",
    "BenchmarkEvaluator",
    "BenchmarkScenario",
    "NaiveReactiveController",
    "RandomController",
    "RivalBot",
]
