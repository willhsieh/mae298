"""Battleship search helpers and evaluation tools."""

from .config import (
    BenchmarkConfig,
    GameConfig,
    MonteCarloConfig,
    SearchConfig,
    load_benchmark_config_from_env,
    load_game_config_from_env,
    load_monte_carlo_config_from_env,
    load_search_config_from_env,
)
from .board import generate_board, generate_boards
from .priors import center_bias_prior, edge_bias_prior, uniform_prior
from .search import ALGORITHMS, SearchResult
from .evaluate import benchmark_algorithms

__all__ = [
    "ALGORITHMS",
    "BenchmarkConfig",
    "GameConfig",
    "MonteCarloConfig",
    "SearchConfig",
    "SearchResult",
    "benchmark_algorithms",
    "center_bias_prior",
    "edge_bias_prior",
    "generate_board",
    "generate_boards",
    "load_benchmark_config_from_env",
    "load_game_config_from_env",
    "load_monte_carlo_config_from_env",
    "load_search_config_from_env",
    "uniform_prior",
]
