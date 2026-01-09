from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np

from .board import generate_boards
from .config import BenchmarkConfig, GameConfig, MonteCarloConfig, SearchConfig
from .search import ALGORITHMS, Algorithm


@dataclass
class BenchmarkResult:
    shots: List[int]
    mean: float
    std: float
    median: float
    min_shots: int
    max_shots: int


def _summarize(shots: List[int]) -> BenchmarkResult:
    arr = np.asarray(shots, dtype=float)
    return BenchmarkResult(
        shots=shots,
        mean=float(arr.mean()),
        std=float(arr.std()),
        median=float(np.median(arr)),
        min_shots=int(arr.min()),
        max_shots=int(arr.max()),
    )


def benchmark_algorithms(
    algorithm_names: Iterable[str],
    prior_fn: Callable[[int], np.ndarray],
    benchmark_config: BenchmarkConfig,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> Dict[str, BenchmarkResult]:
    board_rng = np.random.default_rng(benchmark_config.seed)
    boards = generate_boards(benchmark_config.boards, board_rng, game_config)

    results: Dict[str, BenchmarkResult] = {}
    for offset, name in enumerate(algorithm_names, start=1):
        if name not in ALGORITHMS:
            raise KeyError(f"Unknown algorithm: {name}")
        algo: Algorithm = ALGORITHMS[name]
        algo_rng = np.random.default_rng(benchmark_config.seed + offset)
        shots: List[int] = []
        for board in boards:
            prior = prior_fn(game_config.board_size)
            result = algo(board, prior, algo_rng, game_config, search_config, mc_config)
            shots.append(result.shots)
        results[name] = _summarize(shots)
    return results
