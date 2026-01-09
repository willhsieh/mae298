from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battleship.config import (
    load_benchmark_config_from_env,
    load_game_config_from_env,
    load_monte_carlo_config_from_env,
    load_search_config_from_env,
)
from battleship.evaluate import benchmark_algorithms
from battleship.priors import center_bias_prior, edge_bias_prior, uniform_prior
from battleship.search import ALGORITHMS


def _prior_from_env():
    name = os.getenv("BATTLESHIP_PRIOR", "uniform").strip().lower()
    if name == "center":
        return "center", center_bias_prior
    if name == "edge":
        return "edge", edge_bias_prior
    return "uniform", uniform_prior


def _algorithms_from_env():
    value = os.getenv(
        "BATTLESHIP_ALGORITHMS",
        "adjacent,orientation,placement,hunt_target_focus,hunt_target_strict,hunt_target_eig",
    )
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names:
        names = list(ALGORITHMS.keys())
    return names


def main() -> None:
    game_config = load_game_config_from_env()
    search_config = load_search_config_from_env()
    mc_config = load_monte_carlo_config_from_env()
    benchmark_config = load_benchmark_config_from_env()

    prior_name, prior_fn = _prior_from_env()
    algorithm_names = _algorithms_from_env()

    results = benchmark_algorithms(
        algorithm_names,
        prior_fn,
        benchmark_config,
        game_config,
        search_config,
        mc_config,
    )

    print("Battleship Benchmark")
    print(f"Boards: {benchmark_config.boards} | Prior: {prior_name}")
    print("Algorithms:")
    for name in algorithm_names:
        result = results[name]
        print(
            f"- {name:12s} mean={result.mean:6.2f} "
            f"std={result.std:6.2f} median={result.median:6.2f} "
            f"min={result.min_shots:3d} max={result.max_shots:3d}"
        )


if __name__ == "__main__":
    main()
