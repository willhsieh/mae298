from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT))

from battleship.board import generate_board
from battleship.config import BenchmarkConfig, GameConfig, MonteCarloConfig, SearchConfig
from battleship.evaluate import benchmark_algorithms
from battleship.priors import uniform_prior
from battleship.search import hunt_target_trace

ALGORITHMS = [
    "adjacent",
    "orientation",
    "placement",
    "hunt_target_focus",
    "hunt_target_strict",
    "hunt_target_eig",
]

CONFIGS = [
    (
        "10x10_default",
        GameConfig(),
        BenchmarkConfig(boards=50, seed=298),
    ),
]


def plot_benchmark_results(name: str, results, out_dir: Path) -> None:
    labels = ALGORITHMS
    means = [results[key].mean for key in labels]
    stds = [results[key].std for key in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Shots (mean ± std)")
    ax.set_title(f"Battleship Benchmark: {name}")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"benchmark_{name}.png", dpi=150)
    plt.close(fig)


def save_search_snapshot(
    hit_map: np.ndarray,
    prob_map: np.ndarray,
    shots: int,
    hits: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(hit_map, cmap="Greys")
    axes[0].set_title(f"Search Pattern (Shots: {shots}, Hits: {hits})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(prob_map, cmap="viridis")
    axes[1].set_title("Probability Map")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    search_config = SearchConfig()
    mc_config = MonteCarloConfig()

    for name, game_config, bench_config in CONFIGS:
        results = benchmark_algorithms(
            ALGORITHMS,
            uniform_prior,
            bench_config,
            game_config,
            search_config,
            mc_config,
        )
        plot_benchmark_results(name, results, out_dir)

    # Generate search snapshots for the best model (hunt_target_strict config).
    rng = np.random.default_rng(298)
    game_config = GameConfig()
    board = generate_board(rng, game_config)
    prior = uniform_prior(game_config.board_size)
    result, trace = hunt_target_trace(
        board,
        prior,
        rng,
        game_config,
        snapshot_every=10,
    )

    for shots, hits, hit_map, prob_map in trace:
        save_search_snapshot(
            hit_map,
            prob_map,
            shots,
            hits,
            out_dir / f"hunt_target_strict_shot_{shots}.png",
        )

    print(f"Saved benchmark plots and search snapshots to {out_dir}")


if __name__ == "__main__":
    main()
