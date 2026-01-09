from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Tuple

ShipSize = Tuple[int, int]


@dataclass(frozen=True)
class GameConfig:
    board_size: int = 10
    ship_sizes: Tuple[ShipSize, ...] = ((5, 1), (4, 1), (3, 1), (3, 1), (2, 1))
    adjacency_buffer: int = 1


@dataclass(frozen=True)
class SearchConfig:
    noise: float = 0.01
    hit_boost: float = 4.0
    miss_decay: float = 0.25
    show_every: int = 10


@dataclass(frozen=True)
class MonteCarloConfig:
    samples: int = 300
    max_attempts: int = 5000


@dataclass(frozen=True)
class BenchmarkConfig:
    boards: int = 50
    seed: int = 298


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_ship_sizes(value: str) -> Tuple[ShipSize, ...]:
    sizes: list[ShipSize] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" in token:
            parts = token.lower().split("x")
            if len(parts) != 2:
                raise ValueError(f"Invalid ship size token: {token}")
            sizes.append((int(parts[0]), int(parts[1])))
        else:
            sizes.append((int(token), 1))
    if not sizes:
        raise ValueError("No ship sizes parsed")
    return tuple(sizes)


def _env_ship_sizes(key: str, default: Iterable[ShipSize]) -> Tuple[ShipSize, ...]:
    value = os.getenv(key)
    if value is None or not value.strip():
        return tuple(default)
    try:
        return _parse_ship_sizes(value)
    except ValueError:
        return tuple(default)


def load_game_config_from_env(default: GameConfig | None = None) -> GameConfig:
    base = default or GameConfig()
    return GameConfig(
        board_size=_env_int("BATTLESHIP_BOARD_SIZE", base.board_size),
        ship_sizes=_env_ship_sizes("BATTLESHIP_SHIP_SIZES", base.ship_sizes),
        adjacency_buffer=_env_int("BATTLESHIP_ADJACENCY_BUFFER", base.adjacency_buffer),
    )


def load_search_config_from_env(default: SearchConfig | None = None) -> SearchConfig:
    base = default or SearchConfig()
    return SearchConfig(
        noise=_env_float("BATTLESHIP_NOISE", base.noise),
        hit_boost=_env_float("BATTLESHIP_HIT_BOOST", base.hit_boost),
        miss_decay=_env_float("BATTLESHIP_MISS_DECAY", base.miss_decay),
        show_every=_env_int("BATTLESHIP_SHOW_EVERY", base.show_every),
    )


def load_monte_carlo_config_from_env(
    default: MonteCarloConfig | None = None,
) -> MonteCarloConfig:
    base = default or MonteCarloConfig()
    return MonteCarloConfig(
        samples=_env_int("BATTLESHIP_MC_SAMPLES", base.samples),
        max_attempts=_env_int("BATTLESHIP_MC_MAX_ATTEMPTS", base.max_attempts),
    )


def load_benchmark_config_from_env(
    default: BenchmarkConfig | None = None,
) -> BenchmarkConfig:
    base = default or BenchmarkConfig()
    return BenchmarkConfig(
        boards=_env_int("BATTLESHIP_BOARDS", base.boards),
        seed=_env_int("BATTLESHIP_SEED", base.seed),
    )
