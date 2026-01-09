from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .config import GameConfig, ShipSize


def _candidate_positions(
    board: np.ndarray,
    ship_size: ShipSize,
    buffer_size: int,
) -> List[Tuple[int, int]]:
    board_size = board.shape[0]
    height, width = ship_size
    candidates: List[Tuple[int, int]] = []
    for x in range(board_size - height + 1):
        for y in range(board_size - width + 1):
            x0 = max(0, x - buffer_size)
            x1 = min(board_size, x + height + buffer_size)
            y0 = max(0, y - buffer_size)
            y1 = min(board_size, y + width + buffer_size)
            if np.any(board[x0:x1, y0:y1]):
                continue
            candidates.append((x, y))
    return candidates


def _orientations(ship_size: ShipSize) -> List[ShipSize]:
    height, width = ship_size
    if height == width:
        return [ship_size]
    return [ship_size, (width, height)]


def generate_board(
    rng: np.random.Generator,
    config: GameConfig,
) -> np.ndarray:
    board = np.zeros((config.board_size, config.board_size), dtype=np.int8)
    for ship_size in config.ship_sizes:
        placements: List[Tuple[ShipSize, Tuple[int, int]]] = []
        for orientation in _orientations(ship_size):
            for x, y in _candidate_positions(board, orientation, config.adjacency_buffer):
                placements.append((orientation, (x, y)))
        if not placements:
            raise ValueError(
                "No valid placements left; try reducing adjacency buffer or ship sizes."
            )
        orientation, (x, y) = placements[rng.integers(len(placements))]
        height, width = orientation
        board[x : x + height, y : y + width] = 1
    return board


def generate_boards(
    n: int,
    rng: np.random.Generator,
    config: GameConfig,
) -> List[np.ndarray]:
    return [generate_board(rng, config) for _ in range(n)]


def board_matches_observations(board: np.ndarray, hit_map: np.ndarray) -> bool:
    hits = hit_map == 1
    misses = hit_map == -1
    if hits.any() and not np.all(board[hits] == 1):
        return False
    if misses.any() and not np.all(board[misses] == 0):
        return False
    return True


def unknown_mask(hit_map: np.ndarray) -> np.ndarray:
    return hit_map == 0
