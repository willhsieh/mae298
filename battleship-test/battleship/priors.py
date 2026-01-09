from __future__ import annotations

import numpy as np


def _normalize(prob_map: np.ndarray) -> np.ndarray:
    total = float(prob_map.sum())
    if total <= 0:
        raise ValueError("Probability map has non-positive total")
    return prob_map / total


def uniform_prior(board_size: int) -> np.ndarray:
    return np.full((board_size, board_size), 1.0 / (board_size * board_size))


def center_bias_prior(board_size: int, sigma: float | None = None) -> np.ndarray:
    if sigma is None:
        sigma = board_size / 2.5
    grid_x, grid_y = np.meshgrid(np.arange(board_size), np.arange(board_size), indexing="ij")
    center = (board_size - 1) / 2.0
    dist2 = (grid_x - center) ** 2 + (grid_y - center) ** 2
    prob_map = np.exp(-dist2 / (2.0 * sigma * sigma))
    return _normalize(prob_map)


def edge_bias_prior(board_size: int, decay: float | None = None) -> np.ndarray:
    if decay is None:
        decay = 1.25
    grid_x, grid_y = np.meshgrid(np.arange(board_size), np.arange(board_size), indexing="ij")
    dist_to_edge = np.minimum.reduce(
        [grid_x, grid_y, board_size - 1 - grid_x, board_size - 1 - grid_y]
    )
    prob_map = np.exp(-dist_to_edge / decay)
    return _normalize(prob_map)
