from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from .board import board_matches_observations, generate_board
from .config import GameConfig, MonteCarloConfig, SearchConfig, ShipSize


@dataclass
class SearchResult:
    shots: int
    hit_map: np.ndarray
    history: List[Tuple[int, int, bool]]


Algorithm = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.random.Generator,
        GameConfig,
        SearchConfig,
        MonteCarloConfig,
    ],
    SearchResult,
]


def _normalize_prob_map(prob_map: np.ndarray, hit_map: np.ndarray) -> np.ndarray:
    prob_map = prob_map.copy()
    prob_map[hit_map != 0] = 0.0
    total = float(prob_map.sum())
    if total <= 0:
        prob_map = np.where(hit_map == 0, 1.0, 0.0)
        total = float(prob_map.sum())
        if total <= 0:
            return prob_map
    return prob_map / total


def _pick_next_cell(
    prob_map: np.ndarray, hit_map: np.ndarray, rng: np.random.Generator
) -> Tuple[int, int]:
    masked = np.where(hit_map == 0, prob_map, -np.inf)
    max_val = float(np.max(masked))
    if not np.isfinite(max_val):
        candidates = np.argwhere(hit_map == 0)
        idx = rng.integers(len(candidates))
        return int(candidates[idx, 0]), int(candidates[idx, 1])
    candidates = np.argwhere(masked == max_val)
    idx = rng.integers(len(candidates))
    return int(candidates[idx, 0]), int(candidates[idx, 1])


def _entropy(prob_map: np.ndarray, hit_map: np.ndarray) -> float:
    mask = hit_map == 0
    if not mask.any():
        return 0.0
    values = prob_map[mask]
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _pick_next_cell_eig(
    prob_map: np.ndarray,
    hit_map: np.ndarray,
    prior: np.ndarray,
    remaining_lengths: List[int],
    game_config: GameConfig,
    endpoint_boost: float,
    strict: bool,
    diag_prune: bool,
    rng: np.random.Generator,
    top_k: int = 12,
) -> Tuple[int, int]:
    unknown = np.argwhere(hit_map == 0)
    if unknown.size == 0:
        return _pick_next_cell(prob_map, hit_map, rng)

    values = prob_map[hit_map == 0]
    if unknown.shape[0] > top_k:
        idxs = np.argpartition(values, -top_k)[-top_k:]
        candidates = unknown[idxs]
    else:
        candidates = unknown

    best_score = None
    best_cells: List[Tuple[int, int]] = []
    for x, y in candidates:
        p_hit = prob_map[x, y]

        miss_map = hit_map.copy()
        miss_map[x, y] = -1
        miss_prob = _compute_prob_map(
            miss_map,
            prior,
            remaining_lengths,
            game_config,
            endpoint_boost,
            strict,
        )
        miss_entropy = _entropy(miss_prob, miss_map)

        hit_map_next = hit_map.copy()
        hit_map_next[x, y] = 1
        if diag_prune:
            _mark_diagonal_misses(hit_map_next, game_config.adjacency_buffer)
        hit_prob = _compute_prob_map(
            hit_map_next,
            prior,
            remaining_lengths,
            game_config,
            endpoint_boost,
            strict,
        )
        hit_entropy = _entropy(hit_prob, hit_map_next)

        expected_entropy = p_hit * hit_entropy + (1.0 - p_hit) * miss_entropy
        if best_score is None or expected_entropy < best_score - 1e-9:
            best_score = expected_entropy
            best_cells = [(int(x), int(y))]
        elif best_score is not None and abs(expected_entropy - best_score) <= 1e-9:
            best_cells.append((int(x), int(y)))

    if not best_cells:
        return _pick_next_cell(prob_map, hit_map, rng)
    idx = rng.integers(len(best_cells))
    return best_cells[idx]


def _mark_diagonal_misses(hit_map: np.ndarray, buffer_size: int) -> None:
    if buffer_size <= 0:
        return
    board_size = hit_map.shape[0]
    hits = np.argwhere(hit_map == 1)
    for x, y in hits:
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if hit_map[nx, ny] == 0:
                    hit_map[nx, ny] = -1


def _extract_ships(board: np.ndarray) -> List[List[Tuple[int, int]]]:
    visited = np.zeros_like(board, dtype=bool)
    ships: List[List[Tuple[int, int]]] = []
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x, y] != 1 or visited[x, y]:
                continue
            stack = [(x, y)]
            visited[x, y] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                        if board[nx, ny] == 1 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
            ships.append(cells)
    return ships


def _mark_buffer(
    hit_map: np.ndarray, cells: Iterable[Tuple[int, int]], buffer_size: int
) -> None:
    if buffer_size <= 0:
        return
    board_size = hit_map.shape[0]
    for x, y in cells:
        for dx in range(-buffer_size, buffer_size + 1):
            for dy in range(-buffer_size, buffer_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if hit_map[nx, ny] == 0:
                        hit_map[nx, ny] = -1


def _cluster_hits(hit_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    visited = np.zeros_like(hit_mask, dtype=bool)
    clusters: List[List[Tuple[int, int]]] = []
    for x in range(hit_mask.shape[0]):
        for y in range(hit_mask.shape[1]):
            if not hit_mask[x, y] or visited[x, y]:
                continue
            stack = [(x, y)]
            visited[x, y] = True
            cluster: List[Tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cluster.append((cx, cy))
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < hit_mask.shape[0] and 0 <= ny < hit_mask.shape[1]:
                        if hit_mask[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
            clusters.append(cluster)
    return clusters


def _hunt_heatmap(
    hit_map: np.ndarray, remaining_lengths: Iterable[int], parity: int
) -> np.ndarray:
    board_size = hit_map.shape[0]
    blocked = hit_map != 0
    counts = np.zeros((board_size, board_size), dtype=float)
    for length in remaining_lengths:
        orientations = {(length, 1), (1, length)}
        for height, width in orientations:
            for x in range(board_size - height + 1):
                for y in range(board_size - width + 1):
                    if np.any(blocked[x : x + height, y : y + width]):
                        continue
                    counts[x : x + height, y : y + width] += 1.0
    if parity > 1:
        grid_x, grid_y = np.meshgrid(
            np.arange(board_size), np.arange(board_size), indexing="ij"
        )
        mask = (grid_x + grid_y) % parity == 0
        counts = np.where(mask, counts, 0.0)
    return counts


def _target_heatmap(
    hit_map: np.ndarray,
    clusters: Iterable[List[Tuple[int, int]]],
    remaining_lengths: List[int],
    endpoint_boost: float = 1.0,
) -> np.ndarray:
    board_size = hit_map.shape[0]
    counts = np.zeros((board_size, board_size), dtype=float)
    for cluster in clusters:
        cluster_set = set(cluster)
        xs = {x for x, _ in cluster}
        ys = {y for _, y in cluster}
        cluster_len = len(cluster)
        if cluster_len == 1:
            orientations = ("h", "v")
        elif len(xs) == 1:
            orientations = ("h",)
        elif len(ys) == 1:
            orientations = ("v",)
        else:
            continue

        for length in remaining_lengths:
            if length < cluster_len:
                continue
            for orient in orientations:
                if orient == "h":
                    x = next(iter(xs))
                    min_y = min(ys)
                    max_y = max(ys)
                    start_min = max_y - length + 1
                    start_max = min_y
                    for start_y in range(start_min, start_max + 1):
                        if start_y < 0 or start_y + length > board_size:
                            continue
                        invalid = False
                        for y in range(start_y, start_y + length):
                            if hit_map[x, y] == -1:
                                invalid = True
                                break
                            if hit_map[x, y] == 1 and (x, y) not in cluster_set:
                                invalid = True
                                break
                        if invalid:
                            continue
                        counts[x, start_y : start_y + length] += 1.0
                else:
                    y = next(iter(ys))
                    min_x = min(xs)
                    max_x = max(xs)
                    start_min = max_x - length + 1
                    start_max = min_x
                    for start_x in range(start_min, start_max + 1):
                        if start_x < 0 or start_x + length > board_size:
                            continue
                        invalid = False
                        for x in range(start_x, start_x + length):
                            if hit_map[x, y] == -1:
                                invalid = True
                                break
                            if hit_map[x, y] == 1 and (x, y) not in cluster_set:
                                invalid = True
                                break
                        if invalid:
                            continue
                        counts[start_x : start_x + length, y] += 1.0
    if endpoint_boost > 1.0:
        for cluster in clusters:
            xs = {x for x, _ in cluster}
            ys = {y for _, y in cluster}
            candidates: List[Tuple[int, int]] = []
            if len(cluster) == 1:
                x, y = cluster[0]
                candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            elif len(xs) == 1:
                x = next(iter(xs))
                candidates = [(x, min(ys) - 1), (x, max(ys) + 1)]
            elif len(ys) == 1:
                y = next(iter(ys))
                candidates = [(min(xs) - 1, y), (max(xs) + 1, y)]
            for nx, ny in candidates:
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if hit_map[nx, ny] == 0 and counts[nx, ny] > 0:
                        counts[nx, ny] *= endpoint_boost
    return counts


def _target_heatmap_strict(
    hit_map: np.ndarray,
    clusters: Iterable[List[Tuple[int, int]]],
    remaining_lengths: List[int],
    buffer_size: int,
    endpoint_boost: float = 1.0,
) -> np.ndarray:
    board_size = hit_map.shape[0]
    counts = np.zeros((board_size, board_size), dtype=float)
    all_hits = hit_map == 1
    for cluster in clusters:
        cluster_set = set(cluster)
        cluster_mask = np.zeros_like(hit_map, dtype=bool)
        for x, y in cluster:
            cluster_mask[x, y] = True
        outside_hits = all_hits & ~cluster_mask

        xs = {x for x, _ in cluster}
        ys = {y for _, y in cluster}
        cluster_len = len(cluster)
        if cluster_len == 1:
            orientations = ("h", "v")
        elif len(xs) == 1:
            orientations = ("h",)
        elif len(ys) == 1:
            orientations = ("v",)
        else:
            continue

        for length in remaining_lengths:
            if length < cluster_len:
                continue
            for orient in orientations:
                if orient == "h":
                    x = next(iter(xs))
                    min_y = min(ys)
                    max_y = max(ys)
                    start_min = max_y - length + 1
                    start_max = min_y
                    for start_y in range(start_min, start_max + 1):
                        if start_y < 0 or start_y + length > board_size:
                            continue
                        invalid = False
                        for y in range(start_y, start_y + length):
                            if hit_map[x, y] == -1:
                                invalid = True
                                break
                            if hit_map[x, y] == 1 and (x, y) not in cluster_set:
                                invalid = True
                                break
                        if invalid:
                            continue
                        if buffer_size > 0:
                            y0 = max(0, start_y - buffer_size)
                            y1 = min(board_size, start_y + length + buffer_size)
                            x0 = max(0, x - buffer_size)
                            x1 = min(board_size, x + 1 + buffer_size)
                            if outside_hits[x0:x1, y0:y1].any():
                                continue
                        counts[x, start_y : start_y + length] += 1.0
                else:
                    y = next(iter(ys))
                    min_x = min(xs)
                    max_x = max(xs)
                    start_min = max_x - length + 1
                    start_max = min_x
                    for start_x in range(start_min, start_max + 1):
                        if start_x < 0 or start_x + length > board_size:
                            continue
                        invalid = False
                        for x in range(start_x, start_x + length):
                            if hit_map[x, y] == -1:
                                invalid = True
                                break
                            if hit_map[x, y] == 1 and (x, y) not in cluster_set:
                                invalid = True
                                break
                        if invalid:
                            continue
                        if buffer_size > 0:
                            x0 = max(0, start_x - buffer_size)
                            x1 = min(board_size, start_x + length + buffer_size)
                            y0 = max(0, y - buffer_size)
                            y1 = min(board_size, y + 1 + buffer_size)
                            if outside_hits[x0:x1, y0:y1].any():
                                continue
                        counts[start_x : start_x + length, y] += 1.0

    if endpoint_boost > 1.0:
        for cluster in clusters:
            xs = {x for x, _ in cluster}
            ys = {y for _, y in cluster}
            candidates: List[Tuple[int, int]] = []
            if len(cluster) == 1:
                x, y = cluster[0]
                candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            elif len(xs) == 1:
                x = next(iter(xs))
                candidates = [(x, min(ys) - 1), (x, max(ys) + 1)]
            elif len(ys) == 1:
                y = next(iter(ys))
                candidates = [(min(xs) - 1, y), (max(xs) + 1, y)]
            for nx, ny in candidates:
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if hit_map[nx, ny] == 0 and counts[nx, ny] > 0:
                        counts[nx, ny] *= endpoint_boost
    return counts


def _compute_prob_map(
    hit_map: np.ndarray,
    prior: np.ndarray,
    remaining_lengths: List[int],
    game_config: GameConfig,
    endpoint_boost: float,
    strict: bool,
    active_hit_mask: np.ndarray | None = None,
) -> np.ndarray:
    if active_hit_mask is None:
        active_hit_mask = hit_map == 1
    clusters = _cluster_hits(active_hit_mask)
    prob_map = np.zeros_like(prior, dtype=float)
    if clusters:
        if strict:
            prob_map = _target_heatmap_strict(
                hit_map,
                clusters,
                remaining_lengths,
                buffer_size=game_config.adjacency_buffer,
                endpoint_boost=endpoint_boost,
            )
        else:
            prob_map = _target_heatmap(
                hit_map,
                clusters,
                remaining_lengths,
                endpoint_boost=endpoint_boost,
            )
    if not clusters or prob_map.sum() <= 0:
        min_len = min(remaining_lengths) if remaining_lengths else 1
        parity = 2 if min_len >= 2 else 1
        prob_map = _hunt_heatmap(hit_map, remaining_lengths, parity)

    prob_map *= prior
    return _normalize_prob_map(prob_map, hit_map)


def _adjacent_update(
    prob_map: np.ndarray,
    hit_map: np.ndarray,
    x: int,
    y: int,
    hit: bool,
    search_config: SearchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= prob_map.shape[0] or ny >= prob_map.shape[1]:
            continue
        if hit_map[nx, ny] != 0:
            continue
        if hit:
            scale = search_config.hit_boost + rng.normal(0.0, search_config.noise)
        else:
            scale = search_config.miss_decay + rng.normal(0.0, search_config.noise)
        prob_map[nx, ny] *= max(scale, 0.0)
    return _normalize_prob_map(prob_map, hit_map)


def _orientation_update(
    prob_map: np.ndarray,
    hit_map: np.ndarray,
    x: int,
    y: int,
    hit: bool,
    search_config: SearchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if hit:
        adjacent_hits = []
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < prob_map.shape[0] and 0 <= ny < prob_map.shape[1]:
                if hit_map[nx, ny] == 1:
                    adjacent_hits.append((dx, dy))
        horiz = any(dx == 0 for dx, _ in adjacent_hits)
        vert = any(dy == 0 for _, dy in adjacent_hits)
        if horiz and not vert:
            for dx, dy in [(1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < prob_map.shape[0] and 0 <= ny < prob_map.shape[1]:
                    if hit_map[nx, ny] == 0:
                        prob_map[nx, ny] = 0.0
        if vert and not horiz:
            for dx, dy in [(0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < prob_map.shape[0] and 0 <= ny < prob_map.shape[1]:
                    if hit_map[nx, ny] == 0:
                        prob_map[nx, ny] = 0.0

    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= prob_map.shape[0] or ny >= prob_map.shape[1]:
            continue
        if hit_map[nx, ny] != 0:
            continue
        if hit:
            scale = search_config.hit_boost + rng.normal(0.0, search_config.noise)
        else:
            scale = search_config.miss_decay + rng.normal(0.0, search_config.noise)
        prob_map[nx, ny] *= max(scale, 0.0)
    return _normalize_prob_map(prob_map, hit_map)


def adjacent_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    hit_map = np.zeros((game_config.board_size, game_config.board_size), dtype=np.int8)
    prob_map = _normalize_prob_map(prior, hit_map)
    history: List[Tuple[int, int, bool]] = []
    shots = 0
    hits = 0
    total_hits = int(board.sum())
    while hits < total_hits:
        x, y = _pick_next_cell(prob_map, hit_map, rng)
        hit = bool(board[x, y])
        hit_map[x, y] = 1 if hit else -1
        history.append((x, y, hit))
        shots += 1
        if hit:
            hits += 1
        prob_map[x, y] = 0.0
        prob_map = _adjacent_update(prob_map, hit_map, x, y, hit, search_config, rng)
    return SearchResult(shots=shots, hit_map=hit_map, history=history)


def orientation_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    hit_map = np.zeros((game_config.board_size, game_config.board_size), dtype=np.int8)
    prob_map = _normalize_prob_map(prior, hit_map)
    history: List[Tuple[int, int, bool]] = []
    shots = 0
    hits = 0
    total_hits = int(board.sum())
    while hits < total_hits:
        x, y = _pick_next_cell(prob_map, hit_map, rng)
        hit = bool(board[x, y])
        hit_map[x, y] = 1 if hit else -1
        history.append((x, y, hit))
        shots += 1
        if hit:
            hits += 1
        prob_map[x, y] = 0.0
        prob_map = _orientation_update(prob_map, hit_map, x, y, hit, search_config, rng)
    return SearchResult(shots=shots, hit_map=hit_map, history=history)


def _placement_heatmap(
    hit_map: np.ndarray,
    ship_sizes: Iterable[ShipSize],
) -> np.ndarray:
    board_size = hit_map.shape[0]
    hits = hit_map == 1
    misses = hit_map == -1
    has_hits = hits.any()
    counts = np.zeros((board_size, board_size), dtype=float)
    for ship_size in ship_sizes:
        orientations = {ship_size, (ship_size[1], ship_size[0])}
        for height, width in orientations:
            for x in range(board_size - height + 1):
                for y in range(board_size - width + 1):
                    window_miss = misses[x : x + height, y : y + width]
                    if window_miss.any():
                        continue
                    window_hit = hits[x : x + height, y : y + width]
                    if has_hits and not window_hit.any():
                        continue
                    counts[x : x + height, y : y + width] += 1.0
    counts[hit_map != 0] = 0.0
    total = float(counts.sum())
    if total <= 0:
        counts = np.where(hit_map == 0, 1.0, 0.0)
        total = float(counts.sum())
    return counts / total


def placement_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    hit_map = np.zeros((game_config.board_size, game_config.board_size), dtype=np.int8)
    history: List[Tuple[int, int, bool]] = []
    shots = 0
    hits = 0
    total_hits = int(board.sum())
    while hits < total_hits:
        prob_map = _placement_heatmap(hit_map, game_config.ship_sizes)
        prob_map *= prior
        prob_map = _normalize_prob_map(prob_map, hit_map)
        x, y = _pick_next_cell(prob_map, hit_map, rng)
        hit = bool(board[x, y])
        hit_map[x, y] = 1 if hit else -1
        history.append((x, y, hit))
        shots += 1
        if hit:
            hits += 1
    return SearchResult(shots=shots, hit_map=hit_map, history=history)


def _hunt_target_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    diag_prune: bool,
    endpoint_boost: float,
    strict: bool,
    use_eig: bool,
    eig_top_k: int,
    trace: List[Tuple[int, int, np.ndarray, np.ndarray]] | None = None,
    snapshot_every: int = 10,
) -> SearchResult:
    hit_map = np.zeros((game_config.board_size, game_config.board_size), dtype=np.int8)
    history: List[Tuple[int, int, bool]] = []
    ships = _extract_ships(board)
    remaining_lengths = [len(cells) for cells in ships]
    sunk_ids: set[int] = set()
    shots = 0
    hits = 0
    total_hits = int(board.sum())
    last_prob_map = None
    while hits < total_hits:
        newly_sunk: List[List[Tuple[int, int]]] = []
        for idx, cells in enumerate(ships):
            if idx in sunk_ids:
                continue
            if all(hit_map[x, y] == 1 for x, y in cells):
                sunk_ids.add(idx)
                newly_sunk.append(cells)
                length = len(cells)
                if length in remaining_lengths:
                    remaining_lengths.remove(length)
        for cells in newly_sunk:
            _mark_buffer(hit_map, cells, game_config.adjacency_buffer)

        if diag_prune:
            _mark_diagonal_misses(hit_map, game_config.adjacency_buffer)

        active_hit_mask = hit_map == 1
        for idx in sunk_ids:
            for x, y in ships[idx]:
                active_hit_mask[x, y] = False

        prob_map = _compute_prob_map(
            hit_map,
            prior,
            remaining_lengths,
            game_config,
            endpoint_boost,
            strict,
            active_hit_mask=active_hit_mask,
        )
        last_prob_map = prob_map

        if use_eig:
            x, y = _pick_next_cell_eig(
                prob_map,
                hit_map,
                prior,
                remaining_lengths,
                game_config,
                endpoint_boost,
                strict,
                diag_prune,
                rng,
                top_k=eig_top_k,
            )
        else:
            x, y = _pick_next_cell(prob_map, hit_map, rng)
        hit = bool(board[x, y])
        hit_map[x, y] = 1 if hit else -1
        history.append((x, y, hit))
        shots += 1
        if hit:
            hits += 1

        if trace is not None and shots % snapshot_every == 0:
            trace.append((shots, hits, hit_map.copy(), prob_map.copy()))

    if trace is not None and last_prob_map is not None:
        if not trace or trace[-1][0] != shots:
            trace.append((shots, hits, hit_map.copy(), last_prob_map.copy()))

    return SearchResult(shots=shots, hit_map=hit_map, history=history)


def hunt_target_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    return _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=False,
        endpoint_boost=1.0,
        strict=False,
        use_eig=False,
        eig_top_k=12,
    )


def hunt_target_diag_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    return _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=True,
        endpoint_boost=1.0,
        strict=False,
        use_eig=False,
        eig_top_k=12,
    )


def hunt_target_focus_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    return _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=True,
        endpoint_boost=1.8,
        strict=False,
        use_eig=False,
        eig_top_k=12,
    )


def hunt_target_aggressive_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    return _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=True,
        endpoint_boost=3.0,
        strict=False,
        use_eig=False,
        eig_top_k=12,
    )


def hunt_target_strict_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    return _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=True,
        endpoint_boost=1.8,
        strict=True,
        use_eig=False,
        eig_top_k=12,
    )


def hunt_target_eig_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    return _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=True,
        endpoint_boost=1.8,
        strict=True,
        use_eig=True,
        eig_top_k=12,
    )


def hunt_target_trace(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    snapshot_every: int = 10,
) -> Tuple[SearchResult, List[Tuple[int, int, np.ndarray, np.ndarray]]]:
    trace: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    result = _hunt_target_search(
        board,
        prior,
        rng,
        game_config,
        diag_prune=True,
        endpoint_boost=1.8,
        strict=True,
        use_eig=False,
        eig_top_k=12,
        trace=trace,
        snapshot_every=snapshot_every,
    )
    return result, trace


def _sample_consistent_boards(
    hit_map: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    mc_config: MonteCarloConfig,
    target_count: int,
) -> List[np.ndarray]:
    boards: List[np.ndarray] = []
    attempts = 0
    while len(boards) < target_count and attempts < mc_config.max_attempts:
        attempts += 1
        board = generate_board(rng, game_config)
        if board_matches_observations(board, hit_map):
            boards.append(board)
    return boards


def monte_carlo_bayes_search(
    board: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    game_config: GameConfig,
    search_config: SearchConfig,
    mc_config: MonteCarloConfig,
) -> SearchResult:
    hit_map = np.zeros((game_config.board_size, game_config.board_size), dtype=np.int8)
    history: List[Tuple[int, int, bool]] = []
    pool = np.asarray(
        [generate_board(rng, game_config) for _ in range(mc_config.samples)],
        dtype=np.int8,
    )
    weights = np.ones(pool.shape[0], dtype=float)
    shots = 0
    hits = 0
    total_hits = int(board.sum())
    while hits < total_hits:
        weight_sum = float(weights.sum()) if weights.size else 0.0
        if weight_sum > 0.0:
            prob_map = (pool * weights[:, None, None]).sum(axis=0) / weight_sum
            prob_map *= prior
        else:
            resampled = _sample_consistent_boards(
                hit_map, rng, game_config, mc_config, mc_config.samples
            )
            if resampled:
                pool = np.asarray(resampled, dtype=np.int8)
                weights = np.ones(pool.shape[0], dtype=float)
                prob_map = pool.mean(axis=0) * prior
            else:
                prob_map = np.where(hit_map == 0, 1.0, 0.0)
        prob_map = _normalize_prob_map(prob_map, hit_map)
        x, y = _pick_next_cell(prob_map, hit_map, rng)
        hit = bool(board[x, y])
        hit_map[x, y] = 1 if hit else -1
        history.append((x, y, hit))
        shots += 1
        if hit:
            hits += 1
        if weights.size:
            if hit:
                weights *= pool[:, x, y]
            else:
                weights *= 1 - pool[:, x, y]
    return SearchResult(shots=shots, hit_map=hit_map, history=history)


ALGORITHMS: Dict[str, Algorithm] = {
    "adjacent": adjacent_bayes_search,
    "orientation": orientation_bayes_search,
    "placement": placement_bayes_search,
    "hunt_target": hunt_target_bayes_search,
    "hunt_target_diag": hunt_target_diag_bayes_search,
    "hunt_target_focus": hunt_target_focus_bayes_search,
    "hunt_target_aggressive": hunt_target_aggressive_bayes_search,
    "hunt_target_strict": hunt_target_strict_bayes_search,
    "hunt_target_eig": hunt_target_eig_bayes_search,
    "monte_carlo": monte_carlo_bayes_search,
}
