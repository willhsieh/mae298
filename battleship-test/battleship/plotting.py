from __future__ import annotations

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_board(board: np.ndarray) -> None:
    board_size = board.shape[0]
    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(["blue", "red"])
    ax.matshow(board, cmap=cmap)
    ax.grid(True, which="both", color="black", linewidth=1)
    ax.set_xticks(np.arange(-0.5, board_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, board_size, 1), minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_search(hit_map: np.ndarray, prob_map: np.ndarray, shots: int, hits: int) -> None:
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(hit_map, cmap="Greys")
    plt.title(f"Search Pattern (Shots: {shots}, Hits: {hits})")
    plt.subplot(122)
    plt.imshow(prob_map, cmap="viridis")
    plt.title("Probability Map")
    plt.show()
