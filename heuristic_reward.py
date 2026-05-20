# -*- coding: utf-8 -*-
"""
Heuristic reward shaping for Gomoku self-play.

This module keeps the implementation intentionally small and conservative:
- open three: 01110
- open four: 011110
- five in a row: 11111
"""

from __future__ import annotations

import numpy as np


def _count_overlapping(text, pattern):
    count = 0
    start = 0
    while True:
        index = text.find(pattern, start)
        if index == -1:
            return count
        count += 1
        start = index + 1


def _board_to_grid(board):
    grid = np.zeros((board.height, board.width), dtype=np.int8)
    for move, player in board.states.items():
        row = move // board.width
        col = move % board.width
        grid[row, col] = player
    return grid


def _iter_lines(grid):
    for row in grid:
        yield row.tolist()
    for col in grid.T:
        yield col.tolist()
    for offset in range(-grid.shape[0] + 1, grid.shape[1]):
        diag = np.diagonal(grid, offset=offset)
        if len(diag) >= 5:
            yield diag.tolist()
    flipped = np.fliplr(grid)
    for offset in range(-flipped.shape[0] + 1, flipped.shape[1]):
        diag = np.diagonal(flipped, offset=offset)
        if len(diag) >= 5:
            yield diag.tolist()


def _encode_line(line, player):
    chars = []
    for value in line:
        if value == player:
            chars.append("1")
        elif value == 0:
            chars.append("0")
        else:
            chars.append("2")
    return "".join(chars)


def count_patterns(board, player):
    counts = {
        "open_three": 0,
        "open_four": 0,
        "five": 0,
    }
    for line in _iter_lines(_board_to_grid(board)):
        encoded = _encode_line(line, player)
        counts["open_three"] += _count_overlapping(encoded, "01110")
        counts["open_four"] += _count_overlapping(encoded, "011110")
        counts["five"] += _count_overlapping(encoded, "11111")
    return counts


def _weighted_score(counts, config):
    return (
        counts["open_three"] * config["open_three_weight"]
        + counts["open_four"] * config["open_four_weight"]
        + counts["five"] * config["five_weight"]
    )


def compute_heuristic_reward(board, player, config):
    opponent = board.players[0] if player == board.players[1] else board.players[1]
    player_counts = count_patterns(board, player)
    opponent_counts = count_patterns(board, opponent)
    raw_diff = _weighted_score(player_counts, config) - _weighted_score(
        opponent_counts, config
    )
    normalized = float(np.tanh(raw_diff / max(config["normalizer"], 1e-6)))
    detail = {
        "player_counts": player_counts,
        "opponent_counts": opponent_counts,
        "raw_diff": float(raw_diff),
        "normalized_reward": normalized,
    }
    return normalized, detail
