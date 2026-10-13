from __future__ import annotations

import random

import chess


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return self.rng.choice(legal_moves)

    def select_action(self, board: chess.Board, action_mask: list[int] | None = None):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return self.rng.choice(legal_moves)
