from __future__ import annotations

import chess


class TablebaseAgent:
    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return legal_moves[0]

    def select_action(self, board: chess.Board, action_mask: list[int] | None = None):
        return self.choose_move(board)
