from __future__ import annotations

import chess


class GreedyAgent:
    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

        return legal_moves[0]

    def select_action(self, board: chess.Board, action_mask: list[int] | None = None):
        return self.choose_move(board)
