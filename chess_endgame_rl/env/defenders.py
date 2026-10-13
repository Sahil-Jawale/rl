from __future__ import annotations

import random

import chess


class RandomDefender:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available for the defender")
        return self.rng.choice(legal_moves)


class HeuristicDefender:
    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available for the defender")

        captures = []
        safe_moves = []
        for move in legal_moves:
            board.push(move)
            is_capture = board.is_capture(move)
            is_legal = not board.is_check()
            if is_capture:
                captures.append(move)
            elif is_legal:
                safe_moves.append(move)
            board.pop()

        if captures:
            return captures[0]
        if safe_moves:
            return safe_moves[0]
        return legal_moves[0]


class TablebaseOptimalDefender:
    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available for the defender")

        best_move = legal_moves[0]
        best_score = float("-inf")
        for move in legal_moves:
            board.push(move)
            score = self._score_position(board)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _score_position(self, board: chess.Board) -> float:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            return -1e9

        king_distance = chess.square_distance(white_king, black_king)
        score = 0.0

        # Prefer keeping the black king far from the white king.
        score += (king_distance * 2.0)

        # Prefer safe king locations over risky ones.
        if not board.is_check():
            score += 1.0

        # Penalize the white side's immediate tactical threats.
        legal_white = list(board.legal_moves)
        if board.turn == chess.BLACK:
            legal_white = list(board.legal_moves)
        score -= 0.25 * len(legal_white)

        return score


def defender_for(name: str | None, seed: int | None = None):
    name = (name or "random").lower().replace("_", "-")
    if name == "random":
        return RandomDefender(seed)
    if name == "heuristic":
        return HeuristicDefender()
    if name in {"tablebase-optimal", "tablebase_optimal", "optimal"}:
        return TablebaseOptimalDefender()
    raise ValueError(f"Unsupported defender '{name}'")
