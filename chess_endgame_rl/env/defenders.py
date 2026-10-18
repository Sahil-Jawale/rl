"""Defender (Black King) policies for training and evaluation.

During training, the defender plays as the lone Black King trying to
avoid checkmate. Multiple difficulty levels are provided:

- **RandomDefender**: Chooses uniformly from legal moves (easiest).
- **HeuristicDefender**: Uses simple heuristics — captures the rook if
  possible, moves toward the center, avoids edges.
- **SmartDefender**: Combines capture priority, center-seeking, and
  opposition-awareness for a tougher opponent.
"""

from __future__ import annotations

import random

import chess


class RandomDefender:
    """Picks a random legal move for the Black King."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available for the defender")
        return self.rng.choice(legal_moves)


class HeuristicDefender:
    """Uses simple heuristics to play the Black King.
    
    Priority:
    1. Capture the rook if possible (wins the game for Black).
    2. Move toward the center of the board.
    3. Stay away from edges.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available for the defender")

        # Priority 1: Capture the rook
        for move in legal_moves:
            captured = board.piece_at(move.to_square)
            if captured and captured.piece_type == chess.ROOK:
                return move

        # Priority 2: Score each move and pick the best
        best_score = float("-inf")
        best_moves = []
        for move in legal_moves:
            score = self._score_move(board, move)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)

    def _score_move(self, board: chess.Board, move: chess.Move) -> float:
        """Score a move for the Black King (higher = better for Black)."""
        board.push(move)
        score = 0.0

        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        # Prefer center squares
        center_dist = abs(to_rank - 3.5) + abs(to_file - 3.5)
        score -= center_dist * 2.0  # Closer to center = higher score

        # Avoid edges
        edge_dist = min(to_rank, 7 - to_rank, to_file, 7 - to_file)
        score += edge_dist * 3.0

        # Avoid being in check after moving
        if board.is_check():
            score -= 5.0

        # Penalize stalemate (it's a draw, but our defender should try to avoid it)
        if board.is_stalemate():
            score -= 2.0

        board.pop()
        return score


class SmartDefender:
    """A stronger defender that tries to maintain opposition and avoid corners.
    
    Uses:
    - Rook capture priority
    - Opposition awareness (keep distance from White King)
    - Center-seeking behavior
    - Avoidance of back-rank traps
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available for the defender")

        # Priority 1: Capture the rook
        for move in legal_moves:
            captured = board.piece_at(move.to_square)
            if captured and captured.piece_type == chess.ROOK:
                return move

        best_score = float("-inf")
        best_moves = []
        for move in legal_moves:
            score = self._score_position(board, move)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)

    def _score_position(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate a position from Black's perspective."""
        board.push(move)

        white_king = board.king(chess.WHITE)
        black_king = move.to_square
        score = 0.0

        if white_king is not None:
            # Keep distance from White King
            king_distance = chess.square_distance(white_king, black_king)
            score += king_distance * 2.0

            # Try to maintain opposition (same rank or file with odd distance)
            wk_rank = chess.square_rank(white_king)
            wk_file = chess.square_file(white_king)
            bk_rank = chess.square_rank(black_king)
            bk_file = chess.square_file(black_king)

            if wk_rank == bk_rank or wk_file == bk_file:
                if king_distance % 2 == 0:
                    score += 1.0  # Opposition

        # Stay away from edges and corners
        to_rank = chess.square_rank(black_king)
        to_file = chess.square_file(black_king)
        edge_dist = min(to_rank, 7 - to_rank, to_file, 7 - to_file)
        score += edge_dist * 4.0

        # Center preference
        center_dist = max(abs(to_rank - 3.5), abs(to_file - 3.5))
        score -= center_dist * 1.5

        # Avoid being in check
        if board.is_check():
            score -= 3.0

        # Count available legal moves (mobility)
        score += len(list(board.legal_moves)) * 0.5

        board.pop()
        return score


def defender_for(name: str | None, seed: int | None = None):
    """Factory function to create a defender by name.
    
    Args:
        name: One of 'random', 'heuristic', 'smart'.
        seed: Random seed for reproducibility.
    
    Returns:
        A defender instance with a choose_move(board) method.
    """
    name = (name or "random").lower().replace("_", "-")
    if name == "random":
        return RandomDefender(seed)
    if name == "heuristic":
        return HeuristicDefender(seed)
    if name in {"smart", "optimal", "tablebase-optimal", "tablebase_optimal"}:
        return SmartDefender(seed)
    raise ValueError(f"Unsupported defender '{name}'. Use 'random', 'heuristic', or 'smart'.")
