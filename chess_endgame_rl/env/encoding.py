from __future__ import annotations

import chess
import numpy as np

from chess_endgame_rl.env.scenarios import get_scenario

KING_DELTAS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1), (1, 0), (1, 1),
]
ROOK_DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
KNIGHT_DELTAS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


def action_count_for_scenario(scenario_id: str) -> int:
    key = (scenario_id or "").lower().strip()
    scenario = get_scenario(key) if key else {}
    count = scenario.get("action_count")
    if count is not None:
        return int(count)
    # Fallback to the core piece types used in this project for any future scenarios.
    return 8 + 28 + 8


def _relative_target(from_square: int, to_square: int) -> tuple[int, int]:
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    return to_rank - from_rank, to_file - from_file


def move_to_action_id(board: chess.Board, move: chess.Move) -> int | None:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None

    delta = _relative_target(move.from_square, move.to_square)
    if piece.piece_type == chess.KING:
        try:
            return KING_DELTAS.index(delta)
        except ValueError:
            return None

    if piece.piece_type == chess.ROOK:
        if delta[0] == 0 and delta[1] != 0:
            direction_index = 1 if delta[1] > 0 else 3
            distance = abs(delta[1])
            if 1 <= distance <= 7:
                return direction_index * 7 + (distance - 1)
        if delta[1] == 0 and delta[0] != 0:
            direction_index = 0 if delta[0] > 0 else 2
            distance = abs(delta[0])
            if 1 <= distance <= 7:
                return direction_index * 7 + (distance - 1)
        return None

    if piece.piece_type == chess.KNIGHT:
        try:
            return KNIGHT_DELTAS.index(delta)
        except ValueError:
            return None

    return None


def action_to_move(board: chess.Board, action_id: int) -> chess.Move | None:
    for move in board.legal_moves:
        if move_to_action_id(board, move) == action_id:
            return move
    return None


def state_index(board: chess.Board) -> int:
    piece_map = board.piece_map()
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_extra = None
    for square, piece in piece_map.items():
        if piece.color == chess.WHITE and piece.piece_type != chess.KING:
            white_extra = square
            break
    if white_king is None or black_king is None or white_extra is None:
        return 0
    return ((white_king * 64) + white_extra) * 64 + black_king


def legal_action_mask(board: chess.Board, scenario_id: str | None = None) -> np.ndarray:
    count = action_count_for_scenario(scenario_id or "krk")
    mask = np.zeros(count, dtype=np.int8)
    for move in board.legal_moves:
        action_id = move_to_action_id(board, move)
        if action_id is not None and 0 <= action_id < count:
            mask[action_id] = 1
    return mask


def board_to_observation(board: chess.Board, scenario_id: str | None = None) -> np.ndarray:
    """Create a 192-dimensional one-hot observation with three 64-square planes."""
    obs = np.zeros(192, dtype=np.float64)
    if board is None:
        return obs

    scenario = (scenario_id or "krk").lower().strip()
    if scenario and scenario not in {"krk", "knk"}:
        try:
            get_scenario(scenario)
        except KeyError:
            scenario = "krk"

    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            plane_index = 0 if piece.piece_type == chess.KING else 1
        else:
            plane_index = 2
        offset = plane_index * 64 + square
        obs[offset] = 1.0
    return obs
