"""State encoding and action mapping for the KRK chess endgame.

Action space (36 total):
  Actions 0-7:   King moves (8 directions: NW, N, NE, W, E, SW, S, SE)
  Actions 8-35:  Rook moves (4 directions × 7 distances)
    8-14:   Rook north  (1-7 squares)
    15-21:  Rook east   (1-7 squares)
    22-28:  Rook south  (1-7 squares)
    29-35:  Rook west   (1-7 squares)

Observation space (192 dimensions):
  Plane 0 (indices 0-63):   White King position (one-hot)
  Plane 1 (indices 64-127): White Rook position (one-hot)
  Plane 2 (indices 128-191): Black King position (one-hot)
"""

from __future__ import annotations

import chess
import numpy as np

from chess_endgame_rl.env.scenarios import get_scenario

# King can move in 8 directions (rank_delta, file_delta)
KING_DELTAS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Rook moves in 4 directions: North (+rank), East (+file), South (-rank), West (-file)
ROOK_DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Knight deltas (kept for potential future scenarios)
KNIGHT_DELTAS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


def action_count_for_scenario(scenario_id: str) -> int:
    """Return the total number of discrete actions for a scenario."""
    key = (scenario_id or "").lower().strip()
    scenario = get_scenario(key) if key else {}
    count = scenario.get("action_count")
    if count is not None:
        return int(count)
    # Default for KRK: 8 king moves + 4×7 rook moves = 36
    return 36


def _relative_target(from_square: int, to_square: int) -> tuple[int, int]:
    """Compute the (rank_delta, file_delta) between two squares."""
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    return to_rank - from_rank, to_file - from_file


def move_to_action_id(board: chess.Board, move: chess.Move) -> int | None:
    """Map a chess.Move to its discrete action ID.
    
    Returns None if the move doesn't correspond to any known action.
    """
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
        rank_delta, file_delta = delta
        if rank_delta == 0 and file_delta != 0:
            # Horizontal move: East (index 1) or West (index 3)
            direction_index = 1 if file_delta > 0 else 3
            distance = abs(file_delta)
            if 1 <= distance <= 7:
                return 8 + direction_index * 7 + (distance - 1)
        elif file_delta == 0 and rank_delta != 0:
            # Vertical move: North (index 0) or South (index 2)
            direction_index = 0 if rank_delta > 0 else 2
            distance = abs(rank_delta)
            if 1 <= distance <= 7:
                return 8 + direction_index * 7 + (distance - 1)
        return None

    if piece.piece_type == chess.KNIGHT:
        try:
            return 8 + KNIGHT_DELTAS.index(delta)
        except ValueError:
            return None

    return None


def action_to_move(board: chess.Board, action_id: int) -> chess.Move | None:
    """Map a discrete action ID back to a chess.Move on the given board.
    
    Returns None if no legal move corresponds to this action ID.
    """
    for move in board.legal_moves:
        if move_to_action_id(board, move) == action_id:
            return move
    return None


def state_index(board: chess.Board) -> int:
    """Compute a compact integer index for tabular methods.
    
    Encodes the position as: white_king * 64 * 64 + rook * 64 + black_king
    Total states: 64 × 64 × 64 = 262,144
    """
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_extra = None
    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE and piece.piece_type != chess.KING:
            white_extra = square
            break
    if white_king is None or black_king is None or white_extra is None:
        return 0
    return ((white_king * 64) + white_extra) * 64 + black_king


def legal_action_mask(board: chess.Board, scenario_id: str | None = None) -> np.ndarray:
    """Return a binary mask indicating which actions are legal.
    
    Returns:
        A 1D array of shape (action_count,) with 1 for legal actions, 0 otherwise.
    """
    count = action_count_for_scenario(scenario_id or "krk")
    mask = np.zeros(count, dtype=np.int8)
    for move in board.legal_moves:
        action_id = move_to_action_id(board, move)
        if action_id is not None and 0 <= action_id < count:
            mask[action_id] = 1
    return mask


def board_to_observation(board: chess.Board, scenario_id: str | None = None) -> np.ndarray:
    """Create a 192-dimensional one-hot observation with three 64-square planes.
    
    Plane 0: White King
    Plane 1: White Rook (or other non-king White piece)
    Plane 2: Black King
    
    Returns:
        A 1D float32 array of shape (192,).
    """
    obs = np.zeros(192, dtype=np.float32)
    if board is None:
        return obs

    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            plane_index = 0 if piece.piece_type == chess.KING else 1
        else:
            plane_index = 2
        offset = plane_index * 64 + square
        if offset < 192:
            obs[offset] = 1.0
    return obs


PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a chess position into an 18x8x8 binary tensor from side-to-move perspective.
    
    If Black is to move, the board is mirrored vertically and colors are swapped
    so that the active player's pieces always occupy planes 0-5.
    
    Planes:
        0-5:   Own pieces (P, N, B, R, Q, K)
        6-11:  Opponent pieces (P, N, B, R, Q, K)
        12:    Side-to-move (all ones for canonical representation)
        13-16: Castling rights (own K, own Q, opp K, opp Q)
        17:    En-passant target square (if available)
        
    Returns:
        np.ndarray of shape (18, 8, 8) with dtype float32.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    view_board = board if board.turn == chess.WHITE else board.mirror()

    # Planes 0-5: Own pieces (White in view_board)
    for p_idx, pt in enumerate(PIECE_TYPES):
        for sq in view_board.pieces(pt, chess.WHITE):
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[p_idx, r, f] = 1.0

    # Planes 6-11: Opponent pieces (Black in view_board)
    for p_idx, pt in enumerate(PIECE_TYPES):
        for sq in view_board.pieces(pt, chess.BLACK):
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[6 + p_idx, r, f] = 1.0

    # Plane 12: Side-to-move (constant 1.0)
    planes[12, :, :] = 1.0

    # Planes 13-16: Castling rights
    if view_board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if view_board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if view_board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if view_board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # Plane 17: En-passant target square
    if view_board.ep_square is not None:
        r, f = chess.square_rank(view_board.ep_square), chess.square_file(view_board.ep_square)
        planes[17, r, f] = 1.0

    return planes


def move_to_index(move: chess.Move, perspective: chess.Color = chess.WHITE) -> int:
    """Map a chess.Move to a discrete action index in [0, 4095].
    
    Formula: from_sq * 64 + to_sq.
    If perspective is BLACK, the move squares are mirrored vertically.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    if perspective == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    assert 0 <= from_sq < 64 and 0 <= to_sq < 64
    return from_sq * 64 + to_sq


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Map a discrete action index in [0, 4095] back to a chess.Move.
    
    If board.turn is BLACK, square coordinates are un-mirrored.
    Promotions automatically resolve to Queen.
    """
    from_sq = idx // 64
    to_sq = idx % 64
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    piece = board.piece_at(from_sq)
    promotion = None
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask of shape (4096,) indicating legal moves for board.turn."""
    mask = np.zeros(4096, dtype=bool)
    for move in board.legal_moves:
        idx = move_to_index(move, perspective=board.turn)
        mask[idx] = True
    return mask

