from __future__ import annotations

import random
import chess
import numpy as np
import pytest

from chess_endgame_rl.env.encoding import (
    encode_board,
    index_to_move,
    legal_mask,
    move_to_index,
)


def test_encode_board_shape_and_dtype():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    tensor = encode_board(board)
    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == np.float32
    # White King on e1 (rank 0, file 4) -> Plane 5 (King)
    assert tensor[5, 0, 4] == 1.0
    # White Rook on h1 (rank 0, file 7) -> Plane 3 (Rook)
    assert tensor[3, 0, 7] == 1.0
    # Black King on e8 (rank 7, file 4) -> Plane 11 (Opponent King)
    assert tensor[11, 7, 4] == 1.0


def test_color_flip_invariance():
    """A position and its colour-mirrored twin must yield identical tensors and mirrored action indices."""
    board_white = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    # Mirrored position: Black King on e8 becomes White King on e1, White pieces become Black
    board_black = board_white.mirror()
    assert board_black.turn == chess.BLACK

    tensor_w = encode_board(board_white)
    tensor_b = encode_board(board_black)
    np.testing.assert_array_equal(
        tensor_w,
        tensor_b,
        err_msg="Color-mirrored positions must yield identical 18x8x8 tensors from side-to-move perspective.",
    )

    # Move Rh1-h8 for White should match mirrored move Rh8-h1 for Black
    move_w = chess.Move.from_uci("h1h8")
    move_b = chess.Move.from_uci("h8h1")
    idx_w = move_to_index(move_w, perspective=chess.WHITE)
    idx_b = move_to_index(move_b, perspective=chess.BLACK)
    assert idx_w == idx_b, "Mirrored moves from side-to-move perspective must have identical action indices."


def test_legal_mask_count_equals_board_legal_moves():
    """Legal mask true count must exactly equal the number of legal moves."""
    fens = [
        "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
        "4k3/8/8/8/8/8/8/4K2R b - - 0 1",
        "8/8/8/8/4k3/8/8/R3K3 w Q - 0 1",
        "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        mask = legal_mask(board)
        assert mask.shape == (4096,)
        assert mask.dtype == bool
        assert mask.sum() == board.legal_moves.count(), (
            f"Mask count {mask.sum()} != legal moves {board.legal_moves.count()} for {fen}"
        )


def test_action_roundtrip_1000_random_positions():
    """Round-trip every legal move for random legal positions: index_to_move(move_to_index(m)) == m."""
    rng = random.Random(42)
    sample_size = 1000
    for _ in range(sample_size):
        # Generate random 3-piece position
        board = chess.Board(None)
        turn = rng.choice([chess.WHITE, chess.BLACK])
        board.turn = turn
        squares = rng.sample(range(64), 3)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(squares[2], chess.Piece(rng.choice([chess.ROOK, chess.QUEEN]), chess.WHITE))

        if not board.is_valid() or board.is_checkmate() or board.is_stalemate():
            continue

        for move in board.legal_moves:
            idx = move_to_index(move, perspective=board.turn)
            assert 0 <= idx < 4096
            reconstructed = index_to_move(idx, board)
            assert reconstructed == move, (
                f"Round-trip failed: original={move}, reconstructed={reconstructed}, turn={board.turn}"
            )
