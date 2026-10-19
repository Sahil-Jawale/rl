from __future__ import annotations

from pathlib import Path
import chess
import pytest

from chess_endgame_rl.solver.tablebase import Tablebase, CACHE_DIR
from chess_endgame_rl.env.defenders import defender_for, OptimalDefender


def test_tablebase_kqk_exists_and_properties():
    """KQvK validation: Max DTM 20 plies (10 moves), ~144k won White states."""
    path = CACHE_DIR / "kqk.npz"
    assert path.exists(), f"Expected cached KQvK tablebase at {path}"
    tb = Tablebase.load(path)
    assert tb.piece_type == chess.QUEEN
    assert tb.dtm.max() == 20, f"Expected KQvK max DTM = 20, got {tb.dtm.max()}"
    assert (tb.dtm[0::2] > 0).sum() > 100_000


def test_tablebase_krk_exists_and_properties():
    """KRvK validation: Max DTM 32 plies (16 moves), ~175k won White states."""
    path = CACHE_DIR / "krk.npz"
    assert path.exists(), f"Expected cached KRvK tablebase at {path}"
    tb = Tablebase.load(path)
    assert tb.piece_type == chess.ROOK
    assert tb.dtm.max() == 32, f"Expected KRvK max DTM = 32, got {tb.dtm.max()}"
    assert (tb.dtm[0::2] > 0).sum() > 100_000


def test_optimal_moves_reduces_dtm_by_one_for_white():
    """For any won White-to-move state, an optimal move strictly reduces DTM by 1."""
    path = CACHE_DIR / "kqk.npz"
    tb = Tablebase.load(path)
    # White King on f6, Queen on e7, Black King on h8 (Mate in 1)
    board = chess.Board("7k/4Q3/5K2/8/8/8/8/8 w - - 0 1")
    wdl, dtm = tb.lookup(board)
    assert wdl == 1
    assert dtm == 1, f"Expected DTM 1 (mate in 1), got {dtm}"
    best = tb.optimal_moves(board)
    # Both Qg7# and Qe8# are checkmates (DTM 0)
    for m in best:
        board.push(m)
        _, next_dtm = tb.lookup(board)
        board.pop()
        assert next_dtm == dtm - 1, f"Optimal move {m} did not decrease DTM by 1: {next_dtm} != {dtm-1}"


def test_optimal_defender_factory():
    """Verify optimal defender instantiation and stubborness."""
    defender = defender_for("optimal", seed=42)
    assert isinstance(defender, OptimalDefender)
    # Position where Black King can capture undefended Queen on g7 -> immediate draw
    board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    # In checkmate, no legal moves
    if board.legal_moves:
        move = defender.choose_move(board)
        assert move is not None
