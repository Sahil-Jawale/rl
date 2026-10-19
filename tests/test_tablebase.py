from __future__ import annotations

from pathlib import Path
import chess
import numpy as np
import pytest

from chess_endgame_rl.solver.tablebase import (
    SOLVER_VERSION,
    Tablebase,
    CACHE_DIR,
    tablebase_cache_path,
)
from chess_endgame_rl.env.defenders import defender_for, OptimalDefender


def test_tablebase_kqk_exists_and_properties():
    """KQvK coverage, WDL ratio, and max DTM match the spec's sanity targets."""
    path = CACHE_DIR / "kqk.npz"
    assert path.exists(), f"Expected cached KQvK tablebase at {path}"
    tb = Tablebase.load(path)
    assert tb.piece_type == chess.QUEEN
    assert tb.dtm.max() == 20, f"Expected KQvK max DTM = 20, got {tb.dtm.max()}"
    assert abs(int(tb.legal.sum()) - 368_000) < 500
    solved_fraction = (tb.dtm[tb.legal] >= 0).mean()
    assert abs(float(solved_fraction) - 0.937) <= 0.005


def test_tablebase_krk_exists_and_properties():
    """KRvK validation: Max DTM 32 plies (16 moves), ~175k won White states."""
    path = CACHE_DIR / "krk.npz"
    assert path.exists(), f"Expected cached KRvK tablebase at {path}"
    tb = Tablebase.load(path)
    assert tb.piece_type == chess.ROOK
    assert tb.dtm.max() == 32, f"Expected KRvK max DTM = 32, got {tb.dtm.max()}"
    assert (tb.dtm[0::2] > 0).sum() > 100_000


def test_tablebase_rejects_positions_outside_three_piece_scope():
    tb = Tablebase.load(CACHE_DIR / "krk.npz")
    board = chess.Board("4k3/8/8/8/8/8/8/3RK2R w - - 0 1")
    assert tb.lookup(board) == (0, -1)


def test_tablebase_cache_key_includes_solver_version(tmp_path):
    assert tablebase_cache_path(chess.QUEEN).name == f"kqk-v{SOLVER_VERSION}.npz"
    stale_cache = tmp_path / "stale.npz"
    np.savez_compressed(stale_cache, solver_version=SOLVER_VERSION + 1)
    with pytest.raises(ValueError, match="uses solver version"):
        Tablebase.load(stale_cache)


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


def test_optimal_defence_delays_mate_by_one_ply():
    tablebase = Tablebase.load(CACHE_DIR / "kqk.npz")
    board = chess.Board("8/8/8/8/8/8/8/KQk5 b - - 0 1")
    _, dtm = tablebase.lookup(board)
    assert dtm > 0
    for move in tablebase.optimal_moves(board):
        board.push(move)
        _, next_dtm = tablebase.lookup(board)
        board.pop()
        assert next_dtm == dtm - 1


def test_optimal_defender_factory():
    """Verify optimal defender selection agrees with both exact tablebases."""
    defender = defender_for("optimal", seed=42)
    assert isinstance(defender, OptimalDefender)
    for tablebase_name, fen in (
        ("krk.npz", "8/8/8/8/8/8/8/KRk5 b - - 0 1"),
        ("kqk.npz", "8/8/8/8/8/8/8/KQk5 b - - 0 1"),
    ):
        board = chess.Board(fen)
        tablebase = Tablebase.load(CACHE_DIR / tablebase_name)
        assert board.legal_moves.count() > 0
        assert defender.choose_move(board) in tablebase.optimal_moves(board)
