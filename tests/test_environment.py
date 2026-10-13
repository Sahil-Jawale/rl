from __future__ import annotations

import chess

from chess_endgame_rl.env import EndgameEnv, get_scenario, list_scenarios
from chess_endgame_rl.env.encoding import action_to_move, legal_action_mask, move_to_action_id
from chess_endgame_rl.eval.suite import summarize_game_results, tablebase_status


def test_scenarios_are_registered():
    assert set(get_scenario("krk").keys()) >= {"move_cap", "action_count"}
    assert set(get_scenario("knk").keys()) >= {"move_cap", "action_count"}
    assert list_scenarios() == ["krk", "knk"]


def test_environment_generates_valid_board():
    env = EndgameEnv("krk", seed=10)
    assert env.board is not None
    assert env.board.is_valid()
    assert env.board.turn == chess.WHITE
    assert len(list(env.board.legal_moves)) > 0
    assert env.action_count == get_scenario("krk")["action_count"]


def test_action_round_trip_for_king_and_rook():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    move = chess.Move.from_uci("h1h8")
    action_id = move_to_action_id(board, move)
    assert action_id is not None
    assert action_to_move(board, action_id) == move


def test_legal_mask_has_expected_length():
    env = EndgameEnv("knk", seed=1)
    mask = env.legal_action_mask()
    assert mask.shape == (16,)
    assert int(mask.sum()) > 0


def test_summary_metrics_and_tablebase_flag_are_consistent():
    metrics = summarize_game_results([1.0, 0.0, -1.0])
    assert metrics["count"] == 3
    assert metrics["mean"] == 0.0
    assert metrics["positive_rate"] == 1 / 3
    assert metrics["negative_rate"] == 1 / 3
    status = tablebase_status()
    assert isinstance(status["available"], bool)
    assert "message" in status
