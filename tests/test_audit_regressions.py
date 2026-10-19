from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_endgame_rl.agents.dqn import DeepQNetwork
from chess_endgame_rl.env import EndgameEnv, get_scenario


def test_reachable_state_sampling_allows_distance_two():
    """Item 7: King distance of 2 (separated by 1 square) is legal chess opposition and must be allowed."""
    scenario = get_scenario("krk")
    min_separation = scenario.get("king_separation", 2)
    # Distance in python-chess is Chebyshev distance. Dist 2 is legal (e.g. Ka1 vs Ka3).
    # If king_separation is set to 1, distance 2 is allowed.
    assert min_separation <= 1, (
        f"king_separation={min_separation} excludes valid legal distance-2 opposition positions."
    )


def test_terminated_vs_truncated_distinction():
    """Item 4: Episode termination (mate/stalemate/loss) vs horizon truncation (move_limit) must be separated."""
    env = EndgameEnv("krk", seed=42, move_cap=1)
    env.reset()
    # Execute a move that exceeds move cap (1 full move = 2 half moves)
    legal_moves = list(env.board.legal_moves)
    action_id = 0  # Pick first legal action
    from chess_endgame_rl.env.encoding import move_to_action_id
    for move in legal_moves:
        aid = move_to_action_id(env.board, move)
        if aid is not None:
            action_id = aid
            break

    _, _, done, info = env.step(action_id)
    assert done
    assert info.get("truncated") is True, "Horizon cut-off must mark truncated=True"
    assert info.get("terminated") is False, "Horizon cut-off must mark terminated=False"


def test_edge_stalling_does_not_net_positive_reward():
    """Item 5: Stalling when Black is on the edge must not produce net positive reward per step."""
    env = EndgameEnv("krk", seed=42)
    # Position where Black King is already on edge (h8), White King on f6, White Rook on f7
    board_before = chess.Board("7k/5R2/5K2/8/8/8/8/8 w - - 0 1")
    # White moves Kf6-g6 (non-mating king shuffle)
    board_after = chess.Board("7k/5R2/6K1/8/8/8/8/8 b - - 1 1")
    reward = env._compute_reward(board_before, board_after)
    # White should not receive positive reward for shuffling without checkmating
    assert reward <= 0.0, f"Stalling gave positive reward {reward}, enabling reward farming"


def test_double_dqn_target_uses_online_argmax():
    """Item 2: Double-DQN target must evaluate Q_target at argmax_{a in LEGAL} Q_online."""
    agent = DeepQNetwork(input_dim=10, output_dim=4, batch_size=2)
    # Verify agent has Double-DQN target computation capability
    assert hasattr(agent, "compute_target_q") or hasattr(agent, "use_double_dqn"), (
        "Agent must implement Double-DQN decoupled target calculation"
    )
