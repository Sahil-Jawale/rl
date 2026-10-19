from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_endgame_rl.agents.dqn import DeepQNetwork
from chess_endgame_rl.agents.ppo import PPOAgent
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
    agent = DeepQNetwork(
        input_dim=10, output_dim=4, batch_size=2, gamma=0.5, use_double_dqn=True
    )
    online_next = np.array([[10.0, 999.0, 5.0, 1000.0]], dtype=np.float32)
    target_next = np.array([[3.0, 10000.0, 70.0, 12000.0]], dtype=np.float32)
    agent._forward = lambda states, use_target=False: (
        np.empty((1, 1)),
        np.empty((1, 1)),
        target_next if use_target else online_next,
    )

    target_q = agent.compute_target_q(
        current_q=np.zeros((1, 4), dtype=np.float32),
        actions=np.array([0]),
        rewards=np.array([1.0], dtype=np.float32),
        next_states=np.zeros((1, 10), dtype=np.float32),
        terminated=np.array([False]),
        legal_masks=np.array([[1, 0, 1, 0]], dtype=np.int8),
    )
    # Online selects legal action 0; target evaluates action 0, not its own
    # preferred action 2 or either illegal action.
    assert target_q[0, 0] == pytest.approx(2.5)


def test_truncation_bootstraps_but_true_termination_does_not():
    agent = DeepQNetwork(
        input_dim=2, output_dim=2, batch_size=1, gamma=0.9, use_double_dqn=True
    )
    agent._forward = lambda states, use_target=False: (
        np.empty((1, 1)),
        np.empty((1, 1)),
        np.array([[0.0, 4.0]], dtype=np.float32),
    )
    common = {
        "current_q": np.zeros((1, 2), dtype=np.float32),
        "actions": np.array([0]),
        "rewards": np.array([1.0], dtype=np.float32),
        "next_states": np.zeros((1, 2), dtype=np.float32),
        "legal_masks": np.array([[0, 1]], dtype=np.int8),
    }

    truncated_target = agent.compute_target_q(
        **common, terminated=np.array([False])
    )
    terminal_target = agent.compute_target_q(
        **common, terminated=np.array([True])
    )
    assert truncated_target[0, 0] == pytest.approx(4.6)
    assert terminal_target[0, 0] == pytest.approx(1.0)


def test_ppo_bootstraps_at_truncation_boundary():
    agent = PPOAgent(
        input_dim=1,
        output_dim=2,
        hidden1=4,
        hidden2=4,
        n_epochs=1,
        mini_batch_size=1,
    )
    agent.gamma = 0.9
    agent.get_value = lambda state: 2.0 if state[0] == 0.0 else 4.0
    captured_returns = []
    agent._ppo_step = lambda states, actions, old_log_probs, advantages, returns, masks: (
        captured_returns.extend(returns.tolist()) or (0.0, 0.0, 0.0)
    )

    agent.update(
        states=[np.array([0.0], dtype=np.float32)],
        actions=[0],
        rewards=[1.0],
        dones=[True],
        log_probs=[0.0],
        legal_masks=[np.array([1, 1], dtype=np.int8)],
        terminateds=[False],
        next_states=[np.array([1.0], dtype=np.float32)],
    )

    assert captured_returns == pytest.approx([4.6])
