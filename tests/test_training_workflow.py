from __future__ import annotations

import numpy as np

from chess_endgame_rl.agents import PPOAgent
from chess_endgame_rl.eval.suite import evaluate_agent
from chess_endgame_rl.train import _classify_outcome, train_ppo, save_training_results


def test_terminal_outcomes_are_not_reported_as_ongoing():
    assert _classify_outcome({"reason": "piece_captured"}) == "piece_captured"
    assert _classify_outcome({"reason": "stalemate_after_black"}) == "stalemate"
    assert _classify_outcome({"reason": "checkmate_against_white"}) == "loss"
    assert _classify_outcome({"reason": "move_limit"}) == "timeout"


def test_ppo_training_and_evaluation_smoke(tmp_path):
    model_path = tmp_path / "ppo.npz"
    result = train_ppo(episodes=1, seed=7, save_path=str(model_path), verbose=False)
    assert model_path.exists()
    assert result["metrics"]["total_episodes"] == 1
    assert result["scenario"] == "krk"

    agent = PPOAgent.load(model_path)
    metrics = evaluate_agent(agent, "ppo", episodes=2, seed=123)
    assert metrics["episodes"] == 2
    assert sum(metrics["outcomes"].values()) == 2

    result_path = save_training_results(result, tmp_path)
    assert result_path.exists()


def test_ppo_entropy_update_remains_finite_with_illegal_actions_masked():
    agent = PPOAgent(input_dim=2, output_dim=3, hidden1=4, hidden2=4)
    mask = np.array([1, 0, 1], dtype=np.int8)
    agent.update(
        states=[np.array([1.0, 0.0], dtype=np.float32)],
        actions=[0],
        rewards=[1.0],
        dones=[True],
        log_probs=[float(np.log(0.5))],
        legal_masks=[mask],
    )
    assert all(np.isfinite(weight).all() for weight in (
        agent.w1, agent.w2, agent.w_policy, agent.w_value
    ))
