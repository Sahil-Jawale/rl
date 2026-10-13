from __future__ import annotations

import json
from pathlib import Path

from chess_endgame_rl.agents import DeepQNetwork, PPOAgent, TabularQAgent
from chess_endgame_rl.env import EndgameEnv
from chess_endgame_rl.env.encoding import board_to_observation
from chess_endgame_rl.eval.suite import summarize_game_results


def _summarize_training(rewards: list[float]) -> dict:
    metrics = summarize_game_results(rewards)
    return {
        "mean_reward": metrics["mean"],
        "std_reward": metrics["std"],
        "positive_rate": metrics["positive_rate"],
        "negative_rate": metrics["negative_rate"],
        "zero_rate": metrics["zero_rate"],
        "min_reward": metrics["min"],
        "max_reward": metrics["max"],
        "median_reward": metrics["median"],
    }


def run_tabular_q_learning(scenario_id: str = "krk", episodes: int = 250, seed: int = 0):
    env = EndgameEnv(scenario_id=scenario_id, defender="random", seed=seed)
    state_count = 64 * 64 * 64
    action_count = env.action_count
    agent = TabularQAgent(state_count=state_count, action_count=action_count, epsilon=1.0)

    rewards: list[float] = []
    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        legal_mask = env.legal_action_mask()
        episode_reward = 0.0

        for _ in range(100):
            action = agent.select_action(state, legal_mask)
            next_state, reward, done, _ = env.step(action)
            next_mask = env.legal_action_mask()
            agent.update(state, action, reward, next_state, done, legal_mask)
            episode_reward += reward
            state = next_state
            legal_mask = next_mask
            if done:
                break

        rewards.append(episode_reward)
        agent.decay_epsilon(decay_rate=0.995)

    metrics = _summarize_training(rewards)
    return {
        "scenario": scenario_id,
        "algorithm": "q-learning",
        "episodes": episodes,
        "mean_reward": metrics["mean_reward"],
        "rewards": rewards,
        "metrics": metrics,
    }


def run_dqn_training(scenario_id: str = "krk", episodes: int = 60, seed: int = 0):
    env = EndgameEnv(scenario_id=scenario_id, defender="random", seed=seed)
    action_count = env.action_count
    agent = DeepQNetwork(input_dim=192, output_dim=action_count, hidden_dim=64)
    rewards: list[float] = []

    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        obs = board_to_observation(env.board, scenario_id)
        legal_mask = env.legal_action_mask()
        episode_reward = 0.0
        for _ in range(80):
            action = agent.predict(obs, legal_mask)
            next_state, reward, done, _ = env.step(action)
            next_obs = board_to_observation(env.board, scenario_id)
            next_mask = env.legal_action_mask()
            agent.update(obs, action, reward, next_obs, done, legal_mask)
            episode_reward += reward
            obs = next_obs
            state = next_state
            legal_mask = next_mask
            if done:
                break
        rewards.append(episode_reward)
        agent.decay_epsilon()

    metrics = _summarize_training(rewards)
    return {
        "scenario": scenario_id,
        "algorithm": "dqn",
        "episodes": episodes,
        "mean_reward": metrics["mean_reward"],
        "rewards": rewards,
        "metrics": metrics,
    }


def run_ppo_training(scenario_id: str = "krk", episodes: int = 60, seed: int = 0):
    env = EndgameEnv(scenario_id=scenario_id, defender="random", seed=seed)
    action_count = env.action_count
    agent = PPOAgent(input_dim=192, output_dim=action_count, hidden_dim=64)
    rewards: list[float] = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        obs = board_to_observation(env.board, scenario_id)
        legal_mask = env.legal_action_mask()
        episode_reward = 0.0
        history = []
        for _ in range(80):
            action = agent.select_action(obs, legal_mask)
            next_state, reward, done, info = env.step(action)
            next_obs = board_to_observation(env.board, scenario_id)
            next_mask = env.legal_action_mask()
            history.append((obs.copy(), action, reward, done, legal_mask.copy()))
            obs = next_obs
            legal_mask = next_mask
            episode_reward += reward
            if done:
                break

        if history:
            states, actions, rewards_history, dones, legal_history = zip(*history)
            agent.update(list(states), list(actions), list(rewards_history), list(dones), list(legal_history))
        rewards.append(episode_reward)

    metrics = _summarize_training(rewards)
    return {
        "scenario": scenario_id,
        "algorithm": "ppo",
        "episodes": episodes,
        "mean_reward": metrics["mean_reward"],
        "rewards": rewards,
        "metrics": metrics,
    }


def save_result_summary(summary: dict, output_dir: str | Path = "results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{summary['scenario']}_{summary['algorithm']}_summary.json"
    path = output_path / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return path
