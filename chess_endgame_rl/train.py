"""Training pipelines for all three RL agents.

This module provides functions to train each RL agent (Q-Learning, DQN, PPO)
on configured chess endgame environments and save both the trained models and
training metrics for comparative analysis.

Training procedure:
1. Initialize the environment and agent
2. Run episodes: agent plays White (K+R), environment controls Black (K)
3. Track per-episode rewards and checkmate/stalemate/timeout outcomes
4. Save trained model weights and training history to disk
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from chess_endgame_rl.agents import DeepQNetwork, PPOAgent, TabularQAgent
from chess_endgame_rl.env import EndgameEnv
from chess_endgame_rl.env.encoding import board_to_observation, state_index


def _classify_outcome(info: dict) -> str:
    """Classify the outcome of an episode."""
    reason = info.get("reason", "")
    if reason == "checkmate":
        return "checkmate"
    if reason == "checkmate_against_white":
        return "loss"
    if "stalemate" in reason:
        return "stalemate"
    if reason == "move_limit":
        return "timeout"
    if reason == "illegal_move":
        return "illegal"
    if reason in {"rook_captured", "piece_captured"}:
        return "piece_captured"
    if reason in {"insufficient_material", "draw"}:
        return "draw"
    if reason == "game_over":
        return "draw"
    return "unfinished"


def _env_state_index(env: EndgameEnv) -> int:
    if env.board is None:
        raise RuntimeError("Environment state is unavailable before reset.")
    return state_index(env.board)


def _summarize_training(rewards: list[float], outcomes: list[str], move_counts: list[int]) -> dict:
    """Compute training summary statistics."""
    total = len(rewards)
    if total == 0:
        return {
            "mean_reward": 0.0, "std_reward": 0.0,
            "min_reward": 0.0, "max_reward": 0.0,
            "checkmate_rate": 0.0, "stalemate_rate": 0.0,
            "timeout_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0,
            "unfinished_rate": 0.0, "avg_moves": 0.0,
        }

    checkmates = sum(1 for o in outcomes if o == "checkmate")
    stalemates = sum(1 for o in outcomes if o == "stalemate")
    timeouts = sum(1 for o in outcomes if o == "timeout")
    losses = sum(1 for o in outcomes if o in {"loss", "piece_captured", "illegal"})
    draws = sum(1 for o in outcomes if o in {"draw", "stalemate"})
    rewards_arr = np.array(rewards)

    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "checkmate_rate": checkmates / total,
        "stalemate_rate": stalemates / total,
        "timeout_rate": timeouts / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "unfinished_rate": sum(1 for o in outcomes if o == "unfinished") / total,
        "avg_moves": float(np.mean(move_counts)) if move_counts else 0.0,
        "total_episodes": total,
        "checkmates": checkmates,
        "stalemates": stalemates,
        "timeouts": timeouts,
    }


# ======================================================================
# Q-Learning Training
# ======================================================================

def train_q_learning(
    episodes: int = 5000,
    seed: int = 42,
    alpha: float = 0.15,
    gamma: float = 0.99,
    epsilon_decay: float = 0.9995,
    defender: str = "random",
    scenario_id: str = "krk",
    save_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train a Tabular Q-Learning agent on KRK endgame.
    
    Args:
        episodes: Number of training episodes.
        seed: Random seed for reproducibility.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon_decay: Epsilon decay rate per episode.
        defender: Defender policy ('random', 'heuristic', 'smart').
        save_path: Path to save the trained model.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with training results and metrics.
    """
    np.random.seed(seed)
    env = EndgameEnv(scenario_id=scenario_id, defender=defender, seed=seed)
    agent = TabularQAgent(
        state_count=262144,
        action_count=env.action_count,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    outcomes_history = []
    move_counts_history = []
    epsilon_history = []
    start_time = time.time()

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        state = _env_state_index(env)
        legal_mask = env.legal_action_mask()
        episode_reward = 0.0
        info = {}

        for _ in range(100):  # Max steps per episode
            action = agent.select_action(state, legal_mask)
            _, reward, done, info = env.step(action)
            next_state = _env_state_index(env)
            next_mask = env.legal_action_mask() if not done else np.zeros(env.action_count, dtype=np.int8)

            agent.update(state, action, reward, next_state, done, next_mask)
            episode_reward += reward
            state = next_state
            legal_mask = next_mask

            if done:
                break

        rewards_history.append(episode_reward)
        outcomes_history.append(_classify_outcome(info))
        move_counts_history.append(env.move_count)
        epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()

        if verbose and (episode + 1) % 500 == 0:
            recent = rewards_history[-500:]
            recent_mates = sum(1 for o in outcomes_history[-500:] if o == "checkmate")
            elapsed = time.time() - start_time
            print(
                f"  Q-Learning | Episode {episode+1}/{episodes} | "
                f"Avg Reward: {np.mean(recent):.3f} | "
                f"Checkmate Rate: {recent_mates/500:.1%} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    # Save model
    if save_path is None:
        save_path = "models/q_learning.json"
    agent.save(save_path)

    metrics = _summarize_training(rewards_history, outcomes_history, move_counts_history)
    elapsed = time.time() - start_time

    return {
        "algorithm": "Q-Learning",
        "scenario": scenario_id,
        "defender": defender,
        "episodes": episodes,
        "training_time": elapsed,
        "metrics": metrics,
        "rewards_history": rewards_history,
        "outcomes_history": outcomes_history,
        "move_counts_history": move_counts_history,
        "epsilon_history": epsilon_history,
        "model_path": str(save_path),
    }


# ======================================================================
# DQN Training
# ======================================================================

def train_dqn(
    episodes: int = 3000,
    seed: int = 42,
    learning_rate: float = 5e-4,
    gamma: float = 0.99,
    epsilon_decay: float = 0.9995,
    defender: str = "random",
    scenario_id: str = "krk",
    save_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train a Deep Q-Network agent on KRK endgame.
    
    Args:
        episodes: Number of training episodes.
        seed: Random seed for reproducibility.
        learning_rate: Neural network learning rate.
        gamma: Discount factor.
        epsilon_decay: Epsilon decay rate per episode.
        defender: Defender policy.
        save_path: Path to save the trained model.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with training results and metrics.
    """
    np.random.seed(seed)
    env = EndgameEnv(scenario_id=scenario_id, defender=defender, seed=seed)
    agent = DeepQNetwork(
        input_dim=193,
        output_dim=env.action_count,
        hidden1=256,
        hidden2=128,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    outcomes_history = []
    move_counts_history = []
    epsilon_history = []
    start_time = time.time()

    for episode in range(episodes):
        obs = env.reset(seed=seed + episode)
        legal_mask = env.legal_action_mask()
        episode_reward = 0.0
        info = {}

        for _ in range(100):
            action = agent.predict(obs, legal_mask)
            next_obs, reward, done, info = env.step(action)
            next_mask = env.legal_action_mask() if not done else np.zeros(env.action_count, dtype=np.int8)

            agent.update(obs, action, reward, next_obs, done, next_mask)
            episode_reward += reward
            obs = next_obs
            legal_mask = next_mask

            if done:
                break

        rewards_history.append(episode_reward)
        outcomes_history.append(_classify_outcome(info))
        move_counts_history.append(env.move_count)
        epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()

        if verbose and (episode + 1) % 500 == 0:
            recent = rewards_history[-500:]
            recent_mates = sum(1 for o in outcomes_history[-500:] if o == "checkmate")
            elapsed = time.time() - start_time
            print(
                f"  DQN        | Episode {episode+1}/{episodes} | "
                f"Avg Reward: {np.mean(recent):.3f} | "
                f"Checkmate Rate: {recent_mates/500:.1%} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    if save_path is None:
        save_path = "models/dqn.npz"
    agent.save(save_path)

    metrics = _summarize_training(rewards_history, outcomes_history, move_counts_history)
    elapsed = time.time() - start_time

    return {
        "algorithm": "DQN",
        "scenario": scenario_id,
        "defender": defender,
        "episodes": episodes,
        "training_time": elapsed,
        "metrics": metrics,
        "rewards_history": rewards_history,
        "outcomes_history": outcomes_history,
        "move_counts_history": move_counts_history,
        "epsilon_history": epsilon_history,
        "model_path": str(save_path),
    }


# ======================================================================
# PPO Training
# ======================================================================

def train_ppo(
    episodes: int = 3000,
    seed: int = 42,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    defender: str = "random",
    scenario_id: str = "krk",
    save_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train a PPO agent on KRK endgame.
    
    Args:
        episodes: Number of training episodes.
        seed: Random seed for reproducibility.
        learning_rate: Neural network learning rate.
        gamma: Discount factor.
        defender: Defender policy.
        save_path: Path to save the trained model.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with training results and metrics.
    """
    np.random.seed(seed)
    env = EndgameEnv(scenario_id=scenario_id, defender=defender, seed=seed)
    agent = PPOAgent(
        input_dim=193,
        output_dim=env.action_count,
        hidden1=256,
        hidden2=128,
        learning_rate=learning_rate,
        gamma=gamma,
    )

    rewards_history = []
    outcomes_history = []
    move_counts_history = []
    start_time = time.time()

    trajectory_size = 10
    batch_trajectories: list[list] = [[] for _ in range(6)]

    for episode in range(episodes):
        obs = env.reset(seed=seed + episode)
        legal_mask = env.legal_action_mask()
        episode_reward = 0.0
        info = {}

        # Collect trajectory for this episode
        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_dones = []
        traj_log_probs = []
        traj_masks = []

        for _ in range(100):
            action, log_prob = agent.select_action(obs, legal_mask)
            next_obs, reward, done, info = env.step(action)
            next_mask = env.legal_action_mask() if not done else np.zeros(env.action_count, dtype=np.int8)

            traj_states.append(obs.copy())
            traj_actions.append(action)
            traj_rewards.append(reward)
            traj_dones.append(done)
            traj_log_probs.append(log_prob)
            traj_masks.append(legal_mask.copy())

            episode_reward += reward
            obs = next_obs
            legal_mask = next_mask

            if done:
                break

        if traj_states:
            episode_trajectory = (
                traj_states, traj_actions, traj_rewards,
                traj_dones, traj_log_probs, traj_masks,
            )
            for batch, episode_data in zip(batch_trajectories, episode_trajectory):
                batch.extend(episode_data)
            if (episode + 1) % trajectory_size == 0 or episode + 1 == episodes:
                agent.update(*batch_trajectories)
                batch_trajectories = [[] for _ in range(6)]

        rewards_history.append(episode_reward)
        outcomes_history.append(_classify_outcome(info))
        move_counts_history.append(env.move_count)

        if verbose and (episode + 1) % 500 == 0:
            recent = rewards_history[-500:]
            recent_mates = sum(1 for o in outcomes_history[-500:] if o == "checkmate")
            elapsed = time.time() - start_time
            print(
                f"  PPO        | Episode {episode+1}/{episodes} | "
                f"Avg Reward: {np.mean(recent):.3f} | "
                f"Checkmate Rate: {recent_mates/500:.1%} | "
                f"Time: {elapsed:.1f}s"
            )

    if save_path is None:
        save_path = "models/ppo.npz"
    agent.save(save_path)

    metrics = _summarize_training(rewards_history, outcomes_history, move_counts_history)
    elapsed = time.time() - start_time

    return {
        "algorithm": "PPO",
        "scenario": scenario_id,
        "defender": defender,
        "episodes": episodes,
        "training_time": elapsed,
        "metrics": metrics,
        "rewards_history": rewards_history,
        "outcomes_history": outcomes_history,
        "move_counts_history": move_counts_history,
        "model_path": str(save_path),
    }


def save_training_results(results: dict, output_dir: str | Path = "results"):
    """Save training results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    algo_name = results["algorithm"].lower().replace("-", "_").replace(" ", "_")
    scenario = results.get("scenario", "krk")
    filename = f"training_{algo_name}.json" if scenario == "krk" else f"training_{scenario}_{algo_name}.json"
    path = output_path / filename

    # Save a serializable version (exclude numpy arrays)
    serializable = {
        "algorithm": results["algorithm"],
        "scenario": results.get("scenario", "krk"),
        "defender": results.get("defender", "random"),
        "episodes": results["episodes"],
        "training_time": results["training_time"],
        "metrics": results["metrics"],
        "rewards_history": [float(r) for r in results["rewards_history"]],
        "outcomes_history": results["outcomes_history"],
        "move_counts_history": [int(m) for m in results["move_counts_history"]],
        "model_path": results.get("model_path", ""),
    }
    if "epsilon_history" in results:
        serializable["epsilon_history"] = [float(e) for e in results["epsilon_history"]]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path
