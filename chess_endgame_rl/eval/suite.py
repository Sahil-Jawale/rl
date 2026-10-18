from __future__ import annotations

import math
import os
from statistics import median
from typing import Iterable

import chess

from chess_endgame_rl.agents import DeepQNetwork, PPOAgent, TabularQAgent
from chess_endgame_rl.env import EndgameEnv
from chess_endgame_rl.env.encoding import state_index


def generate_test_positions(scenario_id: str, count: int = 50, seed: int = 42):
    """Generate a fixed, reproducible test suite for a single scenario."""
    env = EndgameEnv(scenario_id=scenario_id, defender="random", seed=seed)
    positions: list[str] = []
    for index in range(count):
        env.reset(seed=seed + index)
        positions.append(env.render())
    return positions


def summarize_game_results(results: Iterable[float]):
    values = list(results)
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
            "positive_rate": 0.0,
            "zero_rate": 0.0,
            "negative_rate": 0.0,
        }

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    positive = sum(1 for value in values if value > 0)
    zero = sum(1 for value in values if value == 0)
    negative = sum(1 for value in values if value < 0)

    return {
        "count": len(values),
        "mean": mean_value,
        "std": math.sqrt(variance),
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "positive_rate": positive / len(values),
        "zero_rate": zero / len(values),
        "negative_rate": negative / len(values),
    }


def evaluate_agent(
    agent: TabularQAgent | DeepQNetwork | PPOAgent,
    algorithm: str,
    scenario_id: str = "krk",
    episodes: int = 50,
    seed: int = 42,
    defender: str = "random",
) -> dict:
    """Evaluate a trained policy greedily on reproducible, independent games."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")

    outcomes: list[str] = []
    rewards: list[float] = []
    move_counts: list[int] = []
    normalized_algorithm = algorithm.lower().replace("_", "-")
    if normalized_algorithm == "qlearning":
        normalized_algorithm = "q-learning"

    for episode in range(episodes):
        env = EndgameEnv(scenario_id=scenario_id, defender=defender, seed=seed + episode)
        observation = env.reset(seed=seed + episode)
        total_reward = 0.0
        info: dict = {}

        for _ in range(env.move_cap * 2):
            legal_mask = env.legal_action_mask()
            board = env.board
            if board is None:
                raise RuntimeError("Environment reset did not initialize a board.")
            if isinstance(agent, TabularQAgent):
                if normalized_algorithm != "q-learning":
                    raise ValueError(f"Algorithm '{algorithm}' does not match the loaded agent.")
                action = agent.select_action_greedy(state_index(board), legal_mask)
            elif isinstance(agent, DeepQNetwork):
                if normalized_algorithm != "dqn":
                    raise ValueError(f"Algorithm '{algorithm}' does not match the loaded agent.")
                action = agent.predict_greedy(observation, legal_mask)
            elif isinstance(agent, PPOAgent):
                if normalized_algorithm != "ppo":
                    raise ValueError(f"Algorithm '{algorithm}' does not match the loaded agent.")
                action = agent.select_action_greedy(observation, legal_mask)
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}'")

            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        reason = info.get("reason")
        if reason == "checkmate":
            outcome = "checkmate"
        elif reason == "checkmate_against_white":
            outcome = "loss"
        elif reason in {"stalemate", "stalemate_after_black"}:
            outcome = "stalemate"
        elif reason == "move_limit":
            outcome = "timeout"
        elif reason in {"rook_captured", "piece_captured"}:
            outcome = "piece_captured"
        elif reason in {"draw", "insufficient_material", "game_over"}:
            outcome = "draw"
        elif reason == "illegal_move":
            outcome = "illegal"
        else:
            outcome = "unfinished"

        outcomes.append(outcome)
        rewards.append(total_reward)
        move_counts.append(env.move_count)

    counts = {name: outcomes.count(name) for name in sorted(set(outcomes))}
    return {
        "episodes": episodes,
        "defender": defender,
        "outcomes": counts,
        "checkmates": counts.get("checkmate", 0),
        "checkmate_rate": counts.get("checkmate", 0) / episodes,
        "draw_rate": sum(counts.get(name, 0) for name in ("draw", "stalemate")) / episodes,
        "timeout_rate": counts.get("timeout", 0) / episodes,
        "loss_rate": sum(counts.get(name, 0) for name in ("loss", "piece_captured", "illegal")) / episodes,
        "unfinished_rate": counts.get("unfinished", 0) / episodes,
        "mean_reward": sum(rewards) / episodes,
        "avg_moves": sum(move_counts) / episodes,
    }


def summarize_seed_results(seed_values: Iterable[float]) -> dict:
    values = [float(value) for value in seed_values]
    if not values:
        return {
            "count": 0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "ci95": [0.0, 0.0],
            "seed_values": [],
        }

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(variance)
    margin = 1.96 * std_value / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean_reward": mean_value,
        "std_reward": std_value,
        "ci95": [mean_value - margin, mean_value + margin],
        "seed_values": values,
    }


def summarize_seed_benchmarks(results_by_seed: Iterable[dict]) -> dict:
    """Aggregate a list of seed results into a per-algorithm benchmark summary."""
    rows = list(results_by_seed)
    if not rows:
        return {"count": 0, "mean_reward": 0.0, "ci95": [0.0, 0.0], "seed_values": []}

    seed_values = [float(row.get("mean_reward", 0.0)) for row in rows]
    summary = summarize_seed_results(seed_values)
    summary["rows"] = rows
    return summary


def _tablebase_dirs() -> list[str]:
    candidates: list[str] = []
    for env_name in ("PYTHON_CHESS_TABLEBASES", "CHESS_TABLEBASE_DIR", "TB_PATH", "TABLEBASE_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.extend(part for part in value.split(os.pathsep) if part)
    return sorted({path for path in candidates if path})


def tablebase_status() -> dict:
    """Report whether the optional Syzygy support is enabled in the environment."""
    try:
        import chess.syzygy  # type: ignore

        directories = _tablebase_dirs()
        return {
            "available": True,
            "module": chess.syzygy.__name__,
            "directories": directories,
            "message": "Syzygy support is available if tables are installed locally.",
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {
            "available": False,
            "module": None,
            "directories": [],
            "message": f"Syzygy tablebases are not available in this environment: {exc}",
        }


def probe_tablebase_position(board: chess.Board, directories: Iterable[str] | None = None) -> dict:
    """Return WDL/DTZ metadata for a board if an installed Syzygy tablebase can resolve it."""
    if board is None or not board.is_valid():
        return {"available": False, "reason": "Board is not a valid tablebase position."}

    try:
        import chess.syzygy  # type: ignore

        tb = chess.syzygy.Tablebase()
        for directory in directories or _tablebase_dirs():
            try:
                tb.add_directory(str(directory))
            except Exception:
                continue

        try:
            wdl = tb.probe_wdl(board)
            dtz = tb.probe_dtz(board)
            return {
                "available": True,
                "wdl": wdl,
                "dtz": dtz,
                "message": "Tablebase lookup succeeded.",
            }
        except chess.syzygy.MissingTableError as exc:
            return {
                "available": False,
                "reason": f"No table for this position: {exc}",
            }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {
            "available": False,
            "reason": f"Syzygy tablebase support is unavailable: {exc}",
        }
