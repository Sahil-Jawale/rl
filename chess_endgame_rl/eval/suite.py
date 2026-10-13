from __future__ import annotations

import math
import os
from statistics import median
from typing import Iterable

import chess

from chess_endgame_rl.env import EndgameEnv


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
