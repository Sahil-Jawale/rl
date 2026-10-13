from __future__ import annotations

import chess

from chess_endgame_rl.env.scenarios import get_scenario


def terminal_reward(scenario_id: str, board: chess.Board) -> float:
    scenario = get_scenario((scenario_id or "").lower().strip())
    reward_cfg = scenario.get("reward", {})

    if board.is_checkmate():
        return float(reward_cfg.get("terminal_checkmate", 1.0))
    if board.is_stalemate():
        return float(reward_cfg.get("terminal_stalemate", -0.5))
    if board.is_insufficient_material():
        return float(reward_cfg.get("insufficient_material", 0.0))
    return 0.0


def step_reward(scenario_id: str, board_before: chess.Board, board_after: chess.Board) -> float:
    scenario = get_scenario((scenario_id or "").lower().strip())
    reward_cfg = scenario.get("reward", {})
    reward = float(reward_cfg.get("step_cost", 0.0))

    if board_after.is_checkmate():
        reward += float(reward_cfg.get("terminal_checkmate", 1.0))
    elif board_after.is_stalemate():
        reward += float(reward_cfg.get("terminal_stalemate", -0.5))
    elif board_after.is_insufficient_material():
        reward += float(reward_cfg.get("insufficient_material", 0.0))

    return reward
