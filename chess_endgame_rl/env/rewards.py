from __future__ import annotations

import chess

from chess_endgame_rl.env.scenarios import get_scenario


class PotentialShaping:
    """Potential-based reward shaping: F(s, s') = gamma * Phi(s') - Phi(s).
    
    Guarantees policy invariance (Ng, Harada, Russell 1999).
    With solver ground truth:
        Phi(s) = c * (1 - DTM(s) / DTM_max) for won states, 0 otherwise.
        Phi(terminal) = 0.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        c: float = 1.0,
        dtm_max: int = 32,
        enabled: bool = True,
    ):
        self.gamma = gamma
        self.c = c
        self.dtm_max = dtm_max
        self.enabled = enabled

    def phi(self, dtm: int, is_terminal: bool = False) -> float:
        """Potential function Phi(s) with boundary condition Phi(terminal) = 0."""
        if not self.enabled or is_terminal or dtm < 0:
            return 0.0
        return self.c * (1.0 - min(dtm, self.dtm_max) / float(self.dtm_max))

    def shaping_reward(
        self, dtm_s: int, dtm_next: int, is_terminal: bool = False
    ) -> float:
        """Compute F(s, s') = gamma * Phi(s') - Phi(s)."""
        if not self.enabled:
            return 0.0
        phi_s = self.phi(dtm_s, is_terminal=False)
        phi_next = self.phi(dtm_next, is_terminal=is_terminal)
        return self.gamma * phi_next - phi_s


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


def step_reward(
    scenario_id: str, board_before: chess.Board, board_after: chess.Board
) -> float:
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
