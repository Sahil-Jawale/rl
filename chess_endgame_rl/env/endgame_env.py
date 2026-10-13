from __future__ import annotations

import random

import chess
import numpy as np

from .defenders import defender_for
from .encoding import action_count_for_scenario, action_to_move, legal_action_mask, state_index
from .scenarios import get_scenario, scenario_piece_types


class EndgameEnv:
    def __init__(self, scenario_id: str = "krk", defender: str = "random", seed: int | None = None):
        self.scenario_id = (scenario_id or "krk").lower()
        self.scenario = get_scenario(self.scenario_id)
        self.action_count = int(self.scenario.get("action_count", action_count_for_scenario(self.scenario_id)))
        self.defender_name = (defender or "random").lower()
        self.rng = random.Random(seed)
        self.board: chess.Board | None = None
        self.done = False
        self.move_count = 0
        self._defender = defender_for(self.defender_name, seed=self.rng.randint(0, 10_000_000))
        self.reset()

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.board = self._generate_random_position()
        self.done = False
        self.move_count = 0
        return self._state_observation()

    def _white_piece_types(self) -> list[int]:
        return scenario_piece_types(self.scenario_id, chess.WHITE)

    def _black_piece_types(self) -> list[int]:
        return scenario_piece_types(self.scenario_id, chess.BLACK)

    def _generate_random_position(self) -> chess.Board:
        white_piece_types = self._white_piece_types()
        black_piece_types = self._black_piece_types()
        if not white_piece_types or not black_piece_types:
            raise ValueError(f"Scenario '{self.scenario_id}' does not define valid piece assignments")

        for _ in range(20_000):
            board = chess.Board(None)
            board.turn = chess.WHITE
            board.clear_board()
            squares = list(range(64))
            self.rng.shuffle(squares)

            piece_slots = []
            for piece_type in white_piece_types:
                piece_slots.append((chess.WHITE, piece_type))
            for piece_type in black_piece_types:
                piece_slots.append((chess.BLACK, piece_type))

            for color, piece_type in piece_slots:
                if not squares:
                    break
                square = squares.pop()
                board.set_piece_at(square, chess.Piece(piece_type, color))

            if not board.is_valid() or board.is_checkmate() or board.is_stalemate():
                continue
            if len(list(board.legal_moves)) == 0:
                continue

            king_distance = chess.square_distance(
                board.king(chess.WHITE),
                board.king(chess.BLACK),
            )
            min_distance = self.scenario.get("king_separation", 1)
            if min_distance is not None and king_distance <= int(min_distance):
                continue
            return board
        raise ValueError(f"Unable to generate a legal {self.scenario_id} start position")

    def _state_observation(self):
        if self.board is None:
            raise ValueError("Board not initialized")
        return state_index(self.board)

    def legal_action_mask(self) -> np.ndarray:
        if self.board is None:
            raise ValueError("Board not initialized")
        return legal_action_mask(self.board, self.scenario_id)

    def _white_move_reward(self, board_before: chess.Board, board_after: chess.Board) -> float:
        reward_cfg = self.scenario.get("reward", {})
        reward = float(reward_cfg.get("step_cost", 0.0))
        if board_after.is_checkmate():
            reward += float(reward_cfg.get("terminal_checkmate", 1.0))
        elif board_after.is_stalemate():
            reward += float(reward_cfg.get("terminal_stalemate", -0.5))
        elif board_after.is_insufficient_material() and "insufficient_material" in reward_cfg:
            reward += float(reward_cfg.get("insufficient_material", 0.0))
        return reward

    def step(self, action_id: int):
        if self.board is None or self.done:
            raise ValueError("Environment is done or not initialized")

        legal_moves = list(self.board.legal_moves)
        chosen_move = action_to_move(self.board, action_id)
        if chosen_move is None or chosen_move not in legal_moves:
            raise ValueError(f"Action {action_id} is not legal in the current position")

        board_before = self.board.copy()
        self.board.push(chosen_move)
        self.move_count += 1
        reward = self._white_move_reward(board_before, self.board)
        done = self.board.is_game_over()

        if not done:
            defender_move = self._defender.choose_move(self.board)
            self.board.push(defender_move)
            self.move_count += 1
            if self.board.is_game_over():
                done = True
                reward_cfg = self.scenario.get("reward", {})
                if self.board.is_stalemate() and "defender_stalemate_penalty" in reward_cfg:
                    reward += float(reward_cfg.get("defender_stalemate_penalty", -0.5))
                elif self.board.is_checkmate() and "defender_checkmate_penalty" in reward_cfg:
                    reward += float(reward_cfg.get("defender_checkmate_penalty", -1.0))

        self.done = done
        return self._state_observation(), reward, done, {"legal_moves": legal_moves}

    def render(self):
        if self.board is None:
            return ""
        return self.board.fen()
