"""Chess Endgame RL Environment — King+Rook vs King.

This environment models the KRK (King-Rook vs King) chess endgame where
the RL agent controls the White pieces (King + Rook) and must checkmate
the Black lone King within 50 moves.

State, Action, Reward, Environment formulation:
- **State**: 192-dim one-hot vector (3 planes × 64 squares) encoding positions
  of White King, White Rook, and Black King, plus a scalar for move count.
- **Action**: 36 discrete actions (8 king moves + 28 rook moves along ranks/files).
- **Reward**: Dense shaped reward combining checkmate bonus, stalemate penalty,
  distance-to-edge reward, king-proximity reward, and per-step cost.
- **Environment**: Turn-based; agent makes a White move, then a defender policy
  responds with a Black move. Episode ends on checkmate, stalemate, or 50 moves.
"""

from __future__ import annotations

import random
from typing import Any

import chess
import numpy as np

from .defenders import defender_for
from .encoding import (
    action_count_for_scenario,
    action_to_move,
    board_to_observation,
    legal_action_mask,
    move_to_action_id,
)
from .scenarios import get_scenario, scenario_piece_types


class EndgameEnv:
    """Gym-like environment for KRK chess endgame training.
    
    The RL agent plays White (King + Rook) and must deliver checkmate
    against a Black lone King within `move_cap` half-moves.
    
    Attributes:
        scenario_id: Identifier for the endgame scenario (always 'krk').
        action_count: Number of discrete actions available.
        move_count: Current half-move count in the episode.
        done: Whether the current episode has ended.
        board: The python-chess Board representing the current position.
    """

    def __init__(
        self,
        scenario_id: str = "krk",
        defender: str = "random",
        seed: int | None = None,
        move_cap: int | None = None,
    ):
        self.scenario_id = (scenario_id or "krk").lower()
        self.scenario = get_scenario(self.scenario_id)
        self.action_count = int(
            self.scenario.get("action_count", action_count_for_scenario(self.scenario_id))
        )
        self.defender_name = (defender or "random").lower()
        self.move_cap = move_cap or int(self.scenario.get("move_cap", 50))
        self.rng = random.Random(seed)
        self.board: chess.Board | None = None
        self.done = False
        self.move_count = 0
        self._defender = defender_for(
            self.defender_name, seed=self.rng.randint(0, 10_000_000)
        )
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self.rng.seed(seed)
        self.board = self._generate_random_position()
        self.done = False
        self.move_count = 0
        return self._observation()

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one full turn (White move + Black reply).
        
        Returns:
            observation: The new board state as a feature vector.
            reward: Scalar reward for this transition.
            done: Whether the episode has ended.
            info: Dictionary with additional metadata.
        """
        if self.board is None or self.done:
            raise ValueError("Environment is done or not initialized. Call reset().")

        # --- White (agent) move ---
        legal_moves = list(self.board.legal_moves)
        chosen_move = action_to_move(self.board, action_id)
        if chosen_move is None or chosen_move not in legal_moves:
            # Invalid action → large penalty and episode ends
            self.done = True
            return self._observation(), -10.0, True, {"reason": "illegal_move"}

        board_before = self.board.copy()
        self.board.push(chosen_move)
        self.move_count += 1

        # Calculate reward for this White move
        reward = self._compute_reward(board_before, self.board)

        info: dict[str, Any] = {
            "white_move": chosen_move.uci(),
            "move_count": self.move_count,
        }

        # Check if game is over after White's move
        if self.board.is_game_over():
            self.done = True
            if self.board.is_checkmate():
                info["reason"] = "checkmate"
            elif self.board.is_stalemate():
                info["reason"] = "stalemate"
            else:
                info["reason"] = "game_over"
            return self._observation(), reward, True, info

        # --- Black (defender) move ---
        defender_move = self._defender.choose_move(self.board)
        self.board.push(defender_move)
        self.move_count += 1
        info["black_move"] = defender_move.uci()

        # Check if game is over after Black's move
        if self.board.is_game_over():
            self.done = True
            if self.board.is_stalemate():
                info["reason"] = "stalemate_after_black"
                reward += -5.0  # Stalemate is very bad for White
            return self._observation(), reward, True, info

        # Check 50-move rule: if agent exceeded the move cap, episode ends as a loss
        if self.move_count >= self.move_cap * 2:  # move_cap is in full moves
            self.done = True
            reward += -3.0  # Penalty for failing to mate in time
            info["reason"] = "move_limit"
            return self._observation(), reward, True, info

        return self._observation(), reward, False, info

    def legal_action_mask(self) -> np.ndarray:
        """Return a binary mask of legal actions."""
        if self.board is None:
            raise ValueError("Board not initialized")
        return legal_action_mask(self.board, self.scenario_id)

    def render(self) -> str:
        """Return the FEN string of the current position."""
        if self.board is None:
            return ""
        return self.board.fen()

    def get_observation_dim(self) -> int:
        """Return the dimensionality of the observation vector."""
        return 193  # 3×64 piece planes + 1 normalized move count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        """Build the observation vector for the current board state."""
        if self.board is None:
            return np.zeros(193, dtype=np.float32)
        obs = board_to_observation(self.board, self.scenario_id)
        # Append normalized move count as an extra feature
        move_frac = np.array(
            [self.move_count / max(self.move_cap * 2, 1)], dtype=np.float32
        )
        return np.concatenate([obs.astype(np.float32), move_frac])

    def _white_piece_types(self) -> list[int]:
        return scenario_piece_types(self.scenario_id, chess.WHITE)

    def _black_piece_types(self) -> list[int]:
        return scenario_piece_types(self.scenario_id, chess.BLACK)

    def _generate_random_position(self) -> chess.Board:
        """Generate a random legal KRK starting position."""
        white_piece_types = self._white_piece_types()
        black_piece_types = self._black_piece_types()
        if not white_piece_types or not black_piece_types:
            raise ValueError(
                f"Scenario '{self.scenario_id}' does not define valid piece assignments"
            )

        for _ in range(50_000):
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

            # Ensure kings are far enough apart
            king_distance = chess.square_distance(
                board.king(chess.WHITE),
                board.king(chess.BLACK),
            )
            min_distance = self.scenario.get("king_separation", 2)
            if min_distance is not None and king_distance <= int(min_distance):
                continue

            # Make sure the rook is not immediately capturable
            rook_square = None
            for sq, piece in board.piece_map().items():
                if piece.color == chess.WHITE and piece.piece_type == chess.ROOK:
                    rook_square = sq
                    break
            if rook_square is not None and board.is_attacked_by(chess.BLACK, rook_square):
                continue

            return board

        raise ValueError(
            f"Unable to generate a legal {self.scenario_id} start position"
        )

    def _compute_reward(
        self, board_before: chess.Board, board_after: chess.Board
    ) -> float:
        """Compute a dense shaped reward for a White move.
        
        Reward components:
        1. Terminal checkmate: +10.0 (large bonus, scaled by speed)
        2. Terminal stalemate: -5.0 (severe penalty — stalemate is a draw)
        3. Step cost: -0.02 per move (encourages faster checkmate)
        4. Check bonus: +0.3 for giving check
        5. Distance shaping: reward for pushing Black King toward edges/corners
        6. King proximity: reward for moving White King closer to support mate
        """
        reward_cfg = self.scenario.get("reward", {})
        reward = float(reward_cfg.get("step_cost", -0.02))

        if board_after.is_checkmate():
            # Bonus scales with speed: faster checkmate = higher reward
            base = float(reward_cfg.get("terminal_checkmate", 10.0))
            speed_bonus = max(0, (self.move_cap * 2 - self.move_count)) / (
                self.move_cap * 2
            )
            reward += base + base * speed_bonus * 0.5
            return reward

        if board_after.is_stalemate():
            reward += float(reward_cfg.get("terminal_stalemate", -5.0))
            return reward

        # Check if rook was captured (should not happen in KRK with good play)
        white_rook_exists = False
        for sq, piece in board_after.piece_map().items():
            if piece.color == chess.WHITE and piece.piece_type == chess.ROOK:
                white_rook_exists = True
                break
        if not white_rook_exists:
            reward += float(reward_cfg.get("rook_captured", -8.0))
            return reward

        # Check bonus
        if board_after.is_check():
            reward += float(reward_cfg.get("check_reward", 0.3))

        # --- Reward shaping: push Black King to edges ---
        bk_before = board_before.king(chess.BLACK)
        bk_after = board_after.king(chess.BLACK)
        wk_before = board_before.king(chess.WHITE)
        wk_after = board_after.king(chess.WHITE)

        if bk_before is not None and bk_after is not None:
            # Center distance: how far is Black King from center (higher = better for White)
            def center_distance(sq: int) -> float:
                r, f = chess.square_rank(sq), chess.square_file(sq)
                return max(abs(r - 3.5), abs(f - 3.5))

            cd_before = center_distance(bk_before)
            cd_after = center_distance(bk_after)
            # Reward for pushing Black King further from center
            reward += (cd_after - cd_before) * 0.15

            # Edge bonus: reward for Black King being on edge/corner
            def edge_score(sq: int) -> float:
                r, f = chess.square_rank(sq), chess.square_file(sq)
                edge_dist = min(r, 7 - r, f, 7 - f)
                if edge_dist == 0:
                    return 0.1
                return 0.0

            reward += edge_score(bk_after)

        # King proximity: White King should approach Black King for mating
        if wk_before is not None and wk_after is not None and bk_after is not None:
            dist_before = chess.square_distance(wk_before, bk_after)
            dist_after = chess.square_distance(wk_after, bk_after)
            reward += (dist_before - dist_after) * 0.05

        return reward
