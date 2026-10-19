from __future__ import annotations

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from chess_endgame_rl.env.encoding import encode_board, move_to_index
from chess_endgame_rl.solver.tablebase import Tablebase, id_to_state

ACTION_DIM = 4096


class TablebaseDataset(Dataset):
    """Lazy state/action labels for attacker-to-move positions won by the solver."""

    def __init__(
        self,
        tablebase: Tablebase,
        state_ids: np.ndarray | None = None,
        gamma: float = 0.99,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in the interval (0, 1].")
        self.tablebase = tablebase
        self.gamma = gamma

        if state_ids is None:
            white_legal = tablebase.legal[0::2]
            white_won = tablebase.dtm[0::2] > 0
            state_ids = np.flatnonzero(white_legal & white_won) * 2
        self.state_ids = np.asarray(state_ids, dtype=np.int64)

        if self.state_ids.ndim != 1 or self.state_ids.size == 0:
            raise ValueError("The dataset requires at least one attacker-to-move won state.")
        if np.any(self.state_ids < 0) or np.any(self.state_ids >= tablebase.dtm.size):
            raise ValueError("state_ids contains an out-of-range tablebase index.")
        if np.any(self.state_ids % 2 != 0):
            raise ValueError("All dataset states must have White (the attacker) to move.")
        if not np.all(tablebase.legal[self.state_ids] & (tablebase.dtm[self.state_ids] > 0)):
            raise ValueError("All dataset states must be legal and won by the attacker.")

    def __len__(self) -> int:
        return int(self.state_ids.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        state_id = int(self.state_ids[index])
        white_king, black_king, attacker_piece, _ = id_to_state(state_id)

        board = chess.Board(None)
        board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(attacker_piece, chess.Piece(self.tablebase.piece_type, chess.WHITE))
        board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        moves = list(board.legal_moves)
        legal = np.zeros(ACTION_DIM, dtype=np.bool_)
        optimal = np.zeros(ACTION_DIM, dtype=np.bool_)
        q_targets = np.zeros(ACTION_DIM, dtype=np.float32)
        optimal_moves = self.tablebase.optimal_moves(board)
        if not optimal_moves:
            raise RuntimeError(f"No optimal move found for solved state {state_id}.")

        for move in moves:
            action = move_to_index(move, perspective=board.turn)
            legal[action] = True
            if move in optimal_moves:
                optimal[action] = True

            board.push(move)
            next_wdl, next_dtm = self.tablebase.lookup(board)
            board.pop()
            if next_dtm >= 0:
                q_targets[action] = -float(next_wdl) * self.gamma ** (next_dtm + 1)

        if not np.any(legal) or not np.any(optimal):
            raise RuntimeError(f"Invalid legal/optimal move labels for solved state {state_id}.")

        target_distribution = optimal.astype(np.float32)
        target_distribution /= target_distribution.sum()
        return {
            "state": torch.from_numpy(encode_board(board)),
            "legal_mask": torch.from_numpy(legal),
            "optimal_mask": torch.from_numpy(optimal),
            "target_distribution": torch.from_numpy(target_distribution),
            "q_targets": torch.from_numpy(q_targets),
            "dtm": torch.tensor(int(self.tablebase.dtm[state_id]), dtype=torch.int16),
        }


def split_dataset(
    dataset: TablebaseDataset,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Split distinct positions deterministically into train and validation subsets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two positions are required for a train/validation split.")

    order = np.random.default_rng(seed).permutation(len(dataset))
    validation_size = max(1, int(round(len(dataset) * validation_fraction)))
    validation_size = min(validation_size, len(dataset) - 1)
    validation_indices = order[:validation_size].tolist()
    train_indices = order[validation_size:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)
