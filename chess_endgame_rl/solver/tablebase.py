"""Exact endgame solver using retrograde analysis.

Computes Distance-To-Mate (DTM in plies) and WDL status for every legal
3-piece endgame position (KQvK, KRvK).

Index formula:
    (((wK * 64) + bK) * 64 + piece_sq) * 2 + stm
Total states: 64 * 64 * 64 * 2 = 524,288.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chess
import numpy as np


CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "tablebases"


def state_to_id(wk: int, bk: int, piece_sq: int, stm: int) -> int:
    """Map square coordinates and side-to-move (0=White, 1=Black) to state index."""
    return (((wk * 64) + bk) * 64 + piece_sq) * 2 + stm


def id_to_state(idx: int) -> tuple[int, int, int, int]:
    """Map state index back to (wk, bk, piece_sq, stm)."""
    stm = idx % 2
    idx //= 2
    piece_sq = idx % 64
    idx //= 64
    bk = idx % 64
    wk = idx // 64
    return wk, bk, piece_sq, stm


def board_to_state_id(board: chess.Board, piece_type: int) -> int | None:
    """Map a chess.Board with 3 pieces (wK, bK, wPiece) to its state index."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    pieces = list(board.pieces(piece_type, chess.WHITE))
    if wk is None or bk is None or not pieces:
        return None
    piece_sq = pieces[0]
    stm = 0 if board.turn == chess.WHITE else 1
    return state_to_id(wk, bk, piece_sq, stm)


class Tablebase:
    """Retrograde analysis tablebase for 3-piece chess endgames."""

    def __init__(self, piece_type: int, dtm: np.ndarray, legal: np.ndarray):
        self.piece_type = piece_type
        self.dtm = dtm  # Shape (524288,), int16. -1 = Draw/unsolved, >= 0 = DTM in plies
        self.legal = legal  # Shape (524288,), bool

    @classmethod
    def solve(cls, piece_name: str = "Q") -> Tablebase:
        """Run retrograde analysis to solve the endgame from scratch."""
        pt = chess.QUEEN if piece_name.upper() in ("Q", "KQVK") else chess.ROOK
        total_states = 524288

        dtm = np.full(total_states, -1, dtype=np.int16)
        legal = np.zeros(total_states, dtype=bool)

        board = chess.Board(None)
        checkmates: list[int] = []

        # Graph representations
        # For White states: successors (Black states)
        white_succ: list[list[int]] = [[] for _ in range(262144)]
        # For Black states: legal out-degree and predecessors (from White states)
        black_degree = np.zeros(262144, dtype=np.int16)
        black_preds: list[list[int]] = [[] for _ in range(262144)]
        white_preds: list[list[int]] = [[] for _ in range(262144)]

        # Step 1: Enumerate all legal states and transitions
        for wk in range(64):
            for bk in range(64):
                if chess.square_distance(wk, bk) <= 1:
                    continue
                for piece_sq in range(64):
                    if piece_sq == wk or piece_sq == bk:
                        continue
                    s = ((wk * 64) + bk) * 64 + piece_sq
                    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                    board.set_piece_at(piece_sq, chess.Piece(pt, chess.WHITE))
                    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

                    # White to move (stm=0)
                    board.turn = chess.WHITE
                    if board.is_valid():
                        legal[s * 2] = True
                        for m in board.legal_moves:
                            board.push(m)
                            nwk = board.king(chess.WHITE)
                            nbk = board.king(chess.BLACK)
                            npieces = board.pieces(pt, chess.WHITE)
                            board.pop()
                            if nwk is not None and nbk is not None and npieces:
                                ns = ((nwk * 64) + nbk) * 64 + npieces.pop()
                                white_succ[s].append(ns)
                                white_preds[ns].append(s)

                    # Black to move (stm=1)
                    board.turn = chess.BLACK
                    if board.is_valid():
                        legal[s * 2 + 1] = True
                        if board.is_checkmate():
                            checkmates.append(s)
                            dtm[s * 2 + 1] = 0
                        else:
                            has_capture = False
                            valid_succs: list[int] = []
                            for m in board.legal_moves:
                                if board.is_capture(m):
                                    has_capture = True
                                    break
                                board.push(m)
                                nwk = board.king(chess.WHITE)
                                nbk = board.king(chess.BLACK)
                                npieces = board.pieces(pt, chess.WHITE)
                                board.pop()
                                if nwk is not None and nbk is not None and npieces:
                                    ns = ((nwk * 64) + nbk) * 64 + npieces.pop()
                                    valid_succs.append(ns)

                            if has_capture:
                                # Capturing White's lone piece yields bare kings -> draw
                                black_degree[s] = -9999
                            elif valid_succs:
                                black_degree[s] = len(valid_succs)
                                for ns in valid_succs:
                                    black_preds[ns].append(s)

                    board.clear_board()

        # Step 2: Retrograde BFS propagation
        black_queue = checkmates
        ply = 0

        while black_queue:
            # White moves into solved Black states: won in (d + 1)
            next_white: list[int] = []
            for sb in black_queue:
                d = dtm[sb * 2 + 1]
                for sw in white_preds[sb]:
                    if dtm[sw * 2] == -1:
                        dtm[sw * 2] = d + 1
                        next_white.append(sw)

            if not next_white:
                break

            # Black moves: lost in (d + 1) if all legal moves lead to won White states
            next_black: list[int] = []
            for sw in next_white:
                dw = dtm[sw * 2]
                for sb in black_preds[sw]:
                    if dtm[sb * 2 + 1] == -1 and black_degree[sb] > 0:
                        black_degree[sb] -= 1
                        if black_degree[sb] == 0:
                            # Re-verify max DTM among successors
                            dtm[sb * 2 + 1] = dw + 1
                            next_black.append(sb)

            black_queue = next_black
            ply += 1

        return cls(pt, dtm, legal)

    def save(self, path: Path | str) -> None:
        """Save tablebase arrays to a compressed npz file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(target),
            dtm=self.dtm,
            legal=self.legal,
            piece_type=int(self.piece_type),
        )

    @classmethod
    def load(cls, path: Path | str) -> Tablebase:
        """Load tablebase arrays from a compressed npz file."""
        data = np.load(str(path))
        return cls(
            piece_type=int(data["piece_type"]),
            dtm=data["dtm"],
            legal=data["legal"],
        )

    def lookup(self, board: chess.Board) -> tuple[int, int]:
        """Look up WDL status and DTM in plies for the given position.
        
        Returns:
            (wdl, dtm):
                wdl: +1 if side to move is winning, -1 if losing, 0 if draw
                dtm: distance to mate in plies (>= 0), or -1 if draw
        """
        sid = board_to_state_id(board, self.piece_type)
        if sid is None or not self.legal[sid]:
            return 0, -1

        dtm_val = int(self.dtm[sid])
        if dtm_val < 0:
            return 0, -1

        if board.turn == chess.WHITE:
            return +1, dtm_val
        else:
            return -1, dtm_val

    def optimal_moves(self, board: chess.Board) -> set[chess.Move]:
        """Return the set of moves that strictly minimize DTM for the winning side
        or maximize DTM (longest resistance) for the losing side.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return set()

        wdl, current_dtm = self.lookup(board)

        # If winning (White): optimal moves decrease DTM by 1
        if wdl == 1:
            best_moves: set[chess.Move] = set()
            min_dtm = 9999
            for m in legal_moves:
                board.push(m)
                _, next_dtm = self.lookup(board)
                board.pop()
                if next_dtm >= 0:
                    if next_dtm < min_dtm:
                        min_dtm = next_dtm
                        best_moves = {m}
                    elif next_dtm == min_dtm:
                        best_moves.add(m)
            return best_moves

        # If losing (Black): optimal moves maximize resistance
        if wdl == -1:
            best_moves = set()
            max_dtm = -1
            for m in legal_moves:
                if board.is_capture(m):
                    # Capture wins immediate draw!
                    return {m}
                board.push(m)
                _, next_dtm = self.lookup(board)
                board.pop()
                if next_dtm > max_dtm:
                    max_dtm = next_dtm
                    best_moves = {m}
                elif next_dtm == max_dtm:
                    best_moves.add(m)
            return best_moves

        # If draw: any move that does not blunder into a loss
        safe_moves = set()
        for m in legal_moves:
            board.push(m)
            next_wdl, _ = self.lookup(board)
            board.pop()
            if next_wdl == 0:
                safe_moves.add(m)
        return safe_moves or set(legal_moves)
