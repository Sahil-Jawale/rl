from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_endgame_rl.agents.dueling_ddqn import DuelingDoubleDQN
from chess_endgame_rl.env.defenders import defender_for
from chess_endgame_rl.env.encoding import encode_board, index_to_move, legal_mask
from chess_endgame_rl.env.rewards import PotentialShaping
from chess_endgame_rl.eval.evaluate_all import board_from_state, evaluate_move_quality
from chess_endgame_rl.seed import set_seed
from chess_endgame_rl.solver.tablebase import Tablebase
from chess_endgame_rl.warmstart.dataset import TablebaseDataset, split_dataset


def _terminal_reward(board: chess.Board) -> float:
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def train_finetune(
    tablebase: Tablebase,
    warmstart_checkpoint: Path,
    output_checkpoint: Path,
    *,
    episodes: int = 1_000,
    defender_name: str = "optimal",
    max_plies: int = 100,
    gamma: float = 0.99,
    learning_rate: float = 1e-4,
    epsilon: float = 0.05,
    anchor_weight: float = 0.1,
    shaping_enabled: bool = True,
    batch_size: int = 64,
    eval_every_steps: int = 1_000,
    eval_positions: int = 512,
    early_stop_patience: int = 5,
    seed: int = 0,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if (
        episodes < 1
        or max_plies < 2
        or max_plies % 2 != 0
        or eval_every_steps < 1
        or early_stop_patience < 1
    ):
        raise ValueError("Episodes, horizon, evaluation interval, and patience must be positive.")

    started_at = time.perf_counter()
    set_seed(seed)
    device = torch.device(device_name)
    agent = DuelingDoubleDQN(
        gamma=gamma,
        learning_rate=learning_rate,
        epsilon=epsilon,
        batch_size=batch_size,
        anchor_weight=anchor_weight,
        seed=seed,
        device=device_name,
    )
    agent.load_checkpoint(warmstart_checkpoint)
    potential_shaper = PotentialShaping(
        gamma=gamma,
        dtm_max=32 if tablebase.piece_type == chess.ROOK else 20,
        enabled=shaping_enabled,
    )

    dataset = TablebaseDataset(tablebase, gamma=gamma)
    train_data, validation_data = split_dataset(dataset, seed=seed)
    anchor_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    anchor_iterator = iter(anchor_loader)
    validation_ids = dataset.state_ids[np.asarray(validation_data.indices, dtype=np.int64)]
    validation_size = min(eval_positions, validation_ids.size)
    rng = np.random.default_rng(seed)
    validation_ids = rng.choice(validation_ids, size=validation_size, replace=False)

    start_ids = (
        np.flatnonzero(tablebase.legal[0::2] & (tablebase.dtm[0::2] > 0)) * 2
    ).tolist()
    defender = defender_for(defender_name, seed=seed)
    game_rng = random.Random(seed)
    best_optimal_rate = -1.0
    best_checkpoint_metrics: dict[str, float | int] | None = None
    history: list[dict[str, Any]] = []
    consecutive_degradation = 0
    previous_suboptimal_rate: float | None = None
    environment_steps = 0

    for episode in range(episodes):
        start_id = int(game_rng.choice(start_ids))
        board = board_from_state(tablebase, start_id)
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            state = encode_board(board)
            current_mask = legal_mask(board)
            action = agent.select_action(state, current_mask)
            move = index_to_move(action, board)
            wdl_before, dtm_before = tablebase.lookup(board)
            board.push(move)
            plies += 1
            terminated = board.is_game_over(claim_draw=True)

            if not terminated:
                defender_move = defender.choose_move(board)
                board.push(defender_move)
                plies += 1
                terminated = board.is_game_over(claim_draw=True)

            truncated = not terminated and plies >= max_plies
            reward = _terminal_reward(board) if terminated else -0.001
            wdl_next, dtm_next = tablebase.lookup(board)
            reward += potential_shaper.shaping_reward(
                int(dtm_before),
                int(dtm_next),
                is_terminal=terminated,
                wdl_s=wdl_before,
                wdl_next=wdl_next,
            )
            next_state = encode_board(board)
            next_mask = (
                np.zeros(4096, dtype=np.bool_) if terminated else legal_mask(board)
            )
            agent.remember(
                state,
                action,
                reward,
                current_mask,
                next_state,
                terminated,
                next_mask,
            )

            try:
                anchor_batch = next(anchor_iterator)
            except StopIteration:
                anchor_iterator = iter(anchor_loader)
                anchor_batch = next(anchor_iterator)
            agent.train_step(anchor_batch)
            environment_steps += 1

            if environment_steps % eval_every_steps == 0:
                metrics = evaluate_move_quality(
                    agent.online,
                    tablebase,
                    sample_size=validation_size,
                    seed=seed,
                    device=device,
                    state_ids=validation_ids,
                )
                record = {
                    "episode": episode + 1,
                    "environment_steps": environment_steps,
                    "potential_shaping": shaping_enabled,
                    **metrics,
                }
                history.append(record)
                suboptimal_rate = float(metrics["suboptimal_move_rate"])
                if (
                    previous_suboptimal_rate is not None
                    and suboptimal_rate > previous_suboptimal_rate
                ):
                    consecutive_degradation += 1
                else:
                    consecutive_degradation = 0
                previous_suboptimal_rate = suboptimal_rate

                if float(metrics["optimal_move_rate"]) > best_optimal_rate:
                    best_optimal_rate = float(metrics["optimal_move_rate"])
                    best_checkpoint_metrics = metrics
                    agent.save_checkpoint(
                        output_checkpoint,
                        seed=seed,
                        environment_steps=environment_steps,
                        evaluation_metrics=metrics,
                        defender=defender_name,
                        validation_state_ids=torch.as_tensor(
                            validation_ids, dtype=torch.int64
                        ),
                    )
                if consecutive_degradation >= early_stop_patience:
                    break
            if terminated or truncated:
                break
        if consecutive_degradation >= early_stop_patience:
            break

    if best_checkpoint_metrics is None:
        metrics = evaluate_move_quality(
            agent.online,
            tablebase,
            sample_size=validation_size,
            seed=seed,
            device=device,
            state_ids=validation_ids,
        )
        best_checkpoint_metrics = metrics
        agent.save_checkpoint(
            output_checkpoint,
            seed=seed,
            environment_steps=environment_steps,
            evaluation_metrics=metrics,
            defender=defender_name,
            validation_state_ids=torch.as_tensor(validation_ids, dtype=torch.int64),
        )

    result = {
        "output_checkpoint": str(output_checkpoint),
        "warmstart_checkpoint": str(warmstart_checkpoint),
        "seed": seed,
        "episodes_completed": episode + 1,
        "environment_steps": environment_steps,
        "potential_shaping": shaping_enabled,
        "config": {
            "episodes": episodes,
            "defender": defender_name,
            "max_plies": max_plies,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "anchor_weight": anchor_weight,
            "batch_size": batch_size,
            "target_update_frequency": agent.target_update_frequency,
            "replay_capacity": agent.replay_capacity,
            "eval_every_steps": eval_every_steps,
            "eval_positions": eval_positions,
            "early_stop_patience": early_stop_patience,
            "validation_fraction": 0.1,
            "seed": seed,
            "device": str(device),
        },
        "runtime_seconds": time.perf_counter() - started_at,
        "hardware": platform.platform(),
        "torch_version": torch.__version__,
        "best_validation": best_checkpoint_metrics,
        "history": history,
    }
    output_checkpoint.with_suffix(".metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
