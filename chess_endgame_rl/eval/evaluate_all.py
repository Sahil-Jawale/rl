from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_endgame_rl.env.defenders import defender_for
from chess_endgame_rl.env.encoding import (
    encode_board,
    index_to_move,
    legal_mask,
    move_to_index,
)
from chess_endgame_rl.models.dueling_net import DuelingNet
from chess_endgame_rl.solver.tablebase import (
    Tablebase,
    existing_tablebase_cache_path,
    id_to_state,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "results.json"
README_PATH = ROOT / "README.md"
PROTOCOL_VERSION = 1


def board_from_state(tablebase: Tablebase, state_id: int) -> chess.Board:
    white_king, black_king, attacker_piece, stm = id_to_state(state_id)
    board = chess.Board(None)
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(attacker_piece, chess.Piece(tablebase.piece_type, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    return board


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[DuelingNet, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})
    model = DuelingNet(
        in_channels=int(config.get("in_channels", 18)),
        action_dim=int(config.get("action_dim", 4096)),
        hidden_dim=int(config.get("hidden_dim", 256)),
    ).to(device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} has no model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> DuelingNet:
    model, _ = load_checkpoint(checkpoint_path, device)
    return model


@torch.no_grad()
def choose_model_move(
    model: DuelingNet, board: chess.Board, device: torch.device
) -> chess.Move:
    mask = legal_mask(board)
    if not mask.any():
        raise ValueError("Cannot choose a move in a terminal position.")
    state = torch.as_tensor(
        encode_board(board), dtype=torch.float32, device=device
    ).unsqueeze(0)
    legal = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    q_values, _, _ = model(state, legal)
    action = int(q_values.argmax(dim=1).item())
    from chess_endgame_rl.env.encoding import index_to_move

    move = index_to_move(action, board)
    if move not in board.legal_moves:
        raise RuntimeError(f"Model selected illegal action index {action}.")
    return move


def _outcome(board: chess.Board) -> str:
    if board.is_checkmate():
        return "win" if board.turn == chess.BLACK else "loss"
    return "draw"


def play_game(
    tablebase: Tablebase,
    start_state_id: int,
    defender_name: str,
    *,
    model: DuelingNet | None,
    seed: int,
    device: torch.device,
    max_plies: int = 100,
) -> dict[str, Any]:
    board = board_from_state(tablebase, start_state_id)
    if board.turn != chess.WHITE or board.is_game_over():
        raise ValueError(f"Invalid non-terminal attacker start state: {start_state_id}")
    initial_wdl, initial_dtm = tablebase.lookup(board)
    defender = defender_for(defender_name, seed=seed)
    rng = random.Random(seed)
    plies = 0

    while plies < max_plies and not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            if model is None:
                move = rng.choice(list(board.legal_moves))
            else:
                move = choose_model_move(model, board, device)
        else:
            move = defender.choose_move(board)
        board.push(move)
        plies += 1

    result = _outcome(board) if board.is_game_over(claim_draw=True) else "draw"
    return {
        "outcome": result,
        "plies": plies,
        "initial_dtm": initial_dtm if initial_wdl == 1 else None,
        "truncated": not board.is_game_over(),
    }


def summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(games)
    if total == 0:
        raise ValueError("Cannot summarize an empty evaluation set.")
    wins = sum(game["outcome"] == "win" for game in games)
    draws = sum(game["outcome"] == "draw" for game in games)
    losses = sum(game["outcome"] == "loss" for game in games)
    converted = [game for game in games if game["outcome"] == "win"]
    ratios = [
        game["plies"] / game["initial_dtm"]
        for game in converted
        if game["initial_dtm"] and game["initial_dtm"] > 0
    ]
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_mate_length_plies": (
            float(np.mean([game["plies"] for game in converted])) if converted else None
        ),
        "mean_optimal_dtm_plies": (
            float(np.mean([game["initial_dtm"] for game in converted])) if converted else None
        ),
        "mean_mate_length_to_optimal_dtm": float(np.mean(ratios)) if ratios else None,
    }


@torch.no_grad()
def evaluate_move_quality(
    model: DuelingNet,
    tablebase: Tablebase,
    *,
    sample_size: int = 10_000,
    seed: int = 0,
    device: torch.device,
    state_ids: np.ndarray | None = None,
    batch_size: int = 256,
) -> dict[str, float | int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    winning_ids = np.flatnonzero(
        tablebase.legal[0::2] & (tablebase.dtm[0::2] > 0)
    ) * 2
    if state_ids is not None:
        state_ids = np.asarray(state_ids, dtype=np.int64)
        if (
            state_ids.ndim != 1
            or state_ids.size < sample_size
            or np.any(state_ids % 2 != 0)
            or not np.all(np.isin(state_ids, winning_ids))
        ):
            raise ValueError("Evaluation state_ids must be unique legal attacker-winning positions.")
        if np.unique(state_ids).size != state_ids.size:
            raise ValueError("Evaluation state_ids must not contain duplicates.")
        winning_ids = state_ids
    if sample_size < 1 or sample_size > winning_ids.size:
        raise ValueError(
            f"sample_size must be between 1 and {winning_ids.size} for this tablebase."
        )
    rng = np.random.default_rng(seed)
    selected_ids = rng.choice(winning_ids, size=sample_size, replace=False)

    model.eval()
    selected_actions: list[int] = []
    for start in range(0, sample_size, batch_size):
        batch_ids = selected_ids[start : start + batch_size]
        boards = [board_from_state(tablebase, int(state_id)) for state_id in batch_ids]
        states = torch.as_tensor(
            np.stack([encode_board(board) for board in boards]),
            dtype=torch.float32,
            device=device,
        )
        legal_masks = torch.as_tensor(
            np.stack([legal_mask(board) for board in boards]),
            dtype=torch.bool,
            device=device,
        )
        q_values, _, _ = model(states, legal_masks)
        selected_actions.extend(
            int(action) for action in q_values.argmax(dim=1).cpu().tolist()
        )

    optimal_count = 0
    win_to_draw_count = 0
    for state_id, action in zip(selected_ids, selected_actions, strict=True):
        board = board_from_state(tablebase, int(state_id))
        move = index_to_move(action, board)
        if move not in board.legal_moves:
            raise RuntimeError(f"Model selected illegal action index {action}.")
        action = move_to_index(move, chess.WHITE)
        optimal_actions = {
            move_to_index(candidate, chess.WHITE)
            for candidate in tablebase.optimal_moves(board)
        }
        board.push(move)
        wdl, _ = tablebase.lookup(board)
        if action in optimal_actions:
            optimal_count += 1
        if wdl == 0:
            win_to_draw_count += 1
    return {
        "positions": int(sample_size),
        "optimal_move_rate": optimal_count / sample_size,
        "suboptimal_move_rate": (sample_size - optimal_count) / sample_size,
        "win_to_draw_rate": win_to_draw_count / sample_size,
    }


def evaluate_scenario(
    tablebase: Tablebase,
    *,
    model: DuelingNet,
    rl_model: DuelingNet | None = None,
    games_per_defender: int,
    seed: int,
    device: torch.device,
    quality_positions: int,
    heldout_state_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    eligible = np.flatnonzero(
        tablebase.legal[0::2] & (tablebase.dtm[0::2] > 0)
    ) * 2
    if games_per_defender < 1 or games_per_defender > eligible.size:
        raise ValueError(
            f"games_per_defender must be between 1 and {eligible.size}."
        )
    if heldout_state_ids is not None:
        heldout_state_ids = np.asarray(heldout_state_ids, dtype=np.int64)
        if (
            heldout_state_ids.ndim != 1
            or not np.all(np.isin(heldout_state_ids, eligible))
            or np.unique(heldout_state_ids).size != heldout_state_ids.size
        ):
            raise ValueError("Held-out start states must be unique legal won attacker positions.")
        eligible = heldout_state_ids
    if quality_positions > eligible.size or games_per_defender > eligible.size:
        raise ValueError("Evaluation requests more positions than are available in the held-out set.")
    starts = np.random.default_rng(seed).choice(eligible, size=games_per_defender, replace=False)
    models: dict[str, DuelingNet] = {"model": model}
    if rl_model is not None:
        models["rl_model"] = rl_model
    output: dict[str, Any] = {
        "move_quality": evaluate_move_quality(
            model,
            tablebase,
            sample_size=quality_positions,
            seed=seed,
            device=device,
            state_ids=heldout_state_ids,
        ),
        "defenders": {},
    }
    if rl_model is not None:
        output["rl_move_quality"] = evaluate_move_quality(
            rl_model,
            tablebase,
            sample_size=quality_positions,
            seed=seed,
            device=device,
            state_ids=heldout_state_ids,
        )
    for defender_name in ("random", "optimal"):
        defender_metrics = {}
        for agent_name, agent_model in models.items():
            agent_games = [
                play_game(
                    tablebase,
                    int(state_id),
                    defender_name,
                    model=agent_model,
                    seed=seed + game_idx,
                    device=device,
                )
                for game_idx, state_id in enumerate(starts)
            ]
            defender_metrics[agent_name] = summarize_games(agent_games)
        random_games = [
            play_game(
                tablebase,
                int(state_id),
                defender_name,
                model=None,
                seed=seed + game_idx,
                device=device,
            )
            for game_idx, state_id in enumerate(starts)
        ]
        defender_metrics["random_baseline"] = summarize_games(random_games)
        output["defenders"][defender_name] = defender_metrics
    return output


def results_markdown_table(results: dict[str, Any]) -> str:
    rows = [
        "| Agent | Scenario | Defender | W | D | L | Win rate | Optimal-move % | Win -> draw % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, scenario_data in sorted(results["scenarios"].items()):
        for defender, defender_data in sorted(scenario_data["defenders"].items()):
            for agent_name in ("model", "rl_model", "random_baseline"):
                if agent_name not in defender_data:
                    continue
                metrics = defender_data[agent_name]
                quality_key = "rl_move_quality" if agent_name == "rl_model" else "move_quality"
                optimal_rate = (
                    results["scenarios"][scenario][quality_key]["optimal_move_rate"]
                    if agent_name in {"model", "rl_model"}
                    else None
                )
                draw_rate = metrics["draw_rate"]
                rows.append(
                    "| {} | {} | {} | {} | {} | {} | {:.3f} | {} | {:.3f} |".format(
                        {
                            "model": "Warm start",
                            "rl_model": "Warm start + RL",
                            "random_baseline": "Random baseline",
                        }[agent_name],
                        scenario,
                        defender,
                        metrics["wins"],
                        metrics["draws"],
                        metrics["losses"],
                        metrics["win_rate"],
                        "{:.3f}".format(optimal_rate)
                        if optimal_rate is not None
                        else "N/A",
                        draw_rate,
                    )
                )
    return "\n".join(rows)


def update_readme_table(readme_path: Path, markdown_table: str) -> None:
    start_marker = "<!-- WARMSTART_RESULTS_START -->"
    end_marker = "<!-- WARMSTART_RESULTS_END -->"
    content = readme_path.read_text(encoding="utf-8")
    if start_marker not in content or end_marker not in content:
        raise ValueError("README is missing the warm-start results table markers.")
    start = content.index(start_marker) + len(start_marker)
    end = content.index(end_marker)
    readme_path.write_text(
        content[:start] + "\n\n" + markdown_table + "\n\n" + content[end:],
        encoding="utf-8",
    )


def run_evaluation(
    *,
    games_per_defender: int = 150,
    quality_positions: int = 10_000,
    seed: int = 0,
    device_name: str = "cpu",
    kind: str = "all",
    model_dir: Path = ROOT / "models" / "warmstart",
) -> dict[str, Any]:
    if kind not in {"all", "krk", "kqk"}:
        raise ValueError("kind must be 'all', 'krk', or 'kqk'.")
    device = torch.device(device_name)
    scenarios: dict[str, Any] = {}
    scenario_specs = (
        (chess.ROOK, "krk", "krk"),
        (chess.QUEEN, "kqk", "kqk"),
    )
    for piece_type, name, model_name in scenario_specs:
        if kind != "all" and name != kind:
            continue
        tablebase = Tablebase.load(existing_tablebase_cache_path(piece_type))
        model_path = model_dir / f"{model_name}.pt"
        model, checkpoint = load_checkpoint(model_path, device)
        heldout_state_ids = checkpoint.get("validation_state_ids")
        if heldout_state_ids is None:
            raise ValueError(
                f"Checkpoint {model_path} does not record held-out validation positions."
            )
        heldout_state_ids = np.asarray(heldout_state_ids, dtype=np.int64)
        rl_path = ROOT / "models" / "rl" / f"{model_name}.pt"
        rl_model = load_model(rl_path, device) if rl_path.is_file() else None
        scenarios[name] = evaluate_scenario(
            tablebase,
            model=model,
            rl_model=rl_model,
            games_per_defender=games_per_defender,
            seed=seed,
            device=device,
            quality_positions=quality_positions,
            heldout_state_ids=heldout_state_ids,
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "games_per_defender": games_per_defender,
        "quality_positions": quality_positions,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seeded solver-based evaluation.")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--quality-positions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kind", choices=("all", "krk", "kqk"), default="all")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "warmstart")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument("--skip-readme", action="store_true")
    args = parser.parse_args()

    results = run_evaluation(
        games_per_defender=args.games,
        quality_positions=args.quality_positions,
        seed=args.seed,
        device_name=args.device,
        kind=args.kind,
        model_dir=args.model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if not args.skip_readme:
        update_readme_table(args.readme, results_markdown_table(results))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
