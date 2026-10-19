from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.eval.evaluate_all import (  # noqa: E402
    evaluate_scenario,
    load_checkpoint,
)
from chess_endgame_rl.solver.tablebase import (  # noqa: E402
    Tablebase,
    existing_tablebase_cache_path,
)


def evaluate_checkpoint(
    checkpoint: Path,
    scenario: str,
    *,
    games: int,
    quality_positions: int,
    seed: int,
    device_name: str = "cpu",
) -> dict:
    scenario = scenario.lower()
    if scenario not in {"krk", "kqk"}:
        raise ValueError("scenario must be 'krk' or 'kqk'.")
    piece_type = chess.ROOK if scenario == "krk" else chess.QUEEN
    tablebase = Tablebase.load(existing_tablebase_cache_path(piece_type))
    device = torch.device(device_name)
    model, checkpoint_data = load_checkpoint(checkpoint, device)
    heldout_state_ids = checkpoint_data.get("validation_state_ids")
    if heldout_state_ids is None:
        raise ValueError(f"Checkpoint {checkpoint} has no fixed held-out state set.")
    metrics = evaluate_scenario(
        tablebase,
        model=model,
        games_per_defender=games,
        seed=seed,
        device=device,
        quality_positions=quality_positions,
        heldout_state_ids=np.asarray(heldout_state_ids, dtype=np.int64),
    )
    return {
        "checkpoint": str(checkpoint),
        "scenario": scenario,
        "seed": seed,
        "games_per_defender": games,
        "quality_positions": quality_positions,
        "win_rate_vs_optimal": metrics["defenders"]["optimal"]["model"]["win_rate"],
        "optimal_move_rate": metrics["move_quality"]["optimal_move_rate"],
        "move_quality": metrics["move_quality"],
    }


def check_regression(current: dict, baseline: dict) -> list[str]:
    failures = []
    for key in ("scenario", "seed", "games_per_defender", "quality_positions"):
        if key not in baseline:
            raise ValueError(f"Regression baseline is missing protocol field {key}.")
        if current[key] != baseline[key]:
            raise ValueError(
                f"Evaluation protocol mismatch for {key}: "
                f"{current[key]!r} != baseline {baseline[key]!r}."
            )
    for metric in ("win_rate_vs_optimal", "optimal_move_rate"):
        if metric not in baseline:
            raise ValueError(f"Regression baseline is missing {metric}.")
        if current[metric] < baseline[metric]:
            failures.append(
                f"{metric} regressed: {current[metric]:.4f} < baseline {baseline[metric]:.4f}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check solver-metric checkpoint regression.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=("krk", "kqk"), default="krk")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--quality-positions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "results" / "baseline.json"
    )
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    current = evaluate_checkpoint(
        args.checkpoint,
        args.scenario,
        games=args.games,
        quality_positions=args.quality_positions,
        seed=args.seed,
        device_name=args.device,
    )
    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"updated": str(args.baseline), "metrics": current}, indent=2))
        return
    if not args.baseline.is_file():
        raise FileNotFoundError(
            f"No regression baseline at {args.baseline}; create an explicitly reviewed "
            "baseline with --update."
        )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failures = check_regression(current, baseline)
    print(json.dumps({"metrics": current, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
