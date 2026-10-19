from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.agents.rl_finetuning import train_finetune  # noqa: E402
from chess_endgame_rl.solver.tablebase import (  # noqa: E402
    Tablebase,
    existing_tablebase_cache_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a supervised Dueling network using legal-masked Double DQN."
    )
    parser.add_argument("--kind", choices=("KQvK", "KRvK"), default="KRvK")
    parser.add_argument("--warmstart", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--defender", choices=("random", "optimal"), default="optimal")
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every-steps", type=int, default=1_000)
    parser.add_argument("--eval-positions", type=int, default=512)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    kind = args.kind.upper()
    model_name = "kqk" if kind == "KQVK" else "krk"
    warmstart_path = args.warmstart or ROOT / "models" / "warmstart" / f"{model_name}.pt"
    output_path = args.output or ROOT / "models" / "rl" / f"{model_name}.pt"
    tablebase = Tablebase.load(
        existing_tablebase_cache_path(
            chess.QUEEN if kind == "KQVK" else chess.ROOK
        )
    )
    result = train_finetune(
        tablebase,
        warmstart_path,
        output_path,
        episodes=args.episodes,
        defender_name=args.defender,
        max_plies=args.max_plies,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        anchor_weight=args.anchor_weight,
        shaping_enabled=not args.no_shaping,
        batch_size=args.batch_size,
        eval_every_steps=args.eval_every_steps,
        eval_positions=args.eval_positions,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
