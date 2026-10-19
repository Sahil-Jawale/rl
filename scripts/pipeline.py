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
from chess_endgame_rl.eval.evaluate_all import (  # noqa: E402
    results_markdown_table,
    run_evaluation,
    update_readme_table,
)
from chess_endgame_rl.solver.tablebase import (  # noqa: E402
    Tablebase,
    existing_tablebase_cache_path,
    tablebase_cache_path,
)
from chess_endgame_rl.warmstart.train_supervised import train as train_supervised  # noqa: E402


STAGES = ("solve", "warmstart", "rl", "eval")
KINDS = {
    "KQvK": (chess.QUEEN, "kqk"),
    "KRvK": (chess.ROOK, "krk"),
}


def run_pipeline(
    *,
    kinds: list[str],
    through: str,
    force: bool = False,
    warmstart_epochs: int = 10,
    rl_episodes: int = 1_000,
    evaluation_games: int = 150,
    seed: int = 0,
    readme_path: Path = ROOT / "README.md",
) -> list[dict[str, str]]:
    if through not in STAGES:
        raise ValueError(f"through must be one of {', '.join(STAGES)}.")
    if through == "eval" and set(kinds) != set(KINDS):
        raise ValueError("The combined evaluation stage requires both KQvK and KRvK.")

    completed: list[dict[str, str]] = []
    last_stage = STAGES.index(through)
    for kind in kinds:
        piece_type, model_name = KINDS[kind]
        cache_path = tablebase_cache_path(piece_type)
        warmstart_path = ROOT / "models" / "warmstart" / f"{model_name}.pt"
        rl_path = ROOT / "models" / "rl" / f"{model_name}.pt"

        if last_stage >= STAGES.index("solve"):
            existing_cache = existing_tablebase_cache_path(piece_type)
            if cache_path.exists() and not force:
                completed.append({"stage": "solve", "kind": kind, "status": "skipped"})
            elif existing_cache.exists() and not force:
                tablebase = Tablebase.load(existing_cache)
                tablebase.save(cache_path)
                completed.append({"stage": "solve", "kind": kind, "status": "migrated"})
            else:
                tablebase = Tablebase.solve("Q" if piece_type == chess.QUEEN else "R")
                tablebase.save(cache_path)
                completed.append({"stage": "solve", "kind": kind, "status": "complete"})
        else:
            tablebase = Tablebase.load(existing_tablebase_cache_path(piece_type))

        tablebase = Tablebase.load(existing_tablebase_cache_path(piece_type))
        if last_stage >= STAGES.index("warmstart"):
            if warmstart_path.exists() and not force:
                completed.append({"stage": "warmstart", "kind": kind, "status": "skipped"})
            else:
                train_supervised(
                    tablebase,
                    warmstart_path,
                    epochs=warmstart_epochs,
                    seed=seed,
                )
                completed.append({"stage": "warmstart", "kind": kind, "status": "complete"})

        if last_stage >= STAGES.index("rl"):
            if rl_path.exists() and not force:
                completed.append({"stage": "rl", "kind": kind, "status": "skipped"})
            else:
                if not warmstart_path.exists():
                    raise FileNotFoundError(
                        f"RL requires the supervised checkpoint {warmstart_path}."
                    )
                train_finetune(
                    tablebase,
                    warmstart_path,
                    rl_path,
                    episodes=rl_episodes,
                    seed=seed,
                )
                completed.append({"stage": "rl", "kind": kind, "status": "complete"})

    if last_stage >= STAGES.index("eval"):
        results_path = ROOT / "results" / "results.json"
        if results_path.exists() and not force:
            completed.append({"stage": "eval", "kind": "all", "status": "skipped"})
        else:
            results = run_evaluation(
                games_per_defender=evaluation_games,
                seed=seed,
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            update_readme_table(readme_path, results_markdown_table(results))
            completed.append({"stage": "eval", "kind": "all", "status": "complete"})
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume solve -> warmstart -> RL -> eval.")
    parser.add_argument("--kind", choices=("all", *KINDS), default="all")
    parser.add_argument("--through", choices=STAGES, default="eval")
    parser.add_argument("--warmstart-epochs", type=int, default=10)
    parser.add_argument("--rl-episodes", type=int, default=1_000)
    parser.add_argument("--evaluation-games", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()

    kinds = list(KINDS) if args.kind == "all" else [args.kind]
    print(
        json.dumps(
            run_pipeline(
                kinds=kinds,
                through=args.through,
                force=args.force,
                warmstart_epochs=args.warmstart_epochs,
                rl_episodes=args.rl_episodes,
                evaluation_games=args.evaluation_games,
                seed=args.seed,
                readme_path=args.readme,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
