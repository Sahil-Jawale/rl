from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.env.scenarios import list_scenarios
from chess_endgame_rl.train import (
    save_training_results,
    train_dqn,
    train_ppo,
    train_q_learning,
)


def _runner_for(name: str):
    name = (name or "q-learning").lower().replace("_", "-")
    mapping = {
        "q-learning": train_q_learning,
        "qlearning": train_q_learning,
        "dqn": train_dqn,
        "ppo": train_ppo,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - argparse validation covers this.
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown algorithm '{name}'. Valid choices: {valid}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one RL agent on a configured chess endgame scenario.")
    parser.add_argument("--scenario", choices=list_scenarios(), default="krk", help="Scenario ID to train on")
    parser.add_argument("--algorithm", choices=["q-learning", "dqn", "ppo"], default="q-learning", help="Training algorithm")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=7, help="Fixed random seed")
    parser.add_argument("--defender", choices=["random", "heuristic", "smart"], default="random")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results", help="Directory for training results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models", help="Directory for trained model")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    runner = _runner_for(args.algorithm)
    model_file = {"q-learning": "q_learning.json", "dqn": "dqn.npz", "ppo": "ppo.npz"}[
        args.algorithm.replace("_", "-")
    ]
    model_dir = args.model_dir if args.scenario == "krk" else args.model_dir / args.scenario
    summary = runner(
        scenario_id=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        defender=args.defender,
        save_path=str(model_dir / model_file),
    )
    output_path = save_training_results(summary, args.output_dir)

    print(f"Scenario: {args.scenario}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {summary['metrics']['mean_reward']:.4f}")
    print(f"Saved summary to: {output_path}")

    if args.episodes > 0 and summary.get("metrics"):
        print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
