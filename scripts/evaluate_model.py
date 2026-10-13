from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.eval.suite import generate_test_positions, summarize_game_results
from chess_endgame_rl.train import run_dqn_training, run_ppo_training, run_tabular_q_learning


def _runner_for(name: str):
    name = (name or "q-learning").lower().replace("_", "-")
    return {
        "q-learning": run_tabular_q_learning,
        "qlearning": run_tabular_q_learning,
        "dqn": run_dqn_training,
        "ppo": run_ppo_training,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an evaluation pass for a configured scenario and algorithm.")
    parser.add_argument("--scenario", choices=["krk", "knk"], default="krk")
    parser.add_argument("--algorithm", choices=["q-learning", "dqn", "ppo"], default="q-learning")
    parser.add_argument("--episodes", type=int, default=25, help="Evaluation episodes to run")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--positions", type=int, default=20)
    args = parser.parse_args()

    runner = _runner_for(args.algorithm)
    summary = runner(scenario_id=args.scenario, episodes=args.episodes, seed=args.seed)
    positions = generate_test_positions(args.scenario, count=args.positions, seed=args.seed)
    metrics = summarize_game_results(summary["rewards"])

    print(json.dumps({
        "scenario": args.scenario,
        "algorithm": args.algorithm,
        "episodes": args.episodes,
        "positions": len(positions),
        "summary": summary,
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
