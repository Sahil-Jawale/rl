from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.env.scenarios import list_scenarios
from chess_endgame_rl.train import run_dqn_training, run_ppo_training, run_tabular_q_learning, save_result_summary


def _runner_for(name: str):
    name = (name or "q-learning").lower().replace("_", "-")
    mapping = {
        "q-learning": run_tabular_q_learning,
        "qlearning": run_tabular_q_learning,
        "dqn": run_dqn_training,
        "ppo": run_ppo_training,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - argparse validation covers this.
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown algorithm '{name}'. Valid choices: {valid}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single RL agent on a configured chess endgame scenario.")
    parser.add_argument("--scenario", choices=list_scenarios(), default="krk", help="Scenario ID to train on")
    parser.add_argument("--algorithm", choices=["q-learning", "dqn", "ppo"], default="q-learning", help="Training algorithm")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=7, help="Fixed random seed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results", help="Directory to save the JSON summary")
    args = parser.parse_args()

    runner = _runner_for(args.algorithm)
    summary = runner(scenario_id=args.scenario, episodes=args.episodes, seed=args.seed)
    output_path = save_result_summary(summary, args.output_dir)

    print(f"Scenario: {args.scenario}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {summary['mean_reward']:.4f}")
    print(f"Saved summary to: {output_path}")

    if args.episodes > 0 and summary.get("metrics"):
        print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
