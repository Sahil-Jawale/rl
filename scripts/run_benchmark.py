from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.agents import DeepQNetwork, PPOAgent, TabularQAgent
from chess_endgame_rl.env.scenarios import list_scenarios
from chess_endgame_rl.eval.suite import evaluate_agent
from chess_endgame_rl.train import train_dqn, train_ppo, train_q_learning


TRAINERS = {
    "q-learning": train_q_learning,
    "dqn": train_dqn,
    "ppo": train_ppo,
}
MODEL_LOADERS = {
    "q-learning": TabularQAgent.load,
    "dqn": DeepQNetwork.load,
    "ppo": PPOAgent.load,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare all agents on identical seeded endgame evaluations."
    )
    parser.add_argument("--scenario", choices=list_scenarios(), default="krk")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per agent and seed")
    parser.add_argument("--eval-games", type=int, default=50, help="Held-out games per trained model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23, 41])
    parser.add_argument("--defender", choices=["random", "heuristic", "smart"], default="random")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "benchmark")
    args = parser.parse_args()
    if args.episodes <= 0 or args.eval_games <= 0 or not args.seeds:
        parser.error("--episodes, --eval-games, and --seeds must all be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for algorithm, trainer in TRAINERS.items():
        seed_results = []
        for seed in args.seeds:
            model_path = args.model_dir / args.scenario / f"{algorithm}_{seed}"
            model_path = model_path.with_suffix(".json" if algorithm == "q-learning" else ".npz")
            trained = trainer(
                episodes=args.episodes,
                seed=seed,
                scenario_id=args.scenario,
                defender=args.defender,
                save_path=str(model_path),
                verbose=False,
            )
            agent = MODEL_LOADERS[algorithm](model_path)
            evaluation = evaluate_agent(
                agent,
                algorithm,
                scenario_id=args.scenario,
                episodes=args.eval_games,
                seed=100_000 + seed,
                defender=args.defender,
            )
            seed_results.append({
                "seed": seed,
                "training_metrics": trained["metrics"],
                "evaluation": evaluation,
                "model_path": str(model_path),
            })
            print(
                f"{algorithm:10} seed={seed:>3} "
                f"mates={evaluation['checkmates']}/{args.eval_games} "
                f"draws={evaluation['draw_rate']:.1%} "
                f"timeouts={evaluation['timeout_rate']:.1%}"
            )

        total_games = args.eval_games * len(seed_results)
        total_mates = sum(row["evaluation"]["checkmates"] for row in seed_results)
        results.append({
            "scenario": args.scenario,
            "algorithm": algorithm,
            "training_episodes_per_seed": args.episodes,
            "evaluation_episodes_per_seed": args.eval_games,
            "seeds": args.seeds,
            "defender": args.defender,
            "evaluation": {
                "games": total_games,
                "checkmates": total_mates,
                "checkmate_rate": total_mates / total_games,
                "draw_rate": sum(row["evaluation"]["draw_rate"] for row in seed_results) / len(seed_results),
                "timeout_rate": sum(row["evaluation"]["timeout_rate"] for row in seed_results) / len(seed_results),
                "loss_rate": sum(row["evaluation"]["loss_rate"] for row in seed_results) / len(seed_results),
                "mean_reward": sum(row["evaluation"]["mean_reward"] for row in seed_results) / len(seed_results),
            },
            "seed_results": seed_results,
        })

    result_path = args.output_dir / "benchmark_summary.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved reproducible benchmark results to {result_path}")


if __name__ == "__main__":
    main()
