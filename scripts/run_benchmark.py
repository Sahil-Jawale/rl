from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.env.scenarios import list_scenarios
from chess_endgame_rl.eval.suite import generate_test_positions, summarize_game_results
from chess_endgame_rl.train import run_dqn_training, run_ppo_training, run_tabular_q_learning, save_result_summary


def main() -> None:
    benchmark = []
    scenarios = list_scenarios()
    seeds = [7, 11, 23, 41]

    for scenario in scenarios:
        test_positions = generate_test_positions(scenario, count=20, seed=7)
        for algorithm_name, runner in {
            "q-learning": run_tabular_q_learning,
            "dqn": run_dqn_training,
            "ppo": run_ppo_training,
        }.items():
            seed_results = []
            for seed in seeds:
                summary = runner(scenario_id=scenario, episodes=50 if algorithm_name == "q-learning" else 20, seed=seed)
                save_result_summary(summary, ROOT / "results")
                seed_results.append({
                    "seed": seed,
                    "mean_reward": summary["mean_reward"],
                    "metrics": summary["metrics"],
                    "rewards": summary["rewards"],
                    "algorithm": summary["algorithm"],
                    "scenario": scenario,
                })

            aggregate = {
                "scenario": scenario,
                "algorithm": algorithm_name,
                "mean_reward": sum(item["mean_reward"] for item in seed_results) / len(seed_results),
                "positions": len(test_positions),
                "seeds": len(seed_results),
                "summary": summarize_game_results([value for item in seed_results for value in item["rewards"]]),
                "seed_results": seed_results,
            }
            benchmark.append(aggregate)

        print(f"scenario={scenario} positions={len(test_positions)}")
        for item in benchmark[-3:]:
            print(f"  {item['algorithm']}: mean_reward={item['mean_reward']:.3f} seeds={item['seeds']}")

    result_path = ROOT / "results" / "benchmark_summary.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2)
    print(f"Saved benchmark summary -> {result_path.name}")


if __name__ == "__main__":
    main()
