from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.env.scenarios import list_scenarios
from chess_endgame_rl.train import run_dqn_training, run_ppo_training, run_tabular_q_learning, save_result_summary


def main() -> None:
    for scenario in list_scenarios():
        q_summary = run_tabular_q_learning(scenario_id=scenario, episodes=50, seed=7)
        dqn_summary = run_dqn_training(scenario_id=scenario, episodes=25, seed=7)
        ppo_summary = run_ppo_training(scenario_id=scenario, episodes=25, seed=7)

        for summary in [q_summary, dqn_summary, ppo_summary]:
            path = save_result_summary(summary, ROOT / "results")
            print(f"Saved {summary['algorithm']} result for {summary['scenario']} -> {path.name}")


if __name__ == "__main__":
    main()
