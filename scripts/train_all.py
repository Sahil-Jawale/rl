"""Train all three RL agents and save models + results.

Usage:
    python scripts/train_all.py [--episodes N] [--quick]

The --quick flag uses reduced episodes for testing (500 each).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.train import (
    save_training_results,
    train_dqn,
    train_ppo,
    train_q_learning,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all RL agents for KRK endgame")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--quick", action="store_true", help="Quick training with fewer episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    if args.quick:
        q_episodes = args.episodes or 500
        dqn_episodes = args.episodes or 500
        ppo_episodes = args.episodes or 500
    else:
        q_episodes = args.episodes or 5000
        dqn_episodes = args.episodes or 3000
        ppo_episodes = args.episodes or 3000

    print("=" * 60)
    print("  Learning to Checkmate in Chess Endgames")
    print("  Training RL Agents for KRK (King+Rook vs King)")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    # --- Train Q-Learning ---
    print(f"\n{'='*60}")
    print(f"  Training Q-Learning ({q_episodes} episodes)")
    print(f"{'='*60}")
    q_result = train_q_learning(
        episodes=q_episodes,
        seed=args.seed,
        save_path=str(models_dir / "q_learning.json"),
    )
    save_training_results(q_result, results_dir)
    all_results["Q-Learning"] = q_result["metrics"]
    print(f"\n  [OK] Q-Learning done - Checkmate rate: {q_result['metrics']['checkmate_rate']:.1%}")

    # --- Train DQN ---
    print(f"\n{'='*60}")
    print(f"  Training DQN ({dqn_episodes} episodes)")
    print(f"{'='*60}")
    dqn_result = train_dqn(
        episodes=dqn_episodes,
        seed=args.seed,
        save_path=str(models_dir / "dqn.npz"),
    )
    save_training_results(dqn_result, results_dir)
    all_results["DQN"] = dqn_result["metrics"]
    print(f"\n  [OK] DQN done - Checkmate rate: {dqn_result['metrics']['checkmate_rate']:.1%}")

    # --- Train PPO ---
    print(f"\n{'='*60}")
    print(f"  Training PPO ({ppo_episodes} episodes)")
    print(f"{'='*60}")
    ppo_result = train_ppo(
        episodes=ppo_episodes,
        seed=args.seed,
        save_path=str(models_dir / "ppo.npz"),
    )
    save_training_results(ppo_result, results_dir)
    all_results["PPO"] = ppo_result["metrics"]
    print(f"\n  [OK] PPO done - Checkmate rate: {ppo_result['metrics']['checkmate_rate']:.1%}")

    # --- Summary ---
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"\n  {'Algorithm':<15} {'Checkmate%':>10} {'Stalemate%':>10} {'Timeout%':>10} {'Avg Reward':>10}")
    print(f"  {'-'*55}")
    for algo, metrics in all_results.items():
        print(
            f"  {algo:<15} "
            f"{metrics['checkmate_rate']:>9.1%} "
            f"{metrics['stalemate_rate']:>9.1%} "
            f"{metrics['timeout_rate']:>9.1%} "
            f"{metrics['mean_reward']:>10.3f}"
        )

    # Save combined summary
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_dir}/")
    print(f"  Models saved to {models_dir}/")


if __name__ == "__main__":
    main()
