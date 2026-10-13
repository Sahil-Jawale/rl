from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.env import EndgameEnv


def main() -> None:
    env = EndgameEnv(scenario_id="krk", seed=0)
    print(f"Initial position: {env.render()}")
    mask = env.legal_action_mask()
    print(f"Legal action count: {mask.sum()}")


if __name__ == "__main__":
    main()
