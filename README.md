# Chess Endgame RL

This project compares reinforcement-learning agents on simplified chess endgames with a config-driven environment, legal-action masking, scenario-specific defenders and rewards, and a benchmark-ready workflow for Q-Learning, DQN, and PPO experiments.

## Project status

This repository is runnable as a chess endgame training and evaluation project and includes:

- a scenario configuration file
- a legal-move-aware endgame environment
- Q-learning, DQN, and PPO training implementations
- baseline agents and benchmark/evaluation helpers
- a FastAPI-backed UI that serves a board + results dashboard
- a pytest suite validating the environment API and benchmark-related behavior
- richer benchmark summary metrics (mean, spread, positive/zero/negative rate) and optional tablebase-status reporting in the UI/API

## Getting started

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python scripts\run_benchmark.py
python scripts\run_server.py
```

Then open http://127.0.0.1:8000 in a browser.

## Repository layout

```text
chess_endgame_rl/
  __init__.py
  agents/
    __init__.py
    q_learning.py
    dqn.py
    ppo.py
  baselines/
    __init__.py
    random_agent.py
    greedy_agent.py
    tablebase_agent.py
  env/
    __init__.py
    scenarios.py
    encoding.py
    defenders.py
    rewards.py
    endgame_env.py
  eval/
    __init__.py
    metrics.py
    suite.py
  ui/
    server.py
    static/
      index.html
      styles.css
      app.js
  train.py
  __init__.py
configs/
  scenarios.yaml
results/
  benchmark_summary.json
  .gitkeep
scripts/
  train_all.py
  speed_benchmark.py
  run_benchmark.py
  run_server.py
tests/
  test_environment.py
```

## Notes

- `krk` is treated as the primary learnable variant.
- `knk` is modeled as a draw-control scenario with a zero-win sanity check.
- The project is set up to run end-to-end from scenario configuration through model training, benchmarking, and browser-based inspection.
- The UI is a practical dashboard for scenario selection, result review, and board comparison rather than a full production analytics platform.
