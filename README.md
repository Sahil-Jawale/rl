# Chess Endgame RL

This project compares reinforcement-learning agents on simplified chess endgames with a config-driven environment, legal-action masking, scenario-specific defenders and rewards, and a benchmark-ready workflow for Q-Learning, DQN, and PPO experiments.

## Project status

This repository is runnable as a chess endgame training and evaluation project and includes:

- a scenario configuration file
- a legal-move-aware endgame environment
- Q-learning, DQN, and PPO training implementations
- exact KQvK/KRvK tablebases and solver-based optimal defenders
- a 4096-action Dueling network with tablebase-labelled supervised warm start
- baseline agents and benchmark/evaluation helpers
- a FastAPI-backed UI that serves a board + results dashboard
- a pytest suite validating the environment API and benchmark-related behavior
- richer benchmark summary metrics (mean, spread, positive/zero/negative rate) and optional tablebase-status reporting in the UI/API

## Getting started

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[train]"
python -m pytest -q
python scripts\run_server.py
```

Then open http://127.0.0.1:8000 in a browser.
The play menu lists the legacy DQN baseline and each available tablebase-supervised or RL-fine-tuned Dueling checkpoint. If no warm-start checkpoint exists yet, train one before expecting the new agent to appear as selectable:

```bash
python -m chess_endgame_rl.warmstart.train_supervised --kind KRvK --epochs 10 --seed 7
```

The warm-start checkpoint is a Dueling network trained with supervised exact-solver labels; it is not itself a Double-DQN RL run. It appears as **Dueling network (tablebase warm-start)** after training. The **Dueling Double DQN (fine-tuned)** choice appears only after an RL fine-tuning run has produced `models\rl\krk.pt`. The app defaults to the warm-start model when available; the fine-tuned model remains selectable for direct comparison rather than being treated as stronger without evidence. If no Dueling checkpoint exists, it falls back to the legacy DQN baseline. These agents are experimental; use the Comparative Analysis tab to distinguish validation move quality from game win rate.

For a fast workflow smoke test, run `python scripts\run_benchmark.py --episodes 5 --eval-games 3 --seeds 7`.
This small budget checks the pipeline only; it is not enough training for meaningful checkmate comparisons.
The benchmark uses equal training budgets and reports greedy held-out checkmate, draw, timeout, and loss rates for each seed.
It saves benchmark-only models under `models\benchmark` so it does not overwrite the models used by the website.

## Exact solver and supervised warm start

The primary learnable scenario is `krk`; `kqk` is the secondary solver-backed scenario and `knk` remains the zero-win sanity case. The repository includes cached KQvK and KRvK tablebases. Rebuild a cache with:

```bash
python scripts\build_tablebase.py --kind KRvK
```

Install the optional training dependencies and run the CPU supervised warm start with:

```bash
pip install ".[train]"
python -m chess_endgame_rl.warmstart.train_supervised --kind KRvK
```

The trainer holds out positions for validation, reports optimal-move, keeps-win, and win-to-draw rates, and saves the best-validation checkpoint under `models\warmstart`. The checkpoint’s validation metrics are also visible in Comparative Analysis, including while the training run is in progress.

Fine-tune a warm-start checkpoint with anchored, legal-masked Double DQN:

```bash
python scripts\finetune_ddqn.py --kind KRvK --episodes 1000
```

The RL run saves the best solver-validation checkpoint under `models\rl` and a sibling metrics JSON. It is an RL experiment distinct from supervised pre-training; do not attribute warm-start performance to the RL fine-tuning stage.

Run or resume the full solve → warm-start → RL → evaluation sequence with `python scripts\pipeline.py`. Use `--through warmstart` to stop after supervised training or `--kind KRvK` to run one endgame while developing.

## Warm-start evaluation results

The deterministic evaluation command uses the configured protocol (150 games per defender and 10,000 held-out positions). Select `--kind krk` when only the rook endgame checkpoint is trained; omit it to evaluate both scenarios:

```bash
python -m chess_endgame_rl.eval.evaluate_all --kind krk
```

It writes `results\results.json` and regenerates the table below. Do not fill in or infer metrics before the required checkpoints have actually been evaluated.

<!-- WARMSTART_RESULTS_START -->

| Agent | Scenario | Defender | W | D | L | Win rate | Optimal-move % | Win -> draw % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Warm start | krk | optimal | 124 | 26 | 0 | 0.827 | 0.941 | 0.173 |
| Warm start + RL | krk | optimal | 84 | 66 | 0 | 0.560 | 0.914 | 0.440 |
| Random baseline | krk | optimal | 0 | 150 | 0 | 0.000 | N/A | 1.000 |
| Warm start | krk | random | 144 | 6 | 0 | 0.960 | 0.941 | 0.040 |
| Warm start + RL | krk | random | 121 | 29 | 0 | 0.807 | 0.914 | 0.193 |
| Random baseline | krk | random | 2 | 148 | 0 | 0.013 | N/A | 0.987 |

<!-- WARMSTART_RESULTS_END -->

Problem formulation, implementation choices, current hyperparameter defaults, and model limitations are described in [docs/PROBLEM.md](./docs/PROBLEM.md), [docs/DESIGN.md](./docs/DESIGN.md), [docs/HYPERPARAMETERS.md](./docs/HYPERPARAMETERS.md), and [docs/MODEL_CARD.md](./docs/MODEL_CARD.md).

To make the site reachable on the local network, start it with
`python scripts\run_server.py --host 0.0.0.0 --port 8000`. For internet hosting, deploy the app behind an HTTPS-capable
ASGI hosting provider and configure `HOST` and `PORT` as required by that provider.

## Repository layout

```text
chess_endgame_rl/
  __init__.py
  agents/
    __init__.py
    q_learning.py
    dqn.py
    ppo.py
    dueling_ddqn.py
    rl_finetuning.py
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
  models/
    dueling_net.py
  solver/
    tablebase.py
  warmstart/
    dataset.py
    train_supervised.py
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
  build_tablebase.py
  finetune_ddqn.py
  speed_benchmark.py
  run_benchmark.py
  run_server.py
tests/
  test_audit_regressions.py
  test_encoding.py
  test_environment.py
  test_rewards.py
  test_tablebase.py
  test_warmstart.py
```

## Notes

- `krk` is treated as the primary learnable variant; `kqk` is the secondary exact-solver scenario.
- `knk` is modeled as a draw-control scenario with a zero-win sanity check.
- The project is set up to run end-to-end from scenario configuration through model training, benchmarking, and browser-based inspection.
- The UI is a practical dashboard for scenario selection, result review, and board comparison rather than a full production analytics platform.
