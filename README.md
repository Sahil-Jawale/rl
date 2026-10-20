# Chess Endgame RL: Learning to Convert KRvK

An endgame-learning project focused on **King and Rook versus King (KRvK)**. The attacker tries to checkmate; the lone-king defender tries to resist. A local web app lets you play as the defender and inspect model comparisons.

This is a research and teaching project, not a general-purpose chess engine. The included exact tablebase already solves these small endgames, so it supplies training labels and evaluation ground truth.

## Current agents

The app offers three KRvK attackers:

| Agent | How it learns | What it represents |
|---|---|---|
| **DQN** | Trained from scratch from game rewards. | A compact 193-value board-and-move-count input and 36 legal-move actions. |
| **Dueling network (tablebase warm-start)** | Learns to imitate best-move labels supplied by the exact tablebase. This is supervised learning, not reinforcement learning. | 18 binary 8×8 board planes and 4,096 from-square/to-square action indices. |
| **Dueling Double DQN (fine-tuned)** | Starts from the tablebase-supervised checkpoint and continues learning from rewards with Double-DQN updates. | The same 18-plane input and 4,096-action encoding as the warm-start. |

The fine-tuned model did not outperform its supervised starting checkpoint in the recorded evaluation. This distinction matters: the warm-start copies solver guidance, while the fine-tuned model adds reward-driven updates; neither result shows that reward-only learning from scratch is stronger than the solver.

## What is in the project

- A legal-move-aware KRvK/KQvK environment with random and tablebase-optimal defenders.
- Exact retrograde tablebases used as a source of optimal-move labels and game-theoretic evaluation.
- A convolutional Dueling network, supervised warm-start training, and optional Double-DQN fine-tuning.
- Reproducible evaluation of game outcomes and tablebase move quality.
- A browser interface for playing against available agents, viewing comparisons, and reading a plain-language formulation.

## Recorded KRvK evaluation

Protocol v1 uses seed 7, 150 held-out attacker-winning starting positions per defender, and a 100-ply game horizon. Each of the three agents is evaluated on the same starts. The optimal defender selects a strongest available tablebase defense.

| Agent | Defender | Checkmate wins | Rook captures | Repetitions | Horizon draws | Other draws | Optimal moves |
|---|---|---:|---:|---:|---:|---:|---:|
| DQN | Random | 0 | 73 | 42 | 35 | 0 | — |
| DQN | Tablebase-optimal | 0 | 129 | 21 | 0 | 0 | — |
| Tablebase warm-start | Random | 144 | 5 | 0 | 0 | 1 | 94.06% |
| Tablebase warm-start | Tablebase-optimal | 124 | 19 | 5 | 0 | 2 | 94.06% |
| Warm-start + Double DQN | Random | 121 | 2 | 24 | 0 | 3 | 91.45% |
| Warm-start + Double DQN | Tablebase-optimal | 84 | 29 | 34 | 0 | 3 | 91.45% |

All rows contain 150 games. The outcome columns distinguish reasons games did not end in checkmate. A rook capture leaves bare kings, so it is technically a draw but is not a successful conversion by the attacker. Losses are omitted because the attacking side cannot be checkmated in KRvK.

The **Optimal moves** values are a separate measurement: the fraction of 10,000 held-out positions where the Dueling checkpoint selects a tablebase-optimal move. The DQN uses a different representation and has no comparable move-quality score in this report. Fine-tuning validation is another separate measurement, recorded over 256 positions at training checkpoints.

These are single-seed KRvK results, not an Elo rating, a full-chess measure, or a multi-seed statistical claim. Complete machine-readable results are in [`results/results.json`](results/results.json); fine-tuning validation data is in [`models/rl/krk.metrics.json`](models/rl/krk.metrics.json).

## Run the web app

From the repository root, install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[train]"
```

Start the local server:

```powershell
python scripts\run_server.py --port 8000
```

Open <http://127.0.0.1:8000>. If that port is in use, select another free port. PyTorch is required for loading the Dueling checkpoints. Avoid exposing the development server directly to the public internet.

The UI has three sections:

- **Play Agent:** play as the defending king against a trained attacker.
- **Comparative Analysis:** compare game outcomes under the same protocol; review the separated rook-capture, repetition, horizon, and other-draw categories; inspect Dueling move-quality and fine-tuning validation.
- **RL Formulation:** see the KRvK state, action, reward, and episode definitions in simple language.

## Train and evaluate

The commands below are run from the repository root. Training can overwrite checkpoints and result files; save any results you want to keep before retraining.

Train the supervised warm-start:

```powershell
python -m chess_endgame_rl.warmstart.train_supervised --kind KRvK --epochs 10 --seed 7
```

Optionally fine-tune from that checkpoint:

```powershell
python scripts\finetune_ddqn.py --kind KRvK --episodes 1000 --seed 7
```

Evaluate the KRvK agents using Protocol v1:

```powershell
python -m chess_endgame_rl.eval.evaluate_all --kind krk --games 150 --quality-positions 10000 --seed 7
```

## Tests

Run the full suite:

```powershell
python -m pytest -q
```

Tests cover environment rules and encodings, tablebase behavior, model and training components, evaluation protocols, outcome categorization, and UI endpoints.

## Repository guide

| Path | Purpose |
|---|---|
| `chess_endgame_rl/env/` | Endgame rules, state/action encoding, defenders, and rewards. |
| `chess_endgame_rl/solver/` | Exact KRvK/KQvK retrograde tablebases. |
| `chess_endgame_rl/models/` | Convolutional Dueling network. |
| `chess_endgame_rl/warmstart/` | Tablebase-labelled dataset and supervised training. |
| `chess_endgame_rl/agents/` | DQN and Double-DQN implementations. |
| `chess_endgame_rl/eval/` | Tablebase move-quality and game-outcome evaluation. |
| `chess_endgame_rl/ui/` | FastAPI server and browser application. |
| `configs/` | Scenario and reward configuration. |
| `data/` | Endgame tablebase data and caches. |
| `docs/` | Problem definition, design, hyperparameters, and model card. |
| `models/` | Trained checkpoints and metadata. |
| `results/` | Recorded evaluations and reports. |
| `scripts/` | Training, evaluation, and app entry points. |
| `tests/` | Automated test suite. |

## Limitations

- The project focuses on small, solver-known endgames rather than full chess.
- The strongest recorded policy relies on tablebase supervision.
- Reward-driven fine-tuning did not improve the supervised checkpoint in this run.
- The reported current evaluation covers KRvK and one seed; further seeds and controlled experiments are needed for broader conclusions.
