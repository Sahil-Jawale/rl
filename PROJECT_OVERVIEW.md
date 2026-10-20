# Chess Endgame Learning Project

## What this project does

This project investigates how learned agents can convert small chess endgames into checkmate. Its main scenario is **King and Rook versus King (KRvK)**, where the attacker tries to mate and the lone king defends. It includes:

- A legal-move-aware chess environment and an exact retrograde tablebase for KRvK and KQvK.
- A convolutional Dueling network trained to imitate tablebase-optimal moves.
- An optional Double-DQN reinforcement-learning stage that fine-tunes the supervised checkpoint.
- A reward-trained DQN baseline, evaluated alongside the two Dueling models on the same game protocol; older Q-Learning and PPO artifacts remain archived.
- A local browser app for playing against available checkpoints and reviewing the current model comparison.

This is an endgame research and demonstration project, **not a general chess engine**. The learned models do not use search at inference time. Since the tablebase already solves these endgames exactly, the tablebase acts as ground truth for training and evaluation rather than something the learned models are expected to outperform.

## Current model pipeline

The current Dueling checkpoints share the same convolutional architecture and input/output representation, but they were trained differently:

1. **Tablebase-supervised warm-start:** the solver supplies optimal-move labels, and the network is trained with supervised learning. This checkpoint is a learned policy, **not an RL-trained model**.
2. **Dueling Double DQN fine-tuning:** reinforcement learning starts from the supervised checkpoint. It uses replay, legal-action-masked Double-DQN targets, optional solver-based potential shaping, and a supervised anchor loss. It is **RL fine-tuning, not RL from scratch**.

The fine-tuned checkpoint did not improve the recorded evaluation over the supervised warm-start. The website presents these checkpoints as separate methods and defaults to the warm-start.

### State and action representation

- **State:** 18 binary planes of size 8 × 8 (1,152 input values), represented from the side-to-move perspective. The planes encode own and opponent piece types, side to move, castling rights, and the en-passant target.
- **Action:** 4,096 possible from-square/to-square indices, computed as `from_square * 64 + to_square`.
- **Legality:** a mask excludes illegal moves during selection and value estimation. The encoding supports queen promotion, not underpromotion.
- **Model:** a convolutional Dueling network estimates a state value and action advantages. It selects a move in one forward pass and does not perform tree search.

The DQN baseline uses a compact KRvK representation: 36 actions (8 king moves and 28 rook moves). Its older reward-history and four-seed benchmark use a different protocol; the website's DQN result is a separate evaluation on the same held-out games as the Dueling models.

### RL problem formulation

The attacker seeks checkmate while avoiding draws and losses. A learning transition applies an attacker move and, if the game continues, a defender reply. The defender can be random or tablebase-optimal. The recorded fine-tuning configuration uses a discount factor of 0.99 and a maximum horizon of 100 plies.

The fine-tuning base reward is +1 for attacker checkmate, −1 for attacker checkmated, 0 for a draw, and −0.001 for a nonterminal step. Optional potential-based shaping uses:

`F(s, s') = γ Φ(s') − Φ(s)`

For nonterminal positions the tablebase says are winning for the attacker, `Φ(s) = 1 − min(DTM, DTM_max) / DTM_max`; it is zero for terminal, drawn, or losing states. The supervised anchor loss regularizes updates toward the warm-start's solver-derived behavior. These settings describe **fine-tuning only**; they are not the training objective for the supervised warm-start.

## Recorded KRvK results

Protocol v1 used seed 7, 10,000 held-out positions for move-quality evaluation, and 150 games per defender from held-out attacker-winning positions. The tablebase-optimal defender selects the strongest available defense.

| Agent | Optimal moves | Win-to-draw moves | Vs random (W / rook / rep / horizon / other) | Vs optimal (W / rook / rep / horizon / other) |
|---|---:|---:|---:|---:|
| DQN baseline | — | — | 0 / 73 / 42 / 35 / 0 | 0 / 129 / 21 / 0 / 0 |
| Tablebase-supervised warm-start | 94.06% | 0.52% | 144 / 5 / 0 / 0 / 1 | 124 / 19 / 5 / 0 / 2 |
| Warm-start + Double DQN | 91.45% | 0.48% | 121 / 2 / 24 / 0 / 3 | 84 / 29 / 34 / 0 / 3 |

Move quality and game outcomes are separate measurements: a high optimal-move rate is not itself a game win rate. Optimal-move rates are reported only for the Dueling checkpoints here; game outcomes are reported for all three agents. Outcome counts are wins / rook captures / repetitions / horizon draws / other draws. A rook capture technically ends in a draw but is shown separately because it is not a successful defense by the attacker. Losses are omitted because none occurred and the attacker cannot be checkmated in this endgame. The fine-tuning validation history uses a separate 256-position sample. The evaluation is single-seed and KRvK-specific; it is not an Elo rating or a statistical claim across seeds.

The Q-Learning and PPO reward histories and the four-seed benchmark are retained as historical artifacts, not plotted in the current web comparison. The DQN row above was freshly evaluated under Protocol v1; it is distinct from DQN's historical reward-training runs.

Machine-readable current evaluation results are in [`results/results.json`](results/results.json). Additional experimental details are in [`docs/PROBLEM.md`](docs/PROBLEM.md), [`docs/DESIGN.md`](docs/DESIGN.md), [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md), and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Website

The FastAPI application serves the local chess UI, exposes game and model APIs, and loads available model checkpoints. Training is not required when the committed checkpoints are present.

Start the app from the repository root:

```powershell
python scripts\run_server.py --port 8000
```

Then open <http://127.0.0.1:8000>. Use another free port if 8000 is occupied. The current UI contains:

- **Play Agent:** play the defending king against an available attacker checkpoint.
- **Comparative Analysis:** compare the current checkpoints using solver move quality and W/D/L against random and optimal defenders; inspect the fine-tuning validation trend.
- **RL Formulation:** review the current state/action representation, fine-tuning rewards, and the distinction between supervised warm-start and RL fine-tuning.

## Setup and tests

For a fresh environment, install the project and training dependencies (PyTorch is needed to load the Dueling checkpoints):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[train]"
```

Run the test suite:

```powershell
python -m pytest -q
```

## Repository guide

| Path | Purpose |
|---|---|
| `chess_endgame_rl/env/` | Chess scenarios, state/action encoding, defenders, rewards, and environment. |
| `chess_endgame_rl/solver/` | Exact KRvK/KQvK retrograde tablebase. |
| `chess_endgame_rl/models/` | Convolutional Dueling network architecture. |
| `chess_endgame_rl/warmstart/` | Tablebase-labeled dataset and supervised training. |
| `chess_endgame_rl/agents/` | Legacy agents and Dueling Double-DQN fine-tuning implementation. |
| `chess_endgame_rl/eval/` | Solver-based move-quality and game evaluation. |
| `chess_endgame_rl/ui/` | FastAPI backend and browser UI. |
| `scripts/` | Server, training, evaluation, and pipeline entry points. |
| `configs/` | Scenario configuration. |
| `models/` | Trained checkpoints and their metadata. |
| `results/` | Saved evaluation and training reports. |
| `tests/` | Environment, solver, model, training, evaluation, and UI tests. |

## Scope and limitations

- The project focuses on three-piece endgames; it is not intended for full chess.
- Solver-derived labels and shaping make the experiments measurable but mean the learned policy depends on exact endgame ground truth.
- The reported current-checkpoint evaluation is KRvK, single-seed, and based on a finite game sample.
- Double-DQN fine-tuning did not improve the warm-start in the recorded run. Additional seeds and controlled experiments are needed before making broader claims about performance.
