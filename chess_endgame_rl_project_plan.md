# Chess Endgame RL — Project Plan (KR-K and KN-K)

**Working title:** Comparative Analysis of Reinforcement Learning Algorithms for Chess Endgame Play: Q-Learning, DQN and PPO on King+Rook vs King and King+Knight vs King

All effort and training-budget numbers are rough estimates, not measurements. Items marked **(unverified)** were not checked in research.

---

## 0. Read first: KN-K is a draw

| Scenario | Winnable? | Consequence |
|---|---|---|
| **KR-K** (King + Rook vs King) | **Yes**, forced mate | Normal benchmark: mate rate, moves-to-mate, tablebase comparison |
| **KN-K** (King + Knight vs King) | **No.** A lone knight cannot force checkmate; the position is a draw with correct defense (standard insufficient-material draw) | Every agent should score **0% mate**. It runs as a **control / draw scenario** with different metrics (section 6) |

Consequences for the project:
- Only **one** scenario (KR-K) is a real "can the agent win" test. The cross-scenario comparison is therefore weaker than with two winnable endgames.
- As an RL problem, KN-K is nearly degenerate: the only meaningful event is losing the knight. I defined its reward and metrics around that (3.5, 6), but it will teach less than KR-K.
- If you want two winnable scenarios, swap KN-K for **KQ-K** (King + Queen vs King) with a one-line config change. Not done here, as you asked to remove the others.

---

## 1. Goal and deliverables

For each scenario, train and compare three RL algorithms (tabular Q-Learning, DQN, PPO) = **6 agents**, benchmark them against baselines, and present everything in a UI where pieces move in real time and you can switch scenario, model and defender.

1. A config-driven environment with 2 scenarios.
2. 3 algorithms x 2 scenarios, multi-seed.
3. A fair benchmark: fixed held-out test positions, tablebase ground truth (KR-K), confidence intervals.
4. A real-time web UI.
5. README, results report, viva notes.

---

## 2. Scenarios

| ID | Pieces | Win? | Raw states (White to move) | Actions | Dense Q-table |
|---|---|---|---|---|---|
| `krk` | K+R vs K | Yes | 64^3 = 262,144 | **36** (king 8 + rook 28) | 262,144 x 36 x 4 B = 36 MiB |
| `knk` | K+N vs K | No (draw) | 262,144 | **16** (king 8 + knight 8) | 262,144 x 16 x 4 B = 16 MiB |

Raw counts include illegal placements; legal counts are lower (to be measured). Both tables fit in memory, so no sparse table is needed.

---

## 3. Environment design

### 3.1 Setup
- Library: `python-chess` (`pip install chess`, Python 3.8+; move generation, legal-move validation, engine and tablebase support).
- The agent plays **White**. Black (the lone king) is a scripted **defender**.
- Each episode starts from a uniformly random legal position, White to move, not already terminal.
- Scenario defined in `configs/scenarios.yaml` (pieces, move cap, budgets, reward weights).

### 3.2 Action encoding (piece-type specific, scenario-agnostic code)
- **King:** 8 directions, distance 1 = 8 actions.
- **Rook:** 4 straight directions x distance 1-7 = 28 actions.
- **Knight:** its 8 L-shaped moves = 8 actions. (A direction x distance scheme does not fit knights, so each piece type has its own small encoder.)
- A **legal-move mask** is computed every step; illegal actions are never selected.

### 3.3 Observation
- **Tabular:** integer index from the three piece squares (White king, extra piece, Black king). Optional 8-fold board-symmetry canonicalization.
- **DQN / PPO:** three one-hot planes of 64 = **192 inputs** (White king, extra piece, Black king). Same encoding for both scenarios.

### 3.4 Defender (Black)
| Mode | Used for | Behavior |
|---|---|---|
| `random` | Training (default), both scenarios | Uniform random legal move |
| `heuristic` | Both scenarios; main KN-K evaluation | Captures an unprotected White piece when possible; otherwise avoids walking into check |
| `tablebase_optimal` | **KR-K main evaluation only** | Picks the move maximizing resistance using Syzygy DTZ/WDL probes |

`tablebase_optimal` is meaningless for KN-K (every position is a draw).

### 3.5 Rewards
**KR-K (winnable)**
- Terminal: checkmate +1; stalemate -0.5; rook captured (draw) -0.5; move cap reached 0; step cost -0.01.
- **Main (shaped):** potential-based shaping, `r' = r + gamma * Phi(s') - Phi(s)`, which does not change the optimal policy (Ng, Harada, Russell 1999). `Phi` rewards driving the Black king away from the center and bringing the White king closer. Weights in config.
- **Ablation:** sparse reward (terminal terms only).

**KN-K (control)**
- Knight captured -1; otherwise 0 (no step cost, since there is nothing to finish quickly). Optional small per-step survival bonus.
- No mate reward exists. Expected mate rate: 0%.

### 3.6 Episode end
Checkmate, stalemate, piece captured, or the move cap (per scenario, start with 50; set after measuring). For KN-K, do **not** let python-chess's insufficient-material rule end the episode at move 0; use the move cap.

---

## 4. Models (one agent per scenario per algorithm)

| Algorithm | Library | Input | Model | Starting hyperparameters (untuned) |
|---|---|---|---|---|
| **Tabular Q-Learning** | NumPy | State index | Dense table | alpha 0.1, gamma 0.99, epsilon 1.0 -> 0.05, masked |
| **DQN** | PyTorch (custom, so masking is easy) | 192 one-hot | MLP 192 -> 128 -> 128 -> n_actions | Replay buffer, target network, Huber loss; masked argmax and masked target max |
| **PPO** | `sb3-contrib` `MaskablePPO` **(unverified)** + `ActionMasker` wrapper | 192 one-hot | Actor-critic MLP, 64x64 or 128x128 | gamma 0.99, clip 0.2 |

**Baselines:**
- Random White.
- Greedy 1-ply: mate if available; else avoid stalemate and a hanging piece; then minimize Black king mobility (KR-K). For KN-K, the same greedy rule with "never leave the knight capturable".
- **Tablebase-optimal White** as the KR-K ceiling.

**Fair comparison:** equal environment steps per algorithm; same defender, reward and test suite. Initial budget guess: 300k steps per agent (KR-K), fewer for KN-K; set after the speed benchmark.

---

## 5. Data (what is actually needed)

| Data | Role | Needed? | Status |
|---|---|---|---|
| **Generated positions** (random legal placements) | Training and test positions | Yes | Created by code; test suite = 500 positions per scenario, split from training by position hash |
| **Syzygy tablebases** (WDL `.rtbw`, DTZ `.rtbz`) | KR-K ground truth: win-preserving moves, distance from optimal; optimal defender | KR-K only; **3-piece files only** (KRvK; KNvK optional, all draws) | The full 3-5 piece set is about 939 MiB; the 3-piece subset is much smaller (individual size not verified). Download sources include Lichess and others; the Lichess tablebase page blocked my fetch, so check availability on your machine. `python-chess` can probe Syzygy |
| **Retrograde solver (own code)** | Exact distance-to-mate for KR-K without downloads | Optional alternative | About 80-120 LOC estimate; not built |
| **Lichess open database** (CC0) | Not needed | No | Verified to exist (puzzles, Stockfish evals); not relevant to these endgames |
| **Stockfish** | Optional stronger defender for the UI | Optional | Free engine, driven via `python-chess` (not verified here) |

Note: DTZ is distance to a zeroing move or mate; treat it as a proxy for moves-to-mate, or use an exact solver.

---

## 6. Evaluation protocol

- **Splits:** training positions sampled randomly each episode; a hash-based held-out test suite of 500 start positions per scenario, identical for every agent and seed.
- **Seeds:** 5 minimum, 10 preferred. Average over the suite within each seed, then mean and 95% bootstrap CI across seeds; pairwise comparison on identical positions.

**KR-K metrics**
- Mate rate within the move cap (main metric).
- Mean moves-to-mate versus tablebase optimum.
- Stalemate rate and rook-loss rate.
- % of moves that preserve the win (WDL probe).
- Sample efficiency: mate rate versus environment steps.
- Results versus `random` and `tablebase_optimal` defenders; scores normalized between random White (0) and tablebase-optimal White (1).

**KN-K metrics (control)**
- Mate rate (sanity check: expected 0%; investigate any nonzero value).
- **Knight survival rate** to the cap, and mean plies survived, versus the `random` and `heuristic` (capturing) defenders.
- Reward-hacking check: compare shaped vs unshaped behavior; agents should not do anything strange to farm potential.

**Both:** inference latency per move; learning curves; ablations (sparse vs shaped reward on KR-K, defender strength). No winner is claimed when a CI crosses zero.

---

## 7. UI specification

### 7.1 Features
1. **Scenario selector** (`krk`, `knk`), **model selector** (Q-Learning, DQN, PPO, Random, Greedy, Tablebase-optimal for KR-K), **defender selector**.
2. **Start position:** random, or drag pieces to set a legal custom position.
3. **Playback:** play, pause, single step, restart, speed slider.
4. **Real-time play:** each move streams to the board as it is chosen; pieces slide between squares.
5. **Compare mode:** three boards (Q-Learning, DQN, PPO) from the same start position and the same defender seed.
6. **Move panel:** move list, reward per step, shaped potential, WDL/DTZ evaluation bar (KR-K); a "Draw: no winning plan" badge for KN-K.
7. **Explainability overlay:** arrows for top-k actions with Q-values (Q-Learning, DQN) or probabilities (PPO).
8. **Training tab:** start a training job, stream live metrics into charts; load saved checkpoints for the demo.
9. **Results tab:** tables and curves from benchmark JSON, with CIs.
10. **Export:** game as PGN or animated GIF.

### 7.2 Architecture (recommended)
- **Backend:** FastAPI + WebSocket. Endpoints: `GET /scenarios`, `GET /agents`, `WS /play`, `WS /train`, `GET /results`.
- **Server game loop:** agent picks action -> environment steps -> server sends JSON `{fen, uci, action_id, q_or_probs, reward, potential, wdl, dtz}` paced by the speed slider.
- **Frontend:** single page. Board via **chessboard.js** (MIT license, verified; animated piece moves were not confirmed on its homepage, so check its docs) or a small custom CSS/JS board using CSS transitions. Charts via Chart.js.
- **Rendering helper:** `chess.svg.board()` (verified: last-move highlight and arrows) for exports and thumbnails.
- **Checkpoints:** the UI loads pre-trained models from `results/checkpoints/{scenario}/{model}/seed{n}`; training in the UI is optional and runs as a background subprocess.

### 7.3 Fallback
Streamlit + `chess.svg` redrawn per move with `st.empty()`: simpler, but no smooth piece sliding.

---

## 8. Repo layout

```
chess_endgame_rl/
  configs/scenarios.yaml
  env/            endgame_env.py, scenarios.py, defenders.py, rewards.py, encoding.py
  agents/         q_learning.py, dqn.py, ppo.py
  baselines/      random_agent.py, greedy.py, tablebase_agent.py
  eval/           suite.py, metrics.py, stats.py (bootstrap CI), run_benchmark.py
  tablebases/     (downloaded files, gitignored) + probe.py
  ui/             server.py, static/ (index.html, app.js, styles.css)
  scripts/        speed_benchmark.py, train_all.py
  results/        benchmark JSON, learning curves, checkpoints
  tests/          env rules, masks, encoding round-trip, split disjointness
  README.md  docs/viva_notes.md
```

---

## 9. Workflow additions

- **Speed benchmark first** (steps/second in python-chess); decides whether a custom move generator is needed.
- **Measure random-play mate rate on KR-K first**, to confirm how sparse the reward is.
- **Curriculum** for KR-K: start near mate, widen.
- **Symmetry canonicalization** (8-fold) to shrink the tabular state space.
- **Reproducibility:** fixed seeds, configs saved with every result, git commit in results JSON.
- **Tests:** masks never allow illegal moves; encode/decode round-trip; train/test sets disjoint; shaping telescopes correctly.
- **Viva notes:** why shaping preserves optimality, why KN-K is a draw, why masks, why equal-step budgets.

---

## 10. Timeline and effort (rough, one person)

| Block | Work | Est. |
|---|---|---|
| A | Environment, encoders, defenders, tests | 0.75 day |
| B | Baselines, tablebase probing, evaluation harness | 0.75-1 day |
| C | Three agents, masking, training scripts | 1-1.5 days |
| D | Experiments (2 scenarios x 3 agents x seeds), in background while building the UI | 1 day wall clock |
| E | UI backend + frontend | 1.5-2 days |
| F | Report, README, viva notes | 0.5-1 day |
| **Total** | | **about 4-5 working days** |

Rough size: 1,200-1,700 LOC.

**One-day cut:** KR-K only, three agents, Streamlit fallback UI, 5 seeds, no tablebase-optimal defender. The config-driven design lets KN-K be added later.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| KN-K is a draw: no win metric | Treat as a control with survival metrics; or swap in KQ-K |
| KR-K sparse reward: random play rarely mates (not yet measured) | Potential-based shaping, curriculum, sparse as ablation; measure first |
| python-chess too slow for many training steps | Speed benchmark in block A; custom KR-K/KN-K move generator if needed |
| DQN/PPO fail to learn | Report honestly; tune within an equal budget; show learning curves |
| Illegal moves | Masks everywhere |
| Tablebase files unavailable | Own retrograde solver for KR-K |
| UI latency or desync | Pre-trained checkpoints; server-side pacing; fixed message schema |
| Windows environment friction | `python-chess` is pure Python; keep dependencies pip-only; document setup |

---

## 12. Acceptance checklist

- [ ] `krk` and `knk` selectable by config and in the UI.
- [ ] 3 algorithms trained for each scenario (6 agents).
- [ ] Held-out test suite shared by all agents; tablebase ground truth for KR-K.
- [ ] 5+ seeds, 95% CIs, learning curves.
- [ ] KR-K ablations: sparse vs shaped reward, defender strength.
- [ ] KN-K control metrics (survival, 0% mate sanity check).
- [ ] UI: real-time piece movement, compare mode, results tables from saved results.
- [ ] Tests pass; README explains how to run everything.
- [ ] Limitations stated (simulated defender, tiny endgames, untuned hyperparameters, one winnable scenario).

---

## 13. Open decisions for you

1. Keep KN-K as a draw/control scenario, or swap in KQ-K as a second winnable scenario?
2. UI: FastAPI + JS board (smoother) or Streamlit (faster to build)?
3. Can you download the Syzygy 3-piece tablebases, or should I plan the own-solver route for KR-K?
4. How many days do you actually have?
