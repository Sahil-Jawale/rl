# Task: Rebuild `Sahil-Jawale/rl` after the GMAI design (exact solver + supervised warm start + legal-masked Dueling Double DQN)

> **Audience:** an autonomous coding agent working inside the repo `https://github.com/Sahil-Jawale/rl`.
> **Reference repo (read-only inspiration):** `https://github.com/Luismpso/GMAI` (MIT licence).
> **Do not copy files verbatim.** Re-implement the design in our package layout. If you do borrow any code, keep the MIT attribution.

---

## 0. Read this first

### 0.1 Problem
Our agents (Q-Learning, DQN, PPO in `chess_endgame_rl/agents/`) perform badly on chess endgames (primary scenario `krk`, plus `knk` as a zero-win sanity scenario). We want to restructure the project the way GMAI does it.

### 0.2 What GMAI actually is (verified from its README)
- A **search-free** agent: one forward pass of a **dueling Double DQN** picks the move. No tree search at inference.
- Domains: KQ vs K, KR vs K, KRR vs K (the README says KQvK is the one with published numbers; the others are on its roadmap).
- State: **18 binary 8x8 planes**, always from the **side-to-move perspective**. Planes 0-5 own P,N,B,R,Q,K; 6-11 opponent pieces; 12 side to move; 13-16 castling rights; 17 en-passant square.
- Action space: `from_sq * 64 + to_sq` = **4096** actions, promotions resolve to queen.
- **Exact endgame solver** (`tablebase.py`): retrograde analysis computing distance-to-mate (DTM) for every 3-piece state (~524,288 encodable, ~368,000 legal). Used for (a) optimal supervised labels and (b) a ground-truth evaluation metric.
- **Supervised warm start** from solver labels, then optional RL fine-tuning.
- **Potential-based reward shaping**, `F(s,s') = gamma*Phi(s') - Phi(s)`, `Phi(terminal)=0`.
- **Legal-move masking** inside the learning rule (not only at action selection).
- Reproducible evaluation (seeded) + a **performance-regression test gate** in CI.

### 0.3 Honest caveats the agent must keep in mind (do not hide these in docs)
1. **GMAI's README does not describe Monte Carlo.** Its "Monte Carlo" flavour, if any, is not in the README. Treat the exact solver as the stand-in for "ground truth value" and, if the team wants MC returns, implement them as an *optional* baseline (see section 9).
2. **GMAI's own results are modest** (checkpoint `final.pt`, 150 games/cell, seed 0): KQvK win rate vs a random defender **0.527** (79W/71D/0L), vs a "stubborn" defender **0.347**; move quality **70.0% optimal / 28.0% suboptimal / 2.0% turns a win into a draw**. Its README also states that **RL on top of the supervised warm start currently degrades performance**, so its published checkpoint is warm-start-only.
   - Consequence: "make it exactly like GMAI" gives us a *well-engineered pipeline*, not a strong player. Our acceptance targets (section 7) are therefore set at or above GMAI's published numbers, with stretch goals above them.
3. We have only read our repo's README/layout, not its code. **Phase 1 is an audit.** Do not assume which bug is ours.

---

## 1. Current repo (from README)

```
chess_endgame_rl/
  agents/        q_learning.py  dqn.py  ppo.py
  baselines/     random_agent.py  greedy_agent.py  tablebase_agent.py
  env/           scenarios.py  encoding.py  defenders.py  rewards.py  endgame_env.py
  eval/          metrics.py  suite.py
  ui/            server.py  static/{index.html,styles.css,app.js}
  train.py
configs/scenarios.yaml
results/benchmark_summary.json
scripts/  train_all.py  speed_benchmark.py  run_benchmark.py  run_server.py
tests/    test_environment.py
```
Existing features to **preserve**: config-driven scenarios, legal-action masking, scenario-specific defenders/rewards, FastAPI dashboard, pytest suite, benchmark summary metrics.

---

## 2. Target architecture (additions marked NEW)

```
chess_endgame_rl/
  env/
    encoding.py        # REWRITE: 18x8x8 side-to-move planes + 4096 action index   (Phase 2)
    endgame_env.py     # AUDIT/FIX: terminated vs truncated, perspective, legality  (Phase 1)
    rewards.py         # ADD potential-based shaping with Phi(terminal)=0           (Phase 4)
    defenders.py       # ADD 'optimal' (DTM-maximising) defender using the solver   (Phase 3)
  solver/              # NEW
    tablebase.py       # retrograde-analysis DTM solver + on-disk cache
  models/              # NEW
    dueling_net.py     # conv trunk -> value & advantage heads
  agents/
    dqn.py             # REWRITE as legal-masked Dueling Double DQN
    replay_buffer.py   # NEW: uniform (+ optional prioritized) replay
  warmstart/           # NEW
    dataset.py         # builds (state, optimal-move-set, DTM) from the solver
    train_supervised.py
  eval/
    metrics.py         # ADD solver-based move-quality metrics
    evaluate_all.py    # NEW: seeded, reproducible W/D/L report -> results/results.json
scripts/
  build_tablebase.py   # NEW
  pipeline.py          # NEW: resumable stages (solve -> warmstart -> rl -> eval)
  regression_check.py  # NEW: performance gate
tests/
  test_encoding.py  test_tablebase.py  test_dqn_target.py  test_rewards.py
  test_warmstart.py  test_regression_gate.py
```
Keep Q-Learning and PPO as comparison baselines; do not delete them.

---

## 3. Phase 1 - Audit existing code (do this before writing new code)

Write `docs/AUDIT.md` with findings. For each item below, state **PASS / FAIL / N/A with file+line evidence** and add a regression test for every FAIL.

These are *defect classes common in board-game DQN code* plus the two GMAI explicitly documents (items 1 and 2). Items 3-8 are our suggestions, not claims about GMAI:

1. **Dueling mean over legal actions only.** Advantage baseline must average over *legal* actions. Averaging over all 4096 outputs injects the mean of ~4000 never-trained outputs into every Q. (GMAI-documented.)
2. **Double-DQN target argmax restricted to legal moves.** `a* = argmax_{a in LEGAL(s')} Q_online(s',a)`; `y = r + gamma*(1-terminated)*Q_target(s',a*)`. (GMAI-documented.)
3. **Perspective flip.** In self-play/vs-defender transitions, `s'` is the state where *the opponent* moves. Decide explicitly whether the agent's next decision state is `s'` after the defender's reply (recommended for single-agent vs scripted defender: store `s'` = position after the defender's reply, from the agent's perspective). Check that Q-values are not negated/un-negated inconsistently.
4. **`terminated` vs `truncated`.** Bootstrapping must stop only on true termination (mate/stalemate/insufficient material), *not* on a move-limit truncation.
5. **Terminal reward sign and scale** consistent with the scenario (mate = +1, stalemate = 0 or small negative for the attacker, draw by move limit/repetition = 0 or small negative). A step penalty larger in magnitude than the mate bonus makes "do nothing/draw" optimal.
6. **Legal-action mask applied to exploration too** (epsilon-random picks among legal moves only) and to PPO's logits.
7. **Reachable-state sampling.** Start positions must be legal, not already-terminal, and cover the full state distribution (random placements), not a handful of fixed starts.
8. **Horizon.** Move limit must exceed the optimal DTM with margin. KRK max DTM is 32 plies (16 moves); KQK max DTM is 20 plies (10 moves) [figures from GMAI's solver results; verify with our own solver].

Deliverable: `docs/AUDIT.md` + new failing-then-passing tests. Commit separately: `audit: <summary>`.

---

## 4. Phase 2 - State and action encoding

Implement in `env/encoding.py`:

- `encode_board(board: chess.Board) -> np.ndarray[18,8,8] (float32/uint8)`
  - If black to move, **mirror the board vertically and swap colours** so the network always sees "own pieces" as planes 0-5. Mirror the move indices identically when converting actions.
  - Planes: 0-5 own P,N,B,R,Q,K; 6-11 opponent; 12 side-to-move (constant plane; with perspective flipping it is informative mainly for debugging/consistency); 13-16 castling rights (own K/Q, opp K/Q); 17 en-passant target square.
- `move_to_index(move, perspective) -> int = from_sq*64 + to_sq` (queen promotion only; assert no under-promotion is ever produced in scope).
- `index_to_move(idx, board) -> chess.Move` and `legal_mask(board) -> np.ndarray[4096] bool`.

Tests: round-trip for 10k random legal positions; colour-flip invariance (a position and its colour-mirrored twin yield identical tensors and mirrored action indices); mask count equals `board.legal_moves.count()`.

---

## 5. Phase 3 - Exact solver (`solver/tablebase.py`)

Goal: DTM (in plies) and win/draw/loss for every legal 3-piece position (KQvK, KRvK; KRRvK is 4-piece and optional/stretch). Do **not** require a Syzygy download; compute it. (We may *cross-check* against `baselines/tablebase_agent.py` / Syzygy if it is available.)

**Algorithm (retrograde analysis):**
1. Enumerate all encodable states: `(white_king_sq, black_king_sq, attacker_piece_sq, side_to_move)` = 64*64*64*2 = 524,288. Drop illegal ones (kings adjacent or equal, pieces overlapping, side not to move in check, etc.) -> ~368k legal.
2. Mark terminal states with **black to move**: checkmate -> `lost_in[state] = 0`; stalemate -> draw.
3. Iterate for `n = 0,1,2,...` until no change:
   - A **white-to-move** state is *won in n+1 plies* if some legal white move reaches a black-to-move state with `lost_in == n` (take the minimum).
   - A **black-to-move** state is *lost in n+1 plies* if it has at least one legal move, **every** legal move leads to a white-to-move state that is already won, and `n+1 = 1 + max(win_in of successors)`.
   - Black capturing the attacker's piece leads to bare kings = draw, so such states are not lost.
4. Anything never labelled is a draw.
5. Persist to a compressed `.npz`/`.npy` cache keyed by `(endgame, solver_version)`. Provide `lookup(board) -> (wdl, dtm)` and `optimal_moves(board) -> set[Move]` (moves that preserve the win and minimise DTM for the attacker).

**Validation tests (must pass):**
- KQvK: about 368k legal states, max DTM **20 plies**, "won" fraction about **93.7%** (GMAI-reported; treat as a sanity target, +/-0.5%).
- KRvK: max DTM **32 plies**.
- For every won white-to-move state, the best move strictly reduces DTM by exactly 1 ply for the optimal line; for black-to-move won states the longest defence gives `DTM-1`.
- Spot-check 20 well-known positions (e.g. back-rank mates) against `python-chess` Syzygy if available.

Runtime target: under ~2 minutes for KQvK in pure Python with `python-chess`; use vectorised NumPy or a compact custom move generator if it is slower, since solving requires many successor lookups.

---

## 5b. Defenders

Add to `env/defenders.py`:
- `random` - uniform over legal moves.
- `optimal` (a.k.a. "stubborn" in GMAI; their exact definition is not in the README, so **we define it ourselves**): chooses the defender move that **maximises DTM** (longest resistance), ties broken randomly with a seeded RNG.
- Keep existing scenario-specific defenders.

---

## 6. Phase 4 - Learning stack

### 6.1 Model (`models/dueling_net.py`)
- Input `18x8x8`. Conv trunk (e.g. 3-4 conv layers, 64-128 channels, 3x3, padding 1, BatchNorm/ReLU or GroupNorm), flatten, then
  - **Value head:** `FC -> 1`
  - **Advantage head:** `FC -> 4096`
- `forward(x, legal_mask)`:
  ```
  A_masked_mean = sum(A * mask) / sum(mask)          # legal-only mean
  Q = V + (A - A_masked_mean)
  Q[~mask] = -inf                                     # (or -1e9) for argmax safety
  ```
- Target ~7 ms/move CPU inference; no GPU needed for inference.

### 6.2 Supervised warm start (`warmstart/`)
- Dataset: all legal **attacker-to-move won** states from the solver (symmetry-reduce optionally with the 8-fold board symmetry for pawnless endgames, remembering to transform labels).
- Targets: the **set** of optimal moves per state. Loss options (implement both, ablate):
  1. Cross-entropy over legal moves with a uniform distribution over optimal moves as the soft target.
  2. Q-regression: `Q*(s,a) = +gamma^(DTM(s,a)) ` style target for each legal move, so the network learns a full value ordering (better for later RL fine-tuning).
- Hold out >=10% of states by *position* (not by symmetry copy) for validation.
- Report: top-1 optimal-move accuracy, "keeps win" accuracy, "turns win into draw" rate.

### 6.3 RL fine-tuning (`agents/dqn.py`)
- Double DQN, legal-masked target (see audit items 1-2), Huber loss, target network (hard sync every N steps or Polyak), gradient clipping, replay (uniform first, PER optional).
- Epsilon-greedy over **legal** moves only. Start epsilon low (e.g. 0.05-0.1) when initialised from the warm start; do not reset to 1.0 or you will erase the supervised policy.
- **Anti-degradation measures** (GMAI reports RL *hurts* its warm start, so build these in from day one):
  - Keep an *anchor* loss: add `lambda * CE(policy, supervised_labels)` on a rolling batch of solver-labelled states, annealed down only if evaluation keeps improving. (GMAI has an `ablate_anchor.py` script; we do not know its contents, so design this ablation ourselves.)
  - Evaluate every K steps; keep the **best** checkpoint, not the last.
  - Auto-stop if solver-based "suboptimal move rate" rises for N consecutive evals.

### 6.4 Rewards (`env/rewards.py`)
- Terminal: mate `+1`, draw/stalemate/truncation `0`, loss `-1`. Small per-step cost optional (must be << mate bonus).
- **Potential shaping:** `F(s,s') = gamma*Phi(s') - Phi(s)`, `Phi(terminal) = 0`. A good `Phi` with the solver is `Phi(s) = c * (1 - DTM(s)/DTM_max)` for won states (0 otherwise), which is policy-preserving. Add a unit test verifying the **telescoping identity** over a random trajectory to 1e-9 (GMAI does this).
- Keep a switch to disable shaping for ablations.

---

## 7. Phase 5 - Evaluation and acceptance criteria

Implement `eval/evaluate_all.py` (seeded, deterministic, writes `results/results.json`, regenerates the README results table).

**Protocol**
- For each endgame x defender in {random, optimal}: play N games (>=150 per cell, same seed, same start positions) for the **model** and for a **random-agent baseline**.
- Report **W / D / L and win rate separately.** Do **not** use `(W + 0.5*D)/n` as the headline: random games are mostly draws so it starts near 0.5 and carries no signal.
- Solver-based move quality over a fixed held-out set of attacker-to-move won positions: `% optimal`, `% suboptimal`, `% win->draw`.
- Also log mean mate length vs optimal DTM for converted wins.

**Acceptance targets**

| Metric (KQvK unless stated) | Minimum (= GMAI published) | Target | Stretch |
|---|---|---|---|
| Win rate vs random defender | >= 0.53 | >= 0.90 | >= 0.99 |
| Win rate vs optimal defender | >= 0.35 | >= 0.75 | >= 0.95 |
| Optimal-move rate | >= 70% | >= 92% | >= 98% |
| Win -> draw rate | <= 2% | <= 0.5% | ~0% |
| Random-agent baseline win rate | ~0.007 (sanity) | same | same |
| KRvK (our primary `krk`) | reproduce same pipeline, report numbers | >= 0.85 vs random | >= 0.95 |

The Target and Stretch columns are **our goals, not GMAI's reported results**. They are plausible because a supervised CNN on 368k exactly-labelled positions should fit far better than 70%, but verify empirically and report whatever is achieved honestly.

**Regression gate (`scripts/regression_check.py` + CI):** evaluate a fixed checkpoint on fixed positions with a fixed seed; fail if win rate or optimal-move rate drops below `results/baseline.json`. Support `--update` to rewrite the baseline (baseline updates go in their own commit). This catches code that runs fine but plays worse.

---

## 8. Execution plan (commit per step, run `pytest -q` after each)

1. `audit:` produce `docs/AUDIT.md`, add failing tests, fix bugs. *Exit: all audit tests green.*
2. `encoding:` Phase 2 + tests.
3. `solver:` Phase 3 + validation tests + cached tablebase script `scripts/build_tablebase.py --kind KQvK|KRvK`.
4. `defenders:` add `optimal` defender; update `scenarios.yaml` (add `kqk` scenario; keep `krk` primary, `knk` sanity).
5. `model + warmstart:` Phase 4.1/4.2. *Exit: held-out optimal-move rate reported.*
6. `eval:` Phase 5 evaluation + `results.json`; record the warm-start-only numbers as baseline.
7. `dqn:` legal-masked Dueling DDQN + shaping + anchor loss; compare against warm-start-only. **If RL does not beat warm-start-only on the solver metrics, ship warm-start-only and document it.**
8. `ci:` GitHub Actions (lint with `ruff`, tests on Python 3.10-3.12, regression gate).
9. `ui:` wire the existing FastAPI dashboard to the new checkpoint; show `q_value`, `in_scope`, and solver DTM when available.
10. `docs:` update README (scope and limitations, results table generated from `results.json`), add `docs/MODEL_CARD.md` and `docs/DESIGN.md`.

Make `scripts/pipeline.py` resumable: stages `solve -> warmstart -> rl -> eval`, each skipped if its artefact exists.

---

## 9. Optional: Monte Carlo component (only if the team needs it)

GMAI's README does not use MC, but if the course/report requires "Monte Carlo":
- **First-visit MC value estimation** on the solver-labelled domain as a *baseline* agent in `agents/monte_carlo.py`: tabular over a compact state key `(wK, bK, piece, stm)` (524k states fits in memory), epsilon-soft policy, returns from complete episodes against the random/optimal defender.
- Compare MC vs Q-Learning vs DQN vs warm-start DDQN in the benchmark table. The exact solver provides the ground truth to measure each against.
- Do **not** add MC rollouts/tree search at inference time; that would break the "search-free" property.

---

## 10. Engineering rules
- Python 3.10-3.12, `python-chess`, `torch`, `gymnasium`, `numpy`, `pytest`, `ruff`; FastAPI for serving (already present).
- Everything seeded (`random`, `numpy`, `torch`) via one `set_seed()`; the same checkpoint must always produce the same report.
- No training-only imports inside the serving path (inference should work with the CPU wheel and without Gymnasium).
- Never silently change evaluation protocol after seeing results; if you change it, bump a `PROTOCOL_VERSION` in `results.json`.
- Report negative results. If something degrades the model, write it down in `docs/POSTMORTEM.md` the way GMAI does.

## 11. Definition of done (technical; see also section 12.4 for rubric items)
- [ ] `docs/AUDIT.md` complete, all FAIL items fixed with tests
- [ ] Solver validated (KQvK 20-ply max, ~93.7% won; KRvK 32-ply max)
- [ ] Warm-start model beats GMAI's published minimums on KQvK
- [ ] KRvK pipeline reproduced end-to-end with reported numbers
- [ ] RL fine-tuning either beats warm-start on solver metrics or is documented as not helping
- [ ] `evaluate_all.py` regenerates `results/results.json` and README tables deterministically
- [ ] CI green incl. regression gate
- [ ] README states scope (endgames only), limitations, and measured results honestly

---

## 12. Rubric alignment (25-mark RL mini project)

The project is graded on: **(1) Problem definition & RL formulation - 5**, **(2) Implementation & working of the RL agent - 10**, **(3) Results, analysis & viva - 10**. Everything below exists to make each rubric line visibly satisfied. Produce the artefacts in `docs/` and `results/` as listed.

### 12.1 Problem definition & RL formulation (5 marks) -> `docs/PROBLEM.md`

**Problem statement.** Train an agent that converts won chess endgames (K+R vs K primary; K+Q vs K secondary) into checkmate against a defending side, using only reinforcement learning signals (win/draw/loss and shaped rewards), and measure it against exact ground truth.

**Real-world relevance (write honestly, do not overclaim):**
- Endgame conversion is a clean, fully observable sequential decision problem with a sparse terminal reward, which is the standard hard case for credit assignment.
- Exact endgame tablebases exist, which lets us measure *how close to optimal* an RL agent gets. That is rarely possible in real-world RL.
- The same pipeline (sparse reward, large discrete action space with illegal actions masked, reward shaping that preserves the optimal policy) applies to domains where an exact solver is unavailable, e.g. scheduling, routing, and other constrained combinatorial control tasks.

**Justification for using RL (must be explicit, because a solver exists).** A tablebase already solves these endgames, so the project does not claim RL is the best tool *for chess endgames*. The justification is: (a) the solver is a ground-truth oracle that makes RL's behaviour measurable, (b) tablebases do not scale to larger material or to domains without a perfect model, while a learned value function generalises, (c) the sparse terminal reward makes it a meaningful test of exploration, credit assignment and reward shaping. State this limitation in the report instead of hiding it.

**Formulation table (copy into README and report):**

| Element | Definition in this project |
|---|---|
| Environment | Chess endgame episode (`endgame_env.py`), single learning agent (attacker) vs a scripted defender (`random` or `optimal`); deterministic transitions given the defender policy; random legal start positions; episode ends on checkmate, stalemate, insufficient material, or move limit. |
| State | 18x8x8 binary planes from the side-to-move perspective (own pieces, opponent pieces, side to move, castling, en passant). Fully observable. |
| Action | One of 4096 `from_sq*64+to_sq` indices; only legal moves are selectable (mask); promotions resolve to queen. |
| Reward | Terminal: mate +1, loss -1, draw/stalemate/truncation 0. Optional potential-based shaping `gamma*Phi(s') - Phi(s)` with `Phi(terminal)=0` (policy-preserving). State the exact `Phi` used. |
| Discount | `gamma` stated in config (record the value used). |
| Episode horizon | Move limit chosen above the max optimal DTM with margin (verify with the solver). |

### 12.2 Implementation & working (10 marks)

**Algorithm selection (document the reasoning in `docs/DESIGN.md`):**
- Tabular Q-Learning: baseline; only feasible because 3-piece state space is small (~368k legal states).
- DQN / Dueling Double DQN: function approximation, scales beyond tabular, legal-masked targets.
- PPO: policy-gradient comparison with masked logits.
- (Optional) First-visit Monte Carlo: see section 9.
- Supervised warm start from the solver: **label it clearly as supervised pre-training, not RL.** The RL credit must come from the RL stages. Always report RL-from-scratch results *and* RL-after-warm-start results separately, so it is clear what RL itself achieves.

**Training process (log and report):** episodes/steps, replay size, batch size, target-network sync, epsilon schedule, seeds, wall-clock time, hardware.

**Parameter selection -> `docs/HYPERPARAMETERS.md`.** For every algorithm provide a table of final values *and* how they were chosen. Minimum sweeps (>=3 seeds each, mean +/- std):

| Parameter | Values to try | Reported as |
|---|---|---|
| Learning rate | 1e-3, 3e-4, 1e-4 | final win rate vs optimal defender |
| Gamma | 0.95, 0.99 | win rate, mean mate length |
| Epsilon (start/end/decay) | from-scratch vs low-epsilon warm-start | learning curve |
| Target sync / Polyak tau | 1k, 5k steps or tau 0.005 | stability (variance of eval reward) |
| Reward shaping | on / off | sample efficiency (steps to X% win rate) |
| Anchor loss weight (warm start) | 0, 0.1, 1.0 | does RL degrade the warm start? |
| Replay | uniform vs prioritized (optional) | learning speed |

Record every run in `results/runs/<run_id>/config.json` + `metrics.csv` so tables and plots are regenerated by script, never hand-edited.

**Working agent & demonstration:**
- Keep the FastAPI/browser UI: user picks a scenario, loads a checkpoint, steps through a game, and sees chosen move, Q-value, and (when solver available) DTM and whether the move was optimal.
- Add `scripts/demo.py` (CLI) that plays N games, prints a W/D/L summary, and saves 3 annotated PGNs (a clean win, a slow win, a failure).
- Provide a 2-minute demo script in `docs/DEMO.md` (exact commands, expected output).

### 12.3 Results, analysis & viva (10 marks)

**Required plots (`results/plots/`, generated by `scripts/make_plots.py` from `results/runs/`):**
1. Training curve: eval win rate vs training steps, per algorithm, mean +/- std over seeds.
2. Training curve: solver-based optimal-move rate vs steps (for DQN variants).
3. Bar chart: final W/D/L per algorithm per defender (random, optimal).
4. Shaping on/off and warm-start on/off comparison curves.
5. Mate-length histogram: agent vs optimal DTM for converted wins.
6. Loss/TD-error curve for the DQN (stability evidence).
7. Failure analysis: heat-map or list of the board configurations where the agent turns a win into a draw (e.g. stalemate traps, repetition/move-limit).

**Required comparison table (fill with real measured numbers; no placeholders in the final report):**

| Agent | Scenario | Defender | W | D | L | Win rate | Optimal-move % | Mean mate length / optimal |
|---|---|---|---|---|---|---|---|---|
| Random baseline | krk | random | | | | | | |
| Greedy baseline | krk | random | | | | | | |
| Q-Learning | krk | random / optimal | | | | | | |
| DQN (plain) | krk | random / optimal | | | | | | |
| Dueling DDQN (RL only) | krk | random / optimal | | | | | | |
| Warm start only (supervised) | krk | random / optimal | | | | | | |
| Warm start + RL fine-tune | krk | random / optimal | | | | | | |
| PPO | krk | random / optimal | | | | | | |
| Tablebase (upper bound) | krk | random / optimal | | | | | | |

**Interpretation section must answer:** which algorithm wins and why; what the sparse reward does to learning speed; whether shaping helped (with numbers); whether RL improved on the warm start or degraded it; why the failures happen (draw conversions, move limit, stalemate); and what would change with a longer horizon or bigger material.

**Report (`docs/REPORT.md`, export to PDF) outline:**
1. Problem statement & relevance, and RL justification (12.1)
2. MDP formulation table
3. Method: environment, solver, algorithms, warm start, shaping
4. Experimental setup: hardware, seeds, hyperparameters
5. Results: tables and plots from above
6. Analysis, ablations and failure cases
7. Limitations (scope is endgames only; solver dependence; any negative results)
8. Conclusion and future work
9. References (DQN, Double DQN, Dueling networks, PPO, reward-shaping theory, GMAI repo)

**Viva preparation -> `docs/VIVA.md`.** The agent must draft concise answers, grounded in *our actual results*, to at least these questions (the team must review and own every answer):
1. Why RL when a tablebase solves this exactly? (see 12.1)
2. Define your state, action, reward and environment. Why this reward?
3. Why Dueling Double DQN instead of plain DQN? What does the legal-only mean fix?
4. Why is the target argmax restricted to legal moves?
5. What is potential-based shaping and why does it not change the optimal policy?
6. What does the supervised warm start do and is it still "RL"? How did you separate their contributions?
7. How did you choose hyperparameters? Show the sweep.
8. Why did Q-Learning/PPO/DQN fail or succeed? What were the failure modes?
9. How do you know the agent is good? (solver-based metrics, not just win rate; why `(W+0.5D)/n` is misleading)
10. What is `terminated` vs `truncated` and why does bootstrapping depend on it?
11. What are the limitations (endgames only, 4096 actions, queen-only promotion, solver needed for labels)?
12. What would you do to scale this to KRRvK or more pieces?

### 12.4 Additional acceptance items for the rubric
- [ ] `docs/PROBLEM.md` with problem statement, relevance, explicit RL justification, formulation table
- [ ] `docs/HYPERPARAMETERS.md` with sweeps (>=3 seeds) and final values
- [ ] All 7 plots generated by script from logged runs
- [ ] Comparison table fully populated with measured numbers
- [ ] RL-only and warm-start results reported separately
- [ ] `docs/DEMO.md` + `scripts/demo.py` working end to end
- [ ] `docs/REPORT.md` (+ PDF) following the outline
- [ ] `docs/VIVA.md` reviewed by every team member
