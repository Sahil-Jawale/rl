# Design notes

## Components

- `env/encoding.py` owns the 18-plane side-to-move encoding and the 4096-action representation. The existing compact KRK/KNK action encodings remain available to the legacy agents.
- `solver/tablebase.py` computes WDL and DTM for KQvK/KRvK and stores caches by endgame and solver version. Existing unversioned repository caches are read as version 1.
- `models/dueling_net.py` provides the convolutional value/advantage model. With a legal-action mask, the advantage baseline is computed over legal actions only, and illegal Q-values are excluded from selection.
- `warmstart/` lazily produces solver labels and trains a supervised policy. Cross-entropy over the optimal-move set and legal-action Q regression are distinct selectable objectives. The checkpoint records the position-level held-out state IDs.
- `agents/dueling_ddqn.py` is the new PyTorch RL implementation; `agents/dqn.py`, Q-learning, and PPO remain legacy comparison agents. DDQN selects the next action with the online network and evaluates it with the target network, using the successor legal mask.
- `agents/rl_finetuning.py` trains from a warm-start checkpoint, applies optional solver potential shaping and supervised anchor loss, evaluates on the fixed validation set, saves the best optimal-move checkpoint, and stops after repeated evaluation regressions. The shaping potential is `1 - min(DTM, DTM_max)/DTM_max` only for nonterminal attacker-winning states; it is zero for draws, attacker losses, and terminal states.
- `eval/evaluate_all.py` compares warm-start, available RL checkpoints, and a random attacker on identical held-out start positions against random and optimal defenders. It regenerates the README results table from the JSON report.

## Reproducibility and interpretation

Set Python, NumPy, and PyTorch RNG seeds through `chess_endgame_rl.seed.set_seed`. Evaluation positions are split from solver-labelled positions by the deterministic seed recorded in the checkpoint. Each results file records `PROTOCOL_VERSION`; changing the protocol requires incrementing it.

Supervised pre-training is not reinforcement learning. Report warm-start-only and warm-start-plus-RL results separately. A fine-tuned checkpoint is retained only when its held-out optimal-move rate improves. A negative or neutral result is not evidence that RL helped; use the saved metrics to compare the two checkpoints before making that claim.

The evaluator is intentionally separate from the FastAPI serving path. Training and evaluation import PyTorch; the legacy serving path does not need to load these training utilities.

## Known limitations

- Only KQvK and KRvK are solved; KRRvK and larger-material endgames are out of scope.
- Solver labels and shaping make the experiment measurable but also make the learned policy dependent on an exact oracle.
- Promotion is queen-only and the state/action encoding has no explicit underpromotion action.
- The current scripts provide configurable runs, but a complete multi-seed hyperparameter sweep and end-to-end acceptance results have not yet been produced.
